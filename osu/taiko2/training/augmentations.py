"""Concrete augmentations — exp 45's set, ported to our framework.

Every aug subclasses `PostSampleAugmentation[TaikoDetectionSample]`.
None of them are pre-sample: exp 45 didn't shift the cursor during
training (cursor-offset augs were an exp 44 sibling).

Contract reminder:
  - The sample is a frozen dataclass → each aug returns a NEW sample
    via `dataclasses.replace` when it changes fields.
  - The arrays on the sample (audio + masks) are numpy; augs are free
    to mutate copies but never the originals (the sample is shared
    reference across the pipeline's augmentations — immutability
    matters).
  - Each aug applies with probability `prob`; at rate 0 they're no-ops.
  - Random state is owned per-augmentation (seed=None → non-seeded,
    per-process entropy).

Quick reference of rates used by exp 45:

    MelGainJitter            30%  ±2 dB
    MelGaussianNoise         15%  σ ∈ [0.1, 0.3]
    MelFreqJitter            15%  shift ∈ [-3, +3]
    SpecAugFreq              20%  1 mask, ≤10 bands
    SpecAugTime              20%  1 mask, ≤30 frames
    EventJitter             100%  global ±3, recency 1-2×
    EventDropout              5%  drop 1-2 events
    EventInsertion            3%  add 1 synthetic between two reals
    PartialMetronome          2%  recent half → evenly spaced
    PartialAdvMetronome       2%  older half → dominant-gap spaced
    LargeTimeShift            2%  ±50 shift on 2-4 recent events
    ContextTruncation         5%  keep 8-32 most recent
    ConditioningJitter       10%  ±2% on density trio
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from ..data_samplers.detection import TaikoDetectionPreContext, TaikoDetectionSample
from ..domain.augmentation import PostSampleAugmentation, PreSampleAugmentation
from ..domain.beatmap import RelativeOnset


# ─────────────────────────── base ────────────────────────────────────

@dataclass
class _RngAug(PostSampleAugmentation[TaikoDetectionSample]):
    """Shared plumbing for augs that need a per-instance `random.Random`
    and a probability gate. Subclasses override `_apply(sample, rng)`.

    Stateful (not frozen) because the RNG instance is mutable on draw.
    """
    prob: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {self.prob}")
        self._rng = random.Random(self.seed)

    def apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        if self.prob < 1.0 and self._rng.random() >= self.prob:
            return sample
        return self._apply(sample)

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        raise NotImplementedError


# ─────────────────────────── audio augs ───────────────────────────────

@dataclass
class MelGainJitter(_RngAug):
    """Add a uniform-random dB offset to the whole mel window."""
    range_db: float = 2.0

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        delta = self._rng.uniform(-self.range_db, self.range_db)
        return replace(
            sample,
            audio_past=sample.audio_past + delta,
            audio_future=sample.audio_future + delta,
        )


@dataclass
class MelGaussianNoise(_RngAug):
    """Additive Gaussian noise on the mel values. Std drawn uniformly
    from ``[min_std, max_std]`` per call."""
    min_std: float = 0.1
    max_std: float = 0.3

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        std = self._rng.uniform(self.min_std, self.max_std)
        # Pull a numpy Generator seeded from the python rng so tests are
        # deterministic when `seed` is set.
        ng = np.random.default_rng(self._rng.randint(0, 2**31 - 1))
        past = sample.audio_past + ng.normal(0, std, size=sample.audio_past.shape).astype(np.float32)
        fut = sample.audio_future + ng.normal(0, std, size=sample.audio_future.shape).astype(np.float32)
        return replace(sample, audio_past=past, audio_future=fut)


@dataclass
class MelFreqJitter(_RngAug):
    """Roll the mel bands by a small integer offset. Bins that shift
    off the edge wrap — the model learns small translations don't
    change which onset is which."""
    max_shift: int = 3

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        shift = self._rng.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return sample
        # Roll only the mel rows (first 80); leave coincidence rows
        # (81+) untouched so the two feature spaces don't cross.
        n_mel = 80
        past = sample.audio_past.copy()
        fut = sample.audio_future.copy()
        past[:n_mel] = np.roll(past[:n_mel], shift, axis=0)
        fut[:n_mel] = np.roll(fut[:n_mel], shift, axis=0)
        return replace(sample, audio_past=past, audio_future=fut)


@dataclass
class SpecAugFreq(_RngAug):
    """Zero out a horizontal (freq) band of the mel window."""
    max_bands: int = 10

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        n_bands = sample.audio_past.shape[0]
        width = self._rng.randint(1, max(1, min(self.max_bands, n_bands)))
        start = self._rng.randint(0, max(0, n_bands - width))
        past = sample.audio_past.copy()
        fut = sample.audio_future.copy()
        past[start:start + width, :] = 0.0
        fut[start:start + width, :] = 0.0
        return replace(sample, audio_past=past, audio_future=fut)


@dataclass
class SpecAugTime(_RngAug):
    """Zero out a vertical (time) column on one side of the cursor.

    Chooses past or future with 50/50 probability — across many
    samples both sides get equal coverage.
    """
    max_frames: int = 30

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        side_past = self._rng.random() < 0.5
        arr = sample.audio_past if side_past else sample.audio_future
        n_frames = arr.shape[1]
        width = self._rng.randint(1, max(1, min(self.max_frames, n_frames)))
        start = self._rng.randint(0, max(0, n_frames - width))
        new = arr.copy()
        new[:, start:start + width] = 0.0
        if side_past:
            return replace(sample, audio_past=new)
        return replace(sample, audio_future=new)


# ─────────────────────────── event augs ───────────────────────────────

def _collect_real(
    sample: TaikoDetectionSample,
) -> tuple[list[RelativeOnset], list[int]]:
    """Return `(ordered_real_events, their_slot_indices)` from the
    padded past-events array — helpers for event augs."""
    reals: list[RelativeOnset] = []
    idxs: list[int] = []
    for i, padded in enumerate(sample.past_events_mask):
        if not bool(padded):
            reals.append(sample.past_events[i])
            idxs.append(i)
    return reals, idxs


def _repack_past(
    sample: TaikoDetectionSample,
    new_reals: list[RelativeOnset],
) -> TaikoDetectionSample:
    """Rebuild `past_events` (oldest-first, back-aligned so the newest
    real is at index c_events-1) + mask from a fresh list of real events.
    Mirrors the sampler's `_extract_events(pad_at_start=True, ...)` layout.
    """
    c = len(sample.past_events)
    if len(new_reals) > c:
        new_reals = new_reals[-c:]
    n = len(new_reals)

    placeholder_kind = sample.past_events[0].kind
    placeholder = RelativeOnset(
        time_ms=0, kind=placeholder_kind, bin=0, cursor_offset=0,
    )
    slots: list[RelativeOnset] = []
    mask = np.ones(c, dtype=bool)          # True = padded
    if n > 0:
        pad_n = c - n
        slots.extend([placeholder] * pad_n)
        slots.extend(new_reals)
        mask[pad_n:] = False
    else:
        slots = [placeholder] * c
    return replace(
        sample,
        past_events=tuple(slots),
        past_events_mask=mask,
    )


@dataclass
class EventJitter(_RngAug):
    """Global shift + recency-scaled per-event jitter.

    Per call:
      - `global_shift ∈ {-global_max, …, +global_max}` added to every
        past event.
      - Per-event integer noise in `{-event_max, …, +event_max}` scaled
        linearly from `recency_scale[0]` at the oldest real event to
        `recency_scale[1]` at the newest. Simulates AR-error: recent
        predictions noisier than old GT.
    """
    global_max: int = 3
    event_max: int = 3
    recency_scale: tuple[float, float] = (1.0, 2.0)

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        reals, _ = _collect_real(sample)
        if not reals:
            return sample
        n = len(reals)
        shift = self._rng.randint(-self.global_max, self.global_max)
        scales = np.linspace(self.recency_scale[0], self.recency_scale[1], n)

        new_reals: list[RelativeOnset] = []
        for event, scale in zip(reals, scales):
            per_event = int(round(
                self._rng.randint(-self.event_max, self.event_max) * scale
            ))
            new_off = int(event.cursor_offset) + shift + per_event
            new_bin = int(event.bin) + shift + per_event
            new_reals.append(RelativeOnset(
                time_ms=event.time_ms,
                kind=event.kind,
                bin=new_bin,
                cursor_offset=new_off,
            ))
        # Keep sorted by cursor_offset (jitter can violate order).
        new_reals.sort(key=lambda o: o.cursor_offset)
        return _repack_past(sample, new_reals)


@dataclass
class EventDropout(_RngAug):
    """Drop 1-2 random past events to simulate AR skip errors."""
    drop_min: int = 1
    drop_max: int = 2

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        reals, _ = _collect_real(sample)
        if len(reals) <= self.drop_max + 1:
            return sample
        n_drop = self._rng.randint(self.drop_min, self.drop_max)
        if n_drop >= len(reals):
            return sample
        drop_idxs = set(self._rng.sample(range(len(reals)), n_drop))
        new_reals = [r for i, r in enumerate(reals) if i not in drop_idxs]
        return _repack_past(sample, new_reals)


@dataclass
class EventInsertion(_RngAug):
    """Insert one synthetic event between two existing reals. Simulates
    AR hallucination."""

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        reals, _ = _collect_real(sample)
        if len(reals) < 2:
            return sample
        oldest_off = reals[0].cursor_offset
        newest_off = reals[-1].cursor_offset
        if newest_off - oldest_off <= 1:
            return sample
        lo = int(oldest_off) + 1
        hi = int(newest_off) - 1
        if lo >= hi:
            return sample
        new_off = self._rng.randint(lo, hi)
        # Infer absolute bin: cursor = sample.cursor_bin; bin = cursor + offset
        new_bin = sample.cursor_bin + new_off
        synthetic = RelativeOnset(
            time_ms=0,  # unknown — augmentation is fake data
            kind=reals[0].kind,
            bin=new_bin,
            cursor_offset=new_off,
        )
        merged = sorted(reals + [synthetic], key=lambda o: o.cursor_offset)
        return _repack_past(sample, merged)


@dataclass
class PartialMetronome(_RngAug):
    """Replace the recent half of the event list with an evenly-spaced
    metronomic sequence. Tests whether the model can distinguish real
    music from metronomic corruption."""
    gap_min: int = 10
    gap_max: int = 80

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        reals, _ = _collect_real(sample)
        if len(reals) < 8:
            return sample
        half = len(reals) // 2
        gap = self._rng.randint(self.gap_min, self.gap_max)
        # Recent half: place at offsets -gap*(half-i-1) through 0 relative to last event
        newest_off = int(reals[-1].cursor_offset)
        new_half = [
            RelativeOnset(
                time_ms=0, kind=reals[-1].kind,
                bin=sample.cursor_bin + newest_off - gap * (half - 1 - i),
                cursor_offset=newest_off - gap * (half - 1 - i),
            )
            for i in range(half)
        ]
        merged = list(reals[:-half]) + new_half
        merged.sort(key=lambda o: o.cursor_offset)
        return _repack_past(sample, merged)


@dataclass
class PartialAdvMetronome(_RngAug):
    """Replace the older half with events spaced at the sample's
    dominant gap (modal inter-onset interval, rounded to a multiple of
    ``gap_quant``)."""
    gap_quant: int = 3
    jitter: int = 1

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        reals, _ = _collect_real(sample)
        if len(reals) < 8:
            return sample
        offsets = [int(r.cursor_offset) for r in reals]
        gaps = np.diff(offsets)
        gaps = gaps[gaps > 0]
        if len(gaps) < 2:
            return sample
        quant = (gaps // self.gap_quant) * self.gap_quant
        dominant = Counter(quant.tolist()).most_common(1)[0][0]
        dominant = max(5, int(dominant))
        half = len(reals) // 2
        base_off = int(reals[0].cursor_offset)
        new_half: list[RelativeOnset] = []
        for i in range(half):
            jit = self._rng.randint(-self.jitter, self.jitter)
            off = base_off + (dominant + jit) * i
            new_half.append(RelativeOnset(
                time_ms=0, kind=reals[0].kind,
                bin=sample.cursor_bin + off, cursor_offset=off,
            ))
        merged = new_half + list(reals[half:])
        merged.sort(key=lambda o: o.cursor_offset)
        return _repack_past(sample, merged)


@dataclass
class LargeTimeShift(_RngAug):
    """Shift the 2-4 most recent events by ±N bins — simulates a large
    AR timing error on the latest few predictions."""
    max_shift: int = 50
    n_min: int = 2
    n_max: int = 4

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        reals, _ = _collect_real(sample)
        if len(reals) <= self.n_min:
            return sample
        n_shift = self._rng.randint(self.n_min, min(self.n_max, len(reals)))
        shift = self._rng.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return sample
        keep = list(reals[:-n_shift])
        shifted = [
            RelativeOnset(
                time_ms=o.time_ms, kind=o.kind,
                bin=int(o.bin) + shift,
                cursor_offset=int(o.cursor_offset) + shift,
            )
            for o in reals[-n_shift:]
        ]
        merged = sorted(keep + shifted, key=lambda o: o.cursor_offset)
        return _repack_past(sample, merged)


@dataclass
class ContextTruncation(_RngAug):
    """Keep only the most recent `keep_min..keep_max` real events;
    the rest become padding."""
    keep_min: int = 8
    keep_max: int = 32

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        reals, _ = _collect_real(sample)
        if len(reals) <= self.keep_min:
            return sample
        keep = self._rng.randint(self.keep_min, min(self.keep_max, len(reals)))
        new_reals = reals[-keep:]
        return _repack_past(sample, new_reals)


# ─────────────────────────── conditioning ─────────────────────────────

@dataclass
class ConditioningJitter(_RngAug):
    """Multiply each of the three density fields by an independent
    uniform-random factor in ``[1 - pct, 1 + pct]``."""
    pct: float = 0.02

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        f1 = self._rng.uniform(1 - self.pct, 1 + self.pct)
        f2 = self._rng.uniform(1 - self.pct, 1 + self.pct)
        f3 = self._rng.uniform(1 - self.pct, 1 + self.pct)
        return replace(
            sample,
            density_mean=float(sample.density_mean * f1),
            density_peak=int(round(sample.density_peak * f2)),
            density_std=float(sample.density_std * f3),
        )


# ─────────────────────────── cursor shift (pre-sample) ───────────────

@dataclass
class CursorShift(PreSampleAugmentation[TaikoDetectionPreContext]):
    """Shift the cursor forward by a random amount between the current
    event and the next, so the model sees non-zero offsets during
    training.

    At inference, after a STOP hop the cursor isn't at an event
    boundary — the model needs training examples of this. 30%
    probability by default (matching taiko1 exp 67).

    Because this is a pre-sample augmentation, the shifted cursor
    drives audio extraction and event-offset computation in the
    sampler — no post-hoc audio rolling needed.
    """
    prob: float = 0.30
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def apply(self, ctx: TaikoDetectionPreContext) -> TaikoDetectionPreContext:
        if self._rng.random() >= self.prob:
            return ctx
        ei = ctx.event_idx
        bins = ctx.event_bins
        # Shift forward: random position between current cursor and
        # the next event (the target). If no next event exists, skip.
        if ei >= len(bins):
            return ctx
        next_bin = int(bins[ei])
        gap = next_bin - ctx.cursor_bin
        if gap <= 1:
            return ctx
        shift = self._rng.randint(1, gap - 1)
        return replace(ctx, cursor_bin=ctx.cursor_bin + shift)


# ─────────────────────────── time stretch ────────────────────────────

@dataclass
class TimeStretch(_RngAug):
    """Rescale audio + past/future events along the time axis by a
    per-call factor ``s`` drawn log-uniformly from ``[1/max_scale,
    max_scale]``.

    Semantics: the cursor stays pinned at its original frame index.
    Every other frame / event is pulled towards or pushed away from the
    cursor by factor ``s``. ``s > 1`` expands (slow down content);
    ``s < 1`` compresses (speed up content).

    Audio stretch is mel-frame resampling only — we linearly interpolate
    along the time axis of the already-computed log-mel. This is a valid
    proxy for real audio time-stretch at small factors (the MIR
    literature uses ~±40% mel-frame stretch for data augmentation); it
    stops being realistic at extreme factors because real slowdown /
    speedup shifts frequency content in ways mel-frame resampling does
    not capture. Default ``max_scale = 1.4`` caps us in the realistic
    regime.

    When ``s < 1`` the new window requires source frames outside the
    original ``[0, a_bins + b_bins)`` range — those positions get
    zero-padded, matching the chart-edge padding the model already
    sees at song boundaries.

    Events are rescaled in cursor-relative space (``cursor_offset`` is
    multiplied by ``s``, then rounded). ``time_ms`` and ``bin`` on each
    ``RelativeOnset`` are reset to be consistent with the new offset
    treating the cursor as t=0 — absolute chart positions are not
    meaningful after stretching. Past events whose new offset falls
    outside ``[-a_bins, 0]`` are masked as padding. Future-event target
    derivation (including STOP flips when an in-range onset stretches
    past ``b_pred`` or vice versa) happens downstream in the adapter
    from the updated ``cursor_offset`` — we do not need to change
    ``future_events_mask`` to make STOP derivation work.

    Collisions: when ``s < 1`` can round two originally-distinct events
    onto the same cursor offset, we dedupe by keeping the older event
    (earliest in the oldest-first sequence). The past-event list is
    then re-packed with padding at the start so its length stays
    ``c_events``.
    """
    max_scale: float = 1.4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_scale <= 1.0:
            raise ValueError(
                f"max_scale must be > 1.0, got {self.max_scale}"
            )

    def _apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        log_max = math.log(self.max_scale)
        s = math.exp(self._rng.uniform(-log_max, log_max))

        new_past_audio, new_future_audio = self._stretch_audio(
            sample.audio_past, sample.audio_future, s,
        )
        new_past_events, new_past_mask = self._stretch_past_events(
            sample.past_events, sample.past_events_mask,
            s, a_bins=sample.audio_past.shape[1],
        )
        new_future_events, new_future_mask = self._stretch_future_events(
            sample.future_events, sample.future_events_mask, s,
        )
        return replace(
            sample,
            audio_past=new_past_audio,
            audio_future=new_future_audio,
            past_events=new_past_events,
            past_events_mask=new_past_mask,
            future_events=new_future_events,
            future_events_mask=new_future_mask,
        )

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _stretch_audio(
        audio_past: np.ndarray, audio_future: np.ndarray, s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_bins = audio_past.shape[1]
        b_bins = audio_future.shape[1]
        total = a_bins + b_bins
        if s == 1.0:
            return audio_past.copy(), audio_future.copy()
        full = np.concatenate([audio_past, audio_future], axis=1)       # (F, total)
        F = full.shape[0]
        out_idx = np.arange(total, dtype=np.float32)
        src_idx = a_bins + (out_idx - a_bins) / s                        # cursor pinned at a_bins
        in_range = (src_idx >= 0.0) & (src_idx <= total - 1)
        clipped = np.clip(src_idx, 0.0, total - 1).astype(np.float32)
        lo = np.floor(clipped).astype(np.int64)
        hi = np.minimum(lo + 1, total - 1)
        frac = (clipped - lo).astype(np.float32)
        out = full[:, lo] * (1.0 - frac) + full[:, hi] * frac            # (F, total)
        if not in_range.all():
            out[:, ~in_range] = 0.0
        return (
            np.ascontiguousarray(out[:, :a_bins]),
            np.ascontiguousarray(out[:, a_bins:]),
        )

    @staticmethod
    def _scale_offset(offset: int, s: float) -> int:
        return int(round(offset * s))

    @staticmethod
    def _rebuild_onset(
        old: RelativeOnset, new_offset: int,
    ) -> RelativeOnset:
        # time_ms + bin are set to be self-consistent with the new offset
        # (cursor treated as t=0). Absolute chart positions are lost on
        # stretch; the adapter only reads cursor_offset anyway.
        return replace(
            old,
            cursor_offset=new_offset,
            bin=new_offset,
            time_ms=int(round(new_offset * 5.0)),
        )

    @classmethod
    def _stretch_past_events(
        cls,
        events: tuple[RelativeOnset, ...],
        mask: np.ndarray,
        s: float,
        a_bins: int,
    ) -> tuple[tuple[RelativeOnset, ...], np.ndarray]:
        # Collect real (non-padded) events with their new offsets.
        real: list[RelativeOnset] = []
        for e, m in zip(events, mask):
            if bool(m):
                continue
            new_off = cls._scale_offset(e.cursor_offset, s)
            if new_off > 0 or new_off <= -a_bins:
                # Fell outside the past window after scaling → drop.
                continue
            real.append(cls._rebuild_onset(e, new_off))
        # Dedupe: events with identical new offsets collapse to one;
        # keep the OLDER (earlier in the oldest-first sequence) because
        # its kind is the "first to be heard" in real-time order.
        deduped: list[RelativeOnset] = []
        seen: set[int] = set()
        for e in real:
            if e.cursor_offset in seen:
                continue
            seen.add(e.cursor_offset)
            deduped.append(e)
        # Pad at start (oldest-first convention).
        c_events = len(events)
        pad_n = c_events - len(deduped)
        if pad_n < 0:
            # More surviving events than slots — keep the newest c_events.
            deduped = deduped[-c_events:]
            pad_n = 0
        padding = [cls._rebuild_onset(events[0], 0)] * pad_n
        new_events = tuple(padding + deduped)
        new_mask = np.array(
            [True] * pad_n + [False] * len(deduped),
            dtype=bool,
        )
        return new_events, new_mask

    @classmethod
    def _stretch_future_events(
        cls,
        events: tuple[RelativeOnset, ...],
        mask: np.ndarray,
        s: float,
    ) -> tuple[tuple[RelativeOnset, ...], np.ndarray]:
        # We do NOT change the mask here — the adapter derives STOP
        # from the combination of mask AND offset-vs-b_pred, so a
        # pre-stretch in-range onset that stretches past b_pred will
        # correctly flip to STOP in the target without us touching the
        # mask. Out-of-order events (new_offset < 0, can't happen for
        # s > 0) are the only case we'd need to handle and we don't.
        new_events = tuple(
            cls._rebuild_onset(e, cls._scale_offset(e.cursor_offset, s))
            for e in events
        )
        return new_events, np.array(mask, dtype=bool, copy=True)


# ─────────────────────────── exp 45 bundle ────────────────────────────

def build_exp45_post_augs(*, seed: int | None = None) -> list[PostSampleAugmentation]:
    """The exact augmentation list used by exp 45, in canonical order.

    Audio augs run first (they don't depend on event layout), then
    event augs, then conditioning jitter last.
    """
    rng = random.Random(seed)
    def _s() -> int:
        return rng.randint(0, 2**31 - 1)

    return [
        # Audio
        MelGainJitter(prob=0.30, range_db=2.0, seed=_s()),
        MelGaussianNoise(prob=0.15, min_std=0.1, max_std=0.3, seed=_s()),
        MelFreqJitter(prob=0.15, max_shift=3, seed=_s()),
        SpecAugFreq(prob=0.20, max_bands=10, seed=_s()),
        SpecAugTime(prob=0.20, max_frames=30, seed=_s()),
        # Events
        EventJitter(prob=1.00, global_max=3, event_max=3,
                    recency_scale=(1.0, 2.0), seed=_s()),
        EventDropout(prob=0.05, drop_min=1, drop_max=2, seed=_s()),
        EventInsertion(prob=0.03, seed=_s()),
        PartialMetronome(prob=0.02, seed=_s()),
        PartialAdvMetronome(prob=0.02, seed=_s()),
        LargeTimeShift(prob=0.02, max_shift=50, seed=_s()),
        ContextTruncation(prob=0.05, keep_min=8, keep_max=32, seed=_s()),
        # Conditioning
        ConditioningJitter(prob=0.10, pct=0.02, seed=_s()),
    ]
