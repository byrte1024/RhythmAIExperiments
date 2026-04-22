"""Detection DataSampler: yields audio windows + relative-onset sequences.

Ports the cursor-placement scheme from the old `detection_train.OnsetDataset`
— one sample per `(chart_index, event_index)` pair, cursor anchored at the
previous onset — but drops the next-token target entirely. Future events
are first-class payload (`D_EVENTS` of them), not a target to predict.

Two entry points:
  - `raw_sample(n)`    — deterministic, no perturbation.
  - `augment_sample(n)` — runs the same raw sample through a configurable
                          augmentation pipeline (currently a no-op stub;
                          fill in when we design augmentations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..domain.beatmap import OnsetKind, RelativeOnset
from ..domain.sampling import DataSample, DataSampler, DataSamplerConfig
from ..persistence.manifest import load_manifest
from ..persistence.events import _KIND_ORDER  # shared kind_id → OnsetKind map
from ..splits import chart_ids_for_split


# ─────────────────────────── config ───────────────────────────────────

@dataclass(frozen=True, slots=True)
class TaikoDetectionSamplerConfig(DataSamplerConfig):
    """Config for `TaikoDetectionSampler`.

    Geometric fields (`*_BINS`, `*_EVENTS`) match the old taiko naming:
      - `a_bins` past-audio bins.
      - `b_bins` future-audio bins.
      - `c_events` past events fetched (newest at the end, padded at the start).
      - `d_events` future events fetched (nearest first, padded at the end).

    `min_cursor_bin` skips early-song cursors whose left audio window would
    pad heavily with zeros; matches taiko1's 6000-bin (~30 s) warmup skip.
    """
    dataset_root: Path = field(default=Path("."))
    a_bins: int = 500
    b_bins: int = 500
    c_events: int = 128
    d_events: int = 1
    min_cursor_bin: int = 6000
    # Train/val split. `split ∈ {"all", "train", "val"}`. Splitting is by
    # beatmapset_id — all difficulties of a song land in the same split —
    # and seeded so identical (val_ratio, seed) pairs reproduce identically
    # across runs.
    split: str = "all"
    val_ratio: float = 0.1
    split_seed: int = 42


# ─────────────────────────── sample payload ───────────────────────────

@dataclass(frozen=True, slots=True)
class TaikoDetectionSample(DataSample):
    """One training/eval sample.

    `audio_past` and `audio_future` are kept as **separate** arrays split at
    the cursor frame so downstream code sees the asymmetry explicitly.
    Concatenate with `np.concatenate([past, future], axis=1)` if a combined
    `(F, A+B)` window is needed.

    `past_events` is padded at the **start** (oldest-first convention),
    `future_events` is padded at the **end** (nearest-first). Masks are
    True where a slot is padded.
    """
    chart_id: str
    cursor_bin: int
    audio_past: np.ndarray               # (F, a_bins) float32
    audio_future: np.ndarray             # (F, b_bins) float32
    past_events: tuple[RelativeOnset, ...]   # len == c_events
    past_events_mask: np.ndarray         # (c_events,) bool, True = padded
    future_events: tuple[RelativeOnset, ...]  # len == d_events
    future_events_mask: np.ndarray       # (d_events,) bool, True = padded


# ─────────────────────────── sampler ──────────────────────────────────

class TaikoDetectionSampler(
    DataSampler[TaikoDetectionSample, TaikoDetectionSamplerConfig]
):
    """Sampler over a built taiko2 dataset.

    `load_data` reads the manifest, mmaps every feature array, and loads
    the event arrays (bins + times_ms + kind_ids) for each chart into
    memory. The `(chart_index, event_index)` sample grid is built once,
    filtering out early-song cursors.
    """

    def __init__(self, config: TaikoDetectionSamplerConfig):
        super().__init__(config)
        self._loaded = False
        self._manifest = None
        # Parallel arrays per chart, indexed by chart_idx
        self._chart_ids: list[str] = []
        self._features: list[np.ndarray] = []          # mmap'd (F, T)
        self._event_bins: list[np.ndarray] = []        # (N,) int32
        self._event_times_ms: list[np.ndarray] = []    # (N,) int32
        self._event_kind_ids: list[np.ndarray] = []    # (N,) uint8
        # Flat sample index: (chart_idx, event_idx)
        self._samples: list[tuple[int, int]] = []

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load_data(self) -> None:
        cfg = self.config
        if self._loaded:
            return

        ds_root = Path(cfg.dataset_root).resolve()
        manifest = load_manifest(ds_root / "manifest.json")

        from ..dataset import _safe_filename  # share the same stem logic

        self._manifest = manifest
        self._chart_ids.clear()
        self._features.clear()
        self._event_bins.clear()
        self._event_times_ms.clear()
        self._event_kind_ids.clear()
        self._samples.clear()

        allowed_ids = chart_ids_for_split(
            manifest, cfg.split, cfg.val_ratio, cfg.split_seed,
        )

        for entry in manifest.charts:
            if entry.chart_id not in allowed_ids:
                continue
            feat_path = ds_root / entry.features_path
            evt_path = ds_root / "events" / f"{_safe_filename(entry.chart_id)}.npz"
            if not feat_path.exists() or not evt_path.exists():
                continue
            try:
                features = np.load(feat_path, mmap_mode="r")
                with np.load(evt_path) as data:
                    bins = np.asarray(data["bins"], dtype=np.int64)
                    times = np.asarray(data["times_ms"], dtype=np.int64)
                    kinds = np.asarray(data["kind_ids"], dtype=np.uint8)
            except Exception:
                continue

            chart_idx = len(self._chart_ids)
            self._chart_ids.append(entry.chart_id)
            self._features.append(features)
            self._event_bins.append(bins)
            self._event_times_ms.append(times)
            self._event_kind_ids.append(kinds)

            for ei in self._iter_event_indices(bins):
                cursor = self._cursor_for(bins, ei)
                if cursor >= cfg.min_cursor_bin:
                    self._samples.append((chart_idx, ei))

        self._loaded = True

    def count_samples(self) -> int:
        self._require_loaded()
        return len(self._samples)

    # ── Public sampling API ───────────────────────────────────────────

    def raw_sample(self, n: int) -> TaikoDetectionSample:
        """Deterministic, unaugmented sample."""
        self._require_loaded()
        if n < 0 or n >= len(self._samples):
            raise IndexError(f"sample index {n} out of range ({len(self._samples)})")
        chart_idx, ei = self._samples[n]
        return self._build_sample(sample_id=n, chart_idx=chart_idx, ei=ei)

    def augment_sample(self, n: int) -> TaikoDetectionSample:
        """Same sample as `raw_sample`, passed through the augmentation
        pipeline. Currently a no-op stub — augmentation is a separate
        design decision that depends on model architecture.
        """
        sample = self.raw_sample(n)
        return self._augment(sample)

    def raw_batch(self, n: int) -> list[TaikoDetectionSample]:
        return self._batch(n, augmented=False)

    def augment_batch(self, n: int) -> list[TaikoDetectionSample]:
        return self._batch(n, augmented=True)

    # Satisfy the DataSampler ABC. Default `get_*` routes to raw so any
    # code using the generic interface gets deterministic behavior;
    # training loops should call `augment_*` explicitly.
    def get_sample(self, n: int) -> TaikoDetectionSample:
        return self.raw_sample(n)

    # ── Internals ─────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("call load_data() first")

    def _batch(self, n: int, *, augmented: bool) -> list[TaikoDetectionSample]:
        total = self.count_samples()
        bs = self.config.batch_size
        start = n * bs
        if start < 0 or start >= total:
            raise IndexError(
                f"batch index {n} out of range for {self.count_batches()} batches"
            )
        end = min(start + bs, total)
        if augmented:
            return [self.augment_sample(i) for i in range(start, end)]
        return [self.raw_sample(i) for i in range(start, end)]

    @staticmethod
    def _iter_event_indices(bins: np.ndarray):
        """Same `ei` range taiko1 used: 0..len(bins) inclusive (trailing
        index is the post-last-event anchor). We still emit it so callers
        can reason about end-of-chart samples, even without a STOP target.
        """
        return range(len(bins) + 1)

    @staticmethod
    def _cursor_for(bins: np.ndarray, ei: int) -> int:
        """Cursor placement rule, ported from OnsetDataset.

        ei == 0           → one pre-roll window before the first event
        1 <= ei <= N-1    → cursor sits exactly on the previous onset
        ei == N           → cursor sits on the last onset (end-of-chart)
        """
        if len(bins) == 0:
            return 0
        if ei == 0:
            return int(max(0, int(bins[0]) - 500))  # 500 ≈ b_pred pre-roll
        if ei >= len(bins):
            return int(bins[-1])
        return int(bins[ei - 1])

    def _build_sample(
        self, *, sample_id: int, chart_idx: int, ei: int,
    ) -> TaikoDetectionSample:
        cfg = self.config
        chart_id = self._chart_ids[chart_idx]
        bins = self._event_bins[chart_idx]
        times_ms = self._event_times_ms[chart_idx]
        kind_ids = self._event_kind_ids[chart_idx]
        features = self._features[chart_idx]
        cursor = self._cursor_for(bins, ei)

        audio_past = self._extract_audio(features, cursor - cfg.a_bins, cursor)
        audio_future = self._extract_audio(features, cursor, cursor + cfg.b_bins)

        past_events, past_mask = self._extract_events(
            bins, times_ms, kind_ids,
            cursor=cursor, lo_idx=max(0, ei - cfg.c_events), hi_idx=ei,
            pad_at_start=True, slot_count=cfg.c_events,
        )
        future_lo = ei
        future_hi = min(len(bins), ei + cfg.d_events)
        future_events, future_mask = self._extract_events(
            bins, times_ms, kind_ids,
            cursor=cursor, lo_idx=future_lo, hi_idx=future_hi,
            pad_at_start=False, slot_count=cfg.d_events,
        )

        return TaikoDetectionSample(
            sample_id=sample_id,
            chart_id=chart_id,
            cursor_bin=cursor,
            audio_past=audio_past,
            audio_future=audio_future,
            past_events=past_events,
            past_events_mask=past_mask,
            future_events=future_events,
            future_events_mask=future_mask,
        )

    @staticmethod
    def _extract_audio(
        features: np.ndarray, start: int, end: int,
    ) -> np.ndarray:
        """Slice the feature array with zero-padding outside bounds.

        Matches taiko1's window extractor: float32 return, left/right pad
        with zeros when `start < 0` or `end > T`.
        """
        n_feat, total = features.shape
        pad_left = max(0, -start)
        pad_right = max(0, end - total)
        s = max(0, start)
        e = min(total, end)
        core = features[:, s:e].astype(np.float32, copy=False)
        if pad_left == 0 and pad_right == 0:
            return np.ascontiguousarray(core)
        return np.pad(core, ((0, 0), (pad_left, pad_right)), mode="constant")

    @staticmethod
    def _extract_events(
        bins: np.ndarray,
        times_ms: np.ndarray,
        kind_ids: np.ndarray,
        *,
        cursor: int,
        lo_idx: int,
        hi_idx: int,
        pad_at_start: bool,
        slot_count: int,
    ) -> tuple[tuple[RelativeOnset, ...], np.ndarray]:
        """Build a fixed-size, padded sequence of RelativeOnset."""
        n = max(0, hi_idx - lo_idx)
        n = min(n, slot_count)
        onsets: list[RelativeOnset] = []
        if n > 0:
            use_hi = lo_idx + n
            sub_bins = bins[lo_idx:use_hi]
            sub_times = times_ms[lo_idx:use_hi]
            sub_kinds = kind_ids[lo_idx:use_hi]
            for b, t, k in zip(sub_bins, sub_times, sub_kinds):
                onsets.append(RelativeOnset(
                    time_ms=int(t),
                    kind=_KIND_ORDER[int(k)] if int(k) < len(_KIND_ORDER) else OnsetKind.UNKNOWN,
                    bin=int(b),
                    cursor_offset=int(b) - int(cursor),
                ))

        mask = np.ones(slot_count, dtype=bool)  # True = padded
        if n == 0:
            return tuple([_padding_onset()] * slot_count), mask

        if pad_at_start:
            pad_n = slot_count - n
            padded = [_padding_onset()] * pad_n + onsets
            mask[:pad_n] = True
            mask[pad_n:] = False
        else:
            padded = onsets + [_padding_onset()] * (slot_count - n)
            mask[:n] = False
            mask[n:] = True

        return tuple(padded), mask

    def _augment(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        """No-op placeholder. Swap in an augmentation strategy here.

        Intentionally a stub — augmentation design (event jitter, SpecAug,
        metronome corruption, etc.) depends on the target architecture and
        hasn't been decided for taiko2 yet.
        """
        return sample


def _padding_onset() -> RelativeOnset:
    return RelativeOnset(
        time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=0,
    )
