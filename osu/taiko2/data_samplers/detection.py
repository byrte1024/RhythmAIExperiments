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

import os

from ..domain.augmentation import AugmentationPipeline
from ..domain.beatmap import (
    AudioRef,
    Density,
    Difficulty,
    Onset,
    OnsetKind,
    RelativeOnset,
    Track,
)
from ..domain.chart import Chart
from ..domain.dataset import ChartEntry
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
    # Minimum cursor-to-cursor distance between samples of the same chart.
    # A candidate cursor is dropped if it falls within this many bins of an
    # already-accepted cursor. Two knobs so the allowed overlap can be
    # asymmetric (e.g. require a longer forward gap than backward). The
    # effective minimum distance between kept cursors is
    # `max(allowed_overlap_forward, allowed_overlap_back)`.
    # Defaults — None here — are resolved in `__post_init__` to match
    # `b_bins` (forward) and `a_bins` (backward): keeps consecutive samples
    # from having their directional audio windows overlap each other.
    # Set both to 0 to disable filtering entirely (recovers the old
    # taiko1 behavior of one sample per event boundary).
    allowed_overlap_forward: int | None = None
    allowed_overlap_back: int | None = None
    # Keep every Nth sample AFTER overlap filtering. Ported from taiko1's
    # `subsample` flag — useful for smoke tests (`subsample=16` gives
    # ~1/16th the data) or for cheap ablation runs.
    subsample: int = 1
    # Split configuration. `split_ratios` is an ordered tuple of
    # `(name, ratio)` pairs; ratios are the fraction of songs (not charts)
    # that go to that bucket. Splitting is by `beatmapset_id` — all
    # difficulties of a song land in the same split. `split` picks which
    # bucket this sampler serves, or `"all"` to use every chart regardless.
    # `split_seed` pins the shuffle so two samplers sharing the same spec
    # produce disjoint, reproducible buckets.
    split: str = "all"
    split_ratios: tuple[tuple[str, float], ...] = (
        ("train", 0.9),
        ("val", 0.1),
    )
    split_seed: int = 42
    # Per-split field overrides: `{split_name: {field: value, ...}}`.
    # When the sampler loads and `split` matches a key here, the config is
    # replaced with those field overrides for that sampler's lifetime.
    # Lets a single "base" config express e.g. "train uses overlap=500,
    # val uses overlap=50" without building two configs from scratch.
    # Cannot override `split`, `split_ratios`, `split_seed`, or
    # `split_overrides` itself (those define the split mechanism).
    split_overrides: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self):
        if self.allowed_overlap_forward is None:
            object.__setattr__(self, "allowed_overlap_forward", self.b_bins)
        if self.allowed_overlap_back is None:
            object.__setattr__(self, "allowed_overlap_back", self.a_bins)
        if self.allowed_overlap_forward < 0 or self.allowed_overlap_back < 0:
            raise ValueError(
                "allowed_overlap_forward / allowed_overlap_back must be >= 0"
            )
        reserved = {"split", "split_ratios", "split_seed", "split_overrides"}
        for split_name, overrides in self.split_overrides.items():
            for key in overrides:
                if key in reserved:
                    raise ValueError(
                        f"split_overrides[{split_name!r}] cannot override "
                        f"{key!r} (reserved for split mechanism)"
                    )

    def resolve(self, split: str | None = None) -> "TaikoDetectionSamplerConfig":
        """Return a new config with `split_overrides[split]` applied.

        `split=None` uses `self.split`. If no overrides exist for that
        split, returns `self` unchanged. `split_overrides` on the returned
        config is cleared so a re-resolve is a no-op.
        """
        target = split if split is not None else self.split
        overrides = self.split_overrides.get(target, {})
        if split is None and not overrides:
            return self
        from dataclasses import replace as _replace
        return _replace(
            self, split=target, split_overrides={}, **overrides,
        )


# ─────────────────────────── sample payload ───────────────────────────

@dataclass(frozen=True, slots=True)
class TaikoDetectionPreContext:
    """Input to a pre-sample augmentation.

    Carries everything an augmentation might need to decide a new cursor
    or a different event-index slice without reading from disk: the full
    per-chart event arrays are read-only views into sampler state.

    `cursor_bin` is mutable via `dataclasses.replace`; augmentations
    typically return a new context with a shifted cursor.
    """
    chart_idx: int
    event_idx: int
    cursor_bin: int
    event_bins: np.ndarray       # (N,) int64 — chart's full event bins
    event_times_ms: np.ndarray   # (N,) int64
    event_kind_ids: np.ndarray   # (N,) uint8


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
    # Density conditioning — snapshot of the chart's density stats so the
    # adapter can turn each sample into a (B, 3) conditioning vector
    # without a second lookup.
    density_mean: float = 0.0
    density_peak: int = 0
    density_std: float = 0.0


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

    def __init__(
        self,
        config: TaikoDetectionSamplerConfig,
        pipeline: "AugmentationPipeline[TaikoDetectionPreContext, TaikoDetectionSample] | None" = None,
    ):
        super().__init__(config)
        self._loaded = False
        self._manifest = None
        self._pipeline: AugmentationPipeline = (
            pipeline if pipeline is not None else AugmentationPipeline()
        )
        # Parallel arrays per chart, indexed by chart_idx
        self._chart_ids: list[str] = []
        self._chart_entries: list[ChartEntry] = []     # manifest row per chart
        self._features: list[np.ndarray] = []          # mmap'd (F, T)
        self._event_bins: list[np.ndarray] = []        # (N,) int32
        self._event_times_ms: list[np.ndarray] = []    # (N,) int32
        self._event_kind_ids: list[np.ndarray] = []    # (N,) uint8
        # Flat sample index: (chart_idx, event_idx)
        self._samples: list[tuple[int, int]] = []
        # Fast lookup from chart_id to the chart_idx used by the arrays
        self._id_to_idx: dict[str, int] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    def load_data(self, *, progress: bool = False) -> None:
        """Read the dataset into memory and build the sample grid.

        Pass ``progress=True`` to show a tqdm bar over manifest charts
        and print a post-load summary (chart count, skipped, sample
        count). Default is quiet because tests / notebooks / library
        use don't want the noise; CLIs should set it True.
        """
        if self._loaded:
            return

        # Resolve per-split overrides once. After this point the sampler's
        # `self.config` reflects the effective config for its chosen split,
        # so every downstream read (including `count_batches` via
        # `self.config.batch_size`) sees consistent values.
        resolved = self.config.resolve()
        if resolved is not self.config:
            object.__setattr__(self, "config", resolved)
        cfg = self.config

        ds_root = Path(cfg.dataset_root).resolve()
        manifest = load_manifest(ds_root / "manifest.json")

        from ..dataset import _safe_filename  # share the same stem logic

        self._manifest = manifest
        self._chart_ids.clear()
        self._chart_entries.clear()
        self._features.clear()
        self._event_bins.clear()
        self._event_times_ms.clear()
        self._event_kind_ids.clear()
        self._samples.clear()
        self._id_to_idx.clear()

        allowed_ids = chart_ids_for_split(
            manifest, cfg.split, cfg.split_ratios, cfg.split_seed,
        )

        entries_iter = manifest.charts
        if progress:
            try:
                from tqdm.auto import tqdm
                entries_iter = tqdm(
                    list(manifest.charts),
                    desc=f"Loading {cfg.split!r} split",
                    unit="chart",
                )
            except ImportError:
                pass

        for entry in entries_iter:
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
            self._chart_entries.append(entry)
            self._features.append(features)
            self._event_bins.append(bins)
            self._event_times_ms.append(times)
            self._event_kind_ids.append(kinds)
            self._id_to_idx[entry.chart_id] = chart_idx

            # Per-chart overlap filter: cursors within this chart iterate
            # monotonically (ei → cursor is non-decreasing), so we only need
            # the last accepted cursor and the max of the two overlaps as
            # the minimum allowed gap.
            min_gap = max(cfg.allowed_overlap_forward, cfg.allowed_overlap_back)
            last_cursor: int | None = None

            for ei in self._iter_event_indices(bins):
                cursor = self._cursor_for(bins, ei)
                if cursor < cfg.min_cursor_bin:
                    continue
                if last_cursor is not None and min_gap > 0:
                    if cursor - last_cursor < min_gap:
                        continue
                self._samples.append((chart_idx, ei))
                last_cursor = cursor

        pre_subsample = len(self._samples)
        if cfg.subsample > 1:
            self._samples = self._samples[::cfg.subsample]

        self._loaded = True

        if progress:
            kept = len(self._chart_ids)
            total = len(manifest.charts)
            skipped = total - kept
            msg = (
                f"[sampler] split={cfg.split!r}  "
                f"charts={kept}/{total} ({skipped} skipped)  "
                f"samples={len(self._samples):,}"
            )
            if cfg.subsample > 1:
                msg += f"  (subsample={cfg.subsample}, pre-subsample={pre_subsample:,})"
            try:
                from tqdm.auto import tqdm as _tqdm
                _tqdm.write(msg)
            except ImportError:
                print(msg)

    def count_samples(self) -> int:
        self._require_loaded()
        return len(self._samples)

    # ── Public sampling API ───────────────────────────────────────────

    def raw_sample(self, n: int) -> TaikoDetectionSample:
        """Deterministic, unaugmented sample. Pipeline is bypassed entirely."""
        self._require_loaded()
        if n < 0 or n >= len(self._samples):
            raise IndexError(f"sample index {n} out of range ({len(self._samples)})")
        chart_idx, ei = self._samples[n]
        ctx = self._build_context(chart_idx, ei)
        return self._build_sample(sample_id=n, ctx=ctx)

    def augment_sample(self, n: int) -> TaikoDetectionSample:
        """Run the pipeline: pre-augs mutate the extraction context, then
        the sample is built, then post-augs mutate the sample. Empty
        pipeline is equivalent to `raw_sample(n)`.
        """
        self._require_loaded()
        if n < 0 or n >= len(self._samples):
            raise IndexError(f"sample index {n} out of range ({len(self._samples)})")
        chart_idx, ei = self._samples[n]
        ctx = self._build_context(chart_idx, ei)
        ctx = self._pipeline.apply_pre(ctx)
        sample = self._build_sample(sample_id=n, ctx=ctx)
        sample = self._pipeline.apply_post(sample)
        return sample

    def raw_batch(self, n: int) -> list[TaikoDetectionSample]:
        return self._batch(n, augmented=False)

    def augment_batch(self, n: int) -> list[TaikoDetectionSample]:
        return self._batch(n, augmented=True)

    # Satisfy the DataSampler ABC. Default `get_*` routes to raw so any
    # code using the generic interface gets deterministic behavior;
    # training loops should call `augment_*` explicitly.
    def get_sample(self, n: int) -> TaikoDetectionSample:
        return self.raw_sample(n)

    # ── Chart-level access (not sample-level) ─────────────────────────

    def count_charts(self) -> int:
        """Number of charts in the current split (after file-existence
        and split filtering). Distinct from `count_samples` which counts
        the (chart, event-index) sample grid."""
        self._require_loaded()
        return len(self._chart_ids)

    def chart_ids(self) -> tuple[str, ...]:
        """Chart IDs in this split, in sampler index order. Stable across
        runs for the same config."""
        self._require_loaded()
        return tuple(self._chart_ids)

    def get_chart(self, n: int) -> Chart:
        """Return the n-th chart in the current split as a `Chart` object.

        Reconstructs the `Track` from the manifest entry + the stored
        (times_ms, kind_ids) event arrays — so onset kinds round-trip
        (DON/KA/etc.), not just bins.

        `Chart.audio` is always `None` here: datasets store mel
        spectrograms, not raw audio. Pair with an external audio source
        (or load from the original `.osz`) if playback is needed.
        """
        self._require_loaded()
        if n < 0 or n >= len(self._chart_ids):
            raise IndexError(
                f"chart index {n} out of range ({len(self._chart_ids)} charts)"
            )
        return self._build_chart(n)

    # ── Weighted-sampling utilities ───────────────────────────────────

    def target_bins(self, *, b_pred: int) -> np.ndarray:
        """Per-sample target class id, treating anything ≥ `b_pred` as STOP.

        For sample index `n`:
          - If the sample has no future event in-window, target = b_pred (STOP).
          - Else if `future_events[0].cursor_offset >= b_pred`, target = b_pred.
          - Else target = `future_events[0].cursor_offset`.

        Returns an int64 array of length `count_samples()`. Computed in
        one vectorized pass over `_event_bins`, not by calling
        `raw_sample` (~10× faster on large datasets).
        """
        self._require_loaded()
        stop_idx = int(b_pred)
        out = np.empty(len(self._samples), dtype=np.int64)
        for i, (chart_idx, ei) in enumerate(self._samples):
            bins = self._event_bins[chart_idx]
            if ei >= len(bins):
                out[i] = stop_idx
                continue
            cursor = self._cursor_for(bins, ei)
            offset = int(bins[ei]) - cursor
            if offset < 0 or offset >= b_pred:
                out[i] = stop_idx
            else:
                out[i] = offset
        return out

    def compute_target_weights(
        self,
        *,
        b_pred: int,
        power: float = 0.5,
        stop_boost: float = 1.0,
        cap: float = 1.0,
    ) -> np.ndarray:
        """Per-sample weights for class-balanced training.

        ``weight = min(cap, 1 / (count_of_target_class + 1)^power)``;
        STOP class (index ``b_pred``) optionally multiplied by
        ``stop_boost`` afterwards. Returns float64 of shape
        ``(count_samples,)`` — pass to ``torch.utils.data.
        WeightedRandomSampler`` or any other weighted draw.

        Defaults match taiko1's exp 45 setup (``power=0.5`` sqrt
        weighting, no extra STOP boost). Raise ``power`` to oversample
        rare targets more aggressively; lower it toward 0 for near-
        uniform draws.
        """
        targets = self.target_bins(b_pred=b_pred)
        counts = np.bincount(targets, minlength=b_pred + 1).astype(np.float64)
        per_class_w = 1.0 / (counts + 1.0) ** power
        if stop_boost != 1.0:
            per_class_w[b_pred] *= stop_boost
        weights = np.minimum(per_class_w[targets], cap)
        return weights

    def get_chart_by_id(self, chart_id: str) -> Chart:
        """Return a `Chart` by its `chart_id`. Raises `KeyError` if the
        id isn't in the current split.
        """
        self._require_loaded()
        if chart_id not in self._id_to_idx:
            raise KeyError(f"chart_id {chart_id!r} not in split")
        return self._build_chart(self._id_to_idx[chart_id])

    def _build_chart(self, chart_idx: int) -> Chart:
        entry = self._chart_entries[chart_idx]
        times_ms = self._event_times_ms[chart_idx]
        kind_ids = self._event_kind_ids[chart_idx]

        onsets = tuple(
            Onset(
                time_ms=int(t),
                kind=(
                    _KIND_ORDER[int(k)]
                    if 0 <= int(k) < len(_KIND_ORDER)
                    else OnsetKind.UNKNOWN
                ),
            )
            for t, k in zip(times_ms, kind_ids)
        )

        audio_format = os.path.splitext(entry.audio_filename)[1].lstrip(".").lower()

        track = Track(
            beatmap_id=entry.beatmap_id,
            beatmapset_id=entry.beatmapset_id,
            artist=entry.artist,
            title=entry.title,
            difficulty=Difficulty(
                version=entry.difficulty_version,
                overall_difficulty=entry.overall_difficulty,
                star_rating=entry.star_rating,
            ),
            audio=AudioRef(
                filename=entry.audio_filename,
                format=audio_format,
            ),
            onsets=onsets,
            density=Density(
                mean=entry.density_mean,
                peak=entry.density_peak,
                std=entry.density_std,
                duration_s=entry.duration_s,
                total_events=entry.total_events,
            ),
        )
        return Chart(track=track, audio=None)

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

    def _build_context(self, chart_idx: int, ei: int) -> TaikoDetectionPreContext:
        """Assemble the pre-sample context for augmentation + extraction."""
        bins = self._event_bins[chart_idx]
        return TaikoDetectionPreContext(
            chart_idx=chart_idx,
            event_idx=ei,
            cursor_bin=self._cursor_for(bins, ei),
            event_bins=bins,
            event_times_ms=self._event_times_ms[chart_idx],
            event_kind_ids=self._event_kind_ids[chart_idx],
        )

    def _build_sample(
        self, *, sample_id: int, ctx: TaikoDetectionPreContext,
    ) -> TaikoDetectionSample:
        """Materialize a sample from a (possibly-augmented) context."""
        cfg = self.config
        chart_id = self._chart_ids[ctx.chart_idx]
        bins = ctx.event_bins
        times_ms = ctx.event_times_ms
        kind_ids = ctx.event_kind_ids
        features = self._features[ctx.chart_idx]
        cursor = ctx.cursor_bin
        ei = ctx.event_idx

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

        entry = self._chart_entries[ctx.chart_idx]
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
            density_mean=entry.density_mean,
            density_peak=entry.density_peak,
            density_std=entry.density_std,
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

def _padding_onset() -> RelativeOnset:
    return RelativeOnset(
        time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=0,
    )
