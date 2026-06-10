"""Data sampler for the typing model.

Each sample is a window of 33 onsets (16 past + 1 target + 16 future)
centered on a single onset whose type and strength are the prediction
targets. Onset positions, IOIs, mel patches, and (for past onsets)
D/K + big/normal labels are extracted per sample.

DRUMROLL, SPINNER, and UNKNOWN onsets are filtered out — the typing
model only operates on hits (DON, KA, BIG_DON, BIG_KA).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..dataset import _safe_filename
from ..domain.typing import (
    TYPING_CONTEXT,
    TYPING_MEL_PATCH,
    TypingSample,
)
from ..domain.sampling import DataSampler, DataSamplerConfig
from ..persistence.events import _KIND_ORDER
from ..persistence.manifest import load_manifest
from ..splits import chart_ids_for_split

IDX_DON = 0
IDX_KA = 1
IDX_BDON = 2
IDX_BKA = 3

_HIT_KINDS = frozenset({IDX_DON, IDX_KA, IDX_BDON, IDX_BKA})
_HALF_MEL = TYPING_MEL_PATCH // 2


@dataclass(frozen=True, slots=True)
class TypingSamplerConfig(DataSamplerConfig):
    dataset_root: Path = field(default=Path("."))
    past_context: int = TYPING_CONTEXT
    future_context: int = TYPING_CONTEXT
    mel_patch: int = TYPING_MEL_PATCH
    split: str = "all"
    split_ratios: tuple[tuple[str, float], ...] = (("train", 0.9), ("val", 0.1))
    split_seed: int = 42
    subsample: int = 1
    batch_size: int = 128


def _log_ioi(a: int, b: int) -> float:
    return float(np.log1p(abs(b - a)))


def _compute_iois(bins: np.ndarray, idx: int) -> np.ndarray:
    """Compute [log_ioi_before, log_ioi_after, log_ratio] for onset at idx."""
    n = len(bins)
    if idx > 0:
        ioi_before = _log_ioi(bins[idx - 1], bins[idx])
    else:
        ioi_before = 0.0
    if idx < n - 1:
        ioi_after = _log_ioi(bins[idx], bins[idx + 1])
    else:
        ioi_after = 0.0
    if ioi_before > 0 and ioi_after > 0:
        ratio = ioi_before - ioi_after  # log(before/after) = log(before) - log(after)
    else:
        ratio = 0.0
    return np.array([ioi_before, ioi_after, ratio], dtype=np.float32)


class TypingSampler(DataSampler[TypingSample, TypingSamplerConfig]):

    def __init__(self, config: TypingSamplerConfig):
        super().__init__(config)
        self._chart_ids: list[str] = []
        self._features: list[np.ndarray] = []
        self._event_bins: list[np.ndarray] = []
        self._event_kind_ids: list[np.ndarray] = []
        # Per-chart: indices into the original event array that are hits
        self._hit_indices: list[np.ndarray] = []
        # Global sample list: (chart_idx, hit_position_in_hit_array)
        self._samples: list[tuple[int, int]] = []

    def load_data(self, *, progress: bool = False) -> None:
        cfg = self.config
        ds_root = Path(cfg.dataset_root).resolve()
        manifest = load_manifest(ds_root / "manifest.json")
        allowed = chart_ids_for_split(
            manifest, cfg.split, cfg.split_ratios, cfg.split_seed,
        )

        self._chart_ids.clear()
        self._features.clear()
        self._event_bins.clear()
        self._event_kind_ids.clear()
        self._hit_indices.clear()
        self._samples.clear()

        entries = [e for e in manifest.charts if e.chart_id in allowed]
        if progress:
            try:
                from tqdm.auto import tqdm
                entries = list(tqdm(entries, desc=f"Loading typing {cfg.split!r}", unit="chart"))
            except ImportError:
                pass

        for entry in entries:
            feat_path = ds_root / entry.features_path
            evt_path = ds_root / "events" / f"{_safe_filename(entry.chart_id)}.npz"
            if not feat_path.exists() or not evt_path.exists():
                continue
            try:
                features = np.load(feat_path, mmap_mode="r")
                with np.load(evt_path) as data:
                    bins = np.asarray(data["bins"], dtype=np.int64)
                    kinds = np.asarray(data["kind_ids"], dtype=np.uint8)
            except Exception:
                continue

            # Filter to hits only
            hit_mask = np.isin(kinds, list(_HIT_KINDS))
            hit_idx = np.where(hit_mask)[0]
            if len(hit_idx) < 3:
                continue

            chart_idx = len(self._chart_ids)
            self._chart_ids.append(entry.chart_id)
            self._features.append(features)
            self._event_bins.append(bins[hit_mask])
            self._event_kind_ids.append(kinds[hit_mask])
            self._hit_indices.append(hit_idx)

            for i in range(len(hit_idx)):
                self._samples.append((chart_idx, i))

        if cfg.subsample > 1:
            self._samples = self._samples[::cfg.subsample]

    def count_samples(self) -> int:
        return len(self._samples)

    def _extract_mel_patch(
        self, features: np.ndarray, b: int,
    ) -> np.ndarray:
        """Extract (n_mels, mel_patch) around bin b, zero-pad if needed."""
        n_mels, n_frames = features.shape
        lo = b - _HALF_MEL
        hi = b + _HALF_MEL + 1
        if lo < 0 or hi > n_frames:
            patch = np.zeros((n_mels, self.config.mel_patch), dtype=np.float32)
            src_lo = max(0, lo)
            src_hi = min(n_frames, hi)
            dst_lo = src_lo - lo
            dst_hi = dst_lo + (src_hi - src_lo)
            if src_hi > src_lo:
                patch[:, dst_lo:dst_hi] = features[:, src_lo:src_hi]
            return patch
        return np.asarray(features[:, lo:hi], dtype=np.float32)

    def _kind_to_dk(self, kind_id: int) -> int:
        """Map kind_id to D=0, K=1."""
        if kind_id in (IDX_DON, IDX_BDON):
            return 0
        return 1

    def _kind_to_big(self, kind_id: int) -> int:
        """Map kind_id to normal=0, big=1."""
        if kind_id in (IDX_BDON, IDX_BKA):
            return 1
        return 0

    def get_sample(self, n: int) -> TypingSample:
        chart_idx, hit_pos = self._samples[n]
        bins = self._event_bins[chart_idx]
        kinds = self._event_kind_ids[chart_idx]
        features = self._features[chart_idx]
        pc = self.config.past_context
        fc = self.config.future_context
        n_hits = len(bins)

        target_bin = int(bins[hit_pos])
        target_kind_id = int(kinds[hit_pos])

        # Past context
        past_iois = np.zeros((pc, 3), dtype=np.float32)
        past_kinds = np.zeros(pc, dtype=np.uint8)
        past_bigs = np.zeros(pc, dtype=np.uint8)
        past_mel = np.zeros((pc, features.shape[0], self.config.mel_patch), dtype=np.float32)
        past_mask = np.ones(pc, dtype=bool)  # True = padded

        past_start = max(0, hit_pos - pc)
        past_count = hit_pos - past_start
        for j in range(past_count):
            src = past_start + j
            dst = pc - past_count + j
            past_iois[dst] = _compute_iois(bins, src)
            past_kinds[dst] = self._kind_to_dk(int(kinds[src]))
            past_bigs[dst] = self._kind_to_big(int(kinds[src]))
            past_mel[dst] = self._extract_mel_patch(features, int(bins[src]))
            past_mask[dst] = False

        # Target
        target_iois = _compute_iois(bins, hit_pos)
        target_mel = self._extract_mel_patch(features, target_bin)

        # Future context
        future_iois = np.zeros((fc, 3), dtype=np.float32)
        future_mel = np.zeros((fc, features.shape[0], self.config.mel_patch), dtype=np.float32)
        future_mask = np.ones(fc, dtype=bool)

        future_end = min(n_hits, hit_pos + 1 + fc)
        future_count = future_end - (hit_pos + 1)
        for j in range(future_count):
            src = hit_pos + 1 + j
            future_iois[j] = _compute_iois(bins, src)
            future_mel[j] = self._extract_mel_patch(features, int(bins[src]))
            future_mask[j] = False

        return TypingSample(
            sample_id=n,
            chart_id=self._chart_ids[chart_idx],
            target_idx=hit_pos,
            past_iois=past_iois,
            past_kinds=past_kinds,
            past_bigs=past_bigs,
            past_mel=past_mel,
            past_mask=past_mask,
            target_iois=target_iois,
            target_mel=target_mel,
            future_iois=future_iois,
            future_mel=future_mel,
            future_mask=future_mask,
            target_kind=self._kind_to_dk(target_kind_id),
            target_big=self._kind_to_big(target_kind_id),
        )
