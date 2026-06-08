"""Analyze onset kind (DON/KA/BIG/DRUMROLL/SPINNER) acoustics and patterns.

Loads every chart in a dataset split, extracts mel windows around each
onset, and computes per-kind acoustic statistics + D/K transition
patterns.  The goal is to answer: are DON and KA acoustically
distinguishable, or is typing purely a pattern-level decision?

Usage::

    python -m osu.taiko2.cli.analyze_kinds \\
        --dataset taiko2_v1 --split all \\
        --output osu/taiko2/experiments/023-kind-acoustics/results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..dataset import _safe_filename
from ..persistence.events import _KIND_ORDER
from ..persistence.manifest import load_manifest
from ..splits import chart_ids_for_split


# ─────────────────────────── constants ───────────────────────────────

KIND_NAMES: tuple[str, ...] = tuple(k.value for k in _KIND_ORDER)
# Indices into _KIND_ORDER
IDX_DON = 0
IDX_KA = 1
IDX_BDON = 2
IDX_BKA = 3
IDX_DRUM = 4
IDX_SPIN = 5
IDX_UNK = 6

HIT_KINDS = (IDX_DON, IDX_KA, IDX_BDON, IDX_BKA)
NORMAL_KINDS = (IDX_DON, IDX_KA)
BIG_KINDS = (IDX_BDON, IDX_BKA)

MEL_WINDOW_HALF = 10  # +-10 frames = 21-frame window (105 ms at 5ms/frame)
N_CONTEXT_ONSETS = 8  # preceding/following onsets for pattern context


# ─────────────────────────── per-kind mel accumulators ───────────────

@dataclass
class BandAccumulator:
    """Online mean/var per mel band, supporting batch updates."""
    n: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def init_bands(self, n_bands: int) -> None:
        if self.mean.shape[0] == 0:
            self.mean = np.zeros(n_bands, dtype=np.float64)
            self.m2 = np.zeros(n_bands, dtype=np.float64)

    def update_batch(self, frames: np.ndarray) -> None:
        """Add multiple mel frames (K, n_bands) at once."""
        if frames.shape[0] == 0:
            return
        batch = frames.astype(np.float64)
        b_n = batch.shape[0]
        b_mean = batch.mean(axis=0)
        b_m2 = np.sum((batch - b_mean) ** 2, axis=0)
        if self.n == 0:
            self.n = b_n
            self.mean = b_mean
            self.m2 = b_m2
        else:
            combined_n = self.n + b_n
            delta = b_mean - self.mean
            self.m2 += b_m2 + delta ** 2 * (self.n * b_n / combined_n)
            self.mean += delta * (b_n / combined_n)
            self.n = combined_n

    @property
    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


@dataclass
class WindowAccumulator:
    """Online mean/var for the full mel window (n_bands, window_len)."""
    n: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))

    def update_batch(self, windows: np.ndarray) -> None:
        """Add multiple windows (K, n_bands, window_len) at once."""
        if windows.shape[0] == 0:
            return
        batch = windows.astype(np.float64)
        b_n = batch.shape[0]
        b_mean = batch.mean(axis=0)
        b_m2 = np.sum((batch - b_mean) ** 2, axis=0)
        if self.n == 0:
            self.n = b_n
            self.mean = b_mean
            self.m2 = b_m2
        else:
            combined_n = self.n + b_n
            delta = b_mean - self.mean
            self.m2 += b_m2 + delta ** 2 * (self.n * b_n / combined_n)
            self.mean += delta * (b_n / combined_n)
            self.n = combined_n

    @property
    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.n - 1)


# ─────────────────────────── separability ────────────────────────────

def fisher_lda_score(
    mean_a: np.ndarray, var_a: np.ndarray, n_a: int,
    mean_b: np.ndarray, var_b: np.ndarray, n_b: int,
) -> float:
    """Fisher's linear discriminant ratio (scalar, per-band then averaged).

    J = (mu_a - mu_b)^2 / (var_a + var_b) per band, averaged.
    Higher = more separable.
    """
    denom = var_a + var_b
    denom = np.where(denom < 1e-12, 1e-12, denom)
    j = (mean_a - mean_b) ** 2 / denom
    return float(np.mean(j))


def per_band_ttest(
    mean_a: np.ndarray, var_a: np.ndarray, n_a: int,
    mean_b: np.ndarray, var_b: np.ndarray, n_b: int,
) -> np.ndarray:
    """Welch's t-statistic per band (unsigned)."""
    se = np.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1))
    se = np.where(se < 1e-12, 1e-12, se)
    return np.abs(mean_a - mean_b) / se


# ─────────────────────────── pattern analysis ────────────────────────

def build_transition_matrix(kind_ids: np.ndarray) -> np.ndarray:
    """4x4 transition matrix over (DON, KA, BIG_DON, BIG_KA)."""
    if len(kind_ids) < 2:
        return np.zeros((4, 4), dtype=np.int64)
    a = kind_ids[:-1]
    b = kind_ids[1:]
    hit_mask = (a < 4) & (b < 4)
    a, b = a[hit_mask], b[hit_mask]
    flat = a.astype(np.int64) * 4 + b.astype(np.int64)
    return np.bincount(flat, minlength=16).reshape(4, 4)


def count_ngrams(
    kind_ids: np.ndarray, n: int, dk_only: bool = True,
) -> Counter[tuple[int, ...]]:
    """Count n-grams over kind_ids. If dk_only, map BIG_DON->DON, BIG_KA->KA, skip others."""
    if len(kind_ids) < n:
        return Counter()
    mapped = kind_ids.copy()
    if dk_only:
        mapped[mapped == IDX_BDON] = IDX_DON
        mapped[mapped == IDX_BKA] = IDX_KA
    windows = np.lib.stride_tricks.sliding_window_view(mapped, n)
    if dk_only:
        keep = np.all(windows < 2, axis=1)
        windows = windows[keep]
    # Encode each n-gram as a single int for fast counting
    if n <= 8:
        encoded = np.zeros(len(windows), dtype=np.int64)
        for col in range(n):
            encoded = encoded * 7 + windows[:, col].astype(np.int64)
        unique, counts = np.unique(encoded, return_counts=True)
        # Decode back
        result: Counter[tuple[int, ...]] = Counter()
        for enc, cnt in zip(unique, counts):
            gram: list[int] = []
            v = int(enc)
            for _ in range(n):
                gram.append(v % 7)
                v //= 7
            result[tuple(reversed(gram))] = int(cnt)
        return result
    # Fallback for large n
    counts_out: Counter[tuple[int, ...]] = Counter()
    for row in windows:
        counts_out[tuple(int(x) for x in row)] += 1
    return counts_out


def pattern_repeat_analysis(
    bins: np.ndarray, kind_ids: np.ndarray,
    ioi_tolerance_pct: float = 0.05, min_pattern_len: int = 4,
) -> dict[str, Any]:
    """Find repeated IOI patterns and check if D/K assignment is consistent.

    Uses quantized-IOI hashing: round each IOI to the nearest multiple
    of (tolerance * IOI) and group windows by their quantized hash.
    Windows in the same bucket have IOIs within tolerance of each other.
    """
    L = min_pattern_len
    if len(bins) < L * 2 + 1:
        return {"n_pattern_pairs": 0}

    iois = np.diff(bins).astype(np.float64)
    n_windows = len(iois) - L + 1
    if n_windows < 2:
        return {"n_pattern_pairs": 0}

    # Sliding windows of IOIs: (n_windows, L)
    ioi_windows = np.lib.stride_tricks.sliding_window_view(iois, L)
    # Filter windows with non-positive IOIs
    valid = np.all(ioi_windows > 0, axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return {"n_pattern_pairs": 0}

    ioi_valid = ioi_windows[valid_idx]

    # Quantize: round each IOI to nearest grid point for hashing.
    # Grid size per window = median IOI * tolerance (so IOIs within 5%
    # of each other land in the same or adjacent bucket).
    # Use a coarser grid: round IOIs to nearest 5ms bin for hashing,
    # then verify within each bucket.
    quant = np.round(ioi_valid / 5.0).astype(np.int64) * 5
    # Hash each window into a single value
    hashes = np.zeros(len(quant), dtype=np.int64)
    for col in range(L):
        hashes = hashes * 10007 + quant[:, col]

    # Group by hash
    sort_idx = np.argsort(hashes)
    sorted_hashes = hashes[sort_idx]
    # Find bucket boundaries
    breaks = np.where(np.diff(sorted_hashes) != 0)[0] + 1
    bucket_starts = np.concatenate([[0], breaks])
    bucket_ends = np.concatenate([breaks, [len(sorted_hashes)]])

    # Prepare D/K mapped kinds for the L+1 onset windows
    dk_kinds = kind_ids.copy()
    dk_kinds[dk_kinds == IDX_BDON] = IDX_DON
    dk_kinds[dk_kinds == IDX_BKA] = IDX_KA

    n_pairs = 0
    n_same_kind = 0
    n_flipped_kind = 0
    n_other = 0

    for bs, be in zip(bucket_starts, bucket_ends):
        bucket_size = be - bs
        if bucket_size < 2:
            continue
        bucket_orig_idx = valid_idx[sort_idx[bs:be]]
        bucket_iois = ioi_valid[sort_idx[bs:be]]

        # Within bucket, verify actual IOI match (quantization may group
        # slightly-off patterns). Compare first window against all others.
        ref_ioi = bucket_iois[0]
        for k in range(1, min(bucket_size, 10)):  # cap per-bucket pairs
            ratio = bucket_iois[k] / ref_ioi
            if not np.all(np.abs(ratio - 1.0) < ioi_tolerance_pct):
                continue
            n_pairs += 1
            i = int(bucket_orig_idx[0])
            j = int(bucket_orig_idx[k])
            ki = dk_kinds[i:i + L + 1]
            kj = dk_kinds[j:j + L + 1]
            if np.any(ki >= 2) or np.any(kj >= 2):
                continue
            if np.array_equal(ki, kj):
                n_same_kind += 1
            elif np.array_equal(ki, 1 - kj):
                n_flipped_kind += 1
            else:
                n_other += 1

    total_classified = n_same_kind + n_flipped_kind + n_other
    return {
        "n_pattern_pairs": n_pairs,
        "n_same_kind": n_same_kind,
        "n_flipped_kind": n_flipped_kind,
        "n_other": n_other,
        "same_rate": n_same_kind / total_classified if total_classified else 0.0,
        "flipped_rate": n_flipped_kind / total_classified if total_classified else 0.0,
    }


# ─────────────────────────── main analysis ───────────────────────────

def run_analysis(
    dataset_root: Path,
    split: str,
    output_dir: Path,
    max_charts: int | None = None,
    max_pattern_charts: int = 500,
) -> None:
    ds_root = dataset_root.resolve()
    manifest = load_manifest(ds_root / "manifest.json")
    allowed = chart_ids_for_split(
        manifest, split,
        (("train", 0.9), ("val", 0.1)), 42,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graphs").mkdir(exist_ok=True)

    # ── accumulators ──
    # Per-kind mel at onset frame
    onset_mel: dict[int, BandAccumulator] = {i: BandAccumulator() for i in range(7)}
    # Per-kind mel window (+-MEL_WINDOW_HALF)
    onset_window: dict[int, WindowAccumulator] = {i: WindowAccumulator() for i in range(7)}

    # Transition matrices (corpus-wide)
    trans_4x4 = np.zeros((4, 4), dtype=np.int64)
    # D/K transition (2x2, merging big)
    trans_dk = np.zeros((2, 2), dtype=np.int64)

    # N-gram counters (D/K merged)
    ngram_2: Counter[tuple[int, ...]] = Counter()
    ngram_3: Counter[tuple[int, ...]] = Counter()
    ngram_4: Counter[tuple[int, ...]] = Counter()

    # Per-kind IOI accumulators
    ioi_accum: dict[str, list[int]] = {k: [] for k in KIND_NAMES[:6]}

    # Per-kind energy at onset (sum of mel bands)
    energy_by_kind: dict[int, list[float]] = {i: [] for i in range(6)}

    # Pattern repeat analysis (sampled)
    pattern_results: list[dict[str, Any]] = []

    # Corpus counts
    kind_counts = np.zeros(7, dtype=np.int64)
    n_charts = 0
    n_events_total = 0

    # Per-chart rows for CSV
    per_chart_rows: list[dict[str, Any]] = []

    charts = [e for e in manifest.charts if e.chart_id in allowed]
    if max_charts:
        charts = charts[:max_charts]

    t0 = time.time()

    for entry in tqdm(charts, desc="Analyzing kinds", unit="chart"):
        feat_path = ds_root / entry.features_path
        evt_path = ds_root / "events" / f"{_safe_filename(entry.chart_id)}.npz"
        if not feat_path.exists() or not evt_path.exists():
            continue

        try:
            features = np.load(feat_path, mmap_mode="r")  # (F, T)
            with np.load(evt_path) as data:
                bins = np.asarray(data["bins"], dtype=np.int64)
                kind_ids = np.asarray(data["kind_ids"], dtype=np.uint8)
        except Exception:
            continue

        n_mels, n_frames = features.shape
        if len(bins) < 2:
            continue

        # Filter out UNKNOWN
        mask = kind_ids != IDX_UNK
        bins = bins[mask]
        kind_ids = kind_ids[mask]
        if len(bins) < 2:
            continue

        n_charts += 1
        n_events_total += len(bins)

        # Kind counts (vectorized)
        for kid in range(7):
            kind_counts[kid] += int(np.sum(kind_ids == kid))

        # ── mel extraction (vectorized per kind) ──
        # Read features into memory once (mmap read is the bottleneck)
        hit_mask = kind_ids < 6
        hit_bins = bins[hit_mask]
        hit_kinds = kind_ids[hit_mask]
        frame_ok = (hit_bins >= 0) & (hit_bins < n_frames)

        if np.any(frame_ok):
            ok_bins = hit_bins[frame_ok]
            ok_kinds = hit_kinds[frame_ok]
            # Batch read all onset frames at once: (n_mels, K)
            all_frames = np.asarray(features[:, ok_bins], dtype=np.float32)
            all_energies = all_frames.sum(axis=0)  # (K,)

            for kid in range(6):
                kid_mask = ok_kinds == kid
                kid_count = int(kid_mask.sum())
                if kid_count == 0:
                    continue
                kid_frames = all_frames[:, kid_mask].T  # (K_kid, n_mels)
                onset_mel[kid].init_bands(n_mels)
                onset_mel[kid].update_batch(kid_frames)
                energy_by_kind[kid].extend(all_energies[kid_mask].tolist())

        # Window extraction (vectorized)
        W = 2 * MEL_WINDOW_HALF + 1
        win_ok = (hit_bins >= MEL_WINDOW_HALF) & (hit_bins + MEL_WINDOW_HALF + 1 <= n_frames)
        win_ok &= frame_ok
        if np.any(win_ok):
            wbins = hit_bins[win_ok]
            wkinds = hit_kinds[win_ok]
            # Build index array for all windows at once
            offsets = np.arange(-MEL_WINDOW_HALF, MEL_WINDOW_HALF + 1)
            col_idx = wbins[:, None] + offsets[None, :]  # (K_win, W)
            # Batch read: features[:, col_idx] → (n_mels, K_win, W)
            all_windows = np.asarray(features[:, col_idx.ravel()].reshape(n_mels, len(wbins), W),
                                     dtype=np.float32)
            # all_windows shape: (n_mels, K_win, W) → need (K_win, n_mels, W)
            all_windows = np.moveaxis(all_windows, 1, 0)

            for kid in range(6):
                kid_mask = wkinds == kid
                if not np.any(kid_mask):
                    continue
                onset_window[kid].update_batch(all_windows[kid_mask])

        # ── transition analysis (vectorized) ──
        trans_4x4 += build_transition_matrix(kind_ids)

        # D/K merged transitions (vectorized)
        dk_mapped = kind_ids.copy()
        dk_mapped[dk_mapped == IDX_BDON] = IDX_DON
        dk_mapped[dk_mapped == IDX_BKA] = IDX_KA
        if len(dk_mapped) >= 2:
            a = dk_mapped[:-1]
            b_ = dk_mapped[1:]
            dk_mask = (a < 2) & (b_ < 2)
            if np.any(dk_mask):
                flat = a[dk_mask].astype(np.int64) * 2 + b_[dk_mask].astype(np.int64)
                trans_dk += np.bincount(flat, minlength=4).reshape(2, 2)

        # ── n-grams (vectorized) ──
        ngram_2 += count_ngrams(kind_ids, 2)
        ngram_3 += count_ngrams(kind_ids, 3)
        ngram_4 += count_ngrams(kind_ids, 4)

        # ── IOI by kind (vectorized) ──
        if len(bins) >= 2:
            iois_arr = np.diff(bins)
            ioi_kinds = kind_ids[1:]
            for kid in range(6):
                mask = ioi_kinds == kid
                if np.any(mask):
                    ioi_accum[KIND_NAMES[kid]].extend(iois_arr[mask].tolist())

        # ── pattern repeat (sampled) ──
        if len(pattern_results) < max_pattern_charts:
            pr = pattern_repeat_analysis(bins, kind_ids)
            if pr["n_pattern_pairs"] > 0:
                pattern_results.append(pr)

        # ── per-chart CSV row ──
        chart_n = len(kind_ids)
        c_don = int(np.sum(kind_ids == IDX_DON))
        c_ka = int(np.sum(kind_ids == IDX_KA))
        c_bdon = int(np.sum(kind_ids == IDX_BDON))
        c_bka = int(np.sum(kind_ids == IDX_BKA))
        c_drum = int(np.sum(kind_ids == IDX_DRUM))
        c_spin = int(np.sum(kind_ids == IDX_SPIN))
        c_hits = c_don + c_ka + c_bdon + c_bka
        don_ratio = (c_don + c_bdon) / c_hits if c_hits > 0 else 0.5
        big_ratio = (c_bdon + c_bka) / c_hits if c_hits > 0 else 0.0
        # D/K alternation rate within this chart
        dk_m = dk_mapped  # already computed above
        if len(dk_m) >= 2:
            dk_pairs = dk_m[:-1].astype(np.int64) * 2 + dk_m[1:].astype(np.int64)
            dk_valid = (dk_m[:-1] < 2) & (dk_m[1:] < 2)
            n_dk_valid = int(dk_valid.sum())
            if n_dk_valid > 0:
                n_alt = int(np.sum(((dk_pairs == 1) | (dk_pairs == 2)) & dk_valid))
                alt_rate = n_alt / n_dk_valid
            else:
                alt_rate = 0.0
        else:
            alt_rate = 0.0

        per_chart_rows.append({
            "chart_id": entry.chart_id,
            "beatmap_id": entry.beatmap_id,
            "total_events": chart_n,
            "count_don": c_don,
            "count_ka": c_ka,
            "count_big_don": c_bdon,
            "count_big_ka": c_bka,
            "count_drumroll": c_drum,
            "count_spinner": c_spin,
            "don_ratio": round(don_ratio, 4),
            "big_ratio": round(big_ratio, 4),
            "dk_alternation_rate": round(alt_rate, 4),
            "star_rating": entry.star_rating,
            "density_mean": entry.density_mean,
        })

    elapsed = time.time() - t0
    print(f"\nProcessed {n_charts} charts, {n_events_total:,} events in {elapsed:.1f}s")

    # ── compute results ──
    results: dict[str, Any] = {
        "n_charts": n_charts,
        "n_events": n_events_total,
        "elapsed_s": round(elapsed, 1),
        "mel_window_half": MEL_WINDOW_HALF,
    }

    # Kind distribution
    results["kind_counts"] = {KIND_NAMES[i]: int(kind_counts[i]) for i in range(7)}
    total_hits = sum(int(kind_counts[i]) for i in HIT_KINDS)
    results["kind_fractions"] = {
        KIND_NAMES[i]: round(int(kind_counts[i]) / n_events_total, 4) if n_events_total else 0
        for i in range(6)
    }

    # ── Q1: D vs K mel separability ──
    don_mel = onset_mel[IDX_DON]
    ka_mel = onset_mel[IDX_KA]
    if don_mel.n > 10 and ka_mel.n > 10:
        fisher = fisher_lda_score(
            don_mel.mean, don_mel.var, don_mel.n,
            ka_mel.mean, ka_mel.var, ka_mel.n,
        )
        t_stats = per_band_ttest(
            don_mel.mean, don_mel.var, don_mel.n,
            ka_mel.mean, ka_mel.var, ka_mel.n,
        )
        results["dk_separability"] = {
            "fisher_lda_mean": round(fisher, 6),
            "t_stat_mean": round(float(np.mean(t_stats)), 4),
            "t_stat_max": round(float(np.max(t_stats)), 4),
            "t_stat_per_band": [round(float(x), 4) for x in t_stats],
            "don_n": don_mel.n,
            "ka_n": ka_mel.n,
            "mean_diff_per_band": [round(float(x), 6) for x in (don_mel.mean - ka_mel.mean)],
            "mean_abs_diff": round(float(np.mean(np.abs(don_mel.mean - ka_mel.mean))), 6),
            "mean_relative_diff": round(
                float(np.mean(np.abs(don_mel.mean - ka_mel.mean) /
                              (np.abs(don_mel.mean) + 1e-8))), 6
            ),
        }

    # ── Q3: BIG vs NORMAL separability ──
    for label, big_idx, norm_idx in [
        ("don_big_vs_normal", IDX_BDON, IDX_DON),
        ("ka_big_vs_normal", IDX_BKA, IDX_KA),
    ]:
        big = onset_mel[big_idx]
        norm = onset_mel[norm_idx]
        if big.n > 10 and norm.n > 10:
            fisher = fisher_lda_score(
                big.mean, big.var, big.n,
                norm.mean, norm.var, norm.n,
            )
            t_stats = per_band_ttest(
                big.mean, big.var, big.n,
                norm.mean, norm.var, norm.n,
            )
            results[f"{label}_separability"] = {
                "fisher_lda_mean": round(fisher, 6),
                "t_stat_mean": round(float(np.mean(t_stats)), 4),
                "t_stat_max": round(float(np.max(t_stats)), 4),
                "big_n": big.n,
                "normal_n": norm.n,
                "mean_abs_diff": round(float(np.mean(np.abs(big.mean - norm.mean))), 6),
            }

    # ── Q4: DRUMROLL/SPINNER vs hits ──
    for label, sp_idx in [("drumroll_vs_hits", IDX_DRUM), ("spinner_vs_hits", IDX_SPIN)]:
        sp = onset_mel[sp_idx]
        # Compare against combined DON+KA
        if sp.n > 5 and don_mel.n > 10 and ka_mel.n > 10:
            combined_n = don_mel.n + ka_mel.n
            combined_mean = (don_mel.mean * don_mel.n + ka_mel.mean * ka_mel.n) / combined_n
            combined_var = (don_mel.var * don_mel.n + ka_mel.var * ka_mel.n) / combined_n
            fisher = fisher_lda_score(
                sp.mean, sp.var, sp.n,
                combined_mean, combined_var, combined_n,
            )
            results[f"{label}_separability"] = {
                "fisher_lda_mean": round(fisher, 6),
                "special_n": sp.n,
                "hit_n": combined_n,
            }

    # ── energy by kind ──
    energy_stats: dict[str, dict[str, float]] = {}
    for kid in range(6):
        vals = energy_by_kind[kid]
        if vals:
            arr = np.array(vals)
            energy_stats[KIND_NAMES[kid]] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "n": len(vals),
            }
    results["energy_by_kind"] = energy_stats

    # ── transition matrices ──
    trans_4x4_norm = trans_4x4.astype(np.float64)
    row_sums = trans_4x4_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    trans_4x4_prob = trans_4x4_norm / row_sums

    trans_dk_norm = trans_dk.astype(np.float64)
    dk_sums = trans_dk_norm.sum(axis=1, keepdims=True)
    dk_sums = np.where(dk_sums == 0, 1, dk_sums)
    trans_dk_prob = trans_dk_norm / dk_sums

    results["transition_4x4"] = {
        "counts": trans_4x4.tolist(),
        "probabilities": [[round(x, 4) for x in row] for row in trans_4x4_prob.tolist()],
        "labels": [KIND_NAMES[i] for i in range(4)],
    }
    results["transition_dk"] = {
        "counts": trans_dk.tolist(),
        "probabilities": [[round(x, 4) for x in row] for row in trans_dk_prob.tolist()],
        "labels": ["D", "K"],
        "P(alt|D)": round(float(trans_dk_prob[0, 1]), 4),
        "P(alt|K)": round(float(trans_dk_prob[1, 0]), 4),
        "P(same|D)": round(float(trans_dk_prob[0, 0]), 4),
        "P(same|K)": round(float(trans_dk_prob[1, 1]), 4),
    }

    # ── n-grams ──
    def ngram_to_str(gram: tuple[int, ...]) -> str:
        return "".join("D" if g == 0 else "K" for g in gram)

    results["ngrams"] = {}
    for n, counter in [(2, ngram_2), (3, ngram_3), (4, ngram_4)]:
        total = sum(counter.values())
        top = counter.most_common(20)
        results["ngrams"][f"{n}gram"] = {
            "total": total,
            "top": [
                {"pattern": ngram_to_str(gram), "count": cnt,
                 "pct": round(cnt / total * 100, 2) if total else 0}
                for gram, cnt in top
            ],
        }

    # ── IOI by kind ──
    ioi_stats: dict[str, dict[str, float]] = {}
    for k in KIND_NAMES[:6]:
        vals = ioi_accum[k]
        if vals:
            arr = np.array(vals)
            ioi_stats[k] = {
                "mean": round(float(np.mean(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "n": len(vals),
            }
    results["ioi_by_kind"] = ioi_stats

    # ── Q5: pattern repeat consistency ──
    if pattern_results:
        total_pairs = sum(r["n_pattern_pairs"] for r in pattern_results)
        total_same = sum(r["n_same_kind"] for r in pattern_results)
        total_flip = sum(r["n_flipped_kind"] for r in pattern_results)
        total_other = sum(r["n_other"] for r in pattern_results)
        total_class = total_same + total_flip + total_other
        results["pattern_repeat"] = {
            "n_charts_analyzed": len(pattern_results),
            "n_pattern_pairs": total_pairs,
            "n_classified": total_class,
            "n_same_kind": total_same,
            "n_flipped_kind": total_flip,
            "n_other": total_other,
            "same_rate": round(total_same / total_class, 4) if total_class else 0,
            "flipped_rate": round(total_flip / total_class, 4) if total_class else 0,
            "other_rate": round(total_other / total_class, 4) if total_class else 0,
        }

    # ── per-chart aggregate distributions ──
    def _dist_stats(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {}
        return {
            "n": len(arr),
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "p5": round(float(np.percentile(arr, 5)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    if per_chart_rows:
        for field_name in [
            "don_ratio", "big_ratio", "dk_alternation_rate",
            "count_don", "count_ka", "count_big_don", "count_big_ka",
            "count_drumroll", "count_spinner", "total_events",
        ]:
            vals = np.array([r[field_name] for r in per_chart_rows if r[field_name] is not None])
            if len(vals) > 0:
                results[f"per_chart_{field_name}"] = _dist_stats(vals)

    # ── per-kind energy distributions (full) ──
    energy_full_stats: dict[str, dict[str, float]] = {}
    for kid in range(6):
        vals = energy_by_kind[kid]
        if vals:
            energy_full_stats[KIND_NAMES[kid]] = _dist_stats(np.array(vals))
    results["energy_by_kind_full"] = energy_full_stats

    # ── per-kind IOI distributions (full) ──
    ioi_full_stats: dict[str, dict[str, float]] = {}
    for k in KIND_NAMES[:6]:
        vals = ioi_accum[k]
        if vals:
            ioi_full_stats[k] = _dist_stats(np.array(vals))
    results["ioi_by_kind_full"] = ioi_full_stats

    # ── write per-chart CSV ──
    if per_chart_rows:
        import csv as csv_mod
        csv_path = output_dir / "per_chart.csv"
        fieldnames = list(per_chart_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_chart_rows)
        print(f"Wrote {csv_path} ({len(per_chart_rows)} rows)")

    # ── save mel stats ──
    mel_stats = {}
    for kid in range(6):
        acc = onset_mel[kid]
        if acc.n > 0:
            mel_stats[KIND_NAMES[kid]] = {
                "n": acc.n,
                "mean": acc.mean.tolist(),
                "std": acc.std.tolist(),
            }
    results["mel_at_onset"] = mel_stats

    # ── save window means ──
    window_means = {}
    for kid in range(6):
        acc = onset_window[kid]
        if acc.n > 0:
            window_means[KIND_NAMES[kid]] = {
                "n": acc.n,
                "shape": list(acc.mean.shape),
            }
    results["mel_window_info"] = window_means

    # ── save ──
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {summary_path}")

    # Save window means as npz for plotting
    window_npz: dict[str, np.ndarray] = {}
    for kid in range(6):
        acc = onset_window[kid]
        if acc.n > 0:
            window_npz[f"{KIND_NAMES[kid]}_mean"] = acc.mean.astype(np.float32)
            window_npz[f"{KIND_NAMES[kid]}_var"] = acc.var.astype(np.float32)
    if window_npz:
        npz_path = output_dir / "mel_windows.npz"
        np.savez(npz_path, **window_npz)
        print(f"Wrote {npz_path}")

    # ── graphs ──
    _plot_results(results, output_dir / "graphs", window_npz)

    print(f"\nDone. Results in {output_dir}")


# ─────────────────────────── plotting ────────────────────────────────

def _plot_results(
    results: dict[str, Any], graph_dir: Path,
    window_data: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping graphs")
        return

    # 01 — D vs K mean mel per band
    if "dk_separability" in results and "mel_at_onset" in results:
        mel = results["mel_at_onset"]
        if "don" in mel and "ka" in mel:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            don_mean = np.array(mel["don"]["mean"])
            ka_mean = np.array(mel["ka"]["mean"])
            don_std = np.array(mel["don"]["std"])
            ka_std = np.array(mel["ka"]["std"])
            bands = np.arange(len(don_mean))

            ax = axes[0]
            ax.plot(bands, don_mean, label="DON", color="tab:red", alpha=0.8)
            ax.plot(bands, ka_mean, label="KA", color="tab:blue", alpha=0.8)
            ax.fill_between(bands, don_mean - don_std, don_mean + don_std,
                            color="tab:red", alpha=0.1)
            ax.fill_between(bands, ka_mean - ka_std, ka_mean + ka_std,
                            color="tab:blue", alpha=0.1)
            ax.set_xlabel("Mel band")
            ax.set_ylabel("Mean log-mel energy (dB)")
            ax.set_title("DON vs KA: mean mel spectrum at onset frame")
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            diff = don_mean - ka_mean
            ax.bar(bands, diff, color=np.where(diff > 0, "tab:red", "tab:blue"), alpha=0.7)
            ax.set_xlabel("Mel band")
            ax.set_ylabel("DON - KA (dB)")
            ax.set_title(f"Per-band difference (Fisher LDA = {results['dk_separability']['fisher_lda_mean']:.6f})")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(graph_dir / "01_dk_mel_comparison.png", dpi=150)
            plt.close()
            print(f"  Wrote 01_dk_mel_comparison.png")

    # 02 — BIG vs NORMAL
    if "mel_at_onset" in results:
        mel = results["mel_at_onset"]
        for label, big_name, norm_name in [
            ("don", "big_don", "don"), ("ka", "big_ka", "ka")
        ]:
            if big_name in mel and norm_name in mel:
                fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                big_mean = np.array(mel[big_name]["mean"])
                norm_mean = np.array(mel[norm_name]["mean"])
                bands = np.arange(len(big_mean))
                diff = big_mean - norm_mean
                ax.bar(bands, diff, color=np.where(diff > 0, "tab:orange", "tab:green"), alpha=0.7)
                ax.set_xlabel("Mel band")
                ax.set_ylabel(f"BIG_{label.upper()} - {label.upper()} (dB)")
                sep_key = f"{label}_big_vs_normal_separability"
                fisher_val = results.get(sep_key, {}).get("fisher_lda_mean", "?")
                ax.set_title(f"BIG vs NORMAL {label.upper()} (Fisher = {fisher_val})")
                ax.axhline(0, color="black", linewidth=0.5)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(graph_dir / f"02_{label}_big_vs_normal.png", dpi=150)
                plt.close()
                print(f"  Wrote 02_{label}_big_vs_normal.png")

    # 03 — D/K transition heatmap
    if "transition_dk" in results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, key, title in [
            (axes[0], "transition_dk", "D/K transitions (merged big)"),
            (axes[1], "transition_4x4", "4-way transitions"),
        ]:
            data = results[key]
            probs = np.array(data["probabilities"])
            labels = data["labels"]
            im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Next")
            ax.set_ylabel("Current")
            ax.set_title(title)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{probs[i, j]:.3f}",
                            ha="center", va="center", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(graph_dir / "03_transition_heatmaps.png", dpi=150)
        plt.close()
        print(f"  Wrote 03_transition_heatmaps.png")

    # 04 — n-gram frequencies
    if "ngrams" in results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, n in zip(axes, [2, 3, 4]):
            data = results["ngrams"][f"{n}gram"]
            top = data["top"][:16]
            patterns = [t["pattern"] for t in top]
            pcts = [t["pct"] for t in top]
            bars = ax.barh(range(len(patterns)), pcts, color="steelblue", alpha=0.8)
            ax.set_yticks(range(len(patterns)))
            ax.set_yticklabels(patterns, fontfamily="monospace")
            ax.set_xlabel("% of all n-grams")
            ax.set_title(f"{n}-grams (D/K merged)")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(graph_dir / "04_ngram_frequencies.png", dpi=150)
        plt.close()
        print(f"  Wrote 04_ngram_frequencies.png")

    # 05 — energy by kind
    if "energy_by_kind" in results:
        data = results["energy_by_kind"]
        kinds_with_data = [(k, v) for k, v in data.items() if v["n"] > 0]
        if kinds_with_data:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            names = [k for k, v in kinds_with_data]
            means = [v["mean"] for k, v in kinds_with_data]
            stds = [v["std"] for k, v in kinds_with_data]
            colors = ["tab:red", "tab:blue", "tab:orange", "tab:cyan", "tab:green", "tab:purple"]
            ax.bar(names, means, yerr=stds, color=colors[:len(names)], alpha=0.7, capsize=5)
            ax.set_ylabel("Sum of mel bands at onset frame")
            ax.set_title("Total mel energy at onset, by kind")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(graph_dir / "05_energy_by_kind.png", dpi=150)
            plt.close()
            print(f"  Wrote 05_energy_by_kind.png")

    # 06 — mel window comparison (D vs K, time axis)
    if "don_mean" in window_data and "ka_mean" in window_data:
        don_w = window_data["don_mean"]  # (F, W)
        ka_w = window_data["ka_mean"]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        ax = axes[0]
        im = ax.imshow(don_w, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("DON: mean mel window")
        ax.set_ylabel("Mel band")
        ax.axvline(MEL_WINDOW_HALF, color="white", linestyle="--", alpha=0.7)
        plt.colorbar(im, ax=ax)

        ax = axes[1]
        im = ax.imshow(ka_w, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("KA: mean mel window")
        ax.set_ylabel("Mel band")
        ax.axvline(MEL_WINDOW_HALF, color="white", linestyle="--", alpha=0.7)
        plt.colorbar(im, ax=ax)

        ax = axes[2]
        diff = don_w - ka_w
        vmax = max(abs(diff.min()), abs(diff.max()))
        im = ax.imshow(diff, aspect="auto", origin="lower", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax)
        ax.set_title("DON - KA difference")
        ax.set_ylabel("Mel band")
        ax.set_xlabel(f"Frame offset from onset (0 = -{MEL_WINDOW_HALF}, center = onset)")
        ax.axvline(MEL_WINDOW_HALF, color="black", linestyle="--", alpha=0.7)
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(graph_dir / "06_dk_window_comparison.png", dpi=150)
        plt.close()
        print(f"  Wrote 06_dk_window_comparison.png")

    # 07 — pattern repeat
    if "pattern_repeat" in results:
        pr = results["pattern_repeat"]
        if pr["n_classified"] > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            labels = ["Same D/K", "Flipped D/K", "Other"]
            sizes = [pr["n_same_kind"], pr["n_flipped_kind"], pr["n_other"]]
            colors = ["tab:green", "tab:orange", "tab:gray"]
            ax.bar(labels, sizes, color=colors, alpha=0.8)
            ax.set_ylabel("Count")
            ax.set_title(
                f"When same IOI pattern repeats in a chart: D/K assignment\n"
                f"(n={pr['n_classified']} pairs from {pr['n_charts_analyzed']} charts)"
            )
            for i, (lbl, sz) in enumerate(zip(labels, sizes)):
                total = pr["n_classified"]
                ax.text(i, sz + total * 0.01, f"{sz/total*100:.1f}%",
                        ha="center", fontsize=11)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(graph_dir / "07_pattern_repeat_consistency.png", dpi=150)
            plt.close()
            print(f"  Wrote 07_pattern_repeat_consistency.png")


# ─────────────────────────── CLI ─────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze onset kind acoustics and patterns.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (directory under osu/taiko2/datasets/)",
    )
    parser.add_argument(
        "--split", default="all",
        help="Split to analyze (default: all)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-charts", type=int, default=None,
        help="Limit number of charts (for debugging)",
    )
    parser.add_argument(
        "--max-pattern-charts", type=int, default=500,
        help="Max charts for pattern repeat analysis (expensive)",
    )
    args = parser.parse_args()

    taiko2_root = Path(__file__).resolve().parent.parent
    ds_root = taiko2_root / "datasets" / args.dataset

    if not ds_root.exists():
        print(f"Dataset not found: {ds_root}", file=sys.stderr)
        sys.exit(1)

    run_analysis(
        dataset_root=ds_root,
        split=args.split,
        output_dir=Path(args.output),
        max_charts=args.max_charts,
        max_pattern_charts=args.max_pattern_charts,
    )


if __name__ == "__main__":
    main()
