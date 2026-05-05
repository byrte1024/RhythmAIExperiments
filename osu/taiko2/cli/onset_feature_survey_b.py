"""Onset disagreement / complementarity survey (#011b).

Same mel-domain algorithms as ``onset_feature_survey`` but at a fixed
per-algo best-F1 operating point. Records, for every (chart, GT
onset, algorithm), whether the algorithm caught that onset within
±10 frames. From that boolean matrix derives:

  - pairwise complementarity (Jaccard, marginal-gain)
  - 1 / 2 / 3 / 4-channel union F1 (best by F1, not saturated as
    in #011)
  - per-onset-kind recall (DON / KA / BIG_DON / BIG_KA / DRUMROLL /
    SPINNER) per algorithm
  - per-density-bucket and per-star-rating F1 per algorithm
  - activation-value histograms at TP / FP / FN / near-miss frames

Outputs JSON + CSV under ``--output``. Plot script lives next to
the experiment README.

Usage::

    python -m osu.taiko2.cli.onset_feature_survey_b \\
        --dataset taiko2_v1 \\
        --output osu/taiko2/experiments/011b-onset-disagreement/results \\
        --split val --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..analysis.onset_features import (
    energy,
    hfc_mel,
    log_filtered_flux,
    mel_band_center_freqs_hz,
    normalize_activation,
    peak_pick,
    spectral_flux,
    subband_flux,
    superflux,
)
from ..dataset import _safe_filename
from ..persistence.events import load_events
from ..persistence.manifest import load_manifest
from ..splits import chart_ids_for_split


# ─────────────────────────── Config ───────────────────────────────────


# Best-F1 thresholds @ ±10 frames from #011's full-val results. Fixed
# per algorithm so this experiment uses a stable operating point.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "energy":             0.76,
    "spectral_flux":      0.32,
    "log_filtered_flux":  0.24,
    "hfc_mel":            0.68,
    "superflux":          0.28,
    "subband_sf_4":       0.32,
    "subband_sf_8":       0.32,
}

# Canonical evaluation tolerance — same ±10 frames (±50 ms) used
# throughout #011 and the MIREX standard.
DEFAULT_TOLERANCE = 10

# Onset kind names map for per-kind aggregation. Mirrors
# `persistence.events._KIND_ORDER` index → enum name (verified
# 2026-05-05: order is DON / KA / BIG_DON / BIG_KA / DRUMROLL /
# SPINNER / UNKNOWN; UNKNOWN is the catch-all index 6, not 0).
KIND_NAMES = ("DON", "KA", "BIG_DON", "BIG_KA",
              "DRUMROLL", "SPINNER", "UNKNOWN")

# Density buckets (events / s) and star buckets — quartile cuts on
# the val split would be ideal but are data-dependent; use principled
# fixed cuts that match the human-meaningful "easy / medium / hard"
# segmentation taiko1 used.
DENSITY_BUCKETS = ((-1.0, 1.5, "sparse"),
                   (1.5, 3.0, "medium"),
                   (3.0, 5.0, "dense"),
                   (5.0, 1e9, "very_dense"))
STAR_BUCKETS = ((-1.0, 3.0, "easy"),
                (3.0, 4.0, "medium"),
                (4.0, 5.0, "hard"),
                (5.0, 1e9, "insane_plus"))


# ─────────────────────────── Args ─────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Onset disagreement + sub-analysis survey.",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--split-ratios", default="train:0.9,val:0.1")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-charts", type=int, default=None)
    p.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE,
                   help="Frame tolerance for the canonical eval (default 10 = 50 ms).")
    p.add_argument("--norm-percentile", type=float, default=99.0)
    p.add_argument("--min-peak-distance", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--n-activation-samples", type=int, default=200,
                   help="Per-chart cap on activation values sampled into "
                        "the TP/FP/FN/near-miss histograms (per algorithm). "
                        "0 disables the histogram pass.")
    return p.parse_args(argv)


def _parse_split_ratios(raw: str) -> tuple[tuple[str, float], ...]:
    parts: list[tuple[str, float]] = []
    for frag in raw.split(","):
        name, _, ratio = frag.strip().partition(":")
        if not name or not ratio:
            raise ValueError(f"bad split-ratios fragment {frag!r}")
        parts.append((name.strip(), float(ratio)))
    return tuple(parts)


def _resolve_dataset(name_or_path: str, datasets_dir: Path) -> Path:
    p = Path(name_or_path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (datasets_dir / name_or_path).resolve()


def _bucket(value: float, buckets: tuple) -> str:
    for lo, hi, label in buckets:
        if lo <= value < hi:
            return label
    return "unknown"


# ─────────────────────────── ODF compute ──────────────────────────────


def _compute_activations(
    log_mel: torch.Tensor, freqs_hz: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return (collapsed_envelopes, per_band_envelopes).

    The collapsed dict has the same keys / semantics as #011 — one
    1-D envelope per algorithm, used for peak picking. The per-band
    dict is the (n_bands, T) tensors for sub-band variants, which
    are peak-picked per band so we can aggregate per (band, kind).
    """
    out: dict[str, torch.Tensor] = {}
    out["energy"] = energy(log_mel)
    out["spectral_flux"] = spectral_flux(log_mel)
    out["log_filtered_flux"] = log_filtered_flux(log_mel)
    out["hfc_mel"] = hfc_mel(log_mel, freqs_hz)
    out["superflux"] = superflux(log_mel)

    bands_by_n: dict[str, torch.Tensor] = {}
    for n_bands, name in [(4, "subband_sf_4"), (8, "subband_sf_8")]:
        bands = subband_flux(log_mel, n_bands=n_bands)        # (n, T)
        bands_by_n[name] = bands
        out[name] = bands.sum(dim=0)
    return out, bands_by_n


# ─────────────────────────── Detection record ─────────────────────────


@dataclass(slots=True)
class ChartRecord:
    """Per-chart accumulator. ``caught[algo]`` is a (n_gt,) bool tensor
    saying whether each GT onset was caught by ``algo`` within
    ``tolerance``. ``n_pred[algo]`` and ``n_tp[algo]`` give the algo's
    predicted-peak count and TP count for precision computation.

    ``activation_samples[algo]`` is a small dict of label → list of
    floats sampled from the normalized activation envelope at TP /
    FP / FN / near-miss frames. Capped per chart to keep memory
    bounded.
    """
    chart_id: str
    n_gt: int
    density_mean: float
    star_rating: float
    kinds: np.ndarray             # (n_gt,) uint8 kind ids
    caught: dict[str, np.ndarray]
    n_pred: dict[str, int]
    n_tp: dict[str, int]
    activation_samples: dict[str, dict[str, list[float]]]
    # Per-(sub-band-name, band-idx, kind-name) caught counts. Kept as
    # int counters rather than per-onset matrices to bound memory.
    subband_per_kind_caught: dict[str, dict[int, dict[str, int]]]
    subband_per_kind_total: dict[str, dict[str, int]]


def _evaluate_per_onset(
    pred_frames: np.ndarray,         # (n_pred,) int sorted
    gt_frames: np.ndarray,           # (n_gt,) int sorted
    *,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy nearest-match. Returns (gt_caught, pred_is_tp, gt_match_pred_idx).

    ``gt_caught[i]`` = True if GT onset i has a matching prediction.
    ``pred_is_tp[j]`` = True if prediction j matched some GT.
    ``gt_match_pred_idx[i]`` = index into pred_frames for the matched
        prediction, or -1 if no match.
    """
    n_p, n_g = len(pred_frames), len(gt_frames)
    gt_caught = np.zeros(n_g, dtype=bool)
    pred_is_tp = np.zeros(n_p, dtype=bool)
    gt_match = np.full(n_g, -1, dtype=np.int64)
    if n_p == 0 or n_g == 0:
        return gt_caught, pred_is_tp, gt_match

    j_start = 0
    for i in range(n_g):
        g = int(gt_frames[i])
        while j_start < n_p and pred_frames[j_start] < g - tolerance:
            j_start += 1
        best_j = -1
        best_d = tolerance + 1
        j = j_start
        while j < n_p and pred_frames[j] <= g + tolerance:
            if not pred_is_tp[j]:
                d = abs(int(pred_frames[j]) - g)
                if d < best_d:
                    best_d = d
                    best_j = j
            j += 1
        if best_j >= 0:
            pred_is_tp[best_j] = True
            gt_caught[i] = True
            gt_match[i] = best_j
    return gt_caught, pred_is_tp, gt_match


def _sample_activations(
    activation_n: torch.Tensor,      # (T,) normalized
    *,
    threshold: float,
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    pred_is_tp: np.ndarray,
    gt_caught: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Sample activation values for the four frame categories.

    - TP: activation at the predicted-peak frames that matched a GT.
    - FP: activation at the predicted-peak frames that didn't match a GT.
    - FN_no_peak: activation at the GT frame, capped at the threshold,
      for GTs that have *no* peak above threshold within ±tolerance.
    - FN_near_miss: activation at the GT frame for GTs that DO have
      a local max within ±tolerance, just below threshold.
    """
    if n_samples <= 0:
        return {"tp": [], "fp": [], "fn": [], "near_miss": []}

    act = activation_n.detach().cpu().numpy()
    T = act.shape[0]

    # TP / FP from predicted frames.
    tp_idx = pred_frames[pred_is_tp]
    fp_idx = pred_frames[~pred_is_tp]

    # FN bins (GT frames missed by the algo).
    miss_idx = gt_frames[~gt_caught]

    # For each missed GT, look at the activation value at gt frame and
    # within ±tolerance. If max in window is just below threshold, it's
    # a "near miss"; if it's much lower, it's "no peak."
    fn_strict: list[float] = []
    fn_near: list[float] = []
    for g in miss_idx:
        lo = max(0, int(g) - 10)
        hi = min(T, int(g) + 10 + 1)
        if hi <= lo:
            continue
        local_max = float(act[lo:hi].max())
        # 0.6 of threshold = arbitrary "close enough" cut for "near miss"
        # vs "no peak". The histogram makes this distinction clearer
        # than the binary classification.
        if local_max >= threshold * 0.6:
            fn_near.append(local_max)
        else:
            fn_strict.append(local_max)

    def _sample(values: np.ndarray | list[float]) -> list[float]:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size <= n_samples:
            return arr.tolist()
        idx = rng.choice(arr.size, size=n_samples, replace=False)
        return arr[idx].tolist()

    return {
        "tp": _sample(act[tp_idx]) if tp_idx.size else [],
        "fp": _sample(act[fp_idx]) if fp_idx.size else [],
        "fn": _sample(np.array(fn_strict)) if fn_strict else [],
        "near_miss": _sample(np.array(fn_near)) if fn_near else [],
    }


def _process_chart(
    *,
    chart_id: str,
    log_mel: torch.Tensor,
    gt_frames_t: torch.Tensor,
    gt_kinds_np: np.ndarray,
    density_mean: float,
    star_rating: float,
    freqs_hz: torch.Tensor,
    thresholds: dict[str, float],
    tolerance: int,
    norm_percentile: float,
    min_peak_distance: int,
    n_activation_samples: int,
    rng: np.random.Generator,
) -> ChartRecord:
    n_gt = int(gt_frames_t.numel())
    gt_np = gt_frames_t.detach().cpu().numpy()

    activations, sub_band_envelopes = _compute_activations(log_mel, freqs_hz)

    caught: dict[str, np.ndarray] = {}
    n_pred: dict[str, int] = {}
    n_tp: dict[str, int] = {}
    activation_samples: dict[str, dict[str, list[float]]] = {}

    for algo, act in activations.items():
        thr = float(thresholds.get(algo, 0.32))
        act_n = normalize_activation(act, percentile=norm_percentile)
        peaks = peak_pick(
            act_n, threshold=thr, min_distance=min_peak_distance,
        )
        peaks_np = peaks.detach().cpu().numpy().astype(np.int64)
        peaks_np.sort()
        gt_caught, pred_is_tp, _ = _evaluate_per_onset(
            peaks_np, gt_np, tolerance=tolerance,
        )
        caught[algo] = gt_caught
        n_pred[algo] = int(peaks_np.size)
        n_tp[algo] = int(pred_is_tp.sum())
        activation_samples[algo] = _sample_activations(
            act_n,
            threshold=thr,
            pred_frames=peaks_np,
            gt_frames=gt_np,
            pred_is_tp=pred_is_tp,
            gt_caught=gt_caught,
            n_samples=n_activation_samples,
            rng=rng,
        )

    # Per sub-band, per kind: how many of each onset kind does each
    # band catch on its own? Bands use the same threshold as the
    # collapsed envelope (best-F1 threshold for the collapsed signal),
    # which is conservative for the per-band view but lets us compare
    # against the collapsed numbers directly.
    sb_caught: dict[str, dict[int, dict[str, int]]] = {}
    sb_total: dict[str, dict[str, int]] = {}
    for sb_name, bands in sub_band_envelopes.items():
        thr = float(thresholds.get(sb_name, 0.32))
        sb_total[sb_name] = defaultdict(int)
        for k_id in gt_kinds_np:
            kind_name = KIND_NAMES[int(k_id)] if int(k_id) < len(KIND_NAMES) else "UNKNOWN"
            sb_total[sb_name][kind_name] += 1
        sb_caught[sb_name] = {}
        for b_idx in range(bands.shape[0]):
            band_act = normalize_activation(
                bands[b_idx], percentile=norm_percentile,
            )
            band_peaks = peak_pick(
                band_act, threshold=thr, min_distance=min_peak_distance,
            )
            band_peaks_np = band_peaks.detach().cpu().numpy().astype(np.int64)
            band_peaks_np.sort()
            gt_caught_b, _, _ = _evaluate_per_onset(
                band_peaks_np, gt_np, tolerance=tolerance,
            )
            counts: dict[str, int] = defaultdict(int)
            for caught_flag, k_id in zip(gt_caught_b, gt_kinds_np):
                if caught_flag:
                    kind_name = KIND_NAMES[int(k_id)] if int(k_id) < len(KIND_NAMES) else "UNKNOWN"
                    counts[kind_name] += 1
            sb_caught[sb_name][b_idx] = dict(counts)

    return ChartRecord(
        chart_id=chart_id,
        n_gt=n_gt,
        density_mean=density_mean,
        star_rating=star_rating,
        kinds=gt_kinds_np,
        caught=caught,
        n_pred=n_pred,
        n_tp=n_tp,
        activation_samples=activation_samples,
        subband_per_kind_caught=sb_caught,
        subband_per_kind_total={k: dict(v) for k, v in sb_total.items()},
    )


# ─────────────────────────── Aggregation ──────────────────────────────


def _aggregate_pairwise(records: list[ChartRecord]) -> dict[str, Any]:
    """Pairwise complementarity across all GT onsets.

    Stacks every chart's `caught[algo]` boolean column to form a
    (total_n_gt, n_algos) matrix, then computes pairwise statistics
    by simple boolean ops.
    """
    if not records:
        return {}
    algos = sorted(records[0].caught.keys())
    columns: list[np.ndarray] = []
    for algo in algos:
        cols_for_algo = [r.caught[algo] for r in records]
        columns.append(np.concatenate(cols_for_algo))
    matrix = np.stack(columns, axis=1)            # (N, n_algos) bool
    total = matrix.shape[0]

    pairs: dict[str, dict[str, float]] = {}
    for i, a in enumerate(algos):
        for j, b in enumerate(algos):
            if j <= i:
                continue
            ai = matrix[:, i]
            bj = matrix[:, j]
            both = int(np.logical_and(ai, bj).sum())
            a_only = int(np.logical_and(ai, ~bj).sum())
            b_only = int(np.logical_and(~ai, bj).sum())
            neither = total - both - a_only - b_only
            recall_a = (both + a_only) / total
            recall_b = (both + b_only) / total
            recall_union = (both + a_only + b_only) / total
            jaccard = both / (both + a_only + b_only) if (both + a_only + b_only) > 0 else 1.0
            marg_b_given_a = b_only / total
            marg_a_given_b = a_only / total
            pairs[f"{a}+{b}"] = {
                "n_both": both, "n_a_only": a_only, "n_b_only": b_only,
                "n_neither": neither, "n_total": total,
                "recall_a": recall_a, "recall_b": recall_b,
                "recall_union": recall_union,
                "jaccard": jaccard,
                "marg_b_given_a": marg_b_given_a,
                "marg_a_given_b": marg_a_given_b,
            }

    return {"algos": algos, "n_total": total, "pairs": pairs,
            "caught_matrix_shape": list(matrix.shape)}


def _aggregate_subset_unions(records: list[ChartRecord]) -> dict[str, Any]:
    """For every subset of size 1..4, compute the union recall + the
    pooled precision (using the algo-level peak counts as the union
    proxy: union peak count = sum of peak counts, conservative upper
    bound) and a pooled F1.

    Recall is exact (boolean OR of caught columns); precision is an
    approximation (it overcounts because peaks shared between algos
    are double-counted) — for the channel-design decision we mostly
    care about recall and the F1 ranking is informative directionally.
    """
    if not records:
        return {}
    from itertools import combinations

    algos = sorted(records[0].caught.keys())
    columns = {a: np.concatenate([r.caught[a] for r in records]) for a in algos}
    n_pred_per_algo = {
        a: sum(r.n_pred[a] for r in records) for a in algos
    }
    n_tp_per_algo = {
        a: sum(r.n_tp[a] for r in records) for a in algos
    }
    total_gt = len(next(iter(columns.values())))

    out: dict[str, dict[str, float]] = {}
    max_size = min(4, len(algos))
    for size in range(1, max_size + 1):
        for combo in combinations(algos, size):
            tag = "+".join(combo)
            union = np.zeros(total_gt, dtype=bool)
            for a in combo:
                union |= columns[a]
            tp = int(union.sum())
            recall = tp / total_gt
            n_union_pred = sum(n_pred_per_algo[a] for a in combo)
            n_union_tp_upper = sum(n_tp_per_algo[a] for a in combo)
            precision_upper = (
                n_union_tp_upper / n_union_pred
                if n_union_pred > 0 else 0.0
            )
            # Recall-side precision lower bound: assume all unique
            # union TP frames are recovered without duplication.
            precision_lower = tp / n_union_pred if n_union_pred > 0 else 0.0
            # Use the more honest recall-bound F1 (single-counts each
            # unique GT detected).
            f1 = (2.0 * precision_lower * recall) / (precision_lower + recall) \
                if (precision_lower + recall) > 0 else 0.0
            out[tag] = {
                "size": size,
                "recall": recall,
                "precision_lower": precision_lower,
                "precision_upper": precision_upper,
                "f1": f1,
                "n_pred_union": n_union_pred,
                "n_tp_unique": tp,
            }
    return out


def _aggregate_per_kind(records: list[ChartRecord]) -> dict[str, Any]:
    """Per-(algo, kind) recall over all GT onsets."""
    if not records:
        return {}
    algos = sorted(records[0].caught.keys())
    totals: dict[str, int] = defaultdict(int)
    caught: dict[str, dict[str, int]] = {a: defaultdict(int) for a in algos}
    for r in records:
        for kid in r.kinds:
            kname = KIND_NAMES[int(kid)] if int(kid) < len(KIND_NAMES) else "UNKNOWN"
            totals[kname] += 1
        for a in algos:
            cflags = r.caught[a]
            for caught_flag, kid in zip(cflags, r.kinds):
                if caught_flag:
                    kname = KIND_NAMES[int(kid)] if int(kid) < len(KIND_NAMES) else "UNKNOWN"
                    caught[a][kname] += 1

    rows: dict[str, dict[str, dict[str, float]]] = {}
    for a in algos:
        per_kind: dict[str, dict[str, float]] = {}
        for kind, total in totals.items():
            n = caught[a].get(kind, 0)
            per_kind[kind] = {
                "n": int(total),
                "caught": int(n),
                "recall": n / total if total > 0 else 0.0,
            }
        rows[a] = per_kind
    return {"per_kind": rows, "totals": dict(totals)}


def _aggregate_per_bucket(
    records: list[ChartRecord],
    *,
    bucket_field: str,
    buckets: tuple,
) -> dict[str, Any]:
    """Per-(algo, bucket) F1 — recall + an approximation to precision
    using each chart's peak counts.

    Precision here is per-chart-summed n_tp / n_pred (so it's exact
    per algorithm; pooling across charts inside the bucket).
    """
    if not records:
        return {}
    algos = sorted(records[0].caught.keys())
    by_bucket_algo: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0,
                                      "pred": 0}),
    )
    for r in records:
        value = getattr(r, bucket_field)
        bucket = _bucket(float(value), buckets)
        for a in algos:
            cflags = r.caught[a]
            tp = int(cflags.sum())
            gt = int(cflags.size)
            pred = int(r.n_pred[a])
            tp_in_pred = int(r.n_tp[a])
            fp = pred - tp_in_pred
            fn = gt - tp
            by_bucket_algo[bucket][a]["tp"] += tp
            by_bucket_algo[bucket][a]["fp"] += fp
            by_bucket_algo[bucket][a]["fn"] += fn
            by_bucket_algo[bucket][a]["gt"] += gt
            by_bucket_algo[bucket][a]["pred"] += pred

    out: dict[str, dict[str, dict[str, float]]] = {}
    for bucket, by_algo in by_bucket_algo.items():
        out[bucket] = {}
        for a, c in by_algo.items():
            tp, fp, fn, gt, pred = c["tp"], c["fp"], c["fn"], c["gt"], c["pred"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = (2.0 * p * r_) / (p + r_) if (p + r_) > 0 else 0.0
            out[bucket][a] = {
                "tp": tp, "fp": fp, "fn": fn, "n_gt": gt, "n_pred": pred,
                "precision": p, "recall": r_, "f1": f,
            }
    return out


def _aggregate_subband_per_kind(records: list[ChartRecord]) -> dict[str, Any]:
    """Per (sub-band-name, band-idx, kind-name) caught / total / recall."""
    if not records:
        return {}
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    sb_names = sorted(records[0].subband_per_kind_caught.keys())
    for sb_name in sb_names:
        # Discover band indices and kind names.
        band_idxs = sorted({
            b for r in records
            for b in r.subband_per_kind_caught.get(sb_name, {}).keys()
        })
        kind_names = sorted({
            k for r in records
            for k in r.subband_per_kind_total.get(sb_name, {}).keys()
        })
        per_band: dict[str, dict[str, dict[str, float]]] = {}
        for b_idx in band_idxs:
            per_kind: dict[str, dict[str, float]] = {}
            for kname in kind_names:
                total = sum(
                    r.subband_per_kind_total.get(sb_name, {}).get(kname, 0)
                    for r in records
                )
                caught = sum(
                    r.subband_per_kind_caught.get(sb_name, {}).get(b_idx, {}).get(kname, 0)
                    for r in records
                )
                per_kind[kname] = {
                    "n": int(total),
                    "caught": int(caught),
                    "recall": caught / total if total > 0 else 0.0,
                }
            per_band[str(b_idx)] = per_kind
        out[sb_name] = per_band
    return out


def _aggregate_activation_distributions(
    records: list[ChartRecord],
) -> dict[str, dict[str, list[float]]]:
    """Concatenate sampled activation values per (algo, label)."""
    if not records:
        return {}
    algos = sorted(records[0].activation_samples.keys())
    out: dict[str, dict[str, list[float]]] = {a: {"tp": [], "fp": [],
                                                  "fn": [], "near_miss": []}
                                              for a in algos}
    for r in records:
        for a, samples in r.activation_samples.items():
            for label, vals in samples.items():
                out[a][label].extend(vals)
    return out


# ─────────────────────────── Output ───────────────────────────────────


def _write_pairwise_csv(path: Path, pairwise: dict[str, Any]) -> None:
    pairs = pairwise.get("pairs", {})
    if not pairs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair", "n_total", "n_both", "n_a_only", "n_b_only", "n_neither",
        "recall_a", "recall_b", "recall_union", "jaccard",
        "marg_b_given_a", "marg_a_given_b",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tag, m in pairs.items():
            row = {"pair": tag, "n_total": m["n_total"]}
            row.update({k: m[k] for k in fields if k not in ("pair", "n_total")})
            w.writerow(row)


def _write_subset_csv(path: Path, subsets: dict[str, Any]) -> None:
    if not subsets:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subset", "size", "recall", "precision_lower",
              "precision_upper", "f1", "n_pred_union", "n_tp_unique"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tag, m in subsets.items():
            row = {"subset": tag}
            row.update(m)
            w.writerow(row)


# ─────────────────────────── Main ─────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ds_root = _resolve_dataset(args.dataset, args.datasets_dir)
    if not (ds_root / "manifest.json").exists():
        print(f"manifest.json not found under {ds_root}", file=sys.stderr)
        return 2

    manifest = load_manifest(ds_root / "manifest.json")
    sampler_cfg = manifest.sampler_config
    sample_rate = int(sampler_cfg.sample_rate)
    n_fft = int(sampler_cfg.n_fft)
    n_mels = int(sampler_cfg.n_mels)
    f_min = float(sampler_cfg.f_min)
    f_max = float(sampler_cfg.f_max) if sampler_cfg.f_max is not None else sample_rate / 2.0

    print(f"[dataset] root={ds_root}  charts={len(manifest.charts)}  "
          f"sr={sample_rate}  n_mels={n_mels}")

    split_ratios = _parse_split_ratios(args.split_ratios)
    allowed = chart_ids_for_split(
        manifest, args.split, split_ratios, args.split_seed,
    )
    entries = [e for e in manifest.charts if e.chart_id in allowed]
    if args.max_charts is not None:
        entries = entries[: args.max_charts]
    print(f"[split] {args.split!r} -> {len(entries)} charts")
    print(f"[thresholds] (per-algo, fixed) {DEFAULT_THRESHOLDS}")
    print(f"[tolerance] +/-{args.tolerance} frames "
          f"(+/-{args.tolerance * 5} ms)")

    device = torch.device(args.device)
    freqs_hz = mel_band_center_freqs_hz(
        sample_rate, n_fft, n_mels,
        f_min=f_min, f_max=f_max, device=device,
    )

    rng = np.random.default_rng(0)
    records: list[ChartRecord] = []
    n_skipped = 0
    t_start = time.time()

    iterator: Any = entries
    if not args.no_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(entries, desc="Charts", unit="chart")
        except ImportError:
            pass

    for entry in iterator:
        events_path = ds_root / "events" / f"{_safe_filename(entry.chart_id)}.npz"
        if not events_path.exists():
            n_skipped += 1
            continue
        try:
            with np.load(events_path) as data:
                bins = np.asarray(data["bins"], dtype=np.int64)
                kinds = np.asarray(data["kind_ids"], dtype=np.uint8)
        except Exception:
            n_skipped += 1
            continue

        feat_path = ds_root / entry.features_path
        if not feat_path.exists():
            n_skipped += 1
            continue
        try:
            mel_np = np.load(feat_path).astype(np.float32, copy=False)
        except Exception:
            n_skipped += 1
            continue
        log_mel = torch.from_numpy(mel_np).to(device)
        gt_frames_t = torch.as_tensor(bins, device=device)

        try:
            rec = _process_chart(
                chart_id=entry.chart_id,
                log_mel=log_mel,
                gt_frames_t=gt_frames_t,
                gt_kinds_np=kinds,
                density_mean=float(entry.density_mean),
                star_rating=float(entry.star_rating or 0.0),
                freqs_hz=freqs_hz,
                thresholds=DEFAULT_THRESHOLDS,
                tolerance=args.tolerance,
                norm_percentile=args.norm_percentile,
                min_peak_distance=args.min_peak_distance,
                n_activation_samples=args.n_activation_samples,
                rng=rng,
            )
        except Exception as e:
            print(f"  failed on {entry.chart_id}: {e}", file=sys.stderr)
            n_skipped += 1
            continue
        records.append(rec)

    elapsed = time.time() - t_start
    print(f"[done] processed={len(records)}  skipped={n_skipped}  "
          f"elapsed={elapsed:.1f}s")

    # ── Aggregate ──
    pairwise = _aggregate_pairwise(records)
    subsets = _aggregate_subset_unions(records)
    per_kind = _aggregate_per_kind(records)
    per_density = _aggregate_per_bucket(
        records, bucket_field="density_mean", buckets=DENSITY_BUCKETS,
    )
    per_star = _aggregate_per_bucket(
        records, bucket_field="star_rating", buckets=STAR_BUCKETS,
    )
    sb_per_kind = _aggregate_subband_per_kind(records)
    activations = _aggregate_activation_distributions(records)

    # ── Write outputs ──
    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": str(ds_root),
        "split": args.split,
        "n_charts": len(records),
        "n_skipped": n_skipped,
        "tolerance": args.tolerance,
        "thresholds": DEFAULT_THRESHOLDS,
        "elapsed_s": round(elapsed, 2),
        "pairwise": pairwise,
        "subsets": subsets,
        "per_kind": per_kind,
        "per_density": per_density,
        "per_star": per_star,
        "subband_per_kind": sb_per_kind,
        "activation_distributions": activations,
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_pairwise_csv(out_root / "pairwise.csv", pairwise)
    _write_subset_csv(out_root / "subsets.csv", subsets)

    # Per-chart CSV with caught counts per algo + bucket fields.
    per_chart_path = out_root / "per_chart.csv"
    if records:
        algos = sorted(records[0].caught.keys())
        with per_chart_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["chart_id", "n_gt", "density_mean", "star_rating"]
            for a in algos:
                header += [f"{a}_caught", f"{a}_n_pred", f"{a}_n_tp"]
            w.writerow(header)
            for r in records:
                row = [r.chart_id, r.n_gt,
                       round(r.density_mean, 4), round(r.star_rating, 4)]
                for a in algos:
                    row += [int(r.caught[a].sum()), r.n_pred[a], r.n_tp[a]]
                w.writerow(row)

    print(f"[output] {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
