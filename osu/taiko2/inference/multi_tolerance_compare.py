"""Multi-tolerance chart comparison.

``Chart.compare`` returns a single ``ChartComparison`` whose
``matched_rate`` / ``close_rate`` / ``far_rate`` / ``hallucination_rate``
use hardcoded ms thresholds (25 / 50 / 100 / 100). This module
parameterizes the GT-matching part of that comparison over a tuple of
ms tolerances so the AR-corpus pass can report a multi-tolerance sweep
without breaking the single-tolerance interface.

Per-tolerance results are returned as a ``dict[int, ChartComparison]``
keyed by the ms tolerance. The TaikoNation pattern-space metrics
(``over_pspace_*``, ``hi_pspace``, ``dc_human``, ``oc_human``,
``dc_rand``) are computed once and copied across all tolerances since
they don't depend on the tolerance.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..domain.chart import (
    TN_SEED,
    Chart,
    ChartComparison,
    _compute_resolution_comparisons,
    _distributional_comparison,
    _tn_pattern_metrics,
)


def _gt_match_metrics_at(
    self_ms: np.ndarray, other_ms: np.ndarray, tolerance_ms: int,
) -> dict[str, float]:
    """``Chart._gt_match_metrics`` parameterized by a single ms tolerance.

    Definitions used:
      - matched_rate = fraction of GT onsets with a self onset within
        ``tolerance_ms``.
      - close_rate   = fraction of GT onsets with a self onset within
        ``2 * tolerance_ms`` (always >= matched_rate).
      - far_rate     = fraction of GT onsets with NO self onset within
        ``max(100, 4 * tolerance_ms)``.
      - hallucination_rate = fraction of self onsets with NO GT onset
        within ``max(100, 4 * tolerance_ms)``.
      - error_mean / median / density_ratio: independent of tolerance.
    """
    n_self = len(self_ms)
    n_other = len(other_ms)
    if n_self == 0 or n_other == 0:
        return dict(
            matched_rate=0.0, close_rate=0.0, far_rate=0.0,
            hallucination_rate=0.0,
            error_mean_ms=0.0, error_median_ms=0.0,
            density_ratio=0.0,
            precision=0.0, recall=0.0, f1=0.0,
        )
    ps = np.sort(self_ms)
    gs = np.sort(other_ms)

    def _closest(arr: np.ndarray, v: float) -> float:
        i = int(np.searchsorted(arr, v))
        best = float("inf")
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(arr):
                best = min(best, abs(float(arr[j]) - float(v)))
        return best

    gt_err = np.array([_closest(ps, g) for g in gs])
    pe_err = np.array([_closest(gs, p) for p in ps])

    ps_density = n_self / max((ps[-1] - ps[0]) / 1000.0, 0.1) if n_self > 1 else 0.0
    gs_density = n_other / max((gs[-1] - gs[0]) / 1000.0, 0.1) if n_other > 1 else 0.0

    far_thresh = float(max(100, 4 * tolerance_ms))
    close_thresh = float(2 * tolerance_ms)
    matched = float((gt_err <= tolerance_ms).mean())
    prec = float((pe_err <= tolerance_ms).mean()) if n_self > 0 else 0.0
    f1 = (2.0 * prec * matched / (prec + matched)
           if (prec + matched) > 0 else 0.0)
    return dict(
        matched_rate=matched,
        close_rate=float((gt_err <= close_thresh).mean()),
        far_rate=float((gt_err > far_thresh).mean()),
        hallucination_rate=float((pe_err > far_thresh).mean()),
        error_mean_ms=float(gt_err.mean()),
        error_median_ms=float(np.median(gt_err)),
        density_ratio=ps_density / max(gs_density, 0.01),
        precision=prec,
        recall=matched,
        f1=f1,
    )


def compare_at_tolerances(
    chart: Chart,
    gt_chart: Chart,
    tolerances_ms: tuple[int, ...],
    bin_ms: float = 5.0,
    *,
    seed: int = TN_SEED,
) -> dict[int, ChartComparison]:
    """Run ``chart.compare(gt_chart)`` once per ms tolerance.

    ``bin_ms`` is accepted for API symmetry with frame-tolerance
    callers but is otherwise unused (matching is done in ms).
    """
    del bin_ms  # informational only; matching is ms-native
    if not tolerances_ms:
        raise ValueError("tolerances_ms must be non-empty")
    self_ms = np.asarray(
        [o.time_ms for o in chart.track.onsets], dtype=np.float64,
    )
    other_ms = np.asarray(
        [o.time_ms for o in gt_chart.track.onsets], dtype=np.float64,
    )
    tn = _tn_pattern_metrics(self_ms, other_ms, rng_seed=seed)
    fps_cmp = _compute_resolution_comparisons(self_ms, other_ms)

    self_metrics = chart.calculate_metrics()
    gt_metrics = gt_chart.calculate_metrics()
    dist = _distributional_comparison(self_metrics, gt_metrics)

    out: dict[int, ChartComparison] = {}
    for tol in tolerances_ms:
        gt = _gt_match_metrics_at(self_ms, other_ms, int(tol))
        out[int(tol)] = ChartComparison(
            n_self=len(self_ms),
            n_other=len(other_ms),
            matched_rate=gt["matched_rate"],
            close_rate=gt["close_rate"],
            far_rate=gt["far_rate"],
            hallucination_rate=gt["hallucination_rate"],
            error_mean_ms=gt["error_mean_ms"],
            error_median_ms=gt["error_median_ms"],
            density_ratio=gt["density_ratio"],
            precision=gt["precision"],
            recall=gt["recall"],
            f1=gt["f1"],
            over_pspace_self=tn["over_pspace_self"],
            over_pspace_other=tn["over_pspace_other"],
            hi_pspace=tn["hi_pspace"],
            dc_human=tn["dc_human"],
            oc_human=tn["oc_human"],
            dc_rand=tn["dc_rand"],
            gap_hist_tvd=dist["gap_hist_tvd"],
            ratio_hist_tvd=dist["ratio_hist_tvd"],
            density_corr=dist["density_corr"],
            density_mae=dist["density_mae"],
            silence_overlap_f1=dist["silence_overlap_f1"],
            dense_overlap_f1=dist["dense_overlap_f1"],
            gap_peak_iou=dist["gap_peak_iou"],
            ioi_mean_ratio=dist["ioi_mean_ratio"],
            ioi_std_ratio=dist["ioi_std_ratio"],
            streak_fraction_delta=dist["streak_fraction_delta"],
            bpm_ratio=dist["bpm_ratio"],
            fps_comparisons=fps_cmp,
        )
    return out


# Per-tolerance metric fields rolled up by aggregate_multi_tolerance_summaries.
_PER_TOL_FIELDS: tuple[str, ...] = (
    "matched_rate", "close_rate", "far_rate", "hallucination_rate",
    "error_mean_ms", "error_median_ms",
)


def _percentile_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    q = np.percentile(arr, [0, 25, 50, 75, 95, 100])
    return {
        "n": int(arr.size),
        "min":    round(float(q[0]), 4),
        "p25":    round(float(q[1]), 4),
        "median": round(float(q[2]), 4),
        "p75":    round(float(q[3]), 4),
        "p95":    round(float(q[4]), 4),
        "max":    round(float(q[5]), 4),
        "mean":   round(float(arr.mean()), 4),
    }


def aggregate_multi_tolerance_summaries(
    comparisons_per_chart: list[dict[int, ChartComparison]],
    tolerances_ms: tuple[int, ...],
) -> dict[str, Any]:
    """Roll up per-chart multi-tolerance comparisons into a single dict.

    Output shape::

      {
        "tolerances_ms": [...],
        "fields": {
          "matched_rate_at_tol_5":   {"median": ..., "p25": ..., ...},
          "matched_rate_at_tol_10":  {...},
          ...
        }
      }

    Per the spec — one rollup key per (field, tolerance) pair.
    """
    out: dict[str, Any] = {
        "tolerances_ms": list(tolerances_ms),
        "fields": {},
    }
    for tol in tolerances_ms:
        for name in _PER_TOL_FIELDS:
            vals: list[float] = []
            for entry in comparisons_per_chart:
                cmp_ = entry.get(int(tol))
                if cmp_ is None:
                    continue
                v = getattr(cmp_, name, None)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            stats = _percentile_stats(vals)
            if stats:
                out["fields"][f"{name}_at_tol_{int(tol)}"] = stats
    return out
