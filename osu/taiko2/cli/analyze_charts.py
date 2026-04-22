"""Run `Chart.calculate_metrics` over every GT chart in a dataset split.

Complements ``cli/analyze_dataset.py`` (which aggregates corpus-wide
stats and draws graphs): this script writes one rich JSON per chart
with full histograms / timelines / regions, plus a flat CSV summary of
scalar fields for quick cross-chart querying.

Usage::

    python -m osu.taiko2.cli.analyze_charts --dataset taiko2_v1
    python -m osu.taiko2.cli.analyze_charts --dataset taiko2_v1 --split train

Output layout::

    osu/taiko2/analysis/{dataset_name}/
        chart_metrics/
            {safe_chart_id}.json        # one per chart, full ChartMetrics
        chart_metrics.csv               # flat scalar summary
        chart_metrics_summary.json      # per-field min/p25/median/p75/p95/max
        chart_metrics_graphs/           # visual distributions + correlations
            *.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from ..dataset import _safe_filename
from ..domain.chart import ChartMetrics


# ─────────────────────────── serialization ────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / tuple / dict-with-int-keys into plain JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Fields promoted to the flat CSV; rich structures (histograms, timelines,
# region lists) are intentionally excluded — they live only in the JSON.
_CSV_FIELDS: tuple[str, ...] = (
    "total_events", "duration_s", "events_per_sec",
    "count_don", "count_ka", "count_big_don", "count_big_ka",
    "count_drumroll", "count_spinner", "don_ratio",
    "ioi_min_ms", "ioi_max_ms", "ioi_mean_ms", "ioi_median_ms",
    "ioi_std_ms", "ioi_p95_ms", "ioi_p99_ms",
    "short_ioi_count", "short_ioi_pct", "long_gap_count",
    "density_mean", "density_peak", "density_std", "density_min", "density_cv",
    "longest_streak", "mean_streak_len", "streak_event_fraction",
    "estimated_bpm", "dominant_ioi_ms", "over_pspace_self",
    # Gap-distribution shape (scalars; `gap_peaks` tuple is written to
    # the per-chart JSON only, and aggregated across charts for graphs).
    "gap_peak_count", "gap_peak_falloff",
    "gap_random_distance", "gap_metronome_distance",
    # Ratio-distribution shape (gap[i] / gap[i-1], log2-bucketed).
    "ratio_peak_count", "ratio_peak_falloff",
    "ratio_random_distance", "ratio_metronome_distance",
)


def _metrics_to_csv_row(
    chart_id: str,
    beatmap_id: str,
    beatmapset_id: str,
    star_rating: float | None,
    difficulty_version: str,
    overall_difficulty: float,
    metrics: ChartMetrics,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "chart_id": chart_id,
        "beatmap_id": beatmap_id,
        "beatmapset_id": beatmapset_id,
        "difficulty_version": difficulty_version,
        "overall_difficulty": overall_difficulty,
        "star_rating": star_rating if star_rating is not None else "",
    }
    for name in _CSV_FIELDS:
        v = getattr(metrics, name)
        row[name] = "" if v is None else v
    return row


def _metrics_to_json(metrics: ChartMetrics) -> dict[str, Any]:
    return _json_safe(asdict(metrics))


# ─────────────────────────── aggregates ──────────────────────────────

def _compute_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _pcts(name: str) -> dict[str, float] | None:
        vals = [r[name] for r in rows if r[name] != "" and r[name] is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return {
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    summary: dict[str, Any] = {"n_charts": len(rows)}
    for name in _CSV_FIELDS:
        pcts = _pcts(name)
        if pcts is not None:
            summary[name] = pcts
    # Star rating summary (if populated)
    pcts = _pcts("star_rating")
    if pcts is not None:
        summary["star_rating"] = pcts
    return summary


# ─────────────────────────── plotting ────────────────────────────────

def _plot_graphs(
    rows: list[dict[str, Any]],
    aggregated_ioi: dict[int, int],
    graphs_dir: Path,
    *,
    top_gap_peak_ms: list[float] | None = None,
    top_ratio_peak: list[float] | None = None,
    gap_peak_histogram: np.ndarray | None = None,
    ratio_peak_histogram: np.ndarray | None = None,
) -> None:
    """Draw distribution + correlation plots from the collected metrics.

    Focus is on fields that are unique to `Chart.calculate_metrics()`
    (BPM, over_pspace_self, streak_fraction, density_cv, and the new
    gap- / ratio-distribution shape metrics) rather than the basics
    already covered by `cli/analyze_dataset`.

    Extra inputs for the shape-metric plots:
      - ``top_gap_peak_ms[c]`` = center of each chart's #1 gap peak
        (ms); used to produce a distribution of "where real charts'
        dominant IOI lands".
      - ``top_ratio_peak[c]`` = center of each chart's #1 ratio peak
        (linear ratio, not log2); same idea in ratio space.
      - ``gap_peak_histogram`` / ``ratio_peak_histogram``: corpus-wide
        counts over the fixed 200-bucket supports of (count of peaks
        that landed in each bucket across all charts) — tells us what
        the "average" chart's peak structure looks like.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graphs_dir.mkdir(parents=True, exist_ok=True)

    def _numeric(field: str) -> np.ndarray:
        vals = [r[field] for r in rows if r[field] != "" and r[field] is not None]
        return np.asarray(vals, dtype=float) if vals else np.zeros(0)

    def _save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(graphs_dir / name, dpi=150)
        plt.close(fig)

    def _hist(vals, title, xlabel, fname, bins=60, color="#4a90d9",
              range_=None, log_y=False):
        if not len(vals):
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(vals, bins=bins, color=color,
                edgecolor="black", linewidth=0.3, range=range_, log=log_y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Charts")
        ax.set_title(title)
        med = float(np.median(vals))
        ax.axvline(med, color="red", linestyle="--",
                   label=f"Median: {med:.2f}")
        ax.legend()
        _save(fig, fname)

    # 1 — BPM distribution
    _hist(_numeric("estimated_bpm"),
          "Estimated BPM per Chart (Chart.calculate_metrics)",
          "BPM", "01_bpm.png", color="#fcb71e", range_=(50, 260))

    # 2 — TaikoNation self pattern-space diversity
    _hist(_numeric("over_pspace_self"),
          "TaikoNation Self Pattern-Space Diversity",
          "% of 2^8 distinct 8-step patterns (23 ms quantization)",
          "02_over_pspace.png", color="#c76dba")

    # 3 — Streak event fraction
    _hist(_numeric("streak_event_fraction"),
          "Fraction of Events in a Same-Gap Streak (5% tolerance)",
          "Streak Fraction", "03_streak_fraction.png",
          color="#9b6eff", range_=(0.0, 1.0), bins=50)

    # 4 — Density CV
    _hist(_numeric("density_cv"),
          "Density Coefficient of Variation (std / mean)",
          "CV", "04_density_cv.png", color="#e86850", bins=50)

    # 5 — Longest streak
    longest = _numeric("longest_streak")
    if len(longest):
        _hist(longest,
              "Longest Same-Gap Streak per Chart",
              "Events in longest streak", "05_longest_streak.png",
              color="#6bc46d", bins=60)

    # 6 — Short IOI (<20 ms) count — potential double-hit artifacts
    short = _numeric("short_ioi_count")
    if len(short):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(short, bins=40, color="#eb4528",
                edgecolor="black", linewidth=0.3, log=True)
        ax.set_xlabel("Short-IOI (<20 ms) Count")
        ax.set_ylabel("Charts (log)")
        ax.set_title("Potential Double-Hit Artifacts per Chart")
        _save(fig, "06_short_ioi.png")

    # 7 — Long gap (>2 s) count per chart
    longgap = _numeric("long_gap_count")
    if len(longgap):
        _hist(longgap, "Long-Gap (>2 s) Count per Chart",
              "Count", "07_long_gap.png", color="#7bc0ea", bins=60)

    # 8 — Aggregated IOI histogram across all charts (10 ms buckets)
    if aggregated_ioi:
        buckets = sorted(aggregated_ioi.items())
        xs = np.array([b for b, _ in buckets])
        ys = np.array([c for _, c in buckets])
        mask = (xs <= 1000)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(xs[mask], ys[mask], width=10, color="#4a90d9",
               edgecolor="black", linewidth=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("Inter-Onset Interval (ms, 10 ms buckets)")
        ax.set_ylabel("Count across all charts (log)")
        ax.set_title(
            f"Aggregated IOI Histogram (all charts, 0-1000 ms)  "
            f"[{int(ys[mask].sum()):,} events]"
        )
        _save(fig, "08_ioi_hist_all.png")

    # 9 — Star rating vs BPM
    def _scatter(x_field: str, y_field: str, title: str, xlabel: str,
                 ylabel: str, fname: str, color: str = "#4a90d9") -> None:
        xs: list[float] = []
        ys: list[float] = []
        for r in rows:
            xv, yv = r[x_field], r[y_field]
            if xv in ("", None) or yv in ("", None):
                continue
            try:
                xs.append(float(xv))
                ys.append(float(yv))
            except (TypeError, ValueError):
                continue
        if not xs:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(xs, ys, alpha=0.25, s=10, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  (n={len(xs)})")
        _save(fig, fname)

    _scatter(
        "star_rating", "estimated_bpm",
        "Star Rating vs Estimated BPM",
        "Star Rating (★)", "BPM",
        "09_star_vs_bpm.png", color="#fcb71e",
    )

    # 10 — Star rating vs self pattern-space diversity
    _scatter(
        "star_rating", "over_pspace_self",
        "Star Rating vs Self Pattern-Space Diversity",
        "Star Rating (★)", "over_pspace_self (%)",
        "10_star_vs_pspace.png", color="#c76dba",
    )

    # 11 — Star rating vs streak fraction
    _scatter(
        "star_rating", "streak_event_fraction",
        "Star Rating vs Streak Event Fraction",
        "Star Rating (★)", "Streak Fraction",
        "11_star_vs_streak.png", color="#9b6eff",
    )

    # 12 — Density mean vs BPM
    _scatter(
        "density_mean", "estimated_bpm",
        "Density (events/s) vs Estimated BPM",
        "Density (events/s)", "BPM",
        "12_density_vs_bpm.png", color="#6bc46d",
    )

    # ── Gap- / ratio-distribution shape metrics (reference graphs) ──
    # These establish "what real charts look like" for the new four
    # metrics in each of the two distributions.

    # 13 — gap_peak_count histogram
    _hist(_numeric("gap_peak_count"),
          "Gap histogram — peak count per chart",
          "gap_peak_count", "13_gap_peak_count.png",
          color="#4a90d9", bins=range(0, 16))

    # 14 — gap_peak_falloff
    _hist(_numeric("gap_peak_falloff"),
          "Gap histogram — peak falloff (mean c_{i+1} / c_i)",
          "gap_peak_falloff", "14_gap_peak_falloff.png",
          color="#6bc46d", range_=(0.0, 1.0), bins=40)

    # 15 — gap_random_distance
    _hist(_numeric("gap_random_distance"),
          "Gap histogram — distance from uniform (TVD)",
          "gap_random_distance", "15_gap_random_distance.png",
          color="#fcb71e", range_=(0.0, 1.0), bins=40)

    # 16 — gap_metronome_distance
    _hist(_numeric("gap_metronome_distance"),
          "Gap histogram — distance from delta-at-mode (TVD)",
          "gap_metronome_distance", "16_gap_metronome_distance.png",
          color="#e86850", range_=(0.0, 1.0), bins=40)

    # 17 — ratio_peak_count
    _hist(_numeric("ratio_peak_count"),
          "Ratio histogram — peak count per chart",
          "ratio_peak_count", "17_ratio_peak_count.png",
          color="#4a90d9", bins=range(0, 16))

    # 18 — ratio_peak_falloff
    _hist(_numeric("ratio_peak_falloff"),
          "Ratio histogram — peak falloff",
          "ratio_peak_falloff", "18_ratio_peak_falloff.png",
          color="#6bc46d", range_=(0.0, 1.0), bins=40)

    # 19 — ratio_random_distance
    _hist(_numeric("ratio_random_distance"),
          "Ratio histogram — distance from uniform (TVD)",
          "ratio_random_distance", "19_ratio_random_distance.png",
          color="#fcb71e", range_=(0.0, 1.0), bins=40)

    # 20 — ratio_metronome_distance (anchored at 1.0x)
    _hist(_numeric("ratio_metronome_distance"),
          "Ratio histogram — distance from all-ratios-at-1.0x (TVD)",
          "ratio_metronome_distance", "20_ratio_metronome_distance.png",
          color="#c76dba", range_=(0.0, 1.0), bins=40)

    # 21 — #1 gap-peak location across charts (ms). Where does the
    # typical chart's dominant IOI land?
    if top_gap_peak_ms:
        vals = np.asarray(top_gap_peak_ms, dtype=float)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(vals, bins=100, range=(0, 2000),
                color="#7bc0ea", edgecolor="black", linewidth=0.3)
        med = float(np.median(vals))
        ax.axvline(med, color="red", linestyle="--",
                   label=f"Median: {med:.0f} ms")
        ax.set_xlabel("#1 gap-peak center (ms, 10 ms buckets)")
        ax.set_ylabel("Charts")
        ax.set_title("Dominant IOI location across charts")
        ax.legend()
        _save(fig, "21_top_gap_peak_ms.png")

    # 22 — #1 ratio-peak location across charts. Most charts peak at
    # 1.0x (metronome sections); this graph shows the tail — how many
    # charts' dominant ratio is at 2.0x or 0.5x or elsewhere.
    if top_ratio_peak:
        vals = np.asarray(top_ratio_peak, dtype=float)
        # Log2 x-axis so halving and doubling are equidistant from 1.0.
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(np.log2(vals), bins=100, range=(-3.0, 3.0),
                color="#9b6eff", edgecolor="black", linewidth=0.3)
        med_lin = float(np.median(vals))
        ax.axvline(np.log2(med_lin), color="red", linestyle="--",
                   label=f"Median: {med_lin:.2f}x")
        # Tick labels in linear ratios at integer log2 steps.
        xticks = [-3, -2, -1, 0, 1, 2, 3]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{2.0**t:g}x" for t in xticks])
        ax.set_xlabel("#1 ratio-peak center (log2 axis)")
        ax.set_ylabel("Charts")
        ax.set_title("Dominant ratio location across charts")
        ax.legend()
        _save(fig, "22_top_ratio_peak.png")

    # 23 — Corpus-wide gap-peak bucket histogram (how often does each
    # 10 ms bucket appear as a peak across all charts' full peak lists).
    if gap_peak_histogram is not None and gap_peak_histogram.any():
        xs = np.arange(gap_peak_histogram.shape[0]) * 10 + 5
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(xs[:200], gap_peak_histogram[:200], width=10,
               color="#4a90d9", edgecolor="black", linewidth=0.2)
        ax.set_xlabel("Gap bucket center (ms, 10 ms buckets)")
        ax.set_ylabel("Peak occurrences across charts")
        ax.set_title("Gap-peak locations, corpus-wide (all kept peaks)")
        _save(fig, "23_gap_peak_histogram.png")

    # 24 — Corpus-wide ratio-peak bucket histogram (log2 axis).
    if ratio_peak_histogram is not None and ratio_peak_histogram.any():
        n_buckets = ratio_peak_histogram.shape[0]
        log2_axis = np.linspace(-3.0, 3.0, n_buckets, endpoint=False)
        log2_axis = log2_axis + (6.0 / n_buckets) / 2
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(log2_axis, ratio_peak_histogram, width=6.0 / n_buckets,
               color="#c76dba", edgecolor="black", linewidth=0.2)
        xticks = [-3, -2, -1, 0, 1, 2, 3]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{2.0**t:g}x" for t in xticks])
        ax.set_xlabel("Ratio bucket center (log2 axis)")
        ax.set_ylabel("Peak occurrences across charts")
        ax.set_title("Ratio-peak locations, corpus-wide (all kept peaks)")
        _save(fig, "24_ratio_peak_histogram.png")

    # ── Correlations: new shape metrics vs existing chart-level ones ──
    # Which existing metric a given shape metric tracks tells us whether
    # it's redundant (high correlation = it's already captured) or novel
    # (low correlation = it's measuring something else).
    def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    def _pair_scatter(
        x_field: str, y_field: str, title: str, xlabel: str,
        ylabel: str, fname: str, color: str = "#4a90d9",
        log_x: bool = False,
    ) -> None:
        xs: list[float] = []
        ys: list[float] = []
        for r in rows:
            xv, yv = r[x_field], r[y_field]
            if xv in ("", None) or yv in ("", None):
                continue
            try:
                xs.append(float(xv))
                ys.append(float(yv))
            except (TypeError, ValueError):
                continue
        if len(xs) < 3:
            return
        x_arr = np.asarray(xs)
        y_arr = np.asarray(ys)
        r = _corr(x_arr, y_arr)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_arr, y_arr, alpha=0.22, s=10, color=color)
        if log_x and (x_arr > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        corr_tag = f"  r = {r:+.3f}" if r is not None else ""
        ax.set_title(f"{title}  (n={len(xs)}{corr_tag})")
        _save(fig, fname)

    # Star rating is the best "complexity" proxy — harder charts
    # should push every shape metric toward "less metronomic / more
    # varied". Run all eight against it.
    _pair_scatter(
        "star_rating", "gap_peak_count",
        "Star rating vs gap peak count",
        "Star rating (★)", "gap_peak_count",
        "25_star_vs_gap_peak_count.png", color="#4a90d9",
    )
    _pair_scatter(
        "star_rating", "gap_peak_falloff",
        "Star rating vs gap peak falloff",
        "Star rating (★)", "gap_peak_falloff",
        "26_star_vs_gap_peak_falloff.png", color="#6bc46d",
    )
    _pair_scatter(
        "star_rating", "gap_random_distance",
        "Star rating vs gap random-distance (TVD from uniform)",
        "Star rating (★)", "gap_random_distance",
        "27_star_vs_gap_random.png", color="#fcb71e",
    )
    _pair_scatter(
        "star_rating", "gap_metronome_distance",
        "Star rating vs gap metronome-distance",
        "Star rating (★)", "gap_metronome_distance",
        "28_star_vs_gap_metronome.png", color="#e86850",
    )
    _pair_scatter(
        "star_rating", "ratio_peak_count",
        "Star rating vs ratio peak count",
        "Star rating (★)", "ratio_peak_count",
        "29_star_vs_ratio_peak_count.png", color="#4a90d9",
    )
    _pair_scatter(
        "star_rating", "ratio_peak_falloff",
        "Star rating vs ratio peak falloff",
        "Star rating (★)", "ratio_peak_falloff",
        "30_star_vs_ratio_peak_falloff.png", color="#6bc46d",
    )
    _pair_scatter(
        "star_rating", "ratio_random_distance",
        "Star rating vs ratio random-distance",
        "Star rating (★)", "ratio_random_distance",
        "31_star_vs_ratio_random.png", color="#fcb71e",
    )
    _pair_scatter(
        "star_rating", "ratio_metronome_distance",
        "Star rating vs ratio metronome-distance (anchored at 1.0x)",
        "Star rating (★)", "ratio_metronome_distance",
        "32_star_vs_ratio_metronome.png", color="#c76dba",
    )

    # Against streak_event_fraction — a chart where most events sit in
    # a same-gap streak is close to pure metronome; should correlate
    # inversely with both metronome-distances.
    _pair_scatter(
        "streak_event_fraction", "gap_metronome_distance",
        "Streak-event fraction vs gap metronome-distance",
        "streak_event_fraction", "gap_metronome_distance",
        "33_streak_vs_gap_metronome.png", color="#e86850",
    )
    _pair_scatter(
        "streak_event_fraction", "ratio_metronome_distance",
        "Streak-event fraction vs ratio metronome-distance",
        "streak_event_fraction", "ratio_metronome_distance",
        "34_streak_vs_ratio_metronome.png", color="#c76dba",
    )

    # Pspace-diversity captures "how many distinct 8-step patterns";
    # ratio_peak_count captures "how many distinct rhythmic ratios".
    # Related but not redundant.
    _pair_scatter(
        "over_pspace_self", "ratio_peak_count",
        "Pattern-space diversity vs ratio peak count",
        "over_pspace_self (%)", "ratio_peak_count",
        "35_pspace_vs_ratio_peak_count.png", color="#9b6eff",
    )

    # Density mean vs gap peak falloff — denser charts concentrate
    # more gaps into their primary peak? Or spread them more?
    _pair_scatter(
        "density_mean", "gap_peak_falloff",
        "Density mean vs gap peak falloff",
        "Density (events/s)", "gap_peak_falloff",
        "36_density_vs_gap_falloff.png", color="#7bc0ea",
    )

    # Cross-distribution: are the two peak-counts correlated?
    _pair_scatter(
        "gap_peak_count", "ratio_peak_count",
        "Gap peak count vs ratio peak count",
        "gap_peak_count", "ratio_peak_count",
        "37_gap_vs_ratio_peak_count.png", color="#4a90d9",
    )
    _pair_scatter(
        "gap_metronome_distance", "ratio_metronome_distance",
        "Gap metronome-distance vs ratio metronome-distance",
        "gap_metronome_distance", "ratio_metronome_distance",
        "38_gap_vs_ratio_metronome.png", color="#e86850",
    )


# ─────────────────────────── CLI ──────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-chart metrics over a dataset (or split).",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under --datasets-dir) or path.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "analysis")
    p.add_argument("--split", default="all",
                   help="Split name (must exist in split_ratios), or 'all'. "
                        "Default: 'all' (every chart).")
    p.add_argument("--split-ratios", default="train:0.9,val:0.1",
                   help="Ratios as 'name:ratio,name:ratio'. Ignored for 'all'.")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-charts", type=int, default=None,
                   help="Smoke-test limit.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_split_ratios(raw: str) -> tuple[tuple[str, float], ...]:
    spec: list[tuple[str, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad split-ratios fragment {part!r}")
        name, ratio = part.split(":", 1)
        spec.append((name.strip(), float(ratio.strip())))
    return tuple(spec)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ds_root = Path(args.dataset)
    if not ds_root.is_absolute() and not ds_root.exists():
        ds_root = args.datasets_dir / args.dataset
    ds_root = ds_root.resolve()
    if not ds_root.is_dir():
        print(f"ERROR: dataset root not found: {ds_root}", file=sys.stderr)
        return 2

    spec = _parse_split_ratios(args.split_ratios)

    cfg = TaikoDetectionSamplerConfig(
        batch_size=1,
        dataset_root=ds_root,
        split=args.split,
        split_ratios=spec,
        split_seed=args.split_seed,
        min_cursor_bin=0,
        allowed_overlap_forward=0,
        allowed_overlap_back=0,
    )
    sampler = TaikoDetectionSampler(cfg)
    sampler.load_data(progress=True)

    dataset_name = sampler._manifest.name if sampler._manifest else ds_root.name
    out_root = (args.out_dir / dataset_name).resolve()
    charts_dir = out_root / "chart_metrics"
    charts_dir.mkdir(parents=True, exist_ok=True)

    n_charts = sampler.count_charts()
    if args.max_charts is not None:
        n_charts = min(n_charts, args.max_charts)

    print(f"Dataset:  {dataset_name}")
    print(f"Split:    {args.split}  ({n_charts} charts)")
    print(f"Output:   {out_root}")

    # Import the ratio-histogram constants so the corpus aggregator
    # uses the same bucket geometry `_compute_ratio_distribution` does.
    from ..domain.chart import (                                  # noqa: E402
        _GAP_HIST_N_BUCKETS,
        _GAP_HIST_BUCKET_MS,
        _RATIO_LOG2_MIN,
        _RATIO_LOG2_WIDTH,
        _RATIO_N_BUCKETS,
    )

    rows: list[dict[str, Any]] = []
    aggregated_ioi: Counter[int] = Counter()
    top_gap_peak_ms: list[float] = []
    top_ratio_peak: list[float] = []
    gap_peak_hist = np.zeros(_GAP_HIST_N_BUCKETS, dtype=np.int64)
    ratio_peak_hist = np.zeros(_RATIO_N_BUCKETS, dtype=np.int64)

    it = range(n_charts)
    if not args.no_progress:
        it = tqdm(it, desc="Analyzing charts", unit="chart")

    for i in it:
        chart = sampler.get_chart(i)
        metrics = chart.calculate_metrics()

        # Fold this chart's IOI histogram into the corpus aggregate.
        for bucket, count in metrics.ioi_histogram_10ms.items():
            aggregated_ioi[int(bucket)] += int(count)

        # #1 peak per chart (for "typical dominant IOI / ratio" graphs)
        # and the full peak list folded into corpus-wide bucket counts.
        if metrics.gap_peaks:
            top_gap_peak_ms.append(float(metrics.gap_peaks[0][0]))
            for center_ms, _count in metrics.gap_peaks:
                b = int(center_ms) // _GAP_HIST_BUCKET_MS
                if 0 <= b < _GAP_HIST_N_BUCKETS:
                    gap_peak_hist[b] += 1
        if metrics.ratio_peaks:
            top_ratio_peak.append(float(metrics.ratio_peaks[0][0]))
            for center_ratio, _count in metrics.ratio_peaks:
                if center_ratio <= 0:
                    continue
                import math
                b = int(
                    (math.log2(center_ratio) - _RATIO_LOG2_MIN)
                    / _RATIO_LOG2_WIDTH
                )
                if 0 <= b < _RATIO_N_BUCKETS:
                    ratio_peak_hist[b] += 1

        stem = _safe_filename(chart.track.beatmap_id or chart.track.artist + "_" + chart.track.title + "_" + chart.track.difficulty.version)
        json_path = charts_dir / f"{stem}.json"
        try:
            payload = {
                "chart_id": sampler.chart_ids()[i],
                "beatmap_id": chart.track.beatmap_id,
                "beatmapset_id": chart.track.beatmapset_id,
                "artist": chart.track.artist,
                "title": chart.track.title,
                "difficulty_version": chart.track.difficulty.version,
                "overall_difficulty": chart.track.difficulty.overall_difficulty,
                "star_rating": chart.track.difficulty.star_rating,
                "metrics": _metrics_to_json(metrics),
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            tqdm.write(f"  write failed for {stem}: {e}")
            continue

        rows.append(_metrics_to_csv_row(
            chart_id=sampler.chart_ids()[i],
            beatmap_id=chart.track.beatmap_id,
            beatmapset_id=chart.track.beatmapset_id,
            star_rating=chart.track.difficulty.star_rating,
            difficulty_version=chart.track.difficulty.version,
            overall_difficulty=chart.track.difficulty.overall_difficulty,
            metrics=metrics,
        ))

    if not rows:
        print("No charts produced output.", file=sys.stderr)
        return 1

    csv_path = out_root / "chart_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _compute_summary(rows)
    (out_root / "chart_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    graphs_dir = out_root / "chart_metrics_graphs"
    try:
        _plot_graphs(
            rows, dict(aggregated_ioi), graphs_dir,
            top_gap_peak_ms=top_gap_peak_ms,
            top_ratio_peak=top_ratio_peak,
            gap_peak_histogram=gap_peak_hist,
            ratio_peak_histogram=ratio_peak_hist,
        )
        n_graphs = len(list(graphs_dir.glob("*.png")))
    except Exception as e:
        print(f"  WARN: graph rendering failed: {e}", file=sys.stderr)
        n_graphs = 0

    print(f"\nWrote:")
    print(f"  per-chart JSONs -> {charts_dir}  ({len(rows)} files)")
    print(f"  summary CSV     -> {csv_path}")
    print(f"  summary JSON    -> {out_root / 'chart_metrics_summary.json'}")
    print(f"  graphs          -> {graphs_dir}  ({n_graphs} PNG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
