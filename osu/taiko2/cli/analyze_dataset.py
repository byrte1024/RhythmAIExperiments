"""Analyze a built taiko2 dataset — per-chart metrics + aggregate graphs.

Consolidates the dataset-only analyses from the old taiko repo:
  - ``analyze.py``: per-chart metadata, density distributions, event-type
    proportions, burstiness, silent breaks, averaged density profile.
  - ``analyze_metronome_data.py``: inter-onset gap streaks (same-gap runs
    within 5% tolerance) — longest, mean length, fraction of events in
    streaks.

Adds: gap-ms histogram across all charts, gap-ratio histogram (per-chart
musical subdivisions, no cursor/target involved), star-rating and OD
distributions, and a per-chart flat CSV.

**Not** included: anything involving a training cursor, next-onset
targets, A/B/C splits, or inference — those belong in a training/eval
script, not a dataset analyzer.

Output layout::

    analysis/{dataset_name}/
        summary.json       # aggregate statistics
        metrics.csv        # one row per chart, flat fields
        graphs/*.png       # one PNG per plot

Usage::

    python -m osu.taiko2.cli.analyze_dataset --dataset taiko2_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..domain.beatmap import OnsetBinned, OnsetKind
from ..domain.dataset import ChartEntry, DatasetManifest
from ..persistence.events import load_events
from ..persistence.manifest import load_manifest

EVENT_KINDS: tuple[OnsetKind, ...] = (
    OnsetKind.DON,
    OnsetKind.KA,
    OnsetKind.BIG_DON,
    OnsetKind.BIG_KA,
    OnsetKind.DRUMROLL,
    OnsetKind.SPINNER,
)


# ─────────────────────────── per-chart metrics ────────────────────────

def _find_streaks(
    gaps_ms: np.ndarray, tolerance: float = 0.05,
) -> list[tuple[int, int, float]]:
    """Return `[(start_index, length, representative_gap_ms), ...]`.

    A streak is a run of consecutive gaps where each is within `tolerance`
    (relative) of the first gap in the run. Runs of length < 2 are omitted.
    Ported from old `analyze_metronome_data.find_streaks`.
    """
    if len(gaps_ms) < 2:
        return []
    streaks: list[tuple[int, int, float]] = []
    start = 0
    head = float(gaps_ms[0])
    length = 1
    for i in range(1, len(gaps_ms)):
        g = float(gaps_ms[i])
        if head > 0 and abs(g - head) / head <= tolerance:
            length += 1
            continue
        if length >= 2:
            streaks.append((start, length, head))
        start, head, length = i, g, 1
    if length >= 2:
        streaks.append((start, length, head))
    return streaks


def _chart_metrics(
    entry: ChartEntry, events: tuple[OnsetBinned, ...],
) -> dict[str, Any]:
    """Compute a flat row of per-chart metrics. No cursor / target logic."""
    times_ms = np.array([o.time_ms for o in events], dtype=np.int64)
    kinds = [o.kind for o in events]

    counts = Counter(kinds)
    type_counts = {k.value: counts.get(k, 0) for k in EVENT_KINDS}

    # Inter-onset intervals (ms)
    if len(times_ms) >= 2:
        gaps_ms = np.diff(times_ms)
        gaps_positive = gaps_ms[gaps_ms > 0]
    else:
        gaps_ms = np.zeros((0,), dtype=np.int64)
        gaps_positive = gaps_ms

    if len(gaps_positive):
        median_gap = float(np.median(gaps_positive))
        min_gap = float(gaps_positive.min())
        max_gap = float(gaps_positive.max())
        p95_gap = float(np.percentile(gaps_positive, 95))
    else:
        median_gap = min_gap = max_gap = p95_gap = 0.0

    # Silent breaks (> 1s)
    breaks = gaps_ms[gaps_ms > 1000] if len(gaps_ms) else gaps_ms
    num_breaks = int(len(breaks))
    longest_break = int(breaks.max()) if num_breaks else 0

    # Streaks of same-gap
    streaks = _find_streaks(gaps_ms.astype(np.float64), tolerance=0.05) if len(gaps_ms) >= 2 else []
    if streaks:
        longest_streak = max(s[1] for s in streaks)
        mean_streak_len = float(np.mean([s[1] for s in streaks]))
        events_in_streaks = sum(s[1] + 1 for s in streaks)
    else:
        longest_streak = 0
        mean_streak_len = 0.0
        events_in_streaks = 0
    streak_event_fraction = (
        events_in_streaks / len(events) if len(events) else 0.0
    )

    # Don/Ka balance (sides include "big" variants)
    don_total = type_counts[OnsetKind.DON.value] + type_counts[OnsetKind.BIG_DON.value]
    ka_total = type_counts[OnsetKind.KA.value] + type_counts[OnsetKind.BIG_KA.value]
    don_ka_sum = don_total + ka_total
    don_ratio = don_total / don_ka_sum if don_ka_sum else 0.5

    return {
        "chart_id": entry.chart_id,
        "beatmap_id": entry.beatmap_id,
        "beatmapset_id": entry.beatmapset_id,
        "artist": entry.artist,
        "title": entry.title,
        "difficulty_version": entry.difficulty_version,
        "overall_difficulty": entry.overall_difficulty,
        "star_rating": entry.star_rating if entry.star_rating is not None else "",
        "total_events": entry.total_events,
        "duration_s": entry.duration_s,
        "density_mean": entry.density_mean,
        "density_peak": entry.density_peak,
        "density_std": entry.density_std,
        "count_don": type_counts[OnsetKind.DON.value],
        "count_ka": type_counts[OnsetKind.KA.value],
        "count_big_don": type_counts[OnsetKind.BIG_DON.value],
        "count_big_ka": type_counts[OnsetKind.BIG_KA.value],
        "count_drumroll": type_counts[OnsetKind.DRUMROLL.value],
        "count_spinner": type_counts[OnsetKind.SPINNER.value],
        "don_ratio": round(don_ratio, 4),
        "median_gap_ms": median_gap,
        "min_gap_ms": min_gap,
        "max_gap_ms": max_gap,
        "p95_gap_ms": p95_gap,
        "num_breaks_gt1s": num_breaks,
        "longest_break_ms": longest_break,
        "longest_streak": longest_streak,
        "mean_streak_len": round(mean_streak_len, 3),
        "streak_event_fraction": round(streak_event_fraction, 4),
    }


# ─────────────────────────── aggregates ───────────────────────────────

def _compute_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _stats(name: str) -> dict[str, float]:
        vals = np.array([r[name] for r in rows
                         if r[name] != "" and r[name] is not None],
                        dtype=float)
        if not len(vals):
            return {}
        return {
            "min": float(vals.min()),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }

    event_type_totals = {
        k.value: sum(r[f"count_{k.value}"] for r in rows)
        for k in EVENT_KINDS
    }
    total_events = sum(event_type_totals.values())
    event_type_proportions = {
        k: (v / total_events if total_events else 0.0)
        for k, v in event_type_totals.items()
    }

    return {
        "n_charts": len(rows),
        "total_events": total_events,
        "event_type_totals": event_type_totals,
        "event_type_proportions": {k: round(v, 5) for k, v in event_type_proportions.items()},
        "stats": {
            "total_events": _stats("total_events"),
            "duration_s": _stats("duration_s"),
            "density_mean": _stats("density_mean"),
            "density_peak": _stats("density_peak"),
            "density_std": _stats("density_std"),
            "median_gap_ms": _stats("median_gap_ms"),
            "p95_gap_ms": _stats("p95_gap_ms"),
            "overall_difficulty": _stats("overall_difficulty"),
            "star_rating": _stats("star_rating"),
            "don_ratio": _stats("don_ratio"),
            "longest_streak": _stats("longest_streak"),
            "streak_event_fraction": _stats("streak_event_fraction"),
            "num_breaks_gt1s": _stats("num_breaks_gt1s"),
        },
    }


# ─────────────────────────── plotting ─────────────────────────────────

def _plot_all(
    rows: list[dict[str, Any]],
    all_gaps_ms: np.ndarray,
    all_gap_ratios: np.ndarray,
    avg_density_profile: np.ndarray,
    graphs_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graphs_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(graphs_dir / name, dpi=150)
        plt.close(fig)

    def _hist(vals, title, xlabel, fname, bins=80, color="#4a90d9",
              range_=None, log=False):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(vals, bins=bins, color=color, edgecolor="black",
                linewidth=0.3, range=range_, log=log)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of Charts")
        ax.set_title(title)
        if len(vals):
            med = float(np.median(vals))
            ax.axvline(med, color="red", linestyle="--",
                       label=f"Median: {med:.2f}")
            ax.legend()
        _save(fig, fname)

    # 1 — total events
    _hist([r["total_events"] for r in rows],
          "Distribution of Total Events per Chart", "Total Events",
          "01_total_events.png", color="#4a90d9")

    # 2 — duration
    _hist([r["duration_s"] for r in rows],
          "Distribution of Chart Duration", "Duration (s)",
          "02_duration.png", color="#e8834a")

    # 3 — density (events/sec)
    _hist([r["density_mean"] for r in rows],
          "Distribution of Overall Event Density", "Events per Second",
          "03_density_mean.png", color="#6bc46d")

    # 4 — event type pie
    fig, ax = plt.subplots(figsize=(7, 7))
    type_totals = {k.value: sum(r[f"count_{k.value}"] for r in rows)
                   for k in EVENT_KINDS}
    labels = [k for k, v in type_totals.items() if v > 0]
    sizes = [type_totals[k] for k in labels]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Event Type Proportions (all charts)")
    _save(fig, "04_event_type_pie.png")

    # 5 — Don vs Ka
    _hist([r["don_ratio"] for r in rows],
          "Don vs Ka Balance per Chart",
          "Don / (Don + Ka)", "05_don_ka_ratio.png",
          color="#c76dba", bins=60, range_=(0.0, 1.0))

    # 6 — peak vs avg density scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    avgs = [r["density_mean"] for r in rows]
    peaks = [r["density_peak"] for r in rows]
    ax.scatter(avgs, peaks, alpha=0.15, s=8, color="#4a90d9")
    if avgs:
        ax.plot([0, max(avgs)], [0, max(avgs)], "r--", alpha=0.3, label="y=x")
        ax.legend()
    ax.set_xlabel("Mean Density (events/s)")
    ax.set_ylabel("Peak Density (events/s)")
    ax.set_title("Peak vs Mean Density")
    _save(fig, "06_peak_vs_mean_density.png")

    # 7 — median gap
    _hist([r["median_gap_ms"] for r in rows],
          "Distribution of Median Inter-Onset Interval",
          "Median Gap (ms)", "07_median_gap_ms.png",
          color="#fcb71e", range_=(0, 1000))

    # 8 — burstiness (std of per-second density)
    _hist([r["density_std"] for r in rows],
          "Burstiness: Density Variability per Chart",
          "Std Dev of Per-Second Density", "08_burstiness.png",
          color="#e86850")

    # 9 — silent breaks
    breaks = [r["num_breaks_gt1s"] for r in rows]
    if breaks:
        _hist(breaks, "Silent Breaks (>1s gap) per Chart",
              "Number of Breaks", "09_breaks.png",
              color="#7bc0ea", bins=range(0, max(breaks) + 2))

    # 10 — duration vs events
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter([r["duration_s"] for r in rows],
               [r["total_events"] for r in rows],
               alpha=0.15, s=8, color="#6bc46d")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Total Events")
    ax.set_title("Chart Duration vs Total Events")
    _save(fig, "10_duration_vs_events.png")

    # 11 — gap-ms histogram across ALL charts (log-scale)
    if len(all_gaps_ms):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_gaps_ms, bins=200, color="#4a90d9",
                edgecolor="black", linewidth=0.2, range=(0, 1000))
        ax.set_yscale("log")
        ax.set_xlabel("Inter-Onset Gap (ms)")
        ax.set_ylabel("Count (log)")
        ax.set_title(f"Gap Distribution Across All Charts "
                     f"({len(all_gaps_ms):,} gaps, 0–1000 ms)")
        _save(fig, "11_gap_ms_hist.png")

    # 12 — gap-ratio histogram (gap_{n+1} / gap_n), musical subdivisions
    if len(all_gap_ratios):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_gap_ratios, bins=200, color="#c76dba",
                edgecolor="black", linewidth=0.2, range=(0.0, 4.0))
        ax.set_xlabel("Gap Ratio (gap_{n+1} / gap_n)")
        ax.set_ylabel("Count")
        ax.set_title(f"Gap Ratio Distribution "
                     f"({len(all_gap_ratios):,} pairs, clipped to 0-4)")
        for mark in (0.5, 1.0, 2.0):
            ax.axvline(mark, color="gray", linestyle=":", alpha=0.6)
        _save(fig, "12_gap_ratio_hist.png")

    # 13 — averaged density profile (first 5 min)
    if avg_density_profile.size:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(range(len(avg_density_profile)),
                        avg_density_profile, alpha=0.7, color="#4a90d9")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Avg Events per Second")
        ax.set_title("Average Density Profile Across All Charts (first 5 min)")
        _save(fig, "13_avg_density_profile.png")

    # 14 — top 20 densest
    fig, ax = plt.subplots(figsize=(12, 6))
    top = sorted(rows, key=lambda r: r["density_mean"], reverse=True)[:20]
    ax.barh(range(len(top)), [r["density_mean"] for r in top], color="#eb4528")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([r["chart_id"][:60] for r in top], fontsize=7)
    ax.set_xlabel("Mean Density (events/s)")
    ax.set_title("Top 20 Densest Charts")
    ax.invert_yaxis()
    _save(fig, "14_top20_densest.png")

    # 15 — star rating (if populated)
    stars = [r["star_rating"] for r in rows
             if r["star_rating"] not in ("", None)]
    if stars:
        _hist(stars, f"Star Rating Distribution ({len(stars)} charts)",
              "Star Rating", "15_star_rating.png",
              color="#fcb71e", bins=60)

    # 16 — OD distribution
    ods = [r["overall_difficulty"] for r in rows
           if r["overall_difficulty"] is not None]
    if ods:
        _hist(ods, "OverallDifficulty (OD) Distribution", "OD",
              "16_od.png", color="#64c864", bins=40, range_=(0, 10))

    # 17 — longest streak distribution
    _hist([r["longest_streak"] for r in rows],
          "Longest Same-Gap Streak per Chart",
          "Events in Longest Streak", "17_longest_streak.png",
          color="#9b6eff", bins=60)

    # 18 — fraction of events in streaks
    _hist([r["streak_event_fraction"] for r in rows],
          "Fraction of Events in a Same-Gap Streak",
          "Fraction of Events in Streaks", "18_streak_fraction.png",
          color="#ff7b6b", bins=50, range_=(0.0, 1.0))


# ─────────────────────────── loading helpers ──────────────────────────

def _iter_chart_events(
    manifest: DatasetManifest,
    dataset_dir: Path,
    max_charts: int | None,
    progress: bool,
):
    """Yield (entry, events) for every loadable chart."""
    charts = manifest.charts
    if max_charts is not None:
        charts = charts[:max_charts]
    it = tqdm(charts, desc="Loading events", unit="chart") if progress else charts
    for entry in it:
        # events live at {dataset_dir}/events/{safe_chart_id}.npz
        # The safe stem is the one used by prepare_dataset._safe_filename.
        stem = _safe_stem(entry.chart_id)
        events_path = dataset_dir / "events" / f"{stem}.npz"
        if not events_path.exists():
            if progress:
                tqdm.write(f"  missing events: {events_path}")
            continue
        try:
            events = load_events(events_path)
        except Exception as e:
            if progress:
                tqdm.write(f"  failed to load {events_path}: {e}")
            continue
        yield entry, events


def _safe_stem(s: str, max_len: int = 120) -> str:
    """Must mirror `osu.taiko2.dataset._safe_filename` exactly."""
    import hashlib
    for ch in '<>:"/\\|?*\n\r':
        s = s.replace(ch, "_")
    s = s.strip(". ")
    if len(s) > max_len:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        s = s[:max_len - 9] + "_" + h
    return s


def _write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────── CLI ──────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze a taiko2 dataset: per-chart metrics + graphs.",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name (directory under --datasets-dir) OR "
                        "a path to a dataset root.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets",
                   help="Root containing named datasets (default: "
                        "osu/taiko2/datasets).")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "analysis",
                   help="Output root (default: osu/taiko2/analysis). "
                        "Results land in {out_dir}/{dataset_name}/.")
    p.add_argument("--max-charts", type=int, default=None,
                   help="Only analyze the first N charts (smoke-test mode).")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm bars.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ds_root = Path(args.dataset)
    if not ds_root.is_absolute() and not ds_root.exists():
        ds_root = args.datasets_dir / args.dataset
    ds_root = ds_root.resolve()
    if not ds_root.is_dir():
        print(f"ERROR: dataset root not found: {ds_root}", file=sys.stderr)
        return 2

    manifest_path = ds_root / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest missing: {manifest_path}", file=sys.stderr)
        return 2

    manifest = load_manifest(manifest_path)
    dataset_name = manifest.name or ds_root.name
    out_dir = (args.out_dir / dataset_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = out_dir / "graphs"

    print(f"Dataset:  {manifest.name} ({len(manifest.charts)} charts)")
    print(f"Output:   {out_dir}")

    rows: list[dict[str, Any]] = []
    # Aggregate data accumulators for cross-chart plots
    all_gaps_ms: list[np.ndarray] = []
    all_gap_ratios: list[np.ndarray] = []
    profile_max_s = 300
    profile_sum = np.zeros(profile_max_s, dtype=np.float64)
    profile_count = np.zeros(profile_max_s, dtype=np.int64)

    for entry, events in _iter_chart_events(
        manifest, ds_root, args.max_charts, not args.no_progress,
    ):
        rows.append(_chart_metrics(entry, events))

        if len(events) >= 2:
            times_ms = np.array([o.time_ms for o in events], dtype=np.int64)
            gaps = np.diff(times_ms)
            gaps_pos = gaps[gaps > 0]
            all_gaps_ms.append(gaps_pos.astype(np.float64))

            if len(gaps_pos) >= 2:
                ratios = gaps_pos[1:] / gaps_pos[:-1]
                all_gap_ratios.append(ratios)

            # per-second density profile (first 5 min)
            first_ms = int(times_ms[0])
            buckets = np.zeros(profile_max_s, dtype=np.int64)
            for t in times_ms:
                sec = (int(t) - first_ms) // 1000
                if 0 <= sec < profile_max_s:
                    buckets[sec] += 1
            profile_sum += buckets
            profile_count += (buckets > 0) | (np.arange(profile_max_s) * 1000 + first_ms <= int(times_ms[-1]))

    if not rows:
        print("No charts loaded; aborting.", file=sys.stderr)
        return 1

    # Per-chart CSV
    metrics_path = out_dir / "metrics.csv"
    _write_metrics_csv(rows, metrics_path)
    print(f"Wrote per-chart metrics: {metrics_path}")

    # Aggregates JSON
    aggregates = _compute_aggregates(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8",
    )
    print(f"Wrote summary:          {out_dir / 'summary.json'}")

    # Graphs
    avg_profile = np.where(profile_count > 0,
                           profile_sum / np.maximum(profile_count, 1), 0.0)
    gaps_flat = np.concatenate(all_gaps_ms) if all_gaps_ms else np.zeros((0,))
    ratios_flat = np.concatenate(all_gap_ratios) if all_gap_ratios else np.zeros((0,))

    _plot_all(rows, gaps_flat, ratios_flat, avg_profile, graphs_dir)
    n_pngs = len(list(graphs_dir.glob("*.png")))
    print(f"Wrote {n_pngs} graphs to:   {graphs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
