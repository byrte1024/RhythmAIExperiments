"""Cross-analysis of popularity / engagement (from osu! API v2) against
the intrinsic chart metrics we compute ourselves.

Inputs (both produced by sibling CLIs):

  - ``analysis/{dataset}/chart_metrics.csv``   — from `analyze_charts`
  - ``datasets/{dataset}/manifest_engagement.csv`` — from `fetch_engagement`

Outputs land under ``analysis/{dataset}/engagement/``:

  - ``engagement_summary.json``  per-field min/p25/median/p75/p95/max
                                  over every engagement scalar.
  - ``correlations.json``        full matrix of Pearson ``r`` between
                                  every engagement metric and every
                                  chart scalar, sortable by ``abs(r)``.
  - ``correlations_ranked.csv``  flat CSV of the same, one row per
                                  pair, sorted by ``abs(r)`` desc.
  - ``graphs/``                  the top-K scatter plots (K defaults
                                  to 24, controlled by ``--top-k``)
                                  plus one bar-chart overview of the
                                  strongest correlations per
                                  engagement metric.

The point is the "ideal vs average" question from experiment #003 —
knowing which chart metrics predict popularity / user rating lets us
say something about what distinguishes charts humans actually like
from the corpus mean.

Usage::

    python -m osu.taiko2.cli.analyze_engagement --dataset taiko2_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────── field lists ──────────────────────────────

# Engagement fields we plot and correlate. Skips strings (genre,
# language, status) — those are categorical and shown separately.
_ENGAGEMENT_SCALAR_FIELDS: tuple[str, ...] = (
    "playcount", "passcount", "pass_rate",
    "play_count_set", "favourite_count",
    "rating_mean", "rating_count",
    "bpm_set", "nominations_current",
)

# Chart-level scalars worth pairing against engagement. Skips
# denormalised fields (counts that just scale with total_events).
_CHART_SCALAR_FIELDS: tuple[str, ...] = (
    "total_events", "duration_s", "events_per_sec",
    "don_ratio",
    "ioi_mean_ms", "ioi_median_ms", "ioi_std_ms",
    "ioi_p95_ms", "ioi_p99_ms",
    "short_ioi_pct", "long_gap_count",
    "density_mean", "density_peak", "density_std", "density_cv",
    "longest_streak", "mean_streak_len", "streak_event_fraction",
    "estimated_bpm", "dominant_ioi_ms", "over_pspace_self",
    "gap_peak_count", "gap_peak_mass_total",
    "gap_peak_falloff", "gap_random_distance", "gap_metronome_distance",
    "ratio_peak_count", "ratio_peak_mass_total",
    "ratio_peak_falloff", "ratio_random_distance", "ratio_metronome_distance",
    "star_rating",
)


# ─────────────────────────── IO helpers ──────────────────────────────

def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(s: str) -> float | None:
    if s == "" or s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _join_by_beatmap(
    chart_rows: list[dict[str, str]],
    engagement_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    """Inner-join the two CSVs on ``beatmap_id``. Returns the joined
    rows and the count of chart rows without a matching engagement
    entry (for logging)."""
    engagement_by_id = {r["beatmap_id"]: r for r in engagement_rows}
    joined: list[dict[str, Any]] = []
    missing = 0
    for c in chart_rows:
        bid = c.get("beatmap_id")
        if not bid:
            continue
        e = engagement_by_id.get(bid)
        if e is None:
            missing += 1
            continue
        row: dict[str, Any] = {}
        for k, v in c.items():
            row[f"chart__{k}"] = v
        for k, v in e.items():
            row[f"eng__{k}"] = v
        joined.append(row)
    return joined, missing


def _numeric_column(
    rows: list[dict[str, Any]], field: str,
) -> np.ndarray:
    vals = []
    for r in rows:
        v = _as_float(r.get(field, ""))
        if v is None:
            continue
        vals.append(v)
    return np.asarray(vals, dtype=np.float64)


def _aligned(
    rows: list[dict[str, Any]], fx: str, fy: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        vx = _as_float(r.get(fx, ""))
        vy = _as_float(r.get(fy, ""))
        if vx is None or vy is None:
            continue
        xs.append(vx)
        ys.append(vy)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ─────────────────────────── summary stats ───────────────────────────

def _percentiles(vals: np.ndarray) -> dict[str, float] | None:
    if len(vals) == 0:
        return None
    q = np.percentile(vals, [0, 25, 50, 75, 95, 100])
    return {
        "n": int(len(vals)),
        "min":    round(float(q[0]), 4),
        "p25":    round(float(q[1]), 4),
        "median": round(float(q[2]), 4),
        "p75":    round(float(q[3]), 4),
        "p95":    round(float(q[4]), 4),
        "max":    round(float(q[5]), 4),
        "mean":   round(float(vals.mean()), 4),
    }


def _engagement_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n_joined": len(rows), "scalars": {}}
    for field in _ENGAGEMENT_SCALAR_FIELDS:
        vals = _numeric_column(rows, f"eng__{field}")
        pct = _percentiles(vals)
        if pct is not None:
            out["scalars"][field] = pct
    # Categorical fields — top-20 values by count.
    for field in ("eng__status", "eng__genre", "eng__language"):
        from collections import Counter
        counts = Counter(r.get(field, "") for r in rows if r.get(field))
        out.setdefault("categoricals", {})[field[5:]] = (
            counts.most_common(20)
        )
    return out


# ─────────────────────────── correlations ────────────────────────────

def _compute_correlations(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pearson r between every (engagement, chart) pair. Skip pairs
    with < 50 joint samples — too little data to plot a trend."""
    pairs: list[dict[str, Any]] = []
    for e_field in _ENGAGEMENT_SCALAR_FIELDS:
        for c_field in _CHART_SCALAR_FIELDS:
            xs, ys = _aligned(rows, f"eng__{e_field}", f"chart__{c_field}")
            if len(xs) < 50:
                continue
            r = _pearson(xs, ys)
            if r is None:
                continue
            pairs.append({
                "engagement_field": e_field,
                "chart_field": c_field,
                "n": int(len(xs)),
                "r": round(r, 4),
                "abs_r": round(abs(r), 4),
            })
    pairs.sort(key=lambda p: p["abs_r"], reverse=True)
    return pairs


# ─────────────────────────── graphs ──────────────────────────────────

def _plot_top_scatters(
    rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    graphs_dir: Path,
    *,
    top_k: int,
) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graphs_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for i, pair in enumerate(correlations[:top_k], start=1):
        e_field = pair["engagement_field"]
        c_field = pair["chart_field"]
        xs, ys = _aligned(rows, f"eng__{e_field}", f"chart__{c_field}")
        if len(xs) < 50:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(xs, ys, alpha=0.2, s=8, color="#4a90d9")
        # Log-x for heavily-skewed popularity fields.
        if e_field in (
            "playcount", "passcount", "play_count_set",
            "favourite_count", "rating_count",
        ) and (xs > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel(f"eng: {e_field}")
        ax.set_ylabel(f"chart: {c_field}")
        ax.set_title(
            f"{e_field}  vs  {c_field}   "
            f"(n={pair['n']:,}, r={pair['r']:+.3f})"
        )
        ax.grid(True, which="both", alpha=0.15)
        fig.tight_layout()
        fname = (
            f"{i:02d}_{e_field}__vs__{c_field}.png"
            .replace("/", "_")
        )
        fig.savefig(graphs_dir / fname, dpi=140)
        plt.close(fig)
        n_written += 1
    return n_written


def _plot_best_per_engagement_overview(
    correlations: list[dict[str, Any]],
    graphs_dir: Path,
) -> None:
    """One bar chart per engagement metric: the 8 chart metrics most
    strongly correlated with it. Helps surface "popularity tracks
    what?" at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_eng: dict[str, list[dict[str, Any]]] = {}
    for pair in correlations:
        by_eng.setdefault(pair["engagement_field"], []).append(pair)

    for e_field, pairs in by_eng.items():
        top = pairs[:10]
        if not top:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [p["chart_field"] for p in top]
        rs = [p["r"] for p in top]
        colors = [
            "#e86850" if r < 0 else "#4a90d9" for r in rs
        ]
        ax.barh(range(len(top)), rs, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Pearson r  (blue = positive, red = negative)")
        ax.set_title(f"Top chart metrics correlated with eng:{e_field}")
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(graphs_dir / f"_overview_{e_field}.png", dpi=140)
        plt.close(fig)


# ─────────────────────────── CLI ─────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlate engagement against chart metrics.",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under --datasets-dir) or path.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--analysis-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "analysis")
    p.add_argument("--top-k", type=int, default=24,
                   help="How many highest-|r| scatter plots to render.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ds_name = Path(args.dataset).name
    ds_root = (
        args.datasets_dir / args.dataset
        if not Path(args.dataset).is_absolute() else Path(args.dataset)
    )
    engagement_csv = ds_root / "manifest_engagement.csv"
    chart_metrics_csv = args.analysis_dir / ds_name / "chart_metrics.csv"

    for p in (engagement_csv, chart_metrics_csv):
        if not p.exists():
            print(f"ERROR: missing input {p}", file=sys.stderr)
            return 2

    chart_rows = _load_csv(chart_metrics_csv)
    engagement_rows = _load_csv(engagement_csv)
    print(f"chart metrics:   {len(chart_rows):,} rows")
    print(f"engagement rows: {len(engagement_rows):,} rows")

    joined, missing = _join_by_beatmap(chart_rows, engagement_rows)
    print(f"joined:          {len(joined):,} rows  ({missing:,} chart rows had no engagement match)")
    if not joined:
        print("No overlap — nothing to correlate.", file=sys.stderr)
        return 1

    out_dir = args.analysis_dir / ds_name / "engagement"
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = out_dir / "graphs"

    # Summary.
    summary = _engagement_summary(joined)
    (out_dir / "engagement_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Correlations.
    correlations = _compute_correlations(joined)
    (out_dir / "correlations.json").write_text(
        json.dumps(correlations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "correlations_ranked.csv").open(
        "w", newline="", encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["engagement_field", "chart_field", "n", "r", "abs_r"],
        )
        writer.writeheader()
        writer.writerows(correlations)

    n_graphs = _plot_top_scatters(
        joined, correlations, graphs_dir, top_k=args.top_k,
    )
    _plot_best_per_engagement_overview(correlations, graphs_dir)

    print(f"\nWrote:")
    print(f"  {out_dir / 'engagement_summary.json'}")
    print(f"  {out_dir / 'correlations.json'}")
    print(f"  {out_dir / 'correlations_ranked.csv'}")
    print(f"  {graphs_dir}   ({n_graphs} scatter PNGs + overview bars)")

    # Surface top-10 correlations on stdout so the user sees them.
    print("\nTop correlations (|r| desc):")
    print(f"  {'engagement':<22} {'chart':<28} {'n':>7} {'r':>7}")
    for pair in correlations[:10]:
        print(
            f"  {pair['engagement_field']:<22} "
            f"{pair['chart_field']:<28} "
            f"{pair['n']:>7,} {pair['r']:>+7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
