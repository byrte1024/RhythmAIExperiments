"""Build metadata CSV from all parsed onset CSVs + generate analysis graphs."""
import csv
import glob
import os
import json
import math
from collections import Counter

DATA_DIR = "./data"
OUTPUT_CSV = "./metadata.csv"
GRAPHS_DIR = "./graphs"

os.makedirs(GRAPHS_DIR, exist_ok=True)

EVENT_TYPES = ["don", "ka", "big_don", "big_ka", "drumroll", "spinner"]


def load_onsets(csv_path):
    """Load onsets from a parsed CSV, return (audio_file, [(time_ms, type), ...])."""
    audio_file = ""
    onsets = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("# audio:"):
                audio_file = line.split(":", 1)[1].strip()
                continue
            if line.startswith("time_ms"):
                continue
            parts = line.strip().split(",")
            if len(parts) == 2:
                onsets.append((int(parts[0]), parts[1]))
    return audio_file, onsets


def compute_sectional_density(onsets, bucket_ms=1000):
    """Compute per-second event counts. Returns dict of {type: [count_per_sec, ...]} and total."""
    if not onsets:
        return {}, []

    max_time = max(t for t, _ in onsets)
    n_buckets = max_time // bucket_ms + 1

    by_type = {t: [0] * n_buckets for t in EVENT_TYPES}
    total = [0] * n_buckets

    for time_ms, kind in onsets:
        bucket = time_ms // bucket_ms
        if kind in by_type:
            by_type[kind][bucket] += 1
        total[bucket] += 1

    return by_type, total


def analyze_one(csv_path):
    """Analyze a single CSV and return a metadata dict."""
    name = os.path.splitext(os.path.basename(csv_path))[0]
    audio_file, onsets = load_onsets(csv_path)

    if not onsets:
        return None

    counts = Counter(kind for _, kind in onsets)
    total_events = len(onsets)

    first_ms = onsets[0][0]
    last_ms = onsets[-1][0]
    duration_s = (last_ms - first_ms) / 1000.0
    if duration_s <= 0:
        duration_s = 0.001

    # overall density (events per second)
    density_total = total_events / duration_s
    density_by_type = {t: counts.get(t, 0) / duration_s for t in EVENT_TYPES}

    # sectional density stats (per-second buckets)
    sect_by_type, sect_total = compute_sectional_density(onsets)
    active_buckets = [v for v in sect_total if v > 0]

    peak_density = max(sect_total) if sect_total else 0
    avg_active_density = sum(active_buckets) / len(active_buckets) if active_buckets else 0
    density_std = (
        (sum((v - avg_active_density) ** 2 for v in active_buckets) / len(active_buckets)) ** 0.5
        if active_buckets else 0
    )

    # gaps: silent stretches
    gaps = []
    for i in range(1, len(onsets)):
        gap = onsets[i][0] - onsets[i - 1][0]
        if gap > 1000:
            gaps.append(gap)

    # interval stats
    intervals = [onsets[i][0] - onsets[i - 1][0] for i in range(1, len(onsets))]
    median_interval = sorted(intervals)[len(intervals) // 2] if intervals else 0
    min_interval = min(intervals) if intervals else 0
    max_interval = max(intervals) if intervals else 0

    row = {
        "name": name,
        "audio_file": audio_file,
        "total_events": total_events,
        "duration_s": round(duration_s, 2),
        "first_event_ms": first_ms,
        "last_event_ms": last_ms,
        # per-type counts
        **{f"count_{t}": counts.get(t, 0) for t in EVENT_TYPES},
        # per-type density (events/sec)
        **{f"density_{t}": round(density_by_type[t], 3) for t in EVENT_TYPES},
        "density_total": round(density_total, 3),
        # sectional density stats
        "peak_density_per_sec": peak_density,
        "avg_active_density_per_sec": round(avg_active_density, 3),
        "density_std_per_sec": round(density_std, 3),
        # interval stats
        "median_interval_ms": median_interval,
        "min_interval_ms": min_interval,
        "max_interval_ms": max_interval,
        # gap info
        "num_breaks_gt1s": len(gaps),
        "longest_break_ms": max(gaps) if gaps else 0,
        # sectional density as JSON arrays (for graphing)
        "sectional_total": json.dumps(sect_total),
        **{f"sectional_{t}": json.dumps(sect_by_type.get(t, [])) for t in EVENT_TYPES},
    }
    return row


def build_metadata():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"Found {len(csvs)} onset CSVs")

    rows = []
    for i, path in enumerate(csvs, 1):
        row = analyze_one(path)
        if row:
            rows.append(row)
        if i % 500 == 0:
            print(f"  [{i}/{len(csvs)}] analyzed...")

    # write metadata CSV
    if not rows:
        print("No data to write.")
        return rows

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    return rows


def make_graphs(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed, skipping graphs. pip install matplotlib numpy")
        return

    print("Generating graphs...")

    # ── 1. Distribution of total events per chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    totals = [r["total_events"] for r in rows]
    ax.hist(totals, bins=80, color="#4a90d9", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Total Events")
    ax.set_ylabel("Number of Charts")
    ax.set_title("Distribution of Total Events per Chart")
    ax.axvline(np.median(totals), color="red", linestyle="--", label=f"Median: {int(np.median(totals))}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "01_total_events_dist.png"), dpi=150)
    plt.close(fig)

    # ── 2. Distribution of chart duration ──
    fig, ax = plt.subplots(figsize=(10, 5))
    durations = [r["duration_s"] for r in rows]
    ax.hist(durations, bins=80, color="#e8834a", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Number of Charts")
    ax.set_title("Distribution of Chart Duration")
    ax.axvline(np.median(durations), color="red", linestyle="--", label=f"Median: {np.median(durations):.0f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "02_duration_dist.png"), dpi=150)
    plt.close(fig)

    # ── 3. Overall density distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    densities = [r["density_total"] for r in rows]
    ax.hist(densities, bins=80, color="#6bc46d", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Events per Second (overall)")
    ax.set_ylabel("Number of Charts")
    ax.set_title("Distribution of Overall Event Density")
    ax.axvline(np.median(densities), color="red", linestyle="--", label=f"Median: {np.median(densities):.1f}/s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "03_density_dist.png"), dpi=150)
    plt.close(fig)

    # ── 4. Event type proportions (pie) ──
    fig, ax = plt.subplots(figsize=(7, 7))
    type_totals = {t: sum(r[f"count_{t}"] for r in rows) for t in EVENT_TYPES}
    labels = [t for t in EVENT_TYPES if type_totals[t] > 0]
    sizes = [type_totals[t] for t in labels]
    colors = ["#eb4528", "#4490c7", "#ff7b6b", "#7bc0ea", "#fcb71e", "#64c864"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)], startangle=90)
    ax.set_title("Event Type Proportions (all charts)")
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "04_event_type_pie.png"), dpi=150)
    plt.close(fig)

    # ── 5. Don vs Ka ratio per chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    don_counts = [r["count_don"] + r["count_big_don"] for r in rows]
    ka_counts = [r["count_ka"] + r["count_big_ka"] for r in rows]
    ratios = [d / (d + k) if (d + k) > 0 else 0.5 for d, k in zip(don_counts, ka_counts)]
    ax.hist(ratios, bins=60, color="#c76dba", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Don Ratio (don / (don + ka))")
    ax.set_ylabel("Number of Charts")
    ax.set_title("Don vs Ka Balance per Chart")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(np.median(ratios), color="red", linestyle="--", label=f"Median: {np.median(ratios):.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "05_don_ka_ratio.png"), dpi=150)
    plt.close(fig)

    # ── 6. Peak density vs average density scatter ──
    fig, ax = plt.subplots(figsize=(10, 6))
    peaks = [r["peak_density_per_sec"] for r in rows]
    avgs = [r["avg_active_density_per_sec"] for r in rows]
    ax.scatter(avgs, peaks, alpha=0.15, s=8, color="#4a90d9")
    ax.set_xlabel("Avg Active Density (events/sec)")
    ax.set_ylabel("Peak Density (events/sec)")
    ax.set_title("Peak vs Average Density")
    ax.plot([0, max(avgs)], [0, max(avgs)], "r--", alpha=0.3, label="y=x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "06_peak_vs_avg_density.png"), dpi=150)
    plt.close(fig)

    # ── 7. Median interval distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    medians = [r["median_interval_ms"] for r in rows]
    ax.hist(medians, bins=80, color="#fcb71e", edgecolor="black", linewidth=0.3, range=(0, 1000))
    ax.set_xlabel("Median Interval Between Events (ms)")
    ax.set_ylabel("Number of Charts")
    ax.set_title("Distribution of Median Inter-Onset Interval")
    ax.axvline(np.median(medians), color="red", linestyle="--", label=f"Median: {np.median(medians):.0f}ms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "07_median_interval_dist.png"), dpi=150)
    plt.close(fig)

    # ── 8. Density std (burstiness) distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    stds = [r["density_std_per_sec"] for r in rows]
    ax.hist(stds, bins=80, color="#e86850", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Std Dev of Per-Second Density")
    ax.set_ylabel("Number of Charts")
    ax.set_title("Burstiness: How Variable is the Density Across the Song?")
    ax.axvline(np.median(stds), color="red", linestyle="--", label=f"Median: {np.median(stds):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "08_burstiness_dist.png"), dpi=150)
    plt.close(fig)

    # ── 9. Number of breaks (>1s silence) distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    breaks = [r["num_breaks_gt1s"] for r in rows]
    ax.hist(breaks, bins=range(0, max(breaks) + 2), color="#7bc0ea", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Number of Breaks (>1s gap)")
    ax.set_ylabel("Number of Charts")
    ax.set_title("Distribution of Silent Breaks per Chart")
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "09_breaks_dist.png"), dpi=150)
    plt.close(fig)

    # ── 10. Duration vs total events scatter ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(durations, totals, alpha=0.15, s=8, color="#6bc46d")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Total Events")
    ax.set_title("Chart Duration vs Total Events")
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "10_duration_vs_events.png"), dpi=150)
    plt.close(fig)

    # ── 11. Heatmap: average sectional density across all charts (normalized) ──
    fig, ax = plt.subplots(figsize=(12, 4))
    max_secs = 300  # up to 5 minutes
    avg_profile = np.zeros(max_secs)
    chart_count = np.zeros(max_secs)
    for r in rows:
        sect = json.loads(r["sectional_total"])
        for s, v in enumerate(sect[:max_secs]):
            avg_profile[s] += v
            chart_count[s] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_profile = np.where(chart_count > 0, avg_profile / chart_count, 0)
    ax.fill_between(range(max_secs), avg_profile, alpha=0.7, color="#4a90d9")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Avg Events per Second")
    ax.set_title("Average Density Profile Across All Charts (first 5 min)")
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "11_avg_density_profile.png"), dpi=150)
    plt.close(fig)

    # ── 12. Top 20 densest charts ──
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_by_density = sorted(rows, key=lambda r: r["density_total"], reverse=True)[:20]
    names = [r["name"][:50] for r in sorted_by_density]
    vals = [r["density_total"] for r in sorted_by_density]
    ax.barh(range(len(names)), vals, color="#eb4528")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Events per Second")
    ax.set_title("Top 20 Densest Charts")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, "12_top20_densest.png"), dpi=150)
    plt.close(fig)

    print(f"Saved 12 graphs to {GRAPHS_DIR}/")


if __name__ == "__main__":
    rows = build_metadata()
    if rows:
        make_graphs(rows)
