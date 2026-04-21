"""Experiment 67 — Ratio prediction data analysis.

Analyzes the dataset to understand:
1. Dominant gap distributions (what values would the divisor predict?)
2. Cursor offset distribution (how far is cursor from last event?)
3. Ratio distributions given dominant gaps (what R_MIN/R_MAX do we need?)
4. Coverage: what % of targets are reachable via top-1/2/3 dominant gaps?

Usage:
    cd osu/taiko
    python experiments/experiment_67/analyze_ratios.py
"""

import json
import os
import sys

import numpy as np
from collections import Counter
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")

B_PRED = 250
C_EVENTS = 128
MIN_CURSOR_BIN = 6000


def compute_dominant_gaps(gaps, tolerance=0.05):
    """Find dominant gaps (clustered within tolerance) sorted by frequency."""
    if len(gaps) < 2:
        return []
    clusters = {}
    for g in gaps:
        matched = False
        for center in list(clusters.keys()):
            if abs(g - center) / max(center, 1) <= tolerance:
                clusters[center] += 1
                matched = True
                break
        if not matched:
            clusters[g] = 1
    return sorted(clusters.items(), key=lambda x: -x[1])


def main():
    with open(os.path.join(DATASET_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    charts = manifest["charts"]
    evt_dir = os.path.join(DATASET_DIR, "events")

    # Collectors
    all_dominant_gaps = []           # top-1 dominant gap per sample
    all_dominant_gap_coverage = []   # what % of gaps match the dominant
    all_cursor_offsets = []          # distance from cursor to last event
    all_ratios_top1 = []            # target ratio using top-1 dominant gap
    all_ratios_top2 = []            # target ratio using best of top-2
    all_ratios_top3 = []            # target ratio using best of top-3
    all_raw_gaps = []               # all inter-event gaps in dataset
    all_target_bins = []            # raw target bin offsets
    samples_with_offset = 0         # cursor != last event
    samples_total = 0
    samples_no_context = 0          # fewer than 4 past events
    samples_no_dominant = 0         # no clear dominant gap

    # Per-chart analysis
    chart_dominant_gaps = []

    print(f"Analyzing {len(charts)} charts...")

    for ci, chart in enumerate(tqdm(charts, desc="Charts")):
        evt = np.load(os.path.join(evt_dir, chart["event_file"]))
        if len(evt) < 5:
            continue

        # Compute all gaps for this chart
        chart_gaps = np.diff(evt).astype(np.float64)
        chart_gaps = chart_gaps[chart_gaps > 0]
        all_raw_gaps.extend(chart_gaps.tolist())

        # Chart-level dominant gap
        dom = compute_dominant_gaps(chart_gaps)
        if dom:
            chart_dominant_gaps.append(dom[0][0])

        # Per-sample analysis
        for ei in range(len(evt)):
            cursor = max(0, int(evt[0]) - B_PRED) if ei == 0 else int(evt[ei - 1])
            if cursor < MIN_CURSOR_BIN:
                continue

            # Target
            if ei < len(evt):
                target = max(0, int(evt[ei]) - cursor)
                if target >= B_PRED:
                    target = B_PRED  # STOP
            else:
                target = B_PRED  # STOP

            if target == B_PRED:
                continue  # skip STOP for ratio analysis

            samples_total += 1
            all_target_bins.append(target)

            # Cursor offset from last event
            if ei > 0:
                offset = cursor - int(evt[ei - 1])
            else:
                offset = 0
            all_cursor_offsets.append(offset)
            if offset > 0:
                samples_with_offset += 1

            # Past events for gap analysis
            past_start = max(0, ei - C_EVENTS)
            past = evt[past_start:ei].astype(np.float64)
            if len(past) < 4:
                samples_no_context += 1
                continue

            gaps = np.diff(past)
            gaps = gaps[gaps > 0]
            if len(gaps) < 2:
                samples_no_context += 1
                continue

            # Dominant gaps
            dom = compute_dominant_gaps(gaps)
            if not dom or dom[0][1] < 2:
                samples_no_dominant += 1
                continue

            top_gaps = [d[0] for d in dom[:3]]
            top_counts = [d[1] for d in dom[:3]]
            total_gaps = sum(c for _, c in dom)

            all_dominant_gaps.append(top_gaps[0])
            all_dominant_gap_coverage.append(top_counts[0] / total_gaps)

            # Compute ratios for each dominant gap candidate
            # ratio = (target + offset) / divisor
            effective_target = target + offset  # distance from last event to next event

            for k, collector in [(1, all_ratios_top1), (2, all_ratios_top2), (3, all_ratios_top3)]:
                best_ratio = None
                best_err = float("inf")
                for g in top_gaps[:k]:
                    if g > 0:
                        r = effective_target / g
                        err = abs(r - round(r * 4) / 4)  # distance to nearest 0.25
                        if err < best_err:
                            best_err = err
                            best_ratio = r
                if best_ratio is not None:
                    collector.append(best_ratio)

    # ═══════════════════════════════════════════════════════════
    #  Results
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"DATASET RATIO ANALYSIS")
    print(f"{'='*70}")
    print(f"Total non-STOP samples: {samples_total:,}")
    print(f"Samples with cursor offset > 0: {samples_with_offset:,} ({samples_with_offset/max(samples_total,1)*100:.1f}%)")
    print(f"Samples with < 4 past events: {samples_no_context:,} ({samples_no_context/max(samples_total,1)*100:.1f}%)")
    print(f"Samples with no clear dominant gap: {samples_no_dominant:,} ({samples_no_dominant/max(samples_total,1)*100:.1f}%)")

    # Cursor offset distribution
    offsets = np.array(all_cursor_offsets)
    print(f"\n--- Cursor Offset Distribution ---")
    print(f"  Mean: {offsets.mean():.1f} bins ({offsets.mean()*4.99:.1f}ms)")
    print(f"  Median: {np.median(offsets):.0f}")
    print(f"  P90: {np.percentile(offsets, 90):.0f}")
    print(f"  P99: {np.percentile(offsets, 99):.0f}")
    print(f"  Max: {offsets.max():.0f}")
    print(f"  Exactly 0: {(offsets == 0).sum():,} ({(offsets == 0).mean()*100:.1f}%)")
    print(f"  0-5: {((offsets >= 0) & (offsets <= 5)).sum():,} ({((offsets >= 0) & (offsets <= 5)).mean()*100:.1f}%)")
    print(f"  > 20: {(offsets > 20).sum():,} ({(offsets > 20).mean()*100:.1f}%)")

    # Raw gap distribution
    raw = np.array(all_raw_gaps)
    print(f"\n--- Raw Inter-Event Gap Distribution ---")
    print(f"  Mean: {raw.mean():.1f} bins ({raw.mean()*4.99:.1f}ms)")
    print(f"  Median: {np.median(raw):.0f}")
    print(f"  P10: {np.percentile(raw, 10):.0f}")
    print(f"  P25: {np.percentile(raw, 25):.0f}")
    print(f"  P75: {np.percentile(raw, 75):.0f}")
    print(f"  P90: {np.percentile(raw, 90):.0f}")
    print(f"  P99: {np.percentile(raw, 99):.0f}")
    print(f"  Max: {raw.max():.0f}")

    # Dominant gap distribution
    dom_gaps = np.array(all_dominant_gaps)
    dom_cov = np.array(all_dominant_gap_coverage)
    print(f"\n--- Dominant Gap (Top-1) Distribution ---")
    print(f"  N samples: {len(dom_gaps):,}")
    print(f"  Mean: {dom_gaps.mean():.1f} bins ({dom_gaps.mean()*4.99:.1f}ms)")
    print(f"  Median: {np.median(dom_gaps):.0f}")
    print(f"  P10: {np.percentile(dom_gaps, 10):.0f}")
    print(f"  P90: {np.percentile(dom_gaps, 90):.0f}")
    print(f"  Coverage (% of gaps matching dominant): mean={dom_cov.mean():.1%} med={np.median(dom_cov):.1%}")

    # Most common dominant gap values
    dom_rounded = np.round(dom_gaps).astype(int)
    dom_counter = Counter(dom_rounded)
    print(f"\n  Top 20 dominant gap values:")
    for val, count in dom_counter.most_common(20):
        print(f"    gap={val:>4} ({val*4.99:>6.1f}ms): {count:>8,} ({count/len(dom_gaps)*100:>5.1f}%)")

    # Chart-level dominant gaps
    cdg = np.array(chart_dominant_gaps)
    print(f"\n--- Chart-Level Dominant Gap ---")
    print(f"  N charts: {len(cdg)}")
    print(f"  Mean: {cdg.mean():.1f}  Median: {np.median(cdg):.0f}  Min: {cdg.min():.0f}  Max: {cdg.max():.0f}")

    # Ratio distributions
    for k, ratios_list, label in [
        (1, all_ratios_top1, "Top-1 dominant gap"),
        (2, all_ratios_top2, "Best of top-2"),
        (3, all_ratios_top3, "Best of top-3"),
    ]:
        if not ratios_list:
            continue
        ratios = np.array(ratios_list)
        print(f"\n--- Ratio Distribution ({label}) ---")
        print(f"  N: {len(ratios):,}")
        print(f"  Mean: {ratios.mean():.3f}")
        print(f"  Median: {np.median(ratios):.3f}")
        print(f"  Min: {ratios.min():.3f}")
        print(f"  Max: {ratios.max():.3f}")
        print(f"  P1: {np.percentile(ratios, 1):.3f}")
        print(f"  P5: {np.percentile(ratios, 5):.3f}")
        print(f"  P95: {np.percentile(ratios, 95):.3f}")
        print(f"  P99: {np.percentile(ratios, 99):.3f}")

        # Musical ratio buckets
        buckets = [
            (0, 0.3, "< 0.3x"),
            (0.3, 0.6, "~0.5x (double)"),
            (0.6, 0.8, "~0.67x"),
            (0.8, 1.2, "~1.0x (same)"),
            (1.2, 1.6, "~1.33x"),
            (1.6, 2.4, "~2.0x (half)"),
            (2.4, 3.5, "~3.0x"),
            (3.5, 5.0, "~4.0x"),
            (5.0, 100, "> 5.0x"),
        ]
        print(f"  Ratio buckets:")
        for lo, hi, label_b in buckets:
            n = ((ratios >= lo) & (ratios < hi)).sum()
            print(f"    {label_b:<20}: {n:>8,} ({n/len(ratios)*100:>5.1f}%)")

        # Exact ratio hits (within 5% of clean musical ratios)
        clean_ratios = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.33, 1.5, 2.0, 2.5, 3.0, 4.0]
        print(f"  Clean ratio hits (within 5%):")
        total_clean = 0
        for cr in clean_ratios:
            n = (np.abs(ratios / cr - 1) <= 0.05).sum()
            total_clean += n
            if n > 0:
                print(f"    {cr:>5.2f}x: {n:>8,} ({n/len(ratios)*100:>5.1f}%)")
        print(f"    TOTAL clean: {total_clean:>8,} ({total_clean/len(ratios)*100:>5.1f}%)")

    # Target bin distribution for reference
    tgt = np.array(all_target_bins)
    print(f"\n--- Target Bin Distribution ---")
    print(f"  Mean: {tgt.mean():.1f}  Median: {np.median(tgt):.0f}")
    print(f"  P10: {np.percentile(tgt, 10):.0f}  P90: {np.percentile(tgt, 90):.0f}")

    # Save results
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "n_samples": samples_total,
        "n_with_offset": samples_with_offset,
        "n_no_context": samples_no_context,
        "n_no_dominant": samples_no_dominant,
        "offset_mean": float(offsets.mean()),
        "offset_median": float(np.median(offsets)),
        "offset_pct_zero": float((offsets == 0).mean()),
        "dominant_gap_mean": float(dom_gaps.mean()),
        "dominant_gap_median": float(np.median(dom_gaps)),
        "dominant_coverage_mean": float(dom_cov.mean()),
    }
    if all_ratios_top1:
        r1 = np.array(all_ratios_top1)
        results["ratio_top1_mean"] = float(r1.mean())
        results["ratio_top1_median"] = float(np.median(r1))
        results["ratio_top1_p1"] = float(np.percentile(r1, 1))
        results["ratio_top1_p99"] = float(np.percentile(r1, 99))

    with open(os.path.join(output_dir, "ratio_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Graphs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Cursor offset histogram
    ax = axes[0][0]
    ax.hist(offsets[offsets < 50], bins=50, color="#4a90d9", alpha=0.8)
    ax.set_title(f"Cursor Offset (0={offsets[offsets==0].shape[0]/len(offsets)*100:.0f}%)")
    ax.set_xlabel("Bins from last event")
    ax.grid(True, alpha=0.3)

    # 2. Raw gap histogram
    ax = axes[0][1]
    ax.hist(raw[raw < 200], bins=200, color="#6bc46d", alpha=0.8)
    ax.set_title("Raw Inter-Event Gaps")
    ax.set_xlabel("Gap (bins)")
    ax.grid(True, alpha=0.3)

    # 3. Dominant gap histogram
    ax = axes[0][2]
    ax.hist(dom_gaps[dom_gaps < 150], bins=150, color="#e6a817", alpha=0.8)
    ax.set_title("Dominant Gap (Top-1)")
    ax.set_xlabel("Gap (bins)")
    ax.grid(True, alpha=0.3)

    # 4. Ratio distribution (top-1)
    if all_ratios_top1:
        r1 = np.array(all_ratios_top1)
        ax = axes[1][0]
        ax.hist(r1[(r1 > 0.1) & (r1 < 5)], bins=200, color="#eb4528", alpha=0.8)
        for cr in [0.5, 1.0, 2.0, 3.0, 4.0]:
            ax.axvline(cr, color="black", linestyle="--", alpha=0.3)
        ax.set_title("Ratio (Top-1 dominant gap)")
        ax.set_xlabel("Ratio")
        ax.grid(True, alpha=0.3)

    # 5. Dominant gap coverage
    ax = axes[1][1]
    ax.hist(dom_cov, bins=50, color="#c76dba", alpha=0.8)
    ax.set_title("Dominant Gap Coverage (% of gaps matching)")
    ax.set_xlabel("Coverage")
    ax.grid(True, alpha=0.3)

    # 6. Target bin vs dominant gap scatter
    if len(dom_gaps) > 1000:
        idx = np.random.choice(len(dom_gaps), 50000, replace=False) if len(dom_gaps) > 50000 else np.arange(len(dom_gaps))
        ax = axes[1][2]
        ax.scatter(dom_gaps[idx], np.array(all_target_bins[:len(dom_gaps)])[idx], alpha=0.02, s=1, color="#4a90d9")
        ax.set_xlabel("Dominant gap (bins)")
        ax.set_ylabel("Target bin")
        ax.set_title("Target vs Dominant Gap")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 67: Ratio Prediction Data Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ratio_analysis.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
