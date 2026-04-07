"""Experiment 59-D: Self vs Volunteer Metric Correlations.

Reruns exp 59 + 59-B analysis separately for self-only and volunteer-only
votes to find whether different evaluator types correlate with different metrics.

Usage:
    cd osu/taiko
    python experiments/experiment_59d/analyze_split.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, os.path.join(TAIKO_DIR, "experiments", "experiment_59"))
from analyze_metrics import (
    load_42ar_data, load_53ar_data, load_csv_events_ms, load_mel,
    compute_chart_metrics,
)


def compute_metrics_for_all(all_data):
    seen = {}
    for dp in all_data:
        csv_path = dp["csv_path"]
        if csv_path in seen:
            dp["metrics"] = seen[csv_path]
            continue
        if not os.path.exists(csv_path):
            seen[csv_path] = None
            continue
        events_ms = load_csv_events_ms(csv_path)
        mel = None
        if os.path.exists(dp["mel_path"]):
            try:
                mel = load_mel(dp["mel_path"])
            except Exception:
                pass
        metrics = compute_chart_metrics(events_ms, mel)
        dp["metrics"] = metrics
        seen[csv_path] = metrics


def zscore_and_correlate(data_points):
    """Z-score within (round, song, evaluator) groups, then correlate with score."""
    groups = defaultdict(list)
    for dp in data_points:
        if not dp.get("metrics"):
            continue
        key = (dp["round"], dp["song"], dp["evaluator"])
        groups[key].append(dp)

    # Z-score within each group
    for key, group in groups.items():
        if len(group) < 2:
            continue
        metric_keys = sorted(group[0]["metrics"].keys())
        for mk in metric_keys:
            vals = np.array([dp["metrics"].get(mk, 0) for dp in group], dtype=np.float64)
            std = vals.std()
            mean = vals.mean()
            for i, dp in enumerate(group):
                if "z_metrics" not in dp:
                    dp["z_metrics"] = {}
                dp["z_metrics"][mk] = float((vals[i] - mean) / std) if std > 1e-10 else 0.0

    # Correlate
    metric_keys = None
    for dp in data_points:
        if dp.get("z_metrics"):
            metric_keys = sorted(dp["z_metrics"].keys())
            break
    if not metric_keys:
        return {}

    results = {}
    for mk in metric_keys:
        scores = []
        zvals = []
        for dp in data_points:
            if dp.get("z_metrics") and mk in dp["z_metrics"]:
                z = dp["z_metrics"][mk]
                if not np.isnan(z):
                    scores.append(dp["score"])
                    zvals.append(z)
        if len(scores) >= 8:
            r, p = sp_stats.spearmanr(zvals, scores)
            results[mk] = {"r": float(r), "p": float(p), "n": len(scores)}
    return results


def raw_correlate(data_points):
    """Raw (non-normalized) correlations for comparison."""
    metric_keys = None
    for dp in data_points:
        if dp.get("metrics"):
            metric_keys = sorted(dp["metrics"].keys())
            break
    if not metric_keys:
        return {}

    results = {}
    for mk in metric_keys:
        scores = []
        vals = []
        for dp in data_points:
            if dp.get("metrics") and mk in dp["metrics"]:
                v = dp["metrics"][mk]
                if v is not None and not np.isnan(v):
                    scores.append(dp["score"])
                    vals.append(v)
        if len(scores) >= 8:
            r, p = sp_stats.spearmanr(vals, scores)
            results[mk] = {"r": float(r), "p": float(p), "n": len(scores)}
    return results


def save_graphs(self_z, vol_z, all_z, self_raw, vol_raw, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Get all metric keys
    all_keys = sorted(set(self_z.keys()) | set(vol_z.keys()) | set(all_z.keys()))

    # ── 1. Side-by-side: self vs volunteer z-scored correlations ──
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for i, (label, corrs) in enumerate([("Self (expert)", self_z), ("Volunteers", vol_z)]):
        ax = axes[i]
        sorted_c = sorted(corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
        names = [k for k, v in sorted_c]
        rs = [v["r"] for k, v in sorted_c]
        ps = [v["p"] for k, v in sorted_c]
        colors = ["#6bc46d" if p < 0.05 else "#ff9900" if p < 0.10 else "#cccccc" for p in ps]
        ax.barh(range(len(names)), rs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Spearman r")
        ax.set_title(f"{label} (n={list(corrs.values())[0]['n'] if corrs else 0})\ngreen=p<0.05, orange=p<0.10")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        ax.set_xlim(-0.5, 0.5)

    fig.suptitle("Z-Scored Correlations: Self vs Volunteers", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "self_vs_volunteer_zscore.png"), dpi=150)
    plt.close(fig)

    # ── 2. Direct comparison: same metrics, self r vs volunteer r ──
    fig, ax = plt.subplots(figsize=(10, 10))
    for mk in all_keys:
        sr = self_z.get(mk, {}).get("r", 0)
        vr = vol_z.get(mk, {}).get("r", 0)
        sp = self_z.get(mk, {}).get("p", 1)
        vp = vol_z.get(mk, {}).get("p", 1)

        if sp < 0.05 or vp < 0.05:
            color = "#6bc46d"
            size = 120
        elif sp < 0.10 or vp < 0.10:
            color = "#ff9900"
            size = 80
        else:
            color = "#cccccc"
            size = 40
        ax.scatter(sr, vr, c=color, s=size, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(mk, (sr, vr), fontsize=6, alpha=0.7,
                   xytext=(3, 3), textcoords="offset points")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.plot([-0.5, 0.5], [-0.5, 0.5], "k--", alpha=0.2, label="Agreement line")
    ax.set_xlabel("Self (expert) Spearman r")
    ax.set_ylabel("Volunteer Spearman r")
    ax.set_title("Self vs Volunteer: Per-Metric Correlation Comparison\n(green=significant, orange=marginal, gray=n.s.)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "self_vs_volunteer_scatter.png"), dpi=150)
    plt.close(fig)

    # ── 3. Top metrics table as image ──
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    header = ["Metric", "Self r", "Self p", "Vol r", "Vol p", "All r", "All p"]
    rows = []
    for mk in sorted(all_keys, key=lambda k: abs(all_z.get(k, {}).get("r", 0)), reverse=True)[:12]:
        sr = self_z.get(mk, {}).get("r", 0)
        sp = self_z.get(mk, {}).get("p", 1)
        vr = vol_z.get(mk, {}).get("r", 0)
        vp = vol_z.get(mk, {}).get("p", 1)
        ar = all_z.get(mk, {}).get("r", 0)
        ap = all_z.get(mk, {}).get("p", 1)
        rows.append([mk, f"{sr:+.3f}", f"{sp:.3f}", f"{vr:+.3f}", f"{vp:.3f}", f"{ar:+.3f}", f"{ap:.3f}"])
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    # Color significant cells
    for i, row in enumerate(rows):
        for j, col_idx in [(1, 2), (3, 4), (5, 6)]:
            p_val = float(row[col_idx])
            if p_val < 0.05:
                table[i + 1, j].set_facecolor("#c8f7c8")
                table[i + 1, col_idx].set_facecolor("#c8f7c8")
            elif p_val < 0.10:
                table[i + 1, j].set_facecolor("#ffeebb")
                table[i + 1, col_idx].set_facecolor("#ffeebb")
    ax.set_title("Top 12 Metrics: Self vs Volunteer vs All (z-scored)", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_table.png"), dpi=150)
    plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    data_42 = load_42ar_data()
    data_53 = load_53ar_data()
    all_data = data_42 + data_53

    print("Computing metrics...")
    compute_metrics_for_all(all_data)
    valid = sum(1 for dp in all_data if dp.get("metrics"))
    print(f"  {valid} with metrics")

    # Split by evaluator type
    self_data = [dp for dp in all_data if dp["evaluator"] == "self"]
    vol_data = [dp for dp in all_data if dp["evaluator"] != "self"]
    print(f"  Self: {len(self_data)}, Volunteer: {len(vol_data)}")

    # Z-scored correlations for each
    print("\nZ-scored correlations...")
    self_z = zscore_and_correlate(self_data)
    vol_z = zscore_and_correlate(vol_data)
    all_z = zscore_and_correlate(all_data)

    # Raw correlations for comparison
    self_raw = raw_correlate(self_data)
    vol_raw = raw_correlate(vol_data)

    # Print results
    for label, corrs in [("SELF (expert)", self_z), ("VOLUNTEERS", vol_z), ("ALL", all_z)]:
        n = list(corrs.values())[0]["n"] if corrs else 0
        print(f"\n{'='*70}")
        print(f"{label} (n={n})")
        print(f"{'='*70}")
        print(f"{'Metric':>25s} {'r':>8s} {'p':>8s} {'sig':>5s}")
        for key, corr in sorted(corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True):
            sig = "***" if corr["p"] < 0.001 else "**" if corr["p"] < 0.01 else "*" if corr["p"] < 0.05 else "." if corr["p"] < 0.10 else ""
            print(f"{key:>25s} {corr['r']:>+8.3f} {corr['p']:>8.4f} {sig:>5s}")

    # Save
    results = {"self_z": self_z, "volunteer_z": vol_z, "all_z": all_z,
               "self_raw": self_raw, "volunteer_raw": vol_raw}
    with open(os.path.join(output_dir, "split_correlations.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_graphs(self_z, vol_z, all_z, self_raw, vol_raw, output_dir)


if __name__ == "__main__":
    main()
