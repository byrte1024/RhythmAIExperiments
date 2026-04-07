"""Experiment 59-B: Within-Song Normalized Metric Correlation.

Same metrics as exp 59, but normalized within each song to remove the
per-song confound. Also computes pairwise deltas between models on the
same song for more data points.

Usage:
    cd osu/taiko
    python experiments/experiment_59b/analyze_normalized.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Reuse data loading and metric computation from exp 59
sys.path.insert(0, os.path.join(TAIKO_DIR, "experiments", "experiment_59"))
from analyze_metrics import (
    load_42ar_data, load_53ar_data, load_csv_events_ms, load_mel,
    compute_chart_metrics,
)


def compute_metrics_for_all(all_data):
    """Compute metrics for all data points, caching per CSV."""
    seen = {}
    valid = 0
    for dp in all_data:
        csv_path = dp["csv_path"]
        if csv_path in seen:
            dp["metrics"] = seen[csv_path]
            dp["events_ms"] = seen.get(csv_path + "_ev")
            if dp["metrics"]:
                valid += 1
            continue
        if not os.path.exists(csv_path):
            seen[csv_path] = None
            continue
        events_ms = load_csv_events_ms(csv_path)
        dp["events_ms"] = events_ms
        mel = None
        if os.path.exists(dp["mel_path"]):
            try:
                mel = load_mel(dp["mel_path"])
            except Exception:
                pass
        metrics = compute_chart_metrics(events_ms, mel)
        dp["metrics"] = metrics
        seen[csv_path] = metrics
        seen[csv_path + "_ev"] = events_ms
        if metrics:
            valid += 1
    return valid


def zscore_within_song(all_data):
    """For each (round, song, evaluator) group, z-score each metric across models.

    Returns list of data points with "z_metrics" dict added.
    """
    # Group by (round, song, evaluator) — each group has one score per model
    groups = defaultdict(list)
    for dp in all_data:
        if not dp.get("metrics"):
            continue
        key = (dp["round"], dp["song"], dp["evaluator"])
        groups[key].append(dp)

    for key, group in groups.items():
        if len(group) < 2:
            continue
        # get all metric keys
        metric_keys = sorted(group[0]["metrics"].keys())
        for mk in metric_keys:
            vals = [dp["metrics"].get(mk, 0) for dp in group]
            vals = np.array(vals, dtype=np.float64)
            std = vals.std()
            mean = vals.mean()
            for i, dp in enumerate(group):
                if "z_metrics" not in dp:
                    dp["z_metrics"] = {}
                dp["z_metrics"][mk] = float((vals[i] - mean) / std) if std > 1e-10 else 0.0


def compute_pairwise_deltas(all_data):
    """For each pair of models on the same song (same evaluator), compute metric and score deltas.

    Returns list of dicts with delta_score and delta_metrics.
    """
    groups = defaultdict(list)
    for dp in all_data:
        if not dp.get("metrics"):
            continue
        key = (dp["round"], dp["song"], dp["evaluator"])
        groups[key].append(dp)

    deltas = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                d_score = a["score"] - b["score"]
                d_metrics = {}
                for mk in a["metrics"]:
                    va = a["metrics"].get(mk, 0)
                    vb = b["metrics"].get(mk, 0)
                    if va is not None and vb is not None:
                        d_metrics[mk] = va - vb
                deltas.append({
                    "delta_score": d_score,
                    "delta_metrics": d_metrics,
                    "model_a": a["model"],
                    "model_b": b["model"],
                    "song": a["song"],
                    "round": a["round"],
                })
    return deltas


def correlate_zscored(all_data):
    """Correlate z-scored metrics with scores."""
    metric_keys = None
    for dp in all_data:
        if dp.get("z_metrics"):
            metric_keys = sorted(dp["z_metrics"].keys())
            break
    if not metric_keys:
        return {}

    results = {}
    for mk in metric_keys:
        scores = []
        zvals = []
        for dp in all_data:
            if dp.get("z_metrics") and mk in dp["z_metrics"]:
                z = dp["z_metrics"][mk]
                if not np.isnan(z):
                    scores.append(dp["score"])
                    zvals.append(z)
        if len(scores) >= 10:
            r, p = sp_stats.spearmanr(zvals, scores)
            results[mk] = {"r": float(r), "p": float(p), "n": len(scores)}
    return results


def correlate_deltas(deltas):
    """Correlate pairwise metric deltas with score deltas."""
    if not deltas or not deltas[0].get("delta_metrics"):
        return {}

    metric_keys = sorted(deltas[0]["delta_metrics"].keys())
    results = {}
    for mk in metric_keys:
        d_scores = []
        d_vals = []
        for d in deltas:
            v = d["delta_metrics"].get(mk)
            if v is not None and not np.isnan(v):
                d_scores.append(d["delta_score"])
                d_vals.append(v)
        if len(d_scores) >= 10:
            r, p = sp_stats.spearmanr(d_vals, d_scores)
            results[mk] = {"r": float(r), "p": float(p), "n": len(d_scores)}
    return results


def save_graphs(all_data, z_corrs, delta_corrs, deltas, output_dir):
    """Generate analysis graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {
        "exp14": "#4a90d9", "exp35c": "#e6a817", "exp42": "#eb4528",
        "exp44": "#6bc46d", "exp45": "#c76dba", "exp53": "#00cccc",
    }

    # ── 1. Z-score correlations ──
    sorted_z = sorted(z_corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    names = [k for k, v in sorted_z]
    rs = [v["r"] for k, v in sorted_z]
    ps = [v["p"] for k, v in sorted_z]
    colors = ["#6bc46d" if p < 0.05 else "#ff9900" if p < 0.10 else "#cccccc" for p in ps]
    ax.barh(range(len(names)), rs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Spearman r (z-scored metric vs human score)")
    ax.set_title("Within-Song Normalized Correlations\n(green=p<0.05, orange=p<0.10, gray=n.s.)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "zscore_correlations.png"), dpi=150)
    plt.close(fig)

    # ── 2. Pairwise delta correlations ──
    sorted_d = sorted(delta_corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    names = [k for k, v in sorted_d]
    rs = [v["r"] for k, v in sorted_d]
    ps = [v["p"] for k, v in sorted_d]
    colors = ["#6bc46d" if p < 0.05 else "#ff9900" if p < 0.10 else "#cccccc" for p in ps]
    ax.barh(range(len(names)), rs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Spearman r (metric delta vs score delta)")
    ax.set_title("Pairwise Delta Correlations\n(green=p<0.05, orange=p<0.10, gray=n.s.)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "delta_correlations.png"), dpi=150)
    plt.close(fig)

    # ── 3. Top 6 z-score scatter plots ──
    top6 = sorted_z[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (key, corr) in enumerate(top6):
        ax = axes[i]
        for dp in all_data:
            if dp.get("z_metrics") and key in dp["z_metrics"]:
                z = dp["z_metrics"][key]
                if not np.isnan(z):
                    c = model_colors.get(dp["model"], "#999999")
                    ax.scatter(z, dp["score"] + np.random.uniform(-0.15, 0.15),
                              c=c, s=40, alpha=0.6, edgecolors="black", linewidths=0.3)
        ax.set_xlabel(f"z({key})")
        ax.set_ylabel("Human Score")
        ax.set_title(f"{key}\nr={corr['r']:.3f}, p={corr['p']:.3f}")
        ax.set_yticks([1, 2, 3, 4])
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    fig.suptitle("Top 6 Z-Scored Metrics vs Human Score", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "top6_zscore_scatter.png"), dpi=150)
    plt.close(fig)

    # ── 4. Top 6 pairwise delta scatter plots ──
    top6d = sorted_d[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, (key, corr) in enumerate(top6d):
        ax = axes[i]
        for d in deltas:
            v = d["delta_metrics"].get(key)
            if v is not None and not np.isnan(v):
                ax.scatter(v, d["delta_score"] + np.random.uniform(-0.15, 0.15),
                          c="#4a90d9", s=25, alpha=0.4, edgecolors="black", linewidths=0.2)
        ax.set_xlabel(f"delta({key})")
        ax.set_ylabel("Score Delta (A - B)")
        ax.set_title(f"{key}\nr={corr['r']:.3f}, p={corr['p']:.3f}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    fig.suptitle("Top 6 Pairwise Delta Metrics vs Score Delta", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "top6_delta_scatter.png"), dpi=150)
    plt.close(fig)

    # ── 5. Side-by-side comparison: raw vs z-score vs delta ──
    # Load raw correlations from exp 59
    raw_corrs = {}
    raw_path = os.path.join(TAIKO_DIR, "experiments", "experiment_59", "results", "correlations.json")
    if os.path.exists(raw_path):
        with open(raw_path, "r") as f:
            raw_corrs = json.load(f)

    # Get all metric keys present in all three
    all_keys = sorted(set(raw_corrs.keys()) & set(z_corrs.keys()) & set(delta_corrs.keys()))
    if all_keys:
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(all_keys))
        w = 0.25
        raw_rs = [raw_corrs[k]["r"] for k in all_keys]
        z_rs = [z_corrs[k]["r"] for k in all_keys]
        d_rs = [delta_corrs[k]["r"] for k in all_keys]
        ax.barh(x - w, raw_rs, w, label="Raw (59)", color="#cccccc", alpha=0.7)
        ax.barh(x, z_rs, w, label="Z-scored (59-B)", color="#4a90d9", alpha=0.7)
        ax.barh(x + w, d_rs, w, label="Pairwise delta (59-B)", color="#6bc46d", alpha=0.7)
        ax.set_yticks(x)
        ax.set_yticklabels(all_keys, fontsize=7)
        ax.set_xlabel("Spearman r")
        ax.set_title("Correlation Comparison: Raw vs Normalized vs Pairwise")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=9)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "method_comparison.png"), dpi=150)
        plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    data_42 = load_42ar_data()
    data_53 = load_53ar_data()
    all_data = data_42 + data_53
    print(f"  {len(all_data)} vote-model-song entries")

    # Compute metrics
    print("Computing metrics...")
    valid = compute_metrics_for_all(all_data)
    print(f"  {valid} with valid metrics")

    # Z-score within song
    print("Z-scoring within songs...")
    zscore_within_song(all_data)
    n_z = sum(1 for dp in all_data if dp.get("z_metrics"))
    print(f"  {n_z} with z-scored metrics")

    # Pairwise deltas
    print("Computing pairwise deltas...")
    deltas = compute_pairwise_deltas(all_data)
    print(f"  {len(deltas)} pairwise comparisons")

    # Correlations
    print("\nCorrelating z-scored metrics...")
    z_corrs = correlate_zscored(all_data)

    print("Correlating pairwise deltas...")
    delta_corrs = correlate_deltas(deltas)

    # Print results
    print(f"\n{'='*70}")
    print("Z-SCORED CORRELATIONS (within-song normalized)")
    print(f"{'='*70}")
    print(f"{'Metric':>25s} {'r':>8s} {'p':>8s} {'n':>5s} {'sig':>5s}")
    for key, corr in sorted(z_corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True):
        sig = "***" if corr["p"] < 0.001 else "**" if corr["p"] < 0.01 else "*" if corr["p"] < 0.05 else "." if corr["p"] < 0.10 else ""
        print(f"{key:>25s} {corr['r']:>+8.3f} {corr['p']:>8.4f} {corr['n']:>5d} {sig:>5s}")

    print(f"\n{'='*70}")
    print("PAIRWISE DELTA CORRELATIONS")
    print(f"{'='*70}")
    print(f"{'Metric':>25s} {'r':>8s} {'p':>8s} {'n':>5s} {'sig':>5s}")
    for key, corr in sorted(delta_corrs.items(), key=lambda x: abs(x[1]["r"]), reverse=True):
        sig = "***" if corr["p"] < 0.001 else "**" if corr["p"] < 0.01 else "*" if corr["p"] < 0.05 else "." if corr["p"] < 0.10 else ""
        print(f"{key:>25s} {corr['r']:>+8.3f} {corr['p']:>8.4f} {corr['n']:>5d} {sig:>5s}")

    # Save
    results = {
        "z_correlations": z_corrs,
        "delta_correlations": delta_corrs,
        "n_data_points": len(all_data),
        "n_pairwise": len(deltas),
    }
    with open(os.path.join(output_dir, "correlations.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    save_graphs(all_data, z_corrs, delta_corrs, deltas, output_dir)


if __name__ == "__main__":
    main()
