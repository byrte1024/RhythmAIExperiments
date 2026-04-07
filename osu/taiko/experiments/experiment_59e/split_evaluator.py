"""Experiment 59-E: Split Synthetic Evaluators.

Builds separate evaluators for expert (self) and volunteer preferences
using the metrics each group correlates with (from 59-D).

Usage:
    cd osu/taiko
    python experiments/experiment_59e/split_evaluator.py
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

# Evaluator definitions: metric → (sign, weight)
EVALUATORS = {
    "expert": {
        "gap_cv": (+1, 0.384),
        "gap_std": (+1, 0.281),
    },
    "volunteer": {
        "dominant_gap_pct": (-1, 0.420),
        "gap_entropy": (+1, 0.370),
        "max_metro_streak": (-1, 0.351),
    },
    "combined_59b": {
        "gap_std": (+1, 0.299),
        "gap_cv": (+1, 0.289),
        "dominant_gap_pct": (-1, 0.272),
        "max_metro_streak": (-1, 0.269),
    },
    "combined_top2": {
        "gap_std": (+1, 1.0),
        "gap_cv": (+1, 1.0),
    },
}


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
        dp["metrics"] = compute_chart_metrics(events_ms, mel)
        seen[csv_path] = dp["metrics"]


def build_groups(data):
    groups = defaultdict(list)
    for dp in data:
        if not dp.get("metrics"):
            continue
        key = (dp["round"], dp["song"], dp["evaluator"])
        groups[key].append(dp)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def synthetic_rank(group, evaluator_def):
    """Rank models using an evaluator definition."""
    metric_defs = evaluator_def  # {metric: (sign, weight)}

    # Z-score each metric within group
    z_scores = {}
    for mk in metric_defs:
        vals = np.array([dp["metrics"].get(mk, 0) for dp in group], dtype=np.float64)
        std = vals.std()
        mean = vals.mean()
        z_scores[mk] = (vals - mean) / std if std > 1e-10 else np.zeros_like(vals)

    # Compute synthetic score
    scores = np.zeros(len(group))
    for mk, (sign, weight) in metric_defs.items():
        scores += sign * weight * z_scores[mk]

    ranked_indices = np.argsort(-scores)
    return ranked_indices, scores


def evaluate(groups, evaluator_name, evaluator_def):
    """Evaluate an evaluator against human rankings."""
    rank1_matches = 0
    exact_matches = 0
    total_songs = 0
    total_models = 0
    kendall_taus = []
    all_human = []
    all_synth = []

    for key, group in groups.items():
        n = len(group)
        if n < 2:
            continue

        ranked_idx, synth_scores = synthetic_rank(group, evaluator_def)
        human_scores = np.array([dp["score"] for dp in group])
        human_rank = np.argsort(-human_scores)

        total_songs += 1
        if human_rank[0] == ranked_idx[0]:
            rank1_matches += 1

        h_map = np.zeros(n, dtype=int)
        s_map = np.zeros(n, dtype=int)
        for r, idx in enumerate(human_rank):
            h_map[idx] = r + 1
        for r, idx in enumerate(ranked_idx):
            s_map[idx] = r + 1
        for i in range(n):
            total_models += 1
            if h_map[i] == s_map[i]:
                exact_matches += 1

        if n >= 3:
            tau, _ = sp_stats.kendalltau(h_map, s_map)
            if not np.isnan(tau):
                kendall_taus.append(tau)

        all_human.extend(human_scores)
        all_synth.extend(synth_scores)

    if len(all_human) >= 10:
        r, p = sp_stats.spearmanr(all_synth, all_human)
    else:
        r, p = 0, 1

    return {
        "evaluator": evaluator_name,
        "total_songs": total_songs,
        "total_models": total_models,
        "rank1_match_pct": rank1_matches / max(total_songs, 1),
        "exact_rank_match_pct": exact_matches / max(total_models, 1),
        "kendall_tau_mean": float(np.mean(kendall_taus)) if kendall_taus else 0,
        "spearman_r": float(r),
        "spearman_p": float(p),
    }


def save_graphs(results_table, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 1. Main comparison grid ──
    metrics_to_show = ["rank1_match_pct", "exact_rank_match_pct", "kendall_tau_mean", "spearman_r"]
    titles = ["First Place Match", "Exact Rank Match", "Kendall Tau", "Spearman r"]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    # Group by human_type
    human_types = sorted(set(r["human_type"] for r in results_table))
    eval_names = sorted(set(r["evaluator"] for r in results_table))
    n_evals = len(eval_names)
    n_types = len(human_types)
    type_colors = {"self": "#4a90d9", "volunteer": "#6bc46d", "all": "#ff9900"}

    for mi, (metric, title) in enumerate(zip(metrics_to_show, titles)):
        ax = axes[mi]
        x = np.arange(n_evals)
        w = 0.8 / n_types

        for ti, ht in enumerate(human_types):
            vals = []
            for ev in eval_names:
                match = [r for r in results_table if r["evaluator"] == ev and r["human_type"] == ht]
                vals.append(match[0][metric] if match else 0)
            offset = (ti - n_types / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=ht, color=type_colors.get(ht, "#999"), alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(eval_names, fontsize=7, rotation=45, ha="right")
        ax.set_title(title)
        if mi == 0:
            ax.legend(fontsize=8)
        if "pct" in metric:
            ax.axhline(0.25, color="red", linestyle="--", alpha=0.3)

    fig.suptitle("Split Synthetic Evaluators: Performance by Evaluator × Human Type", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "evaluator_grid.png"), dpi=150)
    plt.close(fig)

    # ── 2. Best matchup highlight ──
    fig, ax = plt.subplots(figsize=(10, 6))
    # Find best evaluator for each human type
    for ht in human_types:
        subset = [r for r in results_table if r["human_type"] == ht]
        subset.sort(key=lambda r: r["rank1_match_pct"], reverse=True)
        best = subset[0]
        print(f"  Best for {ht}: {best['evaluator']} ({best['rank1_match_pct']:.0%} #1 match, tau={best['kendall_tau_mean']:.3f})")

    # Table
    ax.axis("off")
    header = ["Evaluator", "Human Type", "#1 Match", "Exact", "Tau", "r", "p"]
    rows = []
    for r in sorted(results_table, key=lambda x: (x["human_type"], -x["rank1_match_pct"])):
        rows.append([
            r["evaluator"], r["human_type"],
            f"{r['rank1_match_pct']:.0%}", f"{r['exact_rank_match_pct']:.0%}",
            f"{r['kendall_tau_mean']:+.3f}", f"{r['spearman_r']:+.3f}", f"{r['spearman_p']:.4f}",
        ])
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.4)
    ax.set_title("Full Results Table", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "results_table.png"), dpi=150)
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

    self_data = [dp for dp in all_data if dp["evaluator"] == "self"]
    vol_data = [dp for dp in all_data if dp["evaluator"] != "self"]
    print(f"  Self: {len(self_data)}, Volunteer: {len(vol_data)}")

    # Build groups for each human type
    self_groups = build_groups(self_data)
    vol_groups = build_groups(vol_data)
    all_groups = build_groups(all_data)

    # Evaluate every evaluator × human type combination
    results_table = []
    for eval_name, eval_def in EVALUATORS.items():
        for ht_name, groups in [("self", self_groups), ("volunteer", vol_groups), ("all", all_groups)]:
            if not groups:
                continue
            r = evaluate(groups, eval_name, eval_def)
            r["human_type"] = ht_name
            results_table.append(r)

    # Print
    print(f"\n{'='*90}")
    print("RESULTS")
    print(f"{'='*90}")
    print(f"{'Evaluator':>18s} {'HumanType':>10s} {'#1':>6s} {'Exact':>6s} {'Tau':>7s} {'r':>7s} {'p':>8s}")
    for r in sorted(results_table, key=lambda x: (x["human_type"], x["evaluator"])):
        print(f"{r['evaluator']:>18s} {r['human_type']:>10s} "
              f"{r['rank1_match_pct']:>5.0%} {r['exact_rank_match_pct']:>5.0%} "
              f"{r['kendall_tau_mean']:>+6.3f} {r['spearman_r']:>+6.3f} {r['spearman_p']:>8.4f}")

    # Save
    with open(os.path.join(output_dir, "split_evaluator_results.json"), "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    save_graphs(results_table, output_dir)


if __name__ == "__main__":
    main()
