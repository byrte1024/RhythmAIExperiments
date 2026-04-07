"""Experiment 59-C: Synthetic Human Evaluator.

Builds a synthetic evaluator from the 4 significant metrics found in 59-B,
then compares its rankings to actual human rankings from 42-AR and 53-AR.

Usage:
    cd osu/taiko
    python experiments/experiment_59c/synthetic_evaluator.py
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

# The 4 significant metrics from 59-B and their signs (+1 = higher is better)
SIGNIFICANT_METRICS = {
    "gap_std": +1,        # r=+0.299
    "gap_cv": +1,         # r=+0.289
    "dominant_gap_pct": -1,  # r=-0.272
    "max_metro_streak": -1,  # r=-0.269
}

WEIGHTING_SCHEMES = {
    "equal": {"gap_std": 1.0, "gap_cv": 1.0, "dominant_gap_pct": 1.0, "max_metro_streak": 1.0},
    "corr_weighted": {"gap_std": 0.299, "gap_cv": 0.289, "dominant_gap_pct": 0.272, "max_metro_streak": 0.269},
    "top2_only": {"gap_std": 1.0, "gap_cv": 1.0, "dominant_gap_pct": 0.0, "max_metro_streak": 0.0},
}


def compute_metrics_for_all(all_data):
    """Compute chart metrics, caching per CSV."""
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


def build_song_groups(all_data):
    """Group data by (round, song, evaluator). Each group = one ranking event."""
    groups = defaultdict(list)
    for dp in all_data:
        if not dp.get("metrics"):
            continue
        key = (dp["round"], dp["song"], dp["evaluator"])
        groups[key].append(dp)
    # Only keep groups with >= 2 models
    return {k: v for k, v in groups.items() if len(v) >= 2}


def synthetic_rank(group, scheme_name):
    """Rank models in a group using synthetic scoring. Returns list sorted best→worst."""
    weights = WEIGHTING_SCHEMES[scheme_name]

    # z-score metrics within group
    metric_vals = {mk: [] for mk in SIGNIFICANT_METRICS}
    for dp in group:
        for mk in SIGNIFICANT_METRICS:
            metric_vals[mk].append(dp["metrics"].get(mk, 0))

    z_scores = {mk: np.array(metric_vals[mk]) for mk in SIGNIFICANT_METRICS}
    for mk in z_scores:
        vals = z_scores[mk]
        std = vals.std()
        mean = vals.mean()
        z_scores[mk] = (vals - mean) / std if std > 1e-10 else np.zeros_like(vals)

    # Compute synthetic score per model
    scores = np.zeros(len(group))
    for mk, sign in SIGNIFICANT_METRICS.items():
        w = weights.get(mk, 0)
        scores += sign * w * z_scores[mk]

    # Rank (highest score = rank 1)
    ranked_indices = np.argsort(-scores)
    return ranked_indices, scores


def evaluate_scheme(groups, scheme_name):
    """Evaluate a weighting scheme against all human rankings."""
    exact_matches = 0  # synthetic rank == human rank for this model
    total_models = 0
    rank1_matches = 0  # synthetic picks same #1 as human
    total_songs = 0
    kendall_taus = []
    score_corrs = []  # correlation between synthetic score and human score

    all_human_scores = []
    all_synth_scores = []

    for key, group in groups.items():
        n = len(group)
        if n < 2:
            continue

        ranked_idx, synth_scores = synthetic_rank(group, scheme_name)

        # Human ranking (from score: 4=best → 1=worst)
        human_scores = np.array([dp["score"] for dp in group])
        human_rank = np.argsort(-human_scores)  # indices sorted best→worst

        # Synthetic ranking
        synth_rank = ranked_idx

        total_songs += 1

        # Rank 1 match
        if human_rank[0] == synth_rank[0]:
            rank1_matches += 1

        # Per-model exact rank match
        human_rank_map = np.zeros(n, dtype=int)
        synth_rank_map = np.zeros(n, dtype=int)
        for r, idx in enumerate(human_rank):
            human_rank_map[idx] = r + 1
        for r, idx in enumerate(synth_rank):
            synth_rank_map[idx] = r + 1
        for i in range(n):
            total_models += 1
            if human_rank_map[i] == synth_rank_map[i]:
                exact_matches += 1

        # Kendall tau
        if n >= 3:
            tau, _ = sp_stats.kendalltau(human_rank_map, synth_rank_map)
            if not np.isnan(tau):
                kendall_taus.append(tau)

        # Score correlation
        all_human_scores.extend(human_scores)
        all_synth_scores.extend(synth_scores)

    # Overall score correlation
    if len(all_human_scores) >= 10:
        overall_r, overall_p = sp_stats.spearmanr(all_synth_scores, all_human_scores)
    else:
        overall_r, overall_p = 0, 1

    return {
        "scheme": scheme_name,
        "total_songs": total_songs,
        "total_models": total_models,
        "rank1_match": rank1_matches,
        "rank1_match_pct": rank1_matches / max(total_songs, 1),
        "exact_rank_match": exact_matches,
        "exact_rank_match_pct": exact_matches / max(total_models, 1),
        "kendall_tau_mean": float(np.mean(kendall_taus)) if kendall_taus else 0,
        "kendall_tau_std": float(np.std(kendall_taus)) if kendall_taus else 0,
        "overall_spearman_r": float(overall_r),
        "overall_spearman_p": float(overall_p),
    }


def evaluate_by_evaluator_type(all_data, groups):
    """Evaluate separately for self vs volunteer rankings."""
    self_groups = {}
    vol_groups = {}
    for key, group in groups.items():
        rnd, song, evaluator = key
        if evaluator == "self":
            self_groups[key] = group
        else:
            vol_groups[key] = group

    results = {}
    for label, grps in [("self", self_groups), ("volunteer", vol_groups), ("all", groups)]:
        if not grps:
            continue
        for scheme in WEIGHTING_SCHEMES:
            r = evaluate_scheme(grps, scheme)
            r["evaluator_type"] = label
            results[f"{label}_{scheme}"] = r
    return results


def save_graphs(groups, by_eval_results, output_dir):
    """Generate analysis graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 1. Scheme comparison bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, metric_name in enumerate(["rank1_match_pct", "exact_rank_match_pct", "kendall_tau_mean"]):
        ax = axes[i]
        labels = []
        values = []
        colors = []
        for key, r in sorted(by_eval_results.items()):
            labels.append(f"{r['evaluator_type']}\n{r['scheme']}")
            values.append(r[metric_name])
            c = "#4a90d9" if r["evaluator_type"] == "self" else "#6bc46d" if r["evaluator_type"] == "volunteer" else "#ff9900"
            colors.append(c)
        ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        titles = {"rank1_match_pct": "First Place Match Rate",
                  "exact_rank_match_pct": "Exact Rank Match Rate",
                  "kendall_tau_mean": "Mean Kendall Tau"}
        ax.set_title(titles.get(metric_name, metric_name))
        if "pct" in metric_name:
            ax.set_ylim(0, 1)
            # random baseline
            n_models = 4  # most common
            random_pct = 1.0 / n_models
            ax.axhline(random_pct, color="red", linestyle="--", alpha=0.5, label=f"Random ({random_pct:.0%})")
            ax.legend(fontsize=7)

    fig.suptitle("Synthetic Evaluator Performance", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scheme_comparison.png"), dpi=150)
    plt.close(fig)

    # ── 2. Per-song synthetic vs human score scatter (best scheme) ──
    best_scheme = "corr_weighted"
    fig, ax = plt.subplots(figsize=(8, 8))
    model_colors = {
        "exp14": "#4a90d9", "exp35c": "#e6a817", "exp42": "#eb4528",
        "exp44": "#6bc46d", "exp45": "#c76dba", "exp53": "#00cccc",
    }
    for key, group in groups.items():
        if len(group) < 2:
            continue
        ranked_idx, synth_scores = synthetic_rank(group, best_scheme)
        for i, dp in enumerate(group):
            c = model_colors.get(dp["model"], "#999999")
            ax.scatter(synth_scores[i] + np.random.uniform(-0.05, 0.05),
                      dp["score"] + np.random.uniform(-0.15, 0.15),
                      c=c, s=50, alpha=0.6, edgecolors="black", linewidths=0.3)

    # Legend
    for model, color in model_colors.items():
        ax.scatter([], [], c=color, s=50, label=model, edgecolors="black")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Synthetic Score (corr_weighted)")
    ax.set_ylabel("Human Score (4=best)")
    ax.set_yticks([1, 2, 3, 4])

    r_val = by_eval_results.get("all_corr_weighted", {}).get("overall_spearman_r", 0)
    p_val = by_eval_results.get("all_corr_weighted", {}).get("overall_spearman_p", 1)
    ax.set_title(f"Synthetic vs Human Score (r={r_val:.3f}, p={p_val:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "synth_vs_human_scatter.png"), dpi=150)
    plt.close(fig)

    # ── 3. Per-song rank comparison (corr_weighted) ──
    song_results = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        rnd, song, evaluator = key
        ranked_idx, synth_scores = synthetic_rank(group, best_scheme)
        human_scores = np.array([dp["score"] for dp in group])
        human_rank = np.argsort(-human_scores)

        human_models = [group[i]["model"] for i in human_rank]
        synth_models = [group[i]["model"] for i in ranked_idx]
        match = human_rank[0] == ranked_idx[0]
        song_results.append({
            "song": song[:25], "evaluator": evaluator[:10], "round": rnd,
            "human_1st": human_models[0], "synth_1st": synth_models[0],
            "match": match,
        })

    if song_results:
        fig, ax = plt.subplots(figsize=(14, max(6, len(song_results) * 0.3)))
        y = range(len(song_results))
        colors = ["#6bc46d" if r["match"] else "#eb4528" for r in song_results]
        labels = [f"{r['song']} ({r['evaluator']})" for r in song_results]
        ax.barh(y, [1 if r["match"] else 0 for r in song_results], color=colors, alpha=0.6)
        for i, r in enumerate(song_results):
            text = f"H:{r['human_1st']}  S:{r['synth_1st']}"
            ax.text(0.02, i, text, va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("First Place Match (green=match, red=miss)")
        matches = sum(1 for r in song_results if r["match"])
        ax.set_title(f"Per-Song First Place: {matches}/{len(song_results)} matches ({matches/len(song_results):.0%})")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "per_song_rank1.png"), dpi=150)
        plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    data_42 = load_42ar_data()
    data_53 = load_53ar_data()
    all_data = data_42 + data_53
    print(f"  {len(all_data)} entries")

    print("Computing metrics...")
    compute_metrics_for_all(all_data)
    valid = sum(1 for dp in all_data if dp.get("metrics"))
    print(f"  {valid} with metrics")

    print("Building song groups...")
    groups = build_song_groups(all_data)
    print(f"  {len(groups)} ranking events")

    print("\nEvaluating schemes...")
    by_eval = evaluate_by_evaluator_type(all_data, groups)

    # Print results
    print(f"\n{'='*80}")
    print("SYNTHETIC EVALUATOR RESULTS")
    print(f"{'='*80}")
    print(f"{'Scheme':>25s} {'Type':>10s} {'#1 Match':>10s} {'Exact':>10s} {'Tau':>8s} {'r':>8s} {'p':>8s}")
    for key, r in sorted(by_eval.items()):
        print(f"{r['scheme']:>25s} {r['evaluator_type']:>10s} "
              f"{r['rank1_match_pct']:>9.0%} {r['exact_rank_match_pct']:>9.0%} "
              f"{r['kendall_tau_mean']:>+7.3f} {r['overall_spearman_r']:>+7.3f} {r['overall_spearman_p']:>8.4f}")

    # Random baselines
    print(f"\n  Random baseline (3 models): 1st match=33%, exact rank=33%, tau=0.000")
    print(f"  Random baseline (4 models): 1st match=25%, exact rank=25%, tau=0.000")

    # Save
    with open(os.path.join(output_dir, "evaluator_results.json"), "w", encoding="utf-8") as f:
        json.dump(by_eval, f, indent=2)

    save_graphs(groups, by_eval, output_dir)


if __name__ == "__main__":
    main()
