"""Experiment 59-F: Evaluator Weight Sweep.

Sweeps through top-N metrics (N=1..8) with varying temperature on weights,
testing every combination against human rankings.

Temperature controls weight sharpness:
  temp=0: equal weights (all metrics weighted 1.0)
  temp=1: weights proportional to |r| from 59-B
  temp=2: weights proportional to |r|^2 (sharper)
  temp=inf: only the single strongest metric

Usage:
    cd osu/taiko
    python experiments/experiment_59f/sweep_evaluator.py
"""

import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

sys.path.insert(0, os.path.join(TAIKO_DIR, "experiments", "experiment_59"))
from analyze_metrics import (
    load_42ar_data, load_53ar_data, load_csv_events_ms, load_mel,
    compute_chart_metrics,
)

# All metrics ranked by |r| from 59-B z-scored analysis (all evaluators)
METRIC_RANKING = [
    ("gap_std",             +1, 0.299),
    ("gap_cv",              +1, 0.289),
    ("dominant_gap_pct",    -1, 0.272),
    ("max_metro_streak",    -1, 0.269),
    ("max_metro_streak_pct",-1, 0.266),
    ("density_std",         -1, 0.198),
    ("gap_entropy",         +1, 0.194),
    ("density",             -1, 0.182),
]

TEMPERATURES = [0, 0.5, 1.0, 1.5, 2.0, 3.0]


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


def evaluate_formula(groups, metric_defs):
    """Evaluate a formula: metric_defs = [(metric_name, sign, weight), ...]"""
    rank1_matches = 0
    total_songs = 0
    exact_matches = 0
    total_models = 0
    kendall_taus = []
    all_human = []
    all_synth = []

    for key, group in groups.items():
        n = len(group)
        if n < 2:
            continue

        # z-score within group
        z_scores = {}
        for mk, sign, w in metric_defs:
            vals = np.array([dp["metrics"].get(mk, 0) for dp in group], dtype=np.float64)
            std = vals.std()
            mean = vals.mean()
            z_scores[mk] = (vals - mean) / std if std > 1e-10 else np.zeros_like(vals)

        # synthetic score
        synth = np.zeros(n)
        for mk, sign, w in metric_defs:
            synth += sign * w * z_scores[mk]

        human_scores = np.array([dp["score"] for dp in group])
        human_rank = np.argsort(-human_scores)
        synth_rank = np.argsort(-synth)

        total_songs += 1
        if human_rank[0] == synth_rank[0]:
            rank1_matches += 1

        h_map = np.zeros(n, dtype=int)
        s_map = np.zeros(n, dtype=int)
        for r, idx in enumerate(human_rank):
            h_map[idx] = r + 1
        for r, idx in enumerate(synth_rank):
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
        all_synth.extend(synth)

    if len(all_human) >= 10:
        r, p = sp_stats.spearmanr(all_synth, all_human)
    else:
        r, p = 0, 1

    return {
        "rank1_pct": rank1_matches / max(total_songs, 1),
        "exact_pct": exact_matches / max(total_models, 1),
        "tau": float(np.mean(kendall_taus)) if kendall_taus else 0,
        "r": float(r),
        "p": float(p),
        "n_songs": total_songs,
    }


def make_formula(top_n, temp):
    """Create a formula from top-N metrics with given temperature."""
    selected = METRIC_RANKING[:top_n]
    if temp == 0:
        return [(mk, sign, 1.0) for mk, sign, r_val in selected]
    else:
        weights = [abs(r_val) ** temp for mk, sign, r_val in selected]
        # normalize so max weight = 1
        max_w = max(weights)
        weights = [w / max_w for w in weights]
        return [(mk, sign, w) for (mk, sign, r_val), w in zip(selected, weights)]


def save_graphs(sweep_results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 1. Heatmap: top-N vs temperature for each audience, metric = rank1_pct ──
    audiences = sorted(set(r["audience"] for r in sweep_results))
    top_ns = sorted(set(r["top_n"] for r in sweep_results))
    temps = sorted(set(r["temp"] for r in sweep_results))

    for metric_name in ["rank1_pct", "tau", "r"]:
        fig, axes = plt.subplots(1, len(audiences), figsize=(6 * len(audiences), 5))
        if len(audiences) == 1:
            axes = [axes]

        for ai, aud in enumerate(audiences):
            ax = axes[ai]
            matrix = np.zeros((len(top_ns), len(temps)))
            for r in sweep_results:
                if r["audience"] == aud:
                    ti = temps.index(r["temp"])
                    ni = top_ns.index(r["top_n"])
                    matrix[ni, ti] = r["result"][metric_name]

            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                          vmin=min(0, matrix.min()), vmax=max(matrix.max(), 0.01))
            ax.set_xticks(range(len(temps)))
            ax.set_xticklabels([f"{t}" for t in temps], fontsize=8)
            ax.set_yticks(range(len(top_ns)))
            ax.set_yticklabels([f"top-{n}" for n in top_ns], fontsize=8)
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Top-N metrics")
            ax.set_title(f"{aud} — {metric_name}")

            # annotate cells
            for ni in range(len(top_ns)):
                for ti in range(len(temps)):
                    val = matrix[ni, ti]
                    ax.text(ti, ni, f"{val:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if abs(val) > 0.4 else "black")

            plt.colorbar(im, ax=ax, shrink=0.8)

        fig.suptitle(f"Evaluator Sweep: {metric_name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"heatmap_{metric_name}.png"), dpi=150)
        plt.close(fig)

    # ── 2. Best formula per audience ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    header = ["Audience", "Top-N", "Temp", "#1 Match", "Exact", "Tau", "r", "p"]
    rows = []
    for aud in audiences:
        aud_results = [r for r in sweep_results if r["audience"] == aud]
        best = max(aud_results, key=lambda r: r["result"]["rank1_pct"] + r["result"]["tau"] * 0.5)
        res = best["result"]
        rows.append([aud, f"top-{best['top_n']}", f"{best['temp']}", f"{res['rank1_pct']:.0%}",
                     f"{res['exact_pct']:.0%}", f"{res['tau']:+.3f}", f"{res['r']:+.3f}", f"{res['p']:.4f}"])
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Best Formula Per Audience (by #1 match + 0.5*tau)", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "best_formulas.png"), dpi=150)
    plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    all_data = load_42ar_data() + load_53ar_data()
    print(f"  {len(all_data)} entries")

    print("Computing metrics...")
    compute_metrics_for_all(all_data)

    self_data = [dp for dp in all_data if dp["evaluator"] == "self"]
    vol_data = [dp for dp in all_data if dp["evaluator"] != "self"]

    audiences = {
        "all": build_groups(all_data),
        "self": build_groups(self_data),
        "volunteer": build_groups(vol_data),
    }

    # Sweep
    print("\nSweeping...")
    sweep_results = []
    top_ns = list(range(1, len(METRIC_RANKING) + 1))

    for aud_name, groups in audiences.items():
        if not groups:
            continue
        for top_n in top_ns:
            for temp in TEMPERATURES:
                formula = make_formula(top_n, temp)
                result = evaluate_formula(groups, formula)
                sweep_results.append({
                    "audience": aud_name,
                    "top_n": top_n,
                    "temp": temp,
                    "formula": [(mk, sign, round(w, 4)) for mk, sign, w in formula],
                    "result": result,
                })

    # Print best per audience
    print(f"\n{'='*90}")
    print("BEST FORMULAS")
    print(f"{'='*90}")
    for aud_name in ["all", "self", "volunteer"]:
        aud_results = [r for r in sweep_results if r["audience"] == aud_name]
        if not aud_results:
            continue

        # Sort by composite score
        aud_results.sort(key=lambda r: r["result"]["rank1_pct"] + r["result"]["tau"] * 0.5, reverse=True)

        print(f"\n  {aud_name.upper()} — Top 5:")
        print(f"  {'top_n':>5s} {'temp':>5s} {'#1':>6s} {'exact':>6s} {'tau':>7s} {'r':>7s} {'p':>8s}  formula")
        for r in aud_results[:5]:
            res = r["result"]
            formula_str = " + ".join(f"{w:.2f}*{'+' if s > 0 else '-'}{mk}" for mk, s, w in r["formula"])
            print(f"  {r['top_n']:>5d} {r['temp']:>5.1f} {res['rank1_pct']:>5.0%} {res['exact_pct']:>5.0%} "
                  f"{res['tau']:>+6.3f} {res['r']:>+6.3f} {res['p']:>8.4f}  {formula_str[:60]}")

    # Save all results
    with open(os.path.join(output_dir, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    save_graphs(sweep_results, output_dir)


if __name__ == "__main__":
    main()
