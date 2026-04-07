"""Experiment 59-G: Synthetic Evaluator Validation on Fresh Data.

Runs 4 models from 53-AR on 30 fresh val songs, computes chart metrics,
and applies the synthetic evaluators to produce a leaderboard. Compares
to 53-AR human rankings to validate generalization.

Usage:
    cd osu/taiko
    python experiments/experiment_59g/validate_fresh.py
"""

import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(TAIKO_DIR, "audio")
BIN_MS = 4.9887

sys.path.insert(0, os.path.join(TAIKO_DIR, "experiments", "experiment_59"))
from analyze_metrics import compute_chart_metrics, load_csv_events_ms, load_mel

MODELS = {
    "exp14": os.path.join(TAIKO_DIR, "runs", "detect_experiment_14", "checkpoints", "best.pt"),
    "exp44": os.path.join(TAIKO_DIR, "runs", "detect_experiment_44", "checkpoints", "best.pt"),
    "exp45": os.path.join(TAIKO_DIR, "runs", "detect_experiment_45", "checkpoints", "best.pt"),
    "exp53": os.path.join(TAIKO_DIR, "runs", "detect_experiment_53", "checkpoints", "best.pt"),
}

# 53-AR song audio filenames to exclude
EXCLUDE_53AR = {
    "arashi", "sakurazaka46", "camellia", "redalice", "courtney_barnett",
    "mon_rovia", "roccow", "supernovayuli", "conan_gray", "miley_cyrus",
}

# Optimized evaluator formulas from 59-F
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

EVALUATORS = {
    "expert": [(mk, s, 1.0) for mk, s, r in METRIC_RANKING[:2]],
    "volunteer": [(mk, s, abs(r)**0.5) for mk, s, r in METRIC_RANKING[:7]],
    "general": [(mk, s, 1.0) for mk, s, r in METRIC_RANKING[:6]],
}

# 53-AR human ranking (ground truth to compare against)
HUMAN_RANKING_53AR = ["exp45", "exp44", "exp53", "exp14"]

# 53-AR used fixed density conditioning across all songs
FIXED_DENSITY_53AR = {"density_mean": 5.75, "density_peak": 11.1, "density_std": 1.5}

# Two density regimes to test
DENSITY_REGIMES = {
    "song_density": None,  # use each song's actual density from manifest
    "fixed_53ar": FIXED_DENSITY_53AR,  # use the exact density from 53-AR
}


def get_val_songs(manifest):
    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    return songs[:n_val], song_to_charts


def find_audio_file(beatmapset_id, artist, title):
    prefix = f"{beatmapset_id} {artist} - {title}"
    for ext in [".mp3", ".ogg", ".wav", ".flac"]:
        path = os.path.join(AUDIO_DIR, prefix + ext)
        if os.path.exists(path):
            return path
    for f in os.listdir(AUDIO_DIR):
        if f.startswith(str(beatmapset_id) + " "):
            return os.path.join(AUDIO_DIR, f)
    return None


def is_excluded(artist, title):
    """Check if song was in 53-AR."""
    combined = (artist + " " + title).lower()
    for exc in EXCLUDE_53AR:
        if exc in combined:
            return True
    return False


def select_songs(manifest, n=30):
    val_songs, song_to_charts = get_val_songs(manifest)
    charts = manifest["charts"]

    candidates = []
    for sid in val_songs:
        idxs = song_to_charts[sid]
        sorted_idxs = sorted(idxs, key=lambda i: charts[i]["density_mean"])
        ci = sorted_idxs[len(sorted_idxs) // 2]
        c = charts[ci]

        if is_excluded(c["artist"], c["title"]):
            continue

        audio_path = find_audio_file(c["beatmapset_id"], c["artist"], c["title"])
        if audio_path is None:
            continue

        candidates.append({
            "chart_idx": ci,
            "beatmapset_id": c["beatmapset_id"],
            "artist": c["artist"],
            "title": c["title"],
            "difficulty": c["difficulty"],
            "density_mean": c["density_mean"],
            "density_peak": c["density_peak"],
            "density_std": c["density_std"],
            "duration_s": c["duration_s"],
            "audio_path": audio_path,
        })

    candidates.sort(key=lambda x: x["density_mean"])
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


def run_inference(checkpoint, song, model_name, output_dir, regime_name, density_override=None, hop_ms=75):
    safe_name = f"{song['beatmapset_id']}_{song['artist'][:20]}_{song['title'][:20]}".replace(" ", "_").replace("/", "_")
    for ch in "*?:<>|\"":
        safe_name = safe_name.replace(ch, "")
    output_csv = os.path.join(output_dir, f"{safe_name}_{model_name}_{regime_name}_predicted.csv")

    if density_override:
        d_mean = density_override["density_mean"]
        d_peak = density_override["density_peak"]
        d_std = density_override["density_std"]
    else:
        d_mean = song["density_mean"]
        d_peak = song["density_peak"]
        d_std = song["density_std"]

    cmd = [
        sys.executable, os.path.join(TAIKO_DIR, "detection_inference.py"),
        "--checkpoint", checkpoint,
        "--audio", song["audio_path"],
        "--output", output_csv,
        "--density-mean", str(d_mean),
        "--density-peak", str(d_peak),
        "--density-std", str(d_std),
        "--hop-ms", str(hop_ms),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"      ERROR: {result.stderr[-200:]}")
        return None
    return output_csv


def synthetic_score(metrics_by_model, evaluator_def):
    """Compute z-scored synthetic scores for models on one song."""
    models = list(metrics_by_model.keys())
    if len(models) < 2:
        return {}

    z_scores = {}
    for mk, sign, w in evaluator_def:
        vals = np.array([metrics_by_model[m].get(mk, 0) for m in models], dtype=np.float64)
        std = vals.std()
        mean = vals.mean()
        z_scores[mk] = (vals - mean) / std if std > 1e-10 else np.zeros_like(vals)

    scores = {}
    for i, m in enumerate(models):
        scores[m] = sum(sign * w * z_scores[mk][i] for mk, sign, w in evaluator_def)
    return scores


def save_graphs(all_regime_data, model_totals, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {
        "exp14": "#4a90d9", "exp44": "#6bc46d", "exp45": "#c76dba", "exp53": "#00cccc",
    }
    eval_names = list(EVALUATORS.keys())
    regime_names = list(all_regime_data.keys())

    # ── 1. Leaderboard: regime × evaluator grid ──
    n_regimes = len(regime_names)
    n_evals = len(eval_names)
    fig, axes = plt.subplots(n_regimes, n_evals + 1, figsize=(5 * (n_evals + 1), 5 * n_regimes))
    if n_regimes == 1:
        axes = [axes]

    for ri, regime in enumerate(regime_names):
        for ei, eval_name in enumerate(eval_names):
            ax = axes[ri][ei]
            totals = model_totals[regime][eval_name]
            models = sorted(totals.keys(), key=lambda m: -totals[m])
            vals = [totals[m] for m in models]
            colors = [model_colors.get(m, "#999") for m in models]
            ax.bar(models, vals, color=colors)
            ax.set_title(f"{regime}\n{eval_name}", fontsize=10)
            if ei == 0:
                ax.set_ylabel("Synthetic score")

        # Human reference column
        ax = axes[ri][n_evals]
        human_pts = {"exp45": 4, "exp44": 3, "exp53": 2, "exp14": 1}
        models = sorted(human_pts.keys(), key=lambda m: -human_pts[m])
        ax.bar(models, [human_pts[m] for m in models], color=[model_colors.get(m, "#999") for m in models])
        ax.set_title(f"53-AR Human\n(reference)", fontsize=10)

    fig.suptitle("Leaderboard: Synthetic Evaluators × Density Regimes vs 53-AR Human", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "leaderboard.png"), dpi=150)
    plt.close(fig)

    # ── 2. Per-song win counts per regime ──
    for regime in regime_names:
        per_song_data = all_regime_data[regime]
        fig, ax = plt.subplots(figsize=(10, 6))
        for ei, eval_name in enumerate(eval_names):
            win_counts = defaultdict(int)
            for song_data in per_song_data:
                scores = song_data.get(f"scores_{eval_name}")
                if scores:
                    winner = max(scores, key=scores.get)
                    win_counts[winner] += 1
            models = sorted(MODELS.keys())
            vals = [win_counts.get(m, 0) for m in models]
            x = np.arange(len(models))
            w = 0.25
            ax.bar(x + (ei - 1) * w, vals, w, label=eval_name,
                   color=["#4a90d9", "#6bc46d", "#ff9900"][ei], alpha=0.8)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(sorted(MODELS.keys()))
        ax.set_ylabel(f"Songs won (out of {len(per_song_data)})")
        ax.set_title(f"Per-Song Wins: {regime}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"win_counts_{regime}.png"), dpi=150)
        plt.close(fig)

    # ── 3. Summary table ──
    fig, ax = plt.subplots(figsize=(14, 4 + len(regime_names) * len(eval_names) * 0.4))
    ax.axis("off")
    header = ["Regime", "Evaluator", "Ranking", "#1", "Top-2", "Full"]
    rows = []
    for regime in regime_names:
        for eval_name in eval_names:
            totals = model_totals[regime][eval_name]
            ranking = sorted(totals.keys(), key=lambda m: -totals[m])
            top1 = "YES" if ranking[0] == HUMAN_RANKING_53AR[0] else "no"
            top2 = "YES" if ranking[:2] == HUMAN_RANKING_53AR[:2] else "no"
            full = "YES" if ranking == HUMAN_RANKING_53AR else "no"
            rows.append([regime, eval_name, " > ".join(ranking), top1, top2, full])
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    for i, row in enumerate(rows):
        for j in [3, 4, 5]:
            color = "#c8f7c8" if row[j] == "YES" else "#ffcccc"
            table[i + 1, j].set_facecolor(color)
    ax.set_title("Summary: All Regime × Evaluator Combinations", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150)
    plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    csv_dir = os.path.join(output_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("Selecting songs...")
    songs = select_songs(manifest, n=30)
    print(f"Selected {len(songs)} songs\n")

    # Run inference for all models × all songs × both density regimes
    total_runs = len(MODELS) * len(songs) * len(DENSITY_REGIMES)
    print(f"{'='*70}")
    print(f"Running AR inference: {len(MODELS)} models x {len(songs)} songs x {len(DENSITY_REGIMES)} regimes = {total_runs} runs")
    print(f"{'='*70}")

    # per_song_data[regime_name] = list of song entries
    all_regime_data = {}

    for regime_name, density_override in DENSITY_REGIMES.items():
        print(f"\n{'='*70}")
        print(f"  REGIME: {regime_name}" + (f" (d={density_override['density_mean']:.1f}/{density_override['density_peak']:.1f})" if density_override else " (per-song)"))
        print(f"{'='*70}")

        per_song_data = []
        for si, song in enumerate(songs):
            d_label = f"d={density_override['density_mean']:.1f}" if density_override else f"d={song['density_mean']:.1f}"
            print(f"\n  [{si+1}/{len(songs)}] {song['artist']} - {song['title']} ({d_label})")
            song_metrics = {}

            for model_name, ckpt_path in MODELS.items():
                try:
                    csv_path = run_inference(ckpt_path, song, model_name, csv_dir,
                                            regime_name, density_override)
                    if csv_path is None:
                        continue
                    events_ms = load_csv_events_ms(csv_path)
                    mel_path = csv_path.replace(".csv", "_mel.npy")
                    mel = None
                    if os.path.exists(mel_path):
                        try:
                            mel = load_mel(mel_path)
                        except Exception:
                            pass
                    metrics = compute_chart_metrics(events_ms, mel)
                    if metrics:
                        song_metrics[model_name] = metrics
                        print(f"      {model_name}: {len(events_ms)} events, gap_std={metrics['gap_std']:.0f}, gap_cv={metrics['gap_cv']:.2f}")
                except Exception as e:
                    print(f"      {model_name}: FAILED ({e})")

            song_entry = {"song": song, "metrics": song_metrics}
            for eval_name, eval_def in EVALUATORS.items():
                scores = synthetic_score(song_metrics, eval_def)
                song_entry[f"scores_{eval_name}"] = scores
            per_song_data.append(song_entry)

        all_regime_data[regime_name] = per_song_data

    # Aggregate leaderboards for each regime
    print(f"\n{'='*70}")
    print("LEADERBOARDS")
    print(f"{'='*70}")

    model_totals = {}  # regime → eval → {model: total}

    for regime_name, per_song_data in all_regime_data.items():
        print(f"\n  ── {regime_name} ──")
        model_totals[regime_name] = {}

        for eval_name in EVALUATORS:
            totals = defaultdict(float)
            win_counts = defaultdict(int)
            for song_data in per_song_data:
                scores = song_data.get(f"scores_{eval_name}", {})
                for m, s in scores.items():
                    totals[m] += s
                if scores:
                    winner = max(scores, key=scores.get)
                    win_counts[winner] += 1

            model_totals[regime_name][eval_name] = dict(totals)
            ranking = sorted(totals.keys(), key=lambda m: -totals[m])

            print(f"\n    {eval_name}:")
            print(f"      {'Model':>8s} {'Total':>8s} {'Wins':>6s}")
            for m in ranking:
                print(f"      {m:>8s} {totals[m]:>+7.1f} {win_counts[m]:>5d}")
            print(f"      Ranking: {' > '.join(ranking)}")
            top1 = ranking[0] == HUMAN_RANKING_53AR[0]
            top2 = ranking[:2] == HUMAN_RANKING_53AR[:2]
            full = ranking == HUMAN_RANKING_53AR
            print(f"      vs 53-AR: #1={'YES' if top1 else 'no'}  top2={'YES' if top2 else 'no'}  full={'YES' if full else 'no'}")

    print(f"\n  53-AR Human: {' > '.join(HUMAN_RANKING_53AR)}")

    # Save results
    save_data = {
        "n_songs": len(songs),
        "model_totals": model_totals,
        "human_ranking_53ar": HUMAN_RANKING_53AR,
    }
    with open(os.path.join(output_dir, "validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)

    save_graphs(all_regime_data, model_totals, output_dir)


if __name__ == "__main__":
    main()
