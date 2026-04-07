"""Experiment 59-H: Extended Model Comparison.

Runs 6 models (2 reference from 53-AR + 4 new) on 30 fresh val songs,
applies synthetic evaluators to rank them. Uses per-song density only
(59-G showed it's more reliable than fixed density).

Usage:
    cd osu/taiko
    python experiments/experiment_59h/extended_comparison.py
"""

import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(TAIKO_DIR, "audio")

sys.path.insert(0, os.path.join(TAIKO_DIR, "experiments", "experiment_59"))
from analyze_metrics import compute_chart_metrics, load_csv_events_ms, load_mel

MODELS = {
    # Reference (from 53-AR human eval)
    "exp44": os.path.join(TAIKO_DIR, "runs", "detect_experiment_44", "checkpoints", "best.pt"),
    "exp53": os.path.join(TAIKO_DIR, "runs", "detect_experiment_53", "checkpoints", "best.pt"),
    # New models to evaluate
    "exp50b": os.path.join(TAIKO_DIR, "runs", "detect_experiment_50b", "checkpoints", "best.pt"),
    "exp51": os.path.join(TAIKO_DIR, "runs", "detect_experiment_51", "checkpoints", "best.pt"),
    "exp55": os.path.join(TAIKO_DIR, "runs", "detect_experiment_55", "checkpoints", "best.pt"),
    "exp58": os.path.join(TAIKO_DIR, "runs", "detect_experiment_58", "checkpoints", "best.pt"),
}

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

# 53-AR known rankings for reference models
REFERENCE_53AR = {"exp44": 2, "exp53": 3}  # rank position

FIXED_DENSITY_53AR = {"density_mean": 5.75, "density_peak": 11.1, "density_std": 1.5}

DENSITY_REGIMES = {
    "song_density": None,
    "fixed_53ar": FIXED_DENSITY_53AR,
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


def select_songs(manifest, n=30):
    val_songs, song_to_charts = get_val_songs(manifest)
    charts = manifest["charts"]
    candidates = []
    for sid in val_songs:
        idxs = song_to_charts[sid]
        sorted_idxs = sorted(idxs, key=lambda i: charts[i]["density_mean"])
        ci = sorted_idxs[len(sorted_idxs) // 2]
        c = charts[ci]
        audio_path = find_audio_file(c["beatmapset_id"], c["artist"], c["title"])
        if audio_path is None:
            continue
        candidates.append({
            "chart_idx": ci,
            "beatmapset_id": c["beatmapset_id"],
            "artist": c["artist"],
            "title": c["title"],
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
    models = list(metrics_by_model.keys())
    if len(models) < 2:
        return {}
    z_scores = {}
    for mk, sign, w in evaluator_def:
        vals = np.array([metrics_by_model[m].get(mk, 0) for m in models], dtype=np.float64)
        std = vals.std(); mean = vals.mean()
        z_scores[mk] = (vals - mean) / std if std > 1e-10 else np.zeros_like(vals)
    scores = {}
    for i, m in enumerate(models):
        scores[m] = sum(sign * w * z_scores[mk][i] for mk, sign, w in evaluator_def)
    return scores


def save_graphs(per_song_data, model_totals, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {
        "exp44": "#6bc46d", "exp53": "#00cccc",
        "exp50b": "#e6a817", "exp51": "#999999",
        "exp55": "#c76dba", "exp58": "#eb4528",
    }
    eval_names = list(EVALUATORS.keys())

    # ── 1. Leaderboard bars per evaluator ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ei, eval_name in enumerate(eval_names):
        ax = axes[ei]
        totals = model_totals[eval_name]
        ranking = sorted(totals.keys(), key=lambda m: -totals[m])
        vals = [totals[m] for m in ranking]
        colors = [model_colors.get(m, "#999") for m in ranking]
        bars = ax.bar(ranking, vals, color=colors)
        ax.set_title(eval_name, fontsize=12)
        ax.set_ylabel("Total synthetic score")
        # mark reference models
        for i, m in enumerate(ranking):
            if m in REFERENCE_53AR:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2)
        ax.axhline(0, color="black", linewidth=0.5)
    fig.suptitle("Extended Model Comparison (30 fresh val songs, per-song density)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "leaderboard.png"), dpi=150)
    plt.close(fig)

    # ── 2. Per-song win counts ──
    fig, ax = plt.subplots(figsize=(12, 6))
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
    ax.set_title("Per-Song Wins by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "win_counts.png"), dpi=150)
    plt.close(fig)

    # ── 3. Per-model metric distributions (gap_std, gap_cv) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = sorted(MODELS.keys())
    for mi, mk in enumerate(["gap_std", "gap_cv"]):
        ax = axes[mi]
        model_vals = {m: [] for m in models}
        for song_data in per_song_data:
            for m, metrics in song_data["metrics"].items():
                if mk in metrics:
                    model_vals[m].append(metrics[mk])
        box_data = [model_vals[m] for m in models if model_vals[m]]
        box_labels = [m for m in models if model_vals[m]]
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for j, m in enumerate(box_labels):
                bp["boxes"][j].set_facecolor(model_colors.get(m, "#cccccc"))
        ax.set_title(mk)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Key Metrics Distribution by Model", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metric_distributions.png"), dpi=150)
    plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    csv_dir = os.path.join(output_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    # Verify checkpoints exist
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"WARNING: {name} checkpoint not found: {path}")

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("Selecting songs...")
    songs = select_songs(manifest, n=30)
    print(f"Selected {len(songs)} songs\n")

    total_runs = len(MODELS) * len(songs) * len(DENSITY_REGIMES)
    print(f"{'='*70}")
    print(f"Running AR inference: {len(MODELS)} models x {len(songs)} songs x {len(DENSITY_REGIMES)} regimes = {total_runs} runs")
    print(f"{'='*70}")

    all_regime_data = {}

    for regime_name, density_override in DENSITY_REGIMES.items():
        print(f"\n{'='*70}")
        d_label = f"d={density_override['density_mean']:.1f}/{density_override['density_peak']:.1f}" if density_override else "per-song"
        print(f"  REGIME: {regime_name} ({d_label})")
        print(f"{'='*70}")

        per_song_data = []
        for si, song in enumerate(songs):
            d_show = f"d={density_override['density_mean']:.1f}" if density_override else f"d={song['density_mean']:.1f}"
            print(f"\n  [{si+1}/{len(songs)}] {song['artist']} - {song['title']} ({d_show})")
            song_metrics = {}

            for model_name, ckpt_path in MODELS.items():
                if not os.path.exists(ckpt_path):
                    continue
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

    # Aggregate
    print(f"\n{'='*70}")
    print("LEADERBOARDS")
    print(f"{'='*70}")

    all_model_totals = {}
    for regime_name, per_song_data in all_regime_data.items():
        print(f"\n  ── {regime_name} ──")
        regime_totals = {}

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

            regime_totals[eval_name] = dict(totals)
            ranking = sorted(totals.keys(), key=lambda m: -totals[m])

            print(f"\n    {eval_name}:")
            print(f"      {'Model':>8s} {'Total':>8s} {'Wins':>6s} {'Note':>15s}")
            for m in ranking:
                note = "(ref)" if m in REFERENCE_53AR else ""
                print(f"      {m:>8s} {totals[m]:>+7.1f} {win_counts[m]:>5d} {note:>15s}")
            print(f"      Ranking: {' > '.join(ranking)}")

        all_model_totals[regime_name] = regime_totals

    # Save
    save_data = {"n_songs": len(songs), "model_totals": all_model_totals}
    with open(os.path.join(output_dir, "comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Use song_density for graphs (the more reliable regime)
    save_graphs(all_regime_data.get("song_density", []),
                all_model_totals.get("song_density", {}), output_dir)


if __name__ == "__main__":
    main()
