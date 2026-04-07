"""Experiment 59-HB: Ground Truth Comparison for 59-H Models.

Reuses the CSVs from 59-H, loads GT events from the dataset, and computes
AR quality metrics (matched/close/far, hallucination, density ratio) for
each model under both density regimes.

Usage:
    cd osu/taiko
    python experiments/experiment_59hb/gt_comparison.py
"""

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(TAIKO_DIR, "audio")
BIN_MS = 4.9887

CSV_DIR = os.path.join(TAIKO_DIR, "experiments", "experiment_59h", "results", "csvs")

MODELS = ["exp44", "exp53", "exp50b", "exp51", "exp55", "exp58"]
REGIMES = ["song_density", "fixed_53ar"]


def load_csv_events_ms(csv_path):
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("time_ms") or not line:
                continue
            parts = line.split(",")
            if parts:
                events.append(int(parts[0]))
    return np.array(events, dtype=np.float64)


def load_gt_events_ms(event_file):
    events = np.load(os.path.join(DATASET_DIR, "events", event_file))
    return events.astype(np.float64) * BIN_MS


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


def get_songs(manifest):
    """Same song selection as 59-H."""
    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    val_songs = songs[:n_val]

    candidates = []
    for sid in val_songs:
        idxs = song_to_charts[sid]
        sorted_idxs = sorted(idxs, key=lambda i: charts[i]["density_mean"])
        ci = sorted_idxs[len(sorted_idxs) // 2]
        c = charts[ci]
        audio_path = find_audio_file(c["beatmapset_id"], c["artist"], c["title"])
        if audio_path is None:
            continue
        candidates.append(c)

    candidates.sort(key=lambda x: x["density_mean"])
    n = 30
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


def _find_closest(sorted_arr, value):
    idx = np.searchsorted(sorted_arr, value)
    best = float("inf")
    for j in [idx - 1, idx, idx + 1]:
        if 0 <= j < len(sorted_arr):
            d = abs(sorted_arr[j] - value)
            if d < best:
                best = d
    return best


def compute_ar_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0 or len(gt_ms) == 0:
        return None

    pred_sorted = np.sort(pred_ms)
    gt_sorted = np.sort(gt_ms)

    gt_errors = np.array([_find_closest(pred_sorted, gt) for gt in gt_sorted])
    pred_errors = np.array([_find_closest(gt_sorted, p) for p in pred_sorted])

    n_pred = len(pred_sorted)
    n_gt = len(gt_sorted)

    if n_pred > 1:
        pred_density = n_pred / max((pred_sorted[-1] - pred_sorted[0]) / 1000.0, 0.1)
    else:
        pred_density = 0.0
    if n_gt > 1:
        gt_density = n_gt / max((gt_sorted[-1] - gt_sorted[0]) / 1000.0, 0.1)
    else:
        gt_density = 0.0

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "pred_gt_ratio": n_pred / max(n_gt, 1),
        "matched_rate": float((gt_errors <= 25).mean()),
        "close_rate": float((gt_errors <= 50).mean()),
        "far_rate": float((gt_errors > 100).mean()),
        "hallucination_rate": float((pred_errors > 100).mean()),
        "gt_error_mean": float(gt_errors.mean()),
        "gt_error_median": float(np.median(gt_errors)),
        "pred_density": pred_density,
        "gt_density": gt_density,
        "density_ratio": pred_density / max(gt_density, 0.01),
    }


def save_graphs(all_results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {
        "exp44": "#6bc46d", "exp53": "#00cccc",
        "exp50b": "#e6a817", "exp51": "#999999",
        "exp55": "#c76dba", "exp58": "#eb4528",
    }

    for regime in REGIMES:
        metrics_to_plot = ["close_rate", "hallucination_rate", "density_ratio", "gt_error_median"]
        titles = ["Close Rate (<50ms)", "Hallucination Rate (>100ms)", "Density Ratio (pred/gt)", "GT Error Median (ms)"]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        for mi, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[mi]
            models = MODELS
            vals = []
            for m in models:
                model_vals = [r[metric] for r in all_results.get(regime, {}).get(m, []) if r]
                vals.append(np.mean(model_vals) if model_vals else 0)
            colors = [model_colors.get(m, "#999") for m in models]
            ax.bar(models, vals, color=colors)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            if "rate" in metric or "ratio" in metric:
                if "density" in metric:
                    ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
                elif "hallucination" not in metric:
                    ax.set_ylim(0, 1)

        fig.suptitle(f"GT Comparison: {regime}", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"gt_comparison_{regime}.png"), dpi=150)
        plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("Loading songs...")
    songs = get_songs(manifest)
    print(f"  {len(songs)} songs")

    # For each song, find its predicted CSV and GT events
    all_results = {}  # regime → model → [metrics per song]

    for regime in REGIMES:
        all_results[regime] = {m: [] for m in MODELS}

        for song in songs:
            safe_name = f"{song['beatmapset_id']}_{song['artist'][:20]}_{song['title'][:20]}".replace(" ", "_").replace("/", "_")
            for ch in "*?:<>|\"":
                safe_name = safe_name.replace(ch, "")

            gt_ms = load_gt_events_ms(song["event_file"])

            for model in MODELS:
                csv_path = os.path.join(CSV_DIR, f"{safe_name}_{model}_{regime}_predicted.csv")
                if not os.path.exists(csv_path):
                    all_results[regime][model].append(None)
                    continue

                pred_ms = load_csv_events_ms(csv_path)
                metrics = compute_ar_metrics(pred_ms, gt_ms)
                all_results[regime][model].append(metrics)

    # Print summary
    for regime in REGIMES:
        print(f"\n{'='*80}")
        print(f"  {regime}")
        print(f"{'='*80}")
        print(f"  {'Model':>8s} {'Match%':>7s} {'Close%':>7s} {'Far%':>6s} {'Hall%':>6s} {'d_ratio':>8s} {'err_med':>8s} {'#pred':>7s} {'#gt':>7s} {'p/g':>5s}")

        for model in MODELS:
            results = [r for r in all_results[regime][model] if r]
            if not results:
                print(f"  {model:>8s}  (no data)")
                continue
            avg = lambda k: np.mean([r[k] for r in results])
            print(f"  {model:>8s} {avg('matched_rate'):>6.1%} {avg('close_rate'):>6.1%} {avg('far_rate'):>5.1%} {avg('hallucination_rate'):>5.1%} "
                  f"{avg('density_ratio'):>7.2f} {avg('gt_error_median'):>7.0f}ms "
                  f"{avg('n_pred'):>6.0f} {avg('n_gt'):>6.0f} {avg('pred_gt_ratio'):>5.2f}")

    # Save
    save_data = {}
    for regime in REGIMES:
        save_data[regime] = {}
        for model in MODELS:
            results = [r for r in all_results[regime][model] if r]
            if not results:
                continue
            avg = lambda k: float(np.mean([r[k] for r in results]))
            save_data[regime][model] = {
                "n_songs": len(results),
                "matched_rate": avg("matched_rate"),
                "close_rate": avg("close_rate"),
                "far_rate": avg("far_rate"),
                "hallucination_rate": avg("hallucination_rate"),
                "density_ratio": avg("density_ratio"),
                "gt_error_median": avg("gt_error_median"),
                "avg_n_pred": avg("n_pred"),
                "avg_n_gt": avg("n_gt"),
                "pred_gt_ratio": avg("pred_gt_ratio"),
            }

    with open(os.path.join(output_dir, "gt_results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    save_graphs(all_results, output_dir)


if __name__ == "__main__":
    main()
