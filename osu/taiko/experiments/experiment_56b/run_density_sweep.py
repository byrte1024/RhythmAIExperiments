"""Experiment 56-B: Density Sensitivity Test.

Runs AR inference on 50 val songs at 3 density scales (0.8x, 1.0x, 1.2x)
to measure how much density conditioning actually affects model output.

Usage:
    cd osu/taiko
    python experiments/experiment_56b/run_density_sweep.py --checkpoint runs/detect_experiment_45/checkpoints/best.pt
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(TAIKO_DIR, "audio")
BIN_MS = 4.9887

SCALES = [0.8, 1.0, 1.2]
SCALE_NAMES = ["0.8x", "1.0x", "1.2x"]


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


def select_songs(manifest, n=50):
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
            "difficulty": c["difficulty"],
            "density_mean": c["density_mean"],
            "density_peak": c["density_peak"],
            "density_std": c["density_std"],
            "duration_s": c["duration_s"],
            "total_events": c["total_events"],
            "event_file": c["event_file"],
            "audio_path": audio_path,
        })
    candidates.sort(key=lambda x: x["density_mean"])
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


def run_inference(checkpoint, song, output_dir, scale, hop_ms=75):
    safe_name = f"{song['beatmapset_id']}_{song['artist'][:20]}_{song['title'][:20]}".replace(" ", "_").replace("/", "_").replace("*", "").replace("?", "").replace(":", "").replace("<", "").replace(">", "").replace("|", "").replace('"', "")
    scale_tag = f"{scale:.1f}x"
    output_csv = os.path.join(output_dir, f"{safe_name}_{scale_tag}_predicted.csv")

    cmd = [
        sys.executable, os.path.join(TAIKO_DIR, "detection_inference.py"),
        "--checkpoint", checkpoint,
        "--audio", song["audio_path"],
        "--output", output_csv,
        "--density-mean", str(song["density_mean"] * scale),
        "--density-peak", str(song["density_peak"] * scale),
        "--density-std", str(song["density_std"] * scale),
        "--hop-ms", str(hop_ms),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"    ERROR ({scale_tag}): {result.stderr[-200:]}")
        return None
    return output_csv


def load_predicted_events(csv_path):
    events_ms = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("time_ms"):
                continue
            parts = line.split(",")
            if len(parts) >= 1:
                events_ms.append(int(parts[0]))
    return np.array(events_ms, dtype=np.float64)


def load_gt_events(event_file):
    events = np.load(os.path.join(DATASET_DIR, "events", event_file))
    return events.astype(np.float64) * BIN_MS


def _find_closest(sorted_arr, value):
    idx = np.searchsorted(sorted_arr, value)
    best_dist = float("inf")
    for j in [idx - 1, idx, idx + 1]:
        if 0 <= j < len(sorted_arr):
            d = abs(sorted_arr[j] - value)
            if d < best_dist:
                best_dist = d
    return best_dist


def compute_ar_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0:
        return {
            "n_pred": 0, "n_gt": len(gt_ms),
            "event_matched_rate": 0.0, "event_close_rate": 0.0, "event_far_rate": 1.0,
            "hallucination_rate": 1.0, "pred_density": 0.0,
            "gt_density": 0.0, "density_ratio": 0.0,
        }

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
        "event_matched_rate": float((gt_errors <= 25).mean()),
        "event_close_rate": float((gt_errors <= 50).mean()),
        "event_far_rate": float((gt_errors > 100).mean()),
        "hallucination_rate": float((pred_errors > 100).mean()),
        "pred_density": pred_density,
        "gt_density": gt_density,
        "density_ratio": pred_density / max(gt_density, 0.01),
        "gt_error_median": float(np.median(gt_errors)),
    }


def save_graphs(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    x = np.arange(n)
    short_names = [f"{r['song']['artist'][:12]} - {r['song']['title'][:12]}" for r in results]

    # ── 1. Event count per scale ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    for si, (scale, name) in enumerate(zip(SCALES, SCALE_NAMES)):
        counts = [r["scales"][name]["n_pred"] for r in results]
        offset = (si - 1) * 0.25
        ax.bar(x + offset, counts, 0.25, label=name, alpha=0.8)
    gt_counts = [r["scales"]["1.0x"]["n_gt"] for r in results]
    ax.plot(x, gt_counts, "k--", linewidth=1.5, label="GT", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Event count")
    ax.set_title("Predicted events per density scale")
    ax.legend(fontsize=8)

    # ── 2. Sensitivity: 1.2x / 0.8x event ratio ──
    ax = axes[0, 1]
    sensitivities = []
    for r in results:
        n_low = r["scales"]["0.8x"]["n_pred"]
        n_high = r["scales"]["1.2x"]["n_pred"]
        sens = n_high / max(n_low, 1)
        sensitivities.append(sens)
    colors = ["#6bc46d" if s > 1.1 else "#ff9900" if s > 0.95 else "#eb4528" for s in sensitivities]
    ax.bar(x, sensitivities, color=colors)
    ax.axhline(1.5, color="green", linestyle="--", alpha=0.3, label="Ideal (1.5 = proportional)")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.3, label="Insensitive (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("1.2x / 0.8x event ratio")
    ax.set_title("Density Sensitivity (ideal=1.5, deaf=1.0)")
    ax.legend(fontsize=8)

    # ── 3. Density ratio per scale ──
    ax = axes[1, 0]
    for si, (scale, name) in enumerate(zip(SCALES, SCALE_NAMES)):
        ratios = [r["scales"][name]["density_ratio"] for r in results]
        offset = (si - 1) * 0.25
        ax.bar(x + offset, ratios, 0.25, label=name, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Pred/GT density ratio")
    ax.set_title("Density adherence per scale")
    ax.legend(fontsize=8)

    # ── 4. Hallucination per scale ──
    ax = axes[1, 1]
    for si, (scale, name) in enumerate(zip(SCALES, SCALE_NAMES)):
        halls = [r["scales"][name]["hallucination_rate"] for r in results]
        offset = (si - 1) * 0.25
        ax.bar(x + offset, halls, 0.25, label=name, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Hallucination per density scale")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 56-B: Density Sensitivity Sweep", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "density_sweep.png"), dpi=150)
    plt.close(fig)

    # ── 5. Scatter: conditioned density vs sensitivity ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    densities = [r["song"]["density_mean"] for r in results]

    ax = axes[0]
    ax.scatter(densities, sensitivities, c="#4a90d9", s=80, edgecolors="black")
    ax.axhline(1.5, color="green", linestyle="--", alpha=0.3)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Conditioned Density (events/sec)")
    ax.set_ylabel("1.2x / 0.8x ratio")
    ax.set_title("Density vs Sensitivity")

    ax = axes[1]
    delta_matched = [r["scales"]["1.2x"]["event_close_rate"] - r["scales"]["0.8x"]["event_close_rate"] for r in results]
    ax.scatter(densities, delta_matched, c="#6bc46d", s=80, edgecolors="black")
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Conditioned Density (events/sec)")
    ax.set_ylabel("Close rate delta (1.2x - 0.8x)")
    ax.set_title("Does higher density help catch rate?")

    ax = axes[2]
    delta_hall = [r["scales"]["1.2x"]["hallucination_rate"] - r["scales"]["0.8x"]["hallucination_rate"] for r in results]
    ax.scatter(densities, delta_hall, c="#eb4528", s=80, edgecolors="black")
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Conditioned Density (events/sec)")
    ax.set_ylabel("Hallucination delta (1.2x - 0.8x)")
    ax.set_title("Does higher density increase hallucination?")

    fig.suptitle("Density Sensitivity Correlations", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "density_sensitivity_correlation.png"), dpi=150)
    plt.close(fig)

    print(f"\nGraphs saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Exp 56-B: Density sensitivity sweep")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--n-songs", type=int, default=50, help="Number of val songs")
    parser.add_argument("--hop-ms", type=float, default=75, help="Hop on STOP (ms)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "results")
    csv_dir = os.path.join(output_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("Selecting val songs...")
    songs = select_songs(manifest, n=args.n_songs)
    print(f"Selected {len(songs)} songs\n")

    results = []
    for i, song in enumerate(songs):
        print(f"[{i+1}/{len(songs)}] {song['artist']} - {song['title']} (d={song['density_mean']:.1f})")
        try:
            gt_ms = load_gt_events(song["event_file"])

            # Save GT CSV once
            safe_name = f"{song['beatmapset_id']}_{song['artist'][:20]}_{song['title'][:20]}".replace(" ", "_").replace("/", "_").replace("*", "").replace("?", "").replace(":", "").replace("<", "").replace(">", "").replace("|", "").replace('"', "")
            gt_csv = os.path.join(csv_dir, f"{safe_name}_gt.csv")
            with open(gt_csv, "w", encoding="utf-8") as f:
                f.write("time_ms,type\n")
                for t in gt_ms:
                    f.write(f"{int(t)},gt\n")

            song_result = {"song": song, "scales": {}}
            for scale, name in zip(SCALES, SCALE_NAMES):
                csv_path = run_inference(args.checkpoint, song, csv_dir, scale, hop_ms=args.hop_ms)
                if csv_path is None:
                    continue
                pred_ms = load_predicted_events(csv_path)
                metrics = compute_ar_metrics(pred_ms, gt_ms)
                song_result["scales"][name] = metrics
                print(f"    {name}: {metrics['n_pred']:>5d} pred  match={metrics['event_matched_rate']:.1%}  close={metrics['event_close_rate']:.1%}  hall={metrics['hallucination_rate']:.1%}  d_ratio={metrics['density_ratio']:.2f}")

            if len(song_result["scales"]) == 3:
                n_low = song_result["scales"]["0.8x"]["n_pred"]
                n_high = song_result["scales"]["1.2x"]["n_pred"]
                sens = n_high / max(n_low, 1)
                print(f"    Sensitivity: {n_low} -> {n_high} ({sens:.2f}x)")
                results.append(song_result)
        except Exception as e:
            print(f"    SKIPPED ({e})")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Song':>35s} {'d_cond':>6} {'n_0.8x':>7} {'n_1.0x':>7} {'n_1.2x':>7} {'n_GT':>6} {'sens':>6} {'cl_0.8':>7} {'cl_1.0':>7} {'cl_1.2':>7}")
    sensitivities = []
    for r in results:
        s = r["song"]
        s08 = r["scales"]["0.8x"]; s10 = r["scales"]["1.0x"]; s12 = r["scales"]["1.2x"]
        name = f"{s['artist'][:15]} - {s['title'][:15]}"
        sens = s12["n_pred"] / max(s08["n_pred"], 1)
        sensitivities.append(sens)
        print(f"{name:>35s} {s['density_mean']:>6.1f} {s08['n_pred']:>7d} {s10['n_pred']:>7d} {s12['n_pred']:>7d} {s10['n_gt']:>6d} {sens:>6.2f} {s08['event_close_rate']:>6.1%} {s10['event_close_rate']:>6.1%} {s12['event_close_rate']:>6.1%}")

    if results:
        avg_sens = np.mean(sensitivities)
        avg_close_08 = np.mean([r["scales"]["0.8x"]["event_close_rate"] for r in results])
        avg_close_10 = np.mean([r["scales"]["1.0x"]["event_close_rate"] for r in results])
        avg_close_12 = np.mean([r["scales"]["1.2x"]["event_close_rate"] for r in results])
        print(f"\n{'AVERAGE':>35s} {'':>6s} {'':>7s} {'':>7s} {'':>7s} {'':>6s} {avg_sens:>6.2f} {avg_close_08:>6.1%} {avg_close_10:>6.1%} {avg_close_12:>6.1%}")
        print(f"\nAvg sensitivity (1.2x/0.8x): {avg_sens:.3f}  (ideal=1.50, deaf=1.00)")

    # Save JSON
    results_json = os.path.join(output_dir, "density_sweep_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_json}")

    save_graphs(results, output_dir)


if __name__ == "__main__":
    main()
