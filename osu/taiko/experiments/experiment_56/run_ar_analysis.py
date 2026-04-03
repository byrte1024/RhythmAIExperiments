"""Experiment 56: Density Conditioning AR Analysis.

Runs full AR inference on 10 val songs using each song's actual chart density,
then compares predicted onsets to ground truth.

Usage:
    cd osu/taiko
    python experiments/experiment_56/run_ar_analysis.py --checkpoint runs/detect_experiment_55/checkpoints/best.pt
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
BIN_MS = 4.9887  # ms per mel frame


def get_val_songs(manifest):
    """Reproduce the training split to get val songs."""
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
    return val_songs, song_to_charts


def find_audio_file(beatmapset_id, artist, title):
    """Find audio file in the audio directory."""
    prefix = f"{beatmapset_id} {artist} - {title}"
    for ext in [".mp3", ".ogg", ".wav", ".flac"]:
        path = os.path.join(AUDIO_DIR, prefix + ext)
        if os.path.exists(path):
            return path
    # Fuzzy: try just the beatmapset_id prefix
    for f in os.listdir(AUDIO_DIR):
        if f.startswith(str(beatmapset_id) + " "):
            return os.path.join(AUDIO_DIR, f)
    return None


def select_songs(manifest, n=10):
    """Select n diverse val songs with available audio."""
    val_songs, song_to_charts = get_val_songs(manifest)
    charts = manifest["charts"]

    candidates = []
    for sid in val_songs:
        idxs = song_to_charts[sid]
        # Pick the chart with median difficulty (by density_mean)
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

    # Sort by density_mean for diversity, pick evenly spaced
    candidates.sort(key=lambda x: x["density_mean"])
    if len(candidates) <= n:
        selected = candidates
    else:
        step = len(candidates) / n
        selected = [candidates[int(i * step)] for i in range(n)]

    return selected


def run_inference(checkpoint, song, output_dir, hop_ms=75):
    """Run AR inference on a single song, return path to output CSV."""
    safe_name = f"{song['beatmapset_id']}_{song['artist'][:20]}_{song['title'][:20]}".replace(" ", "_").replace("/", "_").replace("*", "").replace("?", "").replace(":", "").replace("<", "").replace(">", "").replace("|", "").replace('"', "")
    output_csv = os.path.join(output_dir, f"{safe_name}_predicted.csv")

    cmd = [
        sys.executable, os.path.join(TAIKO_DIR, "detection_inference.py"),
        "--checkpoint", checkpoint,
        "--audio", song["audio_path"],
        "--output", output_csv,
        "--density-mean", str(song["density_mean"]),
        "--density-peak", str(song["density_peak"]),
        "--density-std", str(song["density_std"]),
        "--hop-ms", str(hop_ms),
    ]

    print(f"  Running: {song['artist']} - {song['title']} (d={song['density_mean']:.1f})")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-200:]}")
        return None
    return output_csv


def load_predicted_events(csv_path):
    """Load predicted events from CSV (time_ms,type)."""
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
    """Load ground truth events (bin indices) and convert to ms."""
    events = np.load(os.path.join(DATASET_DIR, "events", event_file))
    return events.astype(np.float64) * BIN_MS


def _find_closest(sorted_arr, value):
    """Find the closest value in a sorted array. Returns (closest_value, distance)."""
    idx = np.searchsorted(sorted_arr, value)
    best_val = None
    best_dist = float("inf")
    for j in [idx - 1, idx, idx + 1]:
        if 0 <= j < len(sorted_arr):
            d = abs(sorted_arr[j] - value)
            if d < best_dist:
                best_dist = d
                best_val = sorted_arr[j]
    return best_val, best_dist


def compute_ar_metrics(pred_ms, gt_ms):
    """Match predicted events to ground truth and compute metrics.

    Uses absolute ms distance for matching:
      MATCHED:   closest GT/pred within 25ms
      CLOSE:     closest GT/pred within 50ms
      FAR:       closest GT/pred > 100ms (effectively unmatched)

    Two perspectives:
      - Event (GT→Pred): for each GT event, how close is nearest prediction?
      - Pred (Pred→GT): for each prediction, how close is nearest GT event?
        Preds with no nearby GT are hallucinations.

    Returns dict with detailed metrics.
    """
    if len(pred_ms) == 0:
        return {
            "n_pred": 0, "n_gt": len(gt_ms),
            "event_matched": 0, "event_close": 0, "event_far": len(gt_ms),
            "event_matched_rate": 0.0, "event_close_rate": 0.0, "event_far_rate": 1.0,
            "pred_matched": 0, "pred_close": 0, "pred_far": 0,
            "hallucination_rate": 1.0,
            "gt_errors_ms": [], "pred_errors_ms": [],
        }

    pred_sorted = np.sort(pred_ms)
    gt_sorted = np.sort(gt_ms)

    # For each GT event, find closest prediction
    gt_errors_ms = []
    for gt in gt_sorted:
        _, dist = _find_closest(pred_sorted, gt)
        gt_errors_ms.append(dist)
    gt_errors_ms = np.array(gt_errors_ms)

    event_matched = int((gt_errors_ms <= 25).sum())
    event_close = int((gt_errors_ms <= 50).sum())
    event_far = int((gt_errors_ms > 100).sum())

    # For each predicted event, find closest GT
    pred_errors_ms = []
    for p in pred_sorted:
        _, dist = _find_closest(gt_sorted, p)
        pred_errors_ms.append(dist)
    pred_errors_ms = np.array(pred_errors_ms)

    pred_matched = int((pred_errors_ms <= 25).sum())
    pred_close = int((pred_errors_ms <= 50).sum())
    pred_far = int((pred_errors_ms > 100).sum())

    n_pred = len(pred_sorted)
    n_gt = len(gt_sorted)

    # Density comparison
    if n_pred > 1:
        duration_s = (pred_sorted[-1] - pred_sorted[0]) / 1000.0
        pred_density = n_pred / max(duration_s, 0.1)
    else:
        pred_density = 0.0

    if n_gt > 1:
        gt_duration_s = (gt_sorted[-1] - gt_sorted[0]) / 1000.0
        gt_density = n_gt / max(gt_duration_s, 0.1)
    else:
        gt_density = 0.0

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "event_matched": event_matched,
        "event_close": event_close,
        "event_far": event_far,
        "event_matched_rate": event_matched / max(n_gt, 1),
        "event_close_rate": event_close / max(n_gt, 1),
        "event_far_rate": event_far / max(n_gt, 1),
        "pred_matched": pred_matched,
        "pred_close": pred_close,
        "pred_far": pred_far,
        "hallucination_rate": pred_far / max(n_pred, 1),
        "pred_density": pred_density,
        "gt_density": gt_density,
        "density_ratio": pred_density / max(gt_density, 0.01),
        "gt_errors_ms": gt_errors_ms.tolist(),
        "pred_errors_ms": pred_errors_ms.tolist(),
        "gt_error_mean": float(gt_errors_ms.mean()),
        "gt_error_median": float(np.median(gt_errors_ms)),
        "gt_error_p90": float(np.percentile(gt_errors_ms, 90)),
        "pred_error_mean": float(pred_errors_ms.mean()),
        "pred_error_median": float(np.median(pred_errors_ms)),
    }


def save_graphs(results, output_dir):
    """Generate analysis graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    names = [f"{r['song']['artist'][:15]}\n{r['song']['title'][:15]}" for r in results]
    short_names = [f"{r['song']['artist'][:12]} - {r['song']['title'][:12]}" for r in results]

    # ── 1. Per-song metrics bar chart ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Hit/Good/Miss rates (event perspective)
    ax = axes[0, 0]
    x = np.arange(n)
    matched_rates = [r["metrics"]["event_matched_rate"] for r in results]
    close_rates = [r["metrics"]["event_close_rate"] for r in results]
    far_rates = [r["metrics"]["event_far_rate"] for r in results]
    ax.bar(x, matched_rates, label="Matched (<25ms)", color="#6bc46d")
    ax.bar(x, [c - m for c, m in zip(close_rates, matched_rates)], bottom=matched_rates, label="Close (25-50ms)", color="#b8e6b9")
    ax.bar(x, far_rates, bottom=close_rates, label="Far (>100ms)", color="#eb4528")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_title("Event Catch Rate (GT perspective)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Hallucination rate
    ax = axes[0, 1]
    hall_rates = [r["metrics"]["hallucination_rate"] for r in results]
    colors = ["#eb4528" if h > 0.5 else "#ff9900" if h > 0.3 else "#6bc46d" for h in hall_rates]
    ax.bar(x, hall_rates, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate (pred events with no GT match)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax.legend(fontsize=8)

    # Density: conditioned vs actual vs predicted
    ax = axes[1, 0]
    cond_density = [r["song"]["density_mean"] for r in results]
    gt_density = [r["metrics"]["gt_density"] for r in results]
    pred_density = [r["metrics"]["pred_density"] for r in results]
    w = 0.25
    ax.bar(x - w, cond_density, w, label="Conditioned (input)", color="#4a90d9")
    ax.bar(x, gt_density, w, label="GT actual", color="#6bc46d")
    ax.bar(x + w, pred_density, w, label="Predicted actual", color="#ff9900")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Events/sec")
    ax.set_title("Density: Conditioned vs GT vs Predicted")
    ax.legend(fontsize=8)

    # Density ratio (pred/gt)
    ax = axes[1, 1]
    ratios = [r["metrics"]["density_ratio"] for r in results]
    colors = ["#eb4528" if abs(r - 1) > 0.5 else "#ff9900" if abs(r - 1) > 0.2 else "#6bc46d" for r in ratios]
    ax.bar(x, ratios, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Pred/GT Density Ratio")
    ax.set_title("Density Ratio (1.0 = perfect match)")
    ax.set_ylim(0, max(ratios) * 1.2 if ratios else 2)

    fig.suptitle("Experiment 56: AR Density Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ar_analysis.png"), dpi=150)
    plt.close(fig)

    # ── 2. Error distribution per song ──
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, r in enumerate(results[:10]):
        ax = axes[i]
        gt_errs = r["metrics"]["gt_errors_ms"]
        if gt_errs:
            ax.hist(gt_errs, bins=50, range=(0, 500), color="#4a90d9", alpha=0.7, label="GT→Pred")
        pred_errs = r["metrics"]["pred_errors_ms"]
        if pred_errs:
            ax.hist(pred_errs, bins=50, range=(0, 500), color="#ff9900", alpha=0.5, label="Pred→GT")
        ax.axvline(50, color="green", linestyle="--", alpha=0.7, label="50ms")
        ax.axvline(100, color="red", linestyle="--", alpha=0.7, label="100ms")
        ax.set_title(f"{r['song']['title'][:18]}", fontsize=8)
        ax.set_xlabel("Error (ms)", fontsize=7)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(len(results), 10):
        axes[j].set_visible(False)

    fig.suptitle("Error Distributions (GT-to-Pred and Pred-to-GT)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_distributions.png"), dpi=150)
    plt.close(fig)

    # ── 3. Scatter: density_mean vs metrics ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    densities = [r["song"]["density_mean"] for r in results]

    ax = axes[0]
    ax.scatter(densities, [r["metrics"]["event_close_rate"] for r in results], c="#6bc46d", s=80, edgecolors="black")
    ax.set_xlabel("Conditioned Density (events/sec)")
    ax.set_ylabel("Event Close Rate (<50ms)")
    ax.set_title("Density vs Catch Rate")

    ax = axes[1]
    ax.scatter(densities, hall_rates, c="#eb4528", s=80, edgecolors="black")
    ax.set_xlabel("Conditioned Density (events/sec)")
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Density vs Hallucination")

    ax = axes[2]
    ax.scatter(densities, ratios, c="#4a90d9", s=80, edgecolors="black")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Conditioned Density (events/sec)")
    ax.set_ylabel("Pred/GT Density Ratio")
    ax.set_title("Density vs Density Adherence")

    fig.suptitle("Density Correlation Analysis", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "density_correlation.png"), dpi=150)
    plt.close(fig)

    print(f"\nGraphs saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Exp 56: AR density analysis on val songs")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--n-songs", type=int, default=50, help="Number of val songs to test")
    parser.add_argument("--hop-ms", type=float, default=75, help="Hop on STOP (ms)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: experiment_56/results/)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Load manifest
    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Select songs
    print("Selecting val songs...")
    songs = select_songs(manifest, n=args.n_songs)
    print(f"Selected {len(songs)} songs:")
    for i, s in enumerate(songs):
        print(f"  {i+1:2d}. {s['artist']} - {s['title']} [{s['difficulty']}]")
        print(f"      d_mean={s['density_mean']:.1f}  d_peak={s['density_peak']}  d_std={s['density_std']:.2f}  dur={s['duration_s']:.0f}s  events={s['total_events']}")

    # Run inference
    print(f"\n{'='*70}")
    print("Running AR inference...")
    print(f"{'='*70}")

    csv_dir = os.path.join(output_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    results = []
    for i, song in enumerate(songs):
        print(f"\n[{i+1}/{len(songs)}]")
        csv_path = run_inference(args.checkpoint, song, csv_dir, hop_ms=args.hop_ms)
        if csv_path is None:
            print(f"    SKIPPED (inference failed)")
            continue

        # Load predictions and ground truth
        pred_ms = load_predicted_events(csv_path)
        gt_ms = load_gt_events(song["event_file"])

        # Save GT events alongside predicted CSV
        gt_csv_path = csv_path.replace("_predicted.csv", "_gt.csv")
        with open(gt_csv_path, "w", encoding="utf-8") as f:
            f.write("time_ms,type\n")
            for t in gt_ms:
                f.write(f"{int(t)},gt\n")

        # Compute metrics
        metrics = compute_ar_metrics(pred_ms, gt_ms)
        results.append({"song": song, "metrics": metrics})

        # Print summary
        m = metrics
        print(f"    GT: {m['n_gt']} events  |  Pred: {m['n_pred']} events  |  Ratio: {m['n_pred']/max(m['n_gt'],1):.2f}x")
        print(f"    Matched(<25ms): {m['event_matched_rate']:.1%}  Close(<50ms): {m['event_close_rate']:.1%}  Far(>100ms): {m['event_far_rate']:.1%}")
        print(f"    Hallucination: {m['hallucination_rate']:.1%}  ({m['pred_far']} of {m['n_pred']} preds)")
        print(f"    Density: cond={song['density_mean']:.1f}  gt={m['gt_density']:.1f}  pred={m['pred_density']:.1f}  ratio={m['density_ratio']:.2f}")
        print(f"    GT error: mean={m['gt_error_mean']:.0f}ms  median={m['gt_error_median']:.0f}ms  p90={m['gt_error_p90']:.0f}ms")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Song':>35s} {'d_cond':>6} {'d_pred':>6} {'ratio':>6} {'Match%':>7} {'Close%':>7} {'HALL%':>6} {'#pred':>6} {'#gt':>5}")
    for r in results:
        s = r["song"]; m = r["metrics"]
        name = f"{s['artist'][:15]} - {s['title'][:15]}"
        print(f"{name:>35s} {s['density_mean']:>6.1f} {m['pred_density']:>6.1f} {m['density_ratio']:>6.2f} {m['event_matched_rate']:>5.1%} {m['event_close_rate']:>6.1%} {m['hallucination_rate']:>5.1%} {m['n_pred']:>6d} {m['n_gt']:>5d}")

    # Averages
    if results:
        avg_hit = np.mean([r["metrics"]["event_matched_rate"] for r in results])
        avg_good = np.mean([r["metrics"]["event_close_rate"] for r in results])
        avg_hall = np.mean([r["metrics"]["hallucination_rate"] for r in results])
        avg_ratio = np.mean([r["metrics"]["density_ratio"] for r in results])
        print(f"{'AVERAGE':>35s} {'':>6s} {'':>6s} {avg_ratio:>6.2f} {avg_hit:>5.1%} {avg_good:>6.1%} {avg_hall:>5.1%}")

    # Save results JSON
    results_json = os.path.join(output_dir, "ar_results.json")
    # Strip large arrays for JSON
    save_results = []
    for r in results:
        m = dict(r["metrics"])
        m.pop("gt_errors_ms", None)
        m.pop("pred_errors_ms", None)
        save_results.append({"song": r["song"], "metrics": m})
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_json}")

    # Generate graphs
    save_graphs(results, output_dir)


if __name__ == "__main__":
    main()
