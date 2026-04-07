"""Experiment 60: DDC Onset Detector Comparison.

Runs the DDC onset detector (Dance Dance Convolution) on val set songs
and compares against GT osu!taiko events and our models' AR output.

Usage:
    cd osu/taiko
    python experiments/experiment_60/compare_ddc.py
"""

import json
import os
import random
import sys
from collections import defaultdict

import ddc_onset
import librosa
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(TAIKO_DIR, "audio")
BIN_MS = 4.9887

# Our models' results from 59-H for comparison
RESULTS_59H = os.path.join(TAIKO_DIR, "experiments", "experiment_59hb", "results", "gt_results.json")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


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
            "event_file": c["event_file"],
            "duration_s": c["duration_s"],
            "audio_path": audio_path,
        })
    candidates.sort(key=lambda x: x["density_mean"])
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


def load_gt_events_ms(event_file):
    events = np.load(os.path.join(DATASET_DIR, "events", event_file))
    return events.astype(np.float64) * BIN_MS


def _find_closest(sorted_arr, value):
    idx = np.searchsorted(sorted_arr, value)
    best = float("inf")
    for j in [idx - 1, idx, idx + 1]:
        if 0 <= j < len(sorted_arr):
            d = abs(sorted_arr[j] - value)
            if d < best:
                best = d
    return best


def compute_metrics(pred_ms, gt_ms):
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


def save_graphs(threshold_results, best_threshold, best_metrics, our_models, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── 1. DDC threshold sweep ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    thrs = [r["threshold"] for r in threshold_results]

    for mi, (metric, title) in enumerate([
        ("close_rate", "Close Rate (<50ms)"),
        ("hallucination_rate", "Hallucination Rate"),
        ("density_ratio", "Density Ratio (pred/gt)"),
        ("gt_error_median", "GT Error Median (ms)")
    ]):
        ax = axes[mi]
        vals = [r[metric] for r in threshold_results]
        ax.plot(thrs, vals, "o-", color="#4a90d9", linewidth=2, markersize=6)
        ax.axvline(best_threshold, color="red", linestyle="--", alpha=0.5, label=f"Best: {best_threshold}")
        ax.set_xlabel("Threshold")
        ax.set_title(title)
        if "ratio" in metric and "density" in metric:
            ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("DDC Onset Detector: Threshold Sweep on Val Set", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "threshold_sweep.png"), dpi=150)
    plt.close(fig)

    # ── 2. DDC vs our models comparison ──
    if our_models:
        all_models = dict(our_models)
        all_models["DDC"] = best_metrics

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        models = sorted(all_models.keys())
        model_colors = {
            "exp44": "#6bc46d", "exp53": "#00cccc", "exp50b": "#e6a817",
            "exp51": "#999999", "exp55": "#c76dba", "exp58": "#eb4528",
            "DDC": "#4a90d9",
        }

        for mi, (metric, title) in enumerate([
            ("close_rate", "Close Rate (<50ms)"),
            ("hallucination_rate", "Hallucination"),
            ("density_ratio", "Density Ratio"),
            ("gt_error_median", "Error Median (ms)")
        ]):
            ax = axes[mi]
            vals = [all_models[m].get(metric, 0) for m in models]
            colors = [model_colors.get(m, "#999") for m in models]
            bars = ax.bar(models, vals, color=colors)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            if "density" in metric:
                ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
            # highlight DDC bar
            for i, m in enumerate(models):
                if m == "DDC":
                    bars[i].set_edgecolor("black")
                    bars[i].set_linewidth(2)

        fig.suptitle("DDC vs Our Models: GT Matching on Val Set (song_density)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "ddc_vs_ours.png"), dpi=150)
        plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def _save_difficulty_graphs(all_threshold_results, best_per_diff, our_models, output_dir):
    """Save per-difficulty analysis graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff_colors = {
        "BEGINNER": "#4a90d9", "EASY": "#6bc46d", "MEDIUM": "#e6a817",
        "HARD": "#ff9900", "CHALLENGE": "#eb4528",
    }

    # ── 1. Threshold sweep curves per difficulty (overlaid) ──
    metrics_to_plot = [
        ("close_rate", "Close Rate (<50ms)"),
        ("hallucination_rate", "Hallucination Rate"),
        ("density_ratio", "Density Ratio"),
        ("gt_error_median", "GT Error Median (ms)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for mi, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[mi]
        for diff_name, results in all_threshold_results.items():
            if not results:
                continue
            thrs = [r["threshold"] for r in results]
            vals = [r[metric] for r in results]
            ax.plot(thrs, vals, "o-", label=diff_name, color=diff_colors.get(diff_name, "#999"),
                    linewidth=2, markersize=4)
        # Add our exp58 as a horizontal reference
        if our_models and "exp58" in our_models:
            ax.axhline(our_models["exp58"].get(metric, 0), color="black", linestyle="--",
                      alpha=0.5, label="exp58 (ours)")
        ax.set_xlabel("Threshold")
        ax.set_title(title)
        ax.legend(fontsize=7)
        if "ratio" in metric and "density" in metric:
            ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    fig.suptitle("DDC All Difficulties: Threshold Sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "difficulty_sweep.png"), dpi=150)
    plt.close(fig)

    # ── 2. Best operating point per difficulty vs our models ──
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    all_entries = {}
    for diff_name, best in best_per_diff.items():
        all_entries[f"DDC_{diff_name[:4]}"] = best
    if our_models:
        for m in ["exp58", "exp44", "exp53"]:
            if m in our_models:
                all_entries[m] = our_models[m]

    models = list(all_entries.keys())
    model_colors_ext = {
        "DDC_BEGI": "#4a90d9", "DDC_EASY": "#6bc46d", "DDC_MEDI": "#e6a817",
        "DDC_HARD": "#ff9900", "DDC_CHAL": "#eb4528",
        "exp58": "#c76dba", "exp44": "#6bc46d", "exp53": "#00cccc",
    }

    for mi, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[mi]
        vals = [all_entries[m].get(metric, 0) for m in models]
        colors = [model_colors_ext.get(m, "#999") for m in models]
        ax.bar(models, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        if "density" in metric and "ratio" in metric:
            ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
    fig.suptitle("DDC Best Per Difficulty vs Our Models", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "difficulty_vs_ours.png"), dpi=150)
    plt.close(fig)

    # ── 3. Summary table image ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    header = ["Model", "Threshold", "Close%", "Far%", "Hall%", "d_ratio", "err_med", "n_pred"]
    rows = []
    for diff_name in ["BEGINNER", "EASY", "MEDIUM", "HARD", "CHALLENGE"]:
        if diff_name in best_per_diff:
            b = best_per_diff[diff_name]
            rows.append([f"DDC {diff_name}", f"{b.get('threshold', 0):.2f}",
                        f"{b['close_rate']:.1%}", f"{b['far_rate']:.1%}",
                        f"{b['hallucination_rate']:.1%}", f"{b['density_ratio']:.2f}",
                        f"{b['gt_error_median']:.0f}ms", f"{b['n_pred']:.0f}"])
    if our_models:
        for m in ["exp58", "exp44", "exp53"]:
            if m in our_models:
                om = our_models[m]
                rows.append([m, "AR", f"{om['close_rate']:.1%}", f"{om['far_rate']:.1%}",
                            f"{om['hallucination_rate']:.1%}", f"{om['density_ratio']:.2f}",
                            f"{om['gt_error_median']:.0f}ms", f"{om.get('avg_n_pred', 0):.0f}"])
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    ax.set_title("Full Comparison: DDC All Difficulties vs Our Models", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary_table.png"), dpi=150)
    plt.close(fig)

    print(f"Difficulty graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("Selecting songs...")
    songs = select_songs(manifest, n=30)
    print(f"Selected {len(songs)} songs\n")

    # Run DDC on all songs × all difficulties
    difficulties = list(ddc_onset.Difficulty)
    print(f"{'='*70}")
    print(f"Running DDC onset detection ({len(difficulties)} difficulties × {len(songs)} songs)...")
    print(f"{'='*70}")

    # all_ddc_results[difficulty_name] = list of per-song dicts
    all_ddc_results = {d.name: [] for d in difficulties}

    for si, song in enumerate(songs):
        print(f"\n[{si+1}/{len(songs)}] {song['artist']} - {song['title']} (d={song['density_mean']:.1f})")
        try:
            audio, sr = librosa.load(song["audio_path"], sr=44100, mono=True)
            gt_ms = load_gt_events_ms(song["event_file"])

            for diff in difficulties:
                salience = ddc_onset.compute_onset_salience(audio, sr, device=device, difficulty=diff)
                peaks = ddc_onset.find_peaks(salience)
                all_ddc_results[diff.name].append({
                    "song": song,
                    "salience": salience,
                    "peaks": peaks,
                    "gt_ms": gt_ms,
                    "duration_s": len(audio) / sr,
                })
            print(f"    GT: {len(gt_ms)} events, peaks: " +
                  "  ".join(f"{d.name[:4]}={len(all_ddc_results[d.name][-1]['peaks'])}" for d in difficulties))
        except Exception as e:
            print(f"    FAILED: {e}")

    # Threshold sweep per difficulty
    print(f"\n{'='*70}")
    print("Threshold sweep per difficulty...")
    print(f"{'='*70}")

    # all_threshold_results[difficulty_name] = list of threshold result dicts
    all_threshold_results = {}

    for diff in difficulties:
        print(f"\n  --- {diff.name} ---")
        diff_results = []
        for thr in THRESHOLDS:
            all_metrics = []
            for ddc in all_ddc_results[diff.name]:
                above = ddc_onset.threshold_peaks(ddc["salience"], ddc["peaks"], thr)
                above = np.array(above, dtype=np.float64)
                pred_ms = above * (1000.0 / ddc_onset.FRAME_RATE)
                metrics = compute_metrics(pred_ms, ddc["gt_ms"])
                if metrics:
                    all_metrics.append(metrics)

            if all_metrics:
                avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
                avg["threshold"] = thr
                avg["n_songs"] = len(all_metrics)
                diff_results.append(avg)
                print(f"    thr={thr:.2f}: close={avg['close_rate']:.1%}  hall={avg['hallucination_rate']:.1%}  "
                      f"d_ratio={avg['density_ratio']:.2f}  err_med={avg['gt_error_median']:.0f}ms  "
                      f"n_pred={avg['n_pred']:.0f}")
        all_threshold_results[diff.name] = diff_results

    # Find best threshold per difficulty
    best_per_diff = {}
    for diff_name, results in all_threshold_results.items():
        if results:
            best = max(results, key=lambda r: r["close_rate"] - r["hallucination_rate"])
            best_per_diff[diff_name] = best
            print(f"\n  Best {diff_name}: thr={best['threshold']} close={best['close_rate']:.1%} hall={best['hallucination_rate']:.1%}")

    # ── DDC Oracle: per-song, pick difficulty whose output density is closest to GT ──
    print(f"\n{'='*70}")
    print("DDC Oracle (pick difficulty by closest density per song)...")
    print(f"{'='*70}")

    from collections import Counter

    # For each song, at each difficulty+threshold, compute predicted density.
    # Pick the difficulty+threshold whose density is closest to GT density.
    oracle_metrics = []
    oracle_choices = []
    n_songs_oracle = len(all_ddc_results.get("CHALLENGE", []))
    for si in range(n_songs_oracle):
        gt_ms = all_ddc_results["CHALLENGE"][si]["gt_ms"]
        gt_sorted = np.sort(gt_ms)
        if len(gt_sorted) < 2:
            continue
        gt_density = len(gt_sorted) / max((gt_sorted[-1] - gt_sorted[0]) / 1000.0, 0.1)

        best_song_metric = None
        best_density_dist = float("inf")
        best_song_choice = None

        for diff in difficulties:
            if si >= len(all_ddc_results[diff.name]):
                continue
            ddc = all_ddc_results[diff.name][si]
            for thr in THRESHOLDS:
                above = ddc_onset.threshold_peaks(ddc["salience"], ddc["peaks"], thr)
                above = np.array(above, dtype=np.float64)
                pred_ms = above * (1000.0 / ddc_onset.FRAME_RATE)
                if len(pred_ms) < 2:
                    continue
                pred_sorted = np.sort(pred_ms)
                pred_density = len(pred_sorted) / max((pred_sorted[-1] - pred_sorted[0]) / 1000.0, 0.1)
                density_dist = abs(pred_density - gt_density)
                if density_dist < best_density_dist:
                    best_density_dist = density_dist
                    metrics = compute_metrics(pred_ms, gt_ms)
                    if metrics:
                        best_song_metric = metrics
                        best_song_choice = (diff.name, thr)

        if best_song_metric:
            oracle_metrics.append(best_song_metric)
            oracle_choices.append(best_song_choice)

    if oracle_metrics:
        oracle_avg = {k: float(np.mean([m[k] for m in oracle_metrics])) for k in oracle_metrics[0]}
        choice_counts = Counter(c[0] for c in oracle_choices)
        thr_counts = Counter(c[1] for c in oracle_choices)
        print(f"  Oracle: close={oracle_avg['close_rate']:.1%}  hall={oracle_avg['hallucination_rate']:.1%}  "
              f"d_ratio={oracle_avg['density_ratio']:.2f}  err_med={oracle_avg['gt_error_median']:.0f}ms")
        print(f"  Difficulty choices: {dict(sorted(choice_counts.items()))}")
        print(f"  Threshold choices:  {dict(sorted(thr_counts.items()))}")
        best_per_diff["ORACLE"] = oracle_avg
    else:
        oracle_avg = {}

    # Use CHALLENGE as the primary comparison (backward compat)
    threshold_results = all_threshold_results.get("CHALLENGE", [])
    best = best_per_diff.get("CHALLENGE", threshold_results[0] if threshold_results else {})
    best_threshold = best.get("threshold", 0.30)

    # Load our models' results for comparison
    our_models = {}
    if os.path.exists(RESULTS_59H):
        with open(RESULTS_59H, "r") as f:
            data_59h = json.load(f)
        if "song_density" in data_59h:
            our_models = data_59h["song_density"]

    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON: DDC vs Our Models (song_density regime)")
    print(f"{'='*70}")
    print(f"  {'Model':>8s} {'Close%':>7s} {'Far%':>6s} {'Hall%':>6s} {'d_ratio':>8s} {'err_med':>8s} {'p/g':>5s}")

    # DDC at best threshold (CHALLENGE)
    print(f"  {'DDC_CHL':>8s} {best['close_rate']:>6.1%} {best['far_rate']:>5.1%} {best['hallucination_rate']:>5.1%} "
          f"{best['density_ratio']:>7.2f} {best['gt_error_median']:>7.0f}ms {best['pred_gt_ratio']:>5.2f}")
    # DDC Oracle
    if oracle_avg:
        print(f"  {'DDC_ORA':>8s} {oracle_avg['close_rate']:>6.1%} {oracle_avg['far_rate']:>5.1%} {oracle_avg['hallucination_rate']:>5.1%} "
              f"{oracle_avg['density_ratio']:>7.2f} {oracle_avg['gt_error_median']:>7.0f}ms {oracle_avg['pred_gt_ratio']:>5.2f}")

    for model in ["exp44", "exp53", "exp50b", "exp51", "exp55", "exp58"]:
        if model in our_models:
            m = our_models[model]
            print(f"  {model:>8s} {m['close_rate']:>6.1%} {m['far_rate']:>5.1%} {m['hallucination_rate']:>5.1%} "
                  f"{m['density_ratio']:>7.2f} {m['gt_error_median']:>7.0f}ms {m['pred_gt_ratio']:>5.2f}")

    # Save
    save_data = {
        "threshold_sweep_challenge": threshold_results,
        "all_threshold_results": all_threshold_results,
        "best_per_difficulty": {k: v for k, v in best_per_diff.items()},
        "best_threshold": best_threshold,
        "best_metrics": best,
        "n_songs": len(all_ddc_results.get("CHALLENGE", [])),
    }
    with open(os.path.join(output_dir, "ddc_results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    save_graphs(threshold_results, best_threshold, best, our_models, output_dir)

    # ── Additional graphs for all difficulties ──
    _save_difficulty_graphs(all_threshold_results, best_per_diff, our_models, output_dir)


if __name__ == "__main__":
    main()
