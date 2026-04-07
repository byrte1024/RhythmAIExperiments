"""Experiment 61: TaikoNation Evaluation Metrics.

Converts our AR-generated charts and GT charts to TaikoNation's binary format
(23ms timesteps) and runs their exact 5 evaluation metrics.

Uses charts from exp 59-H (6 models × 30 songs × song_density regime).

Usage:
    cd osu/taiko
    python experiments/experiment_61/taikonation_eval.py
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
CSV_DIR = os.path.join(TAIKO_DIR, "experiments", "experiment_59h", "results", "csvs")
BIN_MS = 4.9887
TN_STEP_MS = 23  # TaikoNation's timestep

MODELS = ["exp44", "exp53", "exp50b", "exp51", "exp55", "exp58"]

# TaikoNation's published results
PUBLISHED = {
    "TaikoNation": {"over_pspace": 21.328, "hi_pspace": 94.117, "dc_human": 74.987, "dc_rand": 50.405},
    "DDC": {"over_pspace": 15.938, "hi_pspace": 83.160, "dc_human": 77.900, "dc_rand": 49.938},
    "Human Taiko": {"over_pspace": 14.453, "dc_rand": 50.170},
}


# ═══════════════════════════════════════════════════════════════
#  Binary conversion
# ═══════════════════════════════════════════════════════════════

def events_ms_to_binary(events_ms, step_ms=TN_STEP_MS):
    """Convert event times (ms) to TaikoNation binary format.

    Returns array of 0s and 1s at step_ms resolution.
    """
    if len(events_ms) == 0:
        return np.array([], dtype=np.int32)

    max_time = int(max(events_ms)) + step_ms
    n_steps = max_time // step_ms + 1
    binary = np.zeros(n_steps, dtype=np.int32)

    for t in events_ms:
        idx = int(t) // step_ms
        if 0 <= idx < n_steps:
            binary[idx] = 1

    return binary


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


# ═══════════════════════════════════════════════════════════════
#  TaikoNation's exact evaluation metrics
# ═══════════════════════════════════════════════════════════════

def dc_rand(chart, rng):
    """DCRand: % similarity to random noise."""
    noise = rng.integers(low=0, high=2, size=len(chart))
    total = len(chart)
    if total == 0:
        return 0.0
    similarity = (chart == noise).sum()
    return (similarity / total) * 100


def dc_human(ai_chart, human_chart):
    """DCHuman: Direct binary comparison at each timestep."""
    limit = min(len(ai_chart), len(human_chart))
    if limit == 0:
        return 0.0

    # Find first note in human chart
    start = 0
    for i in range(limit):
        if human_chart[i] == 1:
            start = i
            break

    total = limit - start
    if total <= 0:
        return 0.0

    similarity = (ai_chart[start:limit] == human_chart[start:limit]).sum()
    return (similarity / total) * 100


def oc_human(ai_chart, human_chart, buffer=1):
    """OCHuman: Like DCHuman but with ±buffer tolerance for hits."""
    limit = min(len(ai_chart), len(human_chart))
    if limit == 0:
        return 0.0

    start = 0
    for i in range(limit):
        if human_chart[i] == 1:
            start = i
            break

    total = limit - start
    if total <= 0:
        return 0.0

    similarity = 0
    for i in range(start, limit):
        if ai_chart[i] == 1:
            # Check if any human note within ±buffer
            matched = False
            for b in range(-buffer, buffer + 1):
                j = i + b
                if 0 <= j < limit and human_chart[j] == 1:
                    matched = True
                    break
            if matched:
                similarity += 1
        elif ai_chart[i] == 0:
            if human_chart[i] == 0:
                similarity += 1

    return (similarity / total) * 100


def over_pspace(chart, scale=8):
    """Over. P-Space: Unique patterns as % of possibility space."""
    patterns = set()
    last_ind = len(chart) - scale + 1
    if last_ind <= 0:
        return 0.0

    for i in range(last_ind):
        chunk = tuple(chart[i:i + scale])
        patterns.add(chunk)

    return (len(patterns) / 2**scale) * 100


def hi_pspace(ai_chart, human_chart, scale=8):
    """HI P-Space: % of human patterns found in AI chart."""
    ai_patterns = set()
    human_patterns = set()

    for i in range(len(ai_chart) - scale + 1):
        ai_patterns.add(tuple(ai_chart[i:i + scale]))

    for i in range(len(human_chart) - scale + 1):
        human_patterns.add(tuple(human_chart[i:i + scale]))

    if len(human_patterns) == 0:
        return 0.0

    overlap = len(ai_patterns.intersection(human_patterns))
    return (overlap / len(human_patterns)) * 100


# ═══════════════════════════════════════════════════════════════
#  Song selection (same as 59-H)
# ═══════════════════════════════════════════════════════════════

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
        candidates.append(c)
    candidates.sort(key=lambda x: x["density_mean"])
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def save_graphs(model_results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["over_pspace", "hi_pspace", "dc_human", "oc_human", "dc_rand"]
    titles = ["Over. P-Space\n(pattern diversity)", "HI P-Space\n(human pattern overlap)",
              "DCHuman\n(direct match)", "OCHuman\n(buffered match)", "DCRand\n(vs noise)"]

    model_colors = {
        "exp44": "#6bc46d", "exp53": "#00cccc", "exp50b": "#e6a817",
        "exp51": "#999999", "exp55": "#c76dba", "exp58": "#eb4528",
        "Human GT": "#333333", "TaikoNation*": "#4a90d9", "DDC*": "#ff9900",
    }

    fig, axes = plt.subplots(1, 5, figsize=(25, 6))

    all_entries = dict(model_results)
    # Add published baselines
    all_entries["TaikoNation*"] = PUBLISHED["TaikoNation"]
    all_entries["DDC*"] = PUBLISHED["DDC"]

    models_to_plot = MODELS + ["Human GT", "TaikoNation*", "DDC*"]

    for mi, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[mi]
        vals = []
        labels = []
        colors = []
        for m in models_to_plot:
            if m in all_entries and metric in all_entries[m]:
                vals.append(all_entries[m][metric])
                labels.append(m)
                colors.append(model_colors.get(m, "#999"))

        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=60, labelsize=7)

        # Highlight published baselines
        for i, label in enumerate(labels):
            if label.endswith("*"):
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(1.5)
                bars[i].set_hatch("//")

    fig.suptitle("TaikoNation Evaluation Metrics: Our Models vs Published Baselines\n(* = from paper, not re-evaluated)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "taikonation_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(2009000042)  # same seed as TaikoNation

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("Selecting songs...")
    songs = select_songs(manifest, n=30)
    print(f"Selected {len(songs)} songs\n")

    # Compute metrics for each model
    model_results = {m: defaultdict(list) for m in MODELS}
    model_results["Human GT"] = defaultdict(list)
    n_processed = 0

    for si, song in enumerate(songs):
        safe_name = f"{song['beatmapset_id']}_{song['artist'][:20]}_{song['title'][:20]}".replace(" ", "_").replace("/", "_")
        for ch in "*?:<>|\"":
            safe_name = safe_name.replace(ch, "")

        # Load GT
        gt_ms = load_gt_events_ms(song["event_file"])
        gt_binary = events_ms_to_binary(gt_ms)

        if len(gt_binary) < 16:
            continue

        # GT self-metrics
        model_results["Human GT"]["over_pspace"].append(over_pspace(gt_binary))
        model_results["Human GT"]["dc_rand"].append(dc_rand(gt_binary, rng))

        # Each model
        for model in MODELS:
            csv_path = os.path.join(CSV_DIR, f"{safe_name}_{model}_song_density_predicted.csv")
            if not os.path.exists(csv_path):
                continue

            pred_ms = load_csv_events_ms(csv_path)
            if len(pred_ms) < 2:
                continue

            pred_binary = events_ms_to_binary(pred_ms)

            # Ensure same length for comparison
            max_len = max(len(pred_binary), len(gt_binary))
            pred_padded = np.zeros(max_len, dtype=np.int32)
            gt_padded = np.zeros(max_len, dtype=np.int32)
            pred_padded[:len(pred_binary)] = pred_binary
            gt_padded[:len(gt_binary)] = gt_binary

            model_results[model]["dc_rand"].append(dc_rand(pred_padded, rng))
            model_results[model]["dc_human"].append(dc_human(pred_padded, gt_padded))
            model_results[model]["oc_human"].append(oc_human(pred_padded, gt_padded))
            model_results[model]["over_pspace"].append(over_pspace(pred_padded))
            model_results[model]["hi_pspace"].append(hi_pspace(pred_padded, gt_padded))

        n_processed += 1
        if (si + 1) % 10 == 0:
            print(f"  Processed {si+1}/{len(songs)} songs...")

    print(f"\nProcessed {n_processed} songs")

    # Average results
    avg_results = {}
    for model, metrics in model_results.items():
        avg_results[model] = {}
        for mk, vals in metrics.items():
            if vals:
                avg_results[model][mk] = float(np.mean(vals))

    # Print results
    print(f"\n{'='*90}")
    print("RESULTS (averaged over 30 songs)")
    print(f"{'='*90}")
    print(f"{'Model':>15s} {'Over.PS':>8s} {'HI PS':>8s} {'DCHuman':>8s} {'OCHuman':>8s} {'DCRand':>8s}")

    for model in MODELS + ["Human GT"]:
        r = avg_results.get(model, {})
        print(f"{model:>15s} {r.get('over_pspace',0):>7.1f}% {r.get('hi_pspace',0):>7.1f}% "
              f"{r.get('dc_human',0):>7.1f}% {r.get('oc_human',0):>7.1f}% {r.get('dc_rand',0):>7.1f}%")

    print(f"\n  --- Published baselines (from TaikoNation paper, different songs) ---")
    for model, r in PUBLISHED.items():
        print(f"{model:>15s} {r.get('over_pspace',0):>7.1f}% {r.get('hi_pspace',0):>7.1f}% "
              f"{r.get('dc_human',0):>7.1f}% {r.get('oc_human',0):>7.1f}% {r.get('dc_rand',0):>7.1f}%")

    # Save
    save_data = {"model_results": avg_results, "published": PUBLISHED, "n_songs": n_processed}
    with open(os.path.join(output_dir, "taikonation_results.json"), "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)

    save_graphs(avg_results, output_dir)


if __name__ == "__main__":
    main()
