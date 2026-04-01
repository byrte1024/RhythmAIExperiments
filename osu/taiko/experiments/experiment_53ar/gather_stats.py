"""Compare inference stats across all 4 models, including metronome behavior."""
import os
import json
import glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

MODELS = ["exp14", "exp44", "exp45", "exp53"]
BIN_MS = 4.988662131519274
METRONOME_WINDOW_MS = 8000  # 8 second window for metronome measurement
TOLERANCE = 0.05


def compute_metronome_over_time(onset_ms_list, window_ms=METRONOME_WINDOW_MS, step_ms=1000):
    """Compute metronome closeness over time using a sliding window.

    Returns list of (time_ms, closeness) where closeness is % of gaps
    in the window that match the dominant gap (within tolerance).
    """
    if len(onset_ms_list) < 4:
        return []

    onsets = np.array(sorted(onset_ms_list))
    max_time = onsets[-1]
    results = []

    for t in range(int(window_ms), int(max_time), step_ms):
        # gather onsets in window
        mask = (onsets >= t - window_ms) & (onsets <= t)
        window_onsets = onsets[mask]
        if len(window_onsets) < 4:
            continue

        gaps = np.diff(window_onsets)
        gaps = gaps[gaps > 0]
        if len(gaps) < 3:
            continue

        # cluster gaps within tolerance to find dominant
        sorted_gaps = np.sort(gaps)
        clusters = []
        cluster_vals = [sorted_gaps[0]]
        for i in range(1, len(sorted_gaps)):
            centroid = np.mean(cluster_vals)
            if centroid > 0 and abs(sorted_gaps[i] - centroid) / centroid <= TOLERANCE:
                cluster_vals.append(sorted_gaps[i])
            else:
                clusters.append((np.mean(cluster_vals), len(cluster_vals)))
                cluster_vals = [sorted_gaps[i]]
        clusters.append((np.mean(cluster_vals), len(cluster_vals)))
        clusters.sort(key=lambda x: x[1], reverse=True)

        dominant_count = clusters[0][1]
        closeness = dominant_count / len(gaps) * 100  # % in peak
        results.append((t, closeness))

    return results


def main():
    # collect stats per model
    model_stats = {m: {"events": [], "events_per_sec": [], "stops": 0, "duration": 0,
                       "metronome_closeness": []} for m in MODELS}

    for csv_path in sorted(glob.glob(os.path.join(CHARTS_DIR, "*.csv"))):
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in MODELS:
            continue
        song_stem, model = parts

        # read CSV to count events
        with open(csv_path, "r") as f:
            lines = f.readlines()

        # first line is header or audio path
        events = []
        for line in lines[1:]:
            parts_csv = line.strip().split(",")
            if len(parts_csv) >= 2:
                try:
                    events.append(float(parts_csv[0]))
                except ValueError:
                    continue

        if not events:
            continue

        duration_s = max(events) / 1000.0 if events else 0
        eps = len(events) / duration_s if duration_s > 0 else 0

        model_stats[model]["events"].append(len(events))
        model_stats[model]["events_per_sec"].append(eps)
        model_stats[model]["duration"] += duration_s

        # compute metronome behavior over time
        met_timeline = compute_metronome_over_time(events)
        if met_timeline:
            closeness_vals = [c for _, c in met_timeline]
            model_stats[model]["metronome_closeness"].extend(closeness_vals)

    # print comparison
    print(f"\n{'='*70}")
    print(f"  INFERENCE STATS COMPARISON (53-AR)")
    print(f"{'='*70}")
    print(f"\n{'Model':12s} {'Total Evt':>10s} {'Mean eps':>10s} {'Std eps':>10s} {'Songs':>6s}")
    print("-" * 50)

    for model in MODELS:
        s = model_stats[model]
        if not s["events"]:
            print(f"{model:12s} {'no data':>10s}")
            continue
        total = sum(s["events"])
        mean_eps = np.mean(s["events_per_sec"])
        std_eps = np.std(s["events_per_sec"])
        n_songs = len(s["events"])
        print(f"{model:12s} {total:10d} {mean_eps:10.1f} {std_eps:10.1f} {n_songs:6d}")

    # per-song comparison
    print(f"\n{'Song':40s}", end="")
    for m in MODELS:
        print(f" {m:>8s}", end="")
    print()
    print("-" * (40 + 9 * len(MODELS)))

    songs = sorted(set(
        stem.rsplit("_", 1)[0]
        for stem in [os.path.splitext(os.path.basename(f))[0]
                     for f in glob.glob(os.path.join(CHARTS_DIR, "*.csv"))]
        if stem.rsplit("_", 1)[1] in MODELS
    ))

    for song in songs:
        print(f"{song[:40]:40s}", end="")
        for model in MODELS:
            csv_path = os.path.join(CHARTS_DIR, f"{song}_{model}.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    n_events = sum(1 for line in f.readlines()[1:] if line.strip())
                print(f" {n_events:8d}", end="")
            else:
                print(f" {'—':>8s}", end="")
        print()

    # metronome behavior analysis
    print(f"\n{'='*70}")
    print(f"  METRONOME BEHAVIOR (8s sliding window, % of gaps matching dominant)")
    print(f"{'='*70}")
    print(f"\n{'Model':12s} {'Mean':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s} {'N':>6s}")
    print("-" * 54)

    for model in MODELS:
        vals = model_stats[model]["metronome_closeness"]
        if not vals:
            print(f"{model:12s} {'no data':>8s}")
            continue
        arr = np.array(vals)
        print(f"{model:12s} {arr.mean():7.1f}% {np.median(arr):7.1f}% {arr.min():7.1f}% {arr.max():7.1f}% {arr.std():7.1f}% {len(arr):6d}")

    # save full stats to JSON
    stats_out = {}
    for model in MODELS:
        s = model_stats[model]
        vals = s["metronome_closeness"]
        stats_out[model] = {
            "total_events": sum(s["events"]) if s["events"] else 0,
            "mean_events_per_sec": float(np.mean(s["events_per_sec"])) if s["events_per_sec"] else 0,
            "std_events_per_sec": float(np.std(s["events_per_sec"])) if s["events_per_sec"] else 0,
            "metronome_mean": float(np.mean(vals)) if vals else 0,
            "metronome_median": float(np.median(vals)) if vals else 0,
            "metronome_std": float(np.std(vals)) if vals else 0,
            "metronome_min": float(np.min(vals)) if vals else 0,
            "metronome_max": float(np.max(vals)) if vals else 0,
        }
    with open(os.path.join(SCRIPT_DIR, "inference_stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nSaved: inference_stats.json")


if __name__ == "__main__":
    main()
