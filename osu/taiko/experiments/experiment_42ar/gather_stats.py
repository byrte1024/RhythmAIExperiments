"""Gather and compare inference stats across all models for experiment 42-AR.

Usage:
    python experiments/experiment_42ar/gather_stats.py
"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

MODELS = ["exp14", "exp35c", "exp42"]


def load_stats(stats_path):
    with open(stats_path, "r") as f:
        return json.load(f)


def main():
    # collect all stats
    all_stats = {}  # model -> [list of stats dicts]
    all_songs = set()

    for model_name in MODELS:
        chart_dir = os.path.join(CHARTS_DIR, model_name)
        all_stats[model_name] = {}

        if not os.path.exists(chart_dir):
            print(f"WARNING: {chart_dir} not found")
            continue

        for f in sorted(os.listdir(chart_dir)):
            if f.endswith("_stats.json"):
                stem = f.replace("_stats.json", "")
                stats = load_stats(os.path.join(chart_dir, f))
                all_stats[model_name][stem] = stats
                all_songs.add(stem)

    all_songs = sorted(all_songs)
    print(f"Songs: {len(all_songs)}")
    print(f"Models: {MODELS}")

    # per-song comparison
    print(f"\n{'='*90}")
    print(f"  PER-SONG COMPARISON")
    print(f"{'='*90}")
    print(f"  {'Song':40s}", end="")
    for m in MODELS:
        print(f"  {m:>12s}", end="")
    print()
    print(f"  {'-'*40}", end="")
    for _ in MODELS:
        print(f"  {'-'*12}", end="")
    print()

    # events count per song
    print(f"\n  Events generated:")
    for song in all_songs:
        short = song[:38]
        row = f"  {short:40s}"
        for m in MODELS:
            s = all_stats[m].get(song)
            if s:
                row += f"  {s.get('total_events', 0):12d}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    # events per second
    print(f"\n  Events per second:")
    for song in all_songs:
        short = song[:38]
        row = f"  {short:40s}"
        for m in MODELS:
            s = all_stats[m].get(song)
            if s:
                eps = s.get('events_per_sec', 0)
                row += f"  {eps:12.1f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    # stop count
    print(f"\n  STOP predictions:")
    for song in all_songs:
        short = song[:38]
        row = f"  {short:40s}"
        for m in MODELS:
            s = all_stats[m].get(song)
            if s:
                row += f"  {s.get('stop_count', 0):12d}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    # inference time
    print(f"\n  Inference time (seconds):")
    for song in all_songs:
        short = song[:38]
        row = f"  {short:40s}"
        for m in MODELS:
            s = all_stats[m].get(song)
            if s:
                row += f"  {s.get('inference_time_s', 0):12.1f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    # aggregate stats
    print(f"\n{'='*90}")
    print(f"  AGGREGATE STATS")
    print(f"{'='*90}")

    agg_metrics = [
        ("Total events", "total_events", "sum"),
        ("Mean events/song", "total_events", "mean"),
        ("Mean events/sec", "events_per_sec", "mean"),
        ("Std events/sec", "events_per_sec", "std"),
        ("Total STOPs", "stop_count", "sum"),
        ("Mean STOPs/song", "stop_count", "mean"),
        ("Mean STOP ratio", "stop_ratio", "mean"),
        ("Mean inference time (s)", "inference_time_s", "mean"),
        ("Total inference time (s)", "inference_time_s", "sum"),
        ("Mean duration (s)", "duration_s", "mean"),
    ]

    print(f"  {'Metric':35s}", end="")
    for m in MODELS:
        print(f"  {m:>12s}", end="")
    print()
    print(f"  {'-'*35}", end="")
    for _ in MODELS:
        print(f"  {'-'*12}", end="")
    print()

    for label, key, agg in agg_metrics:
        row = f"  {label:35s}"
        for m in MODELS:
            vals = [s.get(key, 0) for s in all_stats[m].values() if s]
            if not vals:
                row += f"  {'N/A':>12s}"
                continue
            if agg == "sum":
                row += f"  {sum(vals):12.0f}"
            elif agg == "mean":
                row += f"  {np.mean(vals):12.1f}"
            elif agg == "std":
                row += f"  {np.std(vals):12.1f}"
        print(row)

    # gap statistics (from offset distribution in stats)
    print(f"\n  Gap distribution (from predicted offsets):")
    for m in MODELS:
        all_offsets = []
        for s in all_stats[m].values():
            if s and "offset_distribution" in s:
                od = s["offset_distribution"]
                all_offsets.extend([float(k) for k, v in od.items() for _ in range(v)])
            elif s and "mean_offset" in s:
                pass  # no raw offsets available

        if all_offsets:
            arr = np.array(all_offsets)
            print(f"    {m}: mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
                  f"std={arr.std():.1f}  min={arr.min():.0f}  max={arr.max():.0f}")
        else:
            # try from summary stats
            means = [s.get("mean_offset", 0) for s in all_stats[m].values() if s and "mean_offset" in s]
            if means:
                print(f"    {m}: mean_offset avg={np.mean(means):.1f}")
            else:
                print(f"    {m}: no offset data available")

    # save
    out_path = os.path.join(SCRIPT_DIR, "results", "inference_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
