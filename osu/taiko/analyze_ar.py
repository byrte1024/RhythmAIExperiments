"""Analyze AR-generated CSVs against ground truth.

Computes GT matching, TaikoNation metrics, and pattern variety metrics.
Works on any directory containing songs.json + csvs/<regime>/*.csv.

Usage:
    cd osu/taiko
    python analyze_ar.py experiment_62 detect_experiment_62_best
    python analyze_ar.py experiment_62 detect_experiment_58_best
    python analyze_ar.py --dir path/to/any/ar_output
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
BIN_MS = 4.9887
TN_STEP_MS = 23


# ═══════════════════════════════════════════════════════════════
#  CSV / GT loading
# ═══════════════════════════════════════════════════════════════

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
#  GT matching metrics
# ═══════════════════════════════════════════════════════════════

def _find_closest(sorted_arr, value):
    idx = np.searchsorted(sorted_arr, value)
    best = float("inf")
    for j in [idx - 1, idx, idx + 1]:
        if 0 <= j < len(sorted_arr):
            d = abs(sorted_arr[j] - value)
            if d < best:
                best = d
    return best


def compute_gt_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0 or len(gt_ms) == 0:
        return None

    pred_sorted = np.sort(pred_ms)
    gt_sorted = np.sort(gt_ms)

    gt_errors = np.array([_find_closest(pred_sorted, gt) for gt in gt_sorted])
    pred_errors = np.array([_find_closest(gt_sorted, p) for p in pred_sorted])

    n_pred, n_gt = len(pred_sorted), len(gt_sorted)

    pred_density = n_pred / max((pred_sorted[-1] - pred_sorted[0]) / 1000.0, 0.1) if n_pred > 1 else 0.0
    gt_density = n_gt / max((gt_sorted[-1] - gt_sorted[0]) / 1000.0, 0.1) if n_gt > 1 else 0.0

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
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


# ═══════════════════════════════════════════════════════════════
#  TaikoNation metrics
# ═══════════════════════════════════════════════════════════════

def events_ms_to_binary(events_ms, step_ms=TN_STEP_MS):
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


def tn_dc_rand(chart, rng):
    noise = rng.integers(low=0, high=2, size=len(chart))
    if len(chart) == 0:
        return 0.0
    return float((chart == noise).sum() / len(chart) * 100)


def tn_dc_human(ai_chart, human_chart):
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
    return float((ai_chart[start:limit] == human_chart[start:limit]).sum() / total * 100)


def tn_oc_human(ai_chart, human_chart, buffer=1):
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
            matched = False
            for b in range(-buffer, buffer + 1):
                j = i + b
                if 0 <= j < limit and human_chart[j] == 1:
                    matched = True
                    break
            if matched:
                similarity += 1
        elif human_chart[i] == 0:
            similarity += 1
    return float(similarity / total * 100)


def tn_over_pspace(chart, scale=8):
    patterns = set()
    last_ind = len(chart) - scale + 1
    if last_ind <= 0:
        return 0.0
    for i in range(last_ind):
        patterns.add(tuple(chart[i:i + scale]))
    return float(len(patterns) / 2**scale * 100)


def tn_hi_pspace(ai_chart, human_chart, scale=8):
    ai_patterns = set()
    human_patterns = set()
    for i in range(len(ai_chart) - scale + 1):
        ai_patterns.add(tuple(ai_chart[i:i + scale]))
    for i in range(len(human_chart) - scale + 1):
        human_patterns.add(tuple(human_chart[i:i + scale]))
    if len(human_patterns) == 0:
        return 0.0
    return float(len(ai_patterns & human_patterns) / len(human_patterns) * 100)


def compute_tn_metrics(pred_ms, gt_ms, rng):
    pred_binary = events_ms_to_binary(pred_ms)
    gt_binary = events_ms_to_binary(gt_ms)
    if len(pred_binary) < 16 or len(gt_binary) < 16:
        return None

    max_len = max(len(pred_binary), len(gt_binary))
    pred_padded = np.zeros(max_len, dtype=np.int32)
    gt_padded = np.zeros(max_len, dtype=np.int32)
    pred_padded[:len(pred_binary)] = pred_binary
    gt_padded[:len(gt_binary)] = gt_binary

    return {
        "over_pspace": tn_over_pspace(pred_padded),
        "hi_pspace": tn_hi_pspace(pred_padded, gt_padded),
        "dc_human": tn_dc_human(pred_padded, gt_padded),
        "oc_human": tn_oc_human(pred_padded, gt_padded),
        "dc_rand": tn_dc_rand(pred_padded, rng),
    }


# ═══════════════════════════════════════════════════════════════
#  Pattern variety metrics
# ═══════════════════════════════════════════════════════════════

def compute_pattern_metrics(events_ms):
    if len(events_ms) < 3:
        return None

    gaps = np.diff(events_ms)
    gaps = gaps[gaps > 0]
    if len(gaps) < 2:
        return None

    m = {}
    m["n_events"] = len(events_ms)
    m["gap_mean"] = float(gaps.mean())
    m["gap_median"] = float(np.median(gaps))
    m["gap_std"] = float(gaps.std())
    m["gap_cv"] = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0

    # Gap entropy
    gap_bins = (gaps / 10).astype(int)
    _, counts = np.unique(gap_bins, return_counts=True)
    probs = counts / counts.sum()
    m["gap_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # Dominant gap %
    gap_clusters = {}
    for g in gaps:
        matched = False
        for center in gap_clusters:
            if abs(g - center) / max(center, 1) <= 0.05:
                gap_clusters[center] += 1
                matched = True
                break
        if not matched:
            gap_clusters[g] = 1
    sorted_clusters = sorted(gap_clusters.values(), reverse=True)
    m["dominant_gap_pct"] = float(sorted_clusters[0] / len(gaps))

    # Longest metronome streak
    max_streak = 1
    streak = 1
    for i in range(1, len(gaps)):
        if abs(gaps[i] - gaps[i-1]) / max(gaps[i-1], 1) <= 0.05:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    m["max_metro_streak"] = max_streak
    m["max_metro_streak_pct"] = float(max_streak / len(gaps))

    # Density
    duration_s = (events_ms[-1] - events_ms[0]) / 1000.0
    m["density"] = len(events_ms) / max(duration_s, 0.1)

    return m


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze AR output CSVs against ground truth")
    parser.add_argument("experiment", nargs="?", help="Experiment name (e.g. experiment_62)")
    parser.add_argument("label", nargs="?", default=None,
                        help="Checkpoint label (e.g. detect_experiment_58_best)")
    parser.add_argument("--dir", default=None, help="Direct path to ar_eval directory")
    args = parser.parse_args()

    if args.dir:
        ar_dir = args.dir
    elif args.experiment and args.label:
        ar_dir = os.path.join(SCRIPT_DIR, "experiments", args.experiment, "ar_eval", args.label)
    elif args.experiment:
        # List available labels
        ar_base = os.path.join(SCRIPT_DIR, "experiments", args.experiment, "ar_eval")
        if os.path.isdir(ar_base):
            labels = [d for d in os.listdir(ar_base) if os.path.isdir(os.path.join(ar_base, d))]
            if labels:
                print(f"Available labels for {args.experiment}:")
                for l in labels:
                    print(f"  {l}")
                sys.exit(0)
        parser.error("Provide both experiment and label, or use --dir")
    else:
        parser.error("Provide experiment + label, or --dir")

    songs_path = os.path.join(ar_dir, "songs.json")
    if not os.path.exists(songs_path):
        print(f"ERROR: songs.json not found in {ar_dir}")
        print("Run run_ar.py first, or provide a directory with songs.json")
        sys.exit(1)

    with open(songs_path, "r", encoding="utf-8") as f:
        songs = json.load(f)

    # Discover regimes from csvs/ subdirectories
    csvs_dir = os.path.join(ar_dir, "csvs")
    if not os.path.isdir(csvs_dir):
        print(f"ERROR: csvs/ directory not found in {ar_dir}")
        sys.exit(1)

    regimes = [d for d in os.listdir(csvs_dir) if os.path.isdir(os.path.join(csvs_dir, d))]
    if not regimes:
        print(f"ERROR: No regime subdirectories in {csvs_dir}")
        sys.exit(1)

    print(f"Analyzing: {ar_dir}")
    print(f"  Songs: {len(songs)}")
    print(f"  Regimes: {', '.join(regimes)}")
    print()

    all_regime_results = {}

    for regime_name in regimes:
        csv_dir = os.path.join(csvs_dir, regime_name)
        tn_rng = np.random.default_rng(2009000042)

        per_song = []
        gt_all, tn_all, pat_all = [], [], []

        pbar = tqdm(songs, desc=regime_name,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        for song in pbar:
            sname = song["safe_name"]
            csv_path = os.path.join(csv_dir, f"{sname}_predicted.csv")

            if not os.path.exists(csv_path):
                per_song.append({"song": sname, "error": "csv not found"})
                continue

            pred_ms = load_csv_events_ms(csv_path)
            gt_ms = load_gt_events_ms(song["event_file"])

            entry = {"song": sname, "n_pred": len(pred_ms), "n_gt": len(gt_ms)}

            gt = compute_gt_metrics(pred_ms, gt_ms)
            if gt:
                entry["gt"] = gt
                gt_all.append(gt)

            tn = compute_tn_metrics(pred_ms, gt_ms, tn_rng)
            if tn:
                entry["taikonation"] = tn
                tn_all.append(tn)

            pat = compute_pattern_metrics(pred_ms)
            if pat:
                entry["pattern"] = pat
                pat_all.append(pat)

            if gt_all:
                avg_close = np.mean([r["close_rate"] for r in gt_all])
                avg_hall = np.mean([r["hallucination_rate"] for r in gt_all])
                pbar.set_postfix_str(f"close={avg_close:.1%} hall={avg_hall:.1%}")

            per_song.append(entry)

        pbar.close()

        # ── Aggregate ──
        regime_summary = {"regime": regime_name, "n_songs": len(gt_all)}

        print(f"\n  {regime_name} ({len(gt_all)} songs):")

        if gt_all:
            avg = lambda k: float(np.mean([r[k] for r in gt_all]))
            print(f"    Close={avg('close_rate'):.1%}  Far={avg('far_rate'):.1%}  "
                  f"Hall={avg('hallucination_rate'):.1%}  d_ratio={avg('density_ratio'):.2f}  "
                  f"err_med={avg('gt_error_median'):.0f}ms")
            regime_summary["gt"] = {k: avg(k) for k in gt_all[0].keys()}

        if tn_all:
            avg = lambda k: float(np.mean([r[k] for r in tn_all]))
            print(f"    P-Space={avg('over_pspace'):.1f}%  HI-PS={avg('hi_pspace'):.1f}%  "
                  f"DCHuman={avg('dc_human'):.1f}%  OCHuman={avg('oc_human'):.1f}%")
            regime_summary["taikonation"] = {k: avg(k) for k in tn_all[0].keys()}

        if pat_all:
            avg = lambda k: float(np.mean([r[k] for r in pat_all]))
            print(f"    gap_std={avg('gap_std'):.1f}  gap_cv={avg('gap_cv'):.3f}  "
                  f"dom_gap={avg('dominant_gap_pct'):.1%}  metro={avg('max_metro_streak'):.1f}")
            regime_summary["pattern"] = {k: avg(k) for k in pat_all[0].keys()}

        regime_summary["per_song"] = per_song
        all_regime_results[regime_name] = regime_summary

    # Save
    results = {"ar_dir": ar_dir, "regimes": all_regime_results}
    out_path = os.path.join(ar_dir, "ar_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
