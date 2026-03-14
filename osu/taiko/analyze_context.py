"""Experiment 27-B: Analyze context pattern repetition from saved predictions.

Loads a .npz file from run_predictions.py and checks how many misses could
be fixed by detecting and continuing repeating gap patterns from context.

Usage:
    python analyze_context.py runs/detect_experiment_27/predictions_epoch_008_sub10.npz
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from detection_train import N_CLASSES

MIN_PATTERN_LEN = 4


def is_hit(pred, target):
    """Check if pred is a HIT for target (≤3% or ±1 frame)."""
    if target == N_CLASSES - 1:
        return pred == target
    frame_err = abs(pred - target)
    pct_err = abs((pred + 1) / (target + 1) - 1.0)
    return pct_err <= 0.03 or frame_err <= 1


def is_miss(pred, target):
    """Check if pred is a MISS (>20% error)."""
    if target == N_CLASSES - 1:
        return pred != target
    pct_err = abs((pred + 1) / (target + 1) - 1.0)
    return pct_err > 0.20


def gaps_match(a, b):
    """Check if two gap values match (within HIT tolerance: ≤3% or ±1)."""
    if a == 0 and b == 0:
        return True
    frame_err = abs(a - b)
    pct_err = abs((a + 1) / (b + 1) - 1.0)
    return pct_err <= 0.03 or frame_err <= 1


def find_pattern_prediction(gaps):
    """Find repeating patterns in gap sequence and predict the next gap.

    gaps: array of gap values in chronological order, ending at cursor.

    Returns (prediction, pattern_len, n_repeats) or (None, 0, 0).

    For each candidate length L (MIN_PATTERN_LEN to N//2):
      - pattern = last L gaps
      - verify the L gaps before them match (≥80%)
      - count how many full cycles back the pattern holds
      - prediction = pattern[0] (next in cycle)
      - prefer patterns with more repetitions
    """
    n = len(gaps)
    if n < MIN_PATTERN_LEN * 2:
        return None, 0, 0

    best_pred = None
    best_len = 0
    best_repeats = 0

    for L in range(MIN_PATTERN_LEN, n // 2 + 1):
        if n < 2 * L:
            continue

        pattern = gaps[n - L: n]
        prev_cycle = gaps[n - 2 * L: n - L]

        matches = sum(1 for a, b in zip(pattern, prev_cycle) if gaps_match(a, b))
        if matches < L * 0.8:
            continue

        repeats = 1
        for r in range(2, n // L):
            start = n - (r + 1) * L
            if start < 0:
                break
            cycle = gaps[start: start + L]
            cycle_matches = sum(1 for a, b in zip(pattern, cycle) if gaps_match(a, b))
            if cycle_matches >= L * 0.8:
                repeats += 1
            else:
                break

        pred = int(pattern[0])

        if repeats > best_repeats or (repeats == best_repeats and L > best_len):
            best_pred = pred
            best_len = L
            best_repeats = repeats

    return best_pred, best_len, best_repeats


def extract_gaps(event_offsets, event_mask):
    """Extract chronological gap sequence from event offsets and mask."""
    valid = ~event_mask
    valid_offsets = event_offsets[valid]

    if len(valid_offsets) < 2:
        return np.array([])

    sorted_offsets = np.sort(valid_offsets)
    gaps = np.diff(sorted_offsets).astype(np.float64)

    # gap from last event to cursor
    last_to_cursor = -sorted_offsets[-1] if sorted_offsets[-1] < 0 else 0
    if last_to_cursor > 0:
        gaps = np.append(gaps, float(last_to_cursor))

    return gaps[gaps > 0]


def main():
    parser = argparse.ArgumentParser(description="Analyze context patterns from saved predictions")
    parser.add_argument("predictions", help="Path to .npz from run_predictions.py")
    args = parser.parse_args()

    data = np.load(args.predictions)
    targets = data["targets"]
    preds = data["preds"]
    event_offsets = data["event_offsets"]
    event_masks = data["event_masks"]

    n_total = len(targets)
    stop = N_CLASSES - 1

    print(f"Loaded {n_total} samples from {args.predictions}")
    print(f"Analyzing patterns (min length: {MIN_PATTERN_LEN})...\n")

    # counters
    n_hit = 0
    n_miss = 0
    n_other = 0
    n_stop_target = 0

    # pattern analysis on all non-STOP samples with sufficient context
    pattern_found = 0
    pattern_correct = 0
    pattern_correct_on_hit = 0
    pattern_correct_on_miss = 0
    pattern_wrong_on_hit = 0
    pattern_wrong_on_miss = 0

    miss_no_pattern = 0
    miss_insufficient_context = 0

    miss_details = []
    no_pattern_details = []

    for i in tqdm(range(n_total), desc="Pattern analysis"):
        t = targets[i]
        p = preds[i]

        sample_is_hit = is_hit(p, t)
        sample_is_miss = is_miss(p, t)

        if t == stop:
            n_stop_target += 1
            if sample_is_hit:
                n_hit += 1
            elif sample_is_miss:
                n_miss += 1
            else:
                n_other += 1
            continue

        if sample_is_hit:
            n_hit += 1
        elif sample_is_miss:
            n_miss += 1
        else:
            n_other += 1

        # extract gaps
        gaps = extract_gaps(event_offsets[i], event_masks[i])

        if len(gaps) < MIN_PATTERN_LEN * 2:
            if sample_is_miss:
                miss_insufficient_context += 1
            continue

        # find pattern
        pattern_pred, pattern_len, n_repeats = find_pattern_prediction(gaps)

        if pattern_pred is None:
            if sample_is_miss:
                miss_no_pattern += 1
                no_pattern_details.append({
                    "idx": i, "target": int(t), "pred": int(p),
                    "n_gaps": len(gaps),
                    "all_gaps": [int(g) for g in gaps],
                    "has_pattern": False,
                })
            continue

        pattern_found += 1
        pattern_is_hit = is_hit(pattern_pred, int(t))

        if pattern_is_hit:
            pattern_correct += 1
            if sample_is_hit:
                pattern_correct_on_hit += 1
            elif sample_is_miss:
                pattern_correct_on_miss += 1
        else:
            if sample_is_hit:
                pattern_wrong_on_hit += 1
            elif sample_is_miss:
                pattern_wrong_on_miss += 1

        if sample_is_miss:
            miss_details.append({
                "idx": i, "target": int(t), "pred": int(p),
                "pattern_pred": pattern_pred, "pattern_len": pattern_len,
                "pattern_repeats": n_repeats,
                "pattern_is_hit": pattern_is_hit,
                "n_gaps": len(gaps),
                "all_gaps": [int(g) for g in gaps],
                "has_pattern": True,
            })

    # results
    print(f"\n{'='*70}")
    print(f"  CONTEXT PATTERN ANALYSIS (min pattern length: {MIN_PATTERN_LEN})")
    print(f"{'='*70}")
    print(f"  Total samples:     {n_total}")
    print(f"  Non-STOP samples:  {n_total - n_stop_target}")
    print(f"  HIT:               {n_hit} ({n_hit/n_total:.1%})")
    print(f"  GOOD (not HIT):    {n_other} ({n_other/n_total:.1%})")
    print(f"  MISS:              {n_miss} ({n_miss/n_total:.1%})")
    print()
    print(f"  Pattern detection (non-STOP, sufficient context):")
    print(f"    Pattern found:     {pattern_found}")
    print(f"    Pattern correct:   {pattern_correct} ({pattern_correct/max(1,pattern_found):.1%})")
    print()
    print(f"  Pattern vs Model (when pattern found):")
    print(f"    Both correct:                {pattern_correct_on_hit}")
    print(f"    Pattern fixes model miss:    {pattern_correct_on_miss}")
    print(f"    Pattern wrong, model right:  {pattern_wrong_on_hit}")
    print(f"    Both wrong:                  {pattern_wrong_on_miss}")
    print()

    # miss breakdown
    miss_with_detail = len(miss_details)
    miss_pattern_fixes = sum(1 for d in miss_details if d["pattern_is_hit"])
    miss_pattern_fails = sum(1 for d in miss_details if not d["pattern_is_hit"])
    print(f"  MISS breakdown ({n_miss} total):")
    print(f"    Pattern found & correct:    {miss_pattern_fixes} ({miss_pattern_fixes/max(1,n_miss):.1%})")
    print(f"    Pattern found & wrong:      {miss_pattern_fails} ({miss_pattern_fails/max(1,n_miss):.1%})")
    print(f"    No pattern detected:        {miss_no_pattern} ({miss_no_pattern/max(1,n_miss):.1%})")
    print(f"    Insufficient context:       {miss_insufficient_context} ({miss_insufficient_context/max(1,n_miss):.1%})")
    n_stop_miss = n_miss - miss_with_detail - miss_no_pattern - miss_insufficient_context
    print(f"    STOP target misses:         {n_stop_miss}")
    print()

    # theoretical improvement
    new_hit = n_hit + pattern_correct_on_miss
    print(f"  Theoretical HIT with perfect pattern oracle:")
    print(f"    Current:     {n_hit/n_total:.1%}")
    print(f"    Potential:   {new_hit/n_total:.1%} (+{pattern_correct_on_miss/n_total:.1%})")
    print()

    # pattern risk: how often would pattern HURT if we always used it?
    if pattern_found > 0:
        print(f"  Pattern risk (if always used when found):")
        print(f"    Would break existing HITs: {pattern_wrong_on_hit} "
              f"({pattern_wrong_on_hit/max(1,pattern_found):.1%} of pattern predictions)")
        net = pattern_correct_on_miss - pattern_wrong_on_hit
        print(f"    Net gain: {net:+d} samples ({net/n_total:+.1%})")
        print()

    # analyze fixing patterns
    if miss_details:
        fixing = [d for d in miss_details if d["pattern_is_hit"]]
        if fixing:
            plens = [d["pattern_len"] for d in fixing]
            prepeats = [d["pattern_repeats"] for d in fixing]
            print(f"  Fixing patterns (n={len(fixing)}):")
            print(f"    Pattern length: mean={np.mean(plens):.1f}, "
                  f"median={np.median(plens):.0f}, "
                  f"min={min(plens)}, max={max(plens)}")
            print(f"    Repeats: mean={np.mean(prepeats):.1f}, "
                  f"median={np.median(prepeats):.0f}")
            len_counts = Counter(plens)
            print(f"    Length distribution: {dict(sorted(len_counts.items()))}")

    # show 20 examples where pattern would fix a miss
    if miss_details:
        fixing = [d for d in miss_details if d["pattern_is_hit"]]
        # pick diverse examples: sample across different pattern lengths
        rng = np.random.default_rng(42)
        if len(fixing) > 20:
            examples = list(rng.choice(fixing, size=20, replace=False))
        else:
            examples = fixing

        if examples:
            print(f"  {'='*70}")
            print(f"  EXAMPLES: Pattern fixes model miss ({len(examples)} shown)")
            print(f"  {'='*70}")
            for j, ex in enumerate(examples):
                gaps = ex["all_gaps"]
                plen = ex["pattern_len"]
                pattern = gaps[-plen:]
                print(f"\n  [{j+1}] target={ex['target']}  model_pred={ex['pred']}  "
                      f"pattern_pred={ex['pattern_pred']}  "
                      f"(pattern len={plen}, repeats={ex['pattern_repeats']}x)")
                # show gap sequence with pattern highlighted
                # mark pattern boundaries with |
                n = len(gaps)
                # show last 3 pattern cycles (or all gaps if shorter)
                show_from = max(0, n - plen * 3)
                shown_gaps = gaps[show_from:]
                # format: group into pattern-length chunks from the end
                chunks = []
                for k in range(len(shown_gaps), 0, -plen):
                    start = max(0, k - plen)
                    chunk = shown_gaps[start:k]
                    chunks.insert(0, chunk)
                gap_str = " | ".join(" ".join(str(g) for g in c) for c in chunks)
                if show_from > 0:
                    gap_str = "... " + gap_str
                print(f"    gaps: {gap_str} | ? ")
                print(f"    pattern: [{' '.join(str(g) for g in pattern)}] → next={pattern[0]}")
            print()

    # show 20 examples where NO pattern was detected (miss with context)
    if no_pattern_details:
        rng2 = np.random.default_rng(123)
        if len(no_pattern_details) > 20:
            np_examples = list(rng2.choice(no_pattern_details, size=20, replace=False))
        else:
            np_examples = no_pattern_details

        print(f"  {'='*70}")
        print(f"  EXAMPLES: No pattern detected — model miss ({len(np_examples)} shown)")
        print(f"  {'='*70}")
        for j, ex in enumerate(np_examples):
            gaps = ex["all_gaps"]
            # show last 20 gaps for readability
            shown = gaps[-20:]
            gap_str = " ".join(str(g) for g in shown)
            if len(gaps) > 20:
                gap_str = "... " + gap_str
            # check: does the target value even appear in the gaps?
            target_in_gaps = any(is_hit(g, ex["target"]) for g in gaps)
            print(f"\n  [{j+1}] target={ex['target']}  model_pred={ex['pred']}  "
                  f"n_gaps={ex['n_gaps']}  target_in_context={'YES' if target_in_gaps else 'NO'}")
            print(f"    gaps: {gap_str} | ?")
            if target_in_gaps:
                matching = [g for g in gaps if is_hit(g, ex["target"])]
                print(f"    (target {ex['target']} appears as gap: {matching[:5]})")
        print()

    # save results
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_27b")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "predictions_file": args.predictions,
        "min_pattern_len": MIN_PATTERN_LEN,
        "n_total": n_total,
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_other": n_other,
        "n_stop_target": n_stop_target,
        "pattern_found": pattern_found,
        "pattern_correct": pattern_correct,
        "pattern_correct_on_hit": pattern_correct_on_hit,
        "pattern_correct_on_miss": pattern_correct_on_miss,
        "pattern_wrong_on_hit": pattern_wrong_on_hit,
        "pattern_wrong_on_miss": pattern_wrong_on_miss,
        "miss_no_pattern": miss_no_pattern,
        "miss_insufficient_context": miss_insufficient_context,
        "current_hit_rate": n_hit / n_total,
        "potential_hit_rate": new_hit / n_total,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out_dir}/results.json")

    with open(os.path.join(out_dir, "miss_details.json"), "w") as f:
        json.dump(miss_details, f)
    print(f"  Miss details saved ({len(miss_details)} entries)")


if __name__ == "__main__":
    main()
