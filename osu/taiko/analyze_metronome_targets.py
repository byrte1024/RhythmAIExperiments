"""Analyze what % of training samples have a target that continues a same-gap streak.

For each sample (ci, ei), the cursor is evt[ei-1], target is evt[ei] - cursor.
The past events are evt[...:ei]. We look at the gaps between consecutive past events
and check if the target gap continues the most recent same-gap streak.

Reports: for streak lengths 1..N, what % of samples continue that streak.
"""

import os
import json
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")

A_BINS = 500
B_BINS = 500
N_CLASSES = 501
MIN_CURSOR_BIN = 6000
BIN_MS = 4.988662131519274
TOLERANCE = 0.05


def main():
    manifest = json.load(open(os.path.join(DS_DIR, "manifest.json")))
    charts = manifest["charts"]

    # Load all events
    evt_dir = os.path.join(DS_DIR, "events")
    all_events = []
    valid_charts = []
    for chart in charts:
        try:
            evt = np.load(os.path.join(evt_dir, chart["event_file"]))
            all_events.append(evt)
            valid_charts.append(chart)
        except FileNotFoundError:
            all_events.append(np.array([], dtype=np.int32))
            valid_charts.append(chart)

    # Build sample index (same as OnsetDataset, subsample=1)
    samples = []
    for ci, evt in enumerate(all_events):
        for ei in range(len(evt)):
            if ei == 0:
                cursor = max(0, int(evt[0]) - B_BINS) if len(evt) > 0 else 0
            else:
                cursor = int(evt[ei - 1])
            if cursor >= MIN_CURSOR_BIN:
                samples.append((ci, ei))
        # STOP sample
        if len(evt) > 0 and int(evt[-1]) >= MIN_CURSOR_BIN:
            samples.append((ci, len(evt)))

    print(f"Total samples (subsample=1): {len(samples)}")

    # For each sample, check if target continues a same-gap streak
    # streak_continues[k] = number of samples where the last k gaps were the same
    #                        AND the target continues it (k+1 same gaps)
    # streak_present[k] = number of samples where context has a streak of k same gaps ending at cursor

    streak_continues = Counter()  # streak length -> count where target continues
    streak_present = Counter()    # streak length -> count where streak exists in context
    total_non_stop = 0
    total_stop = 0

    for idx, (ci, ei) in enumerate(samples):
        if idx % 500000 == 0 and idx > 0:
            print(f"  Processing {idx}/{len(samples)}...")

        evt = all_events[ci]

        # Get target
        if ei == 0:
            cursor = max(0, int(evt[0]) - B_BINS)
        else:
            cursor = int(evt[ei - 1])

        if ei >= len(evt):
            total_stop += 1
            continue  # STOP sample, skip

        target_gap = int(evt[ei]) - cursor
        if target_gap >= B_BINS or target_gap <= 0:
            total_stop += 1
            continue

        total_non_stop += 1

        # Need at least 2 past events to have 1 gap
        if ei < 2:
            continue

        # Compute past gaps (last up to 128 events before cursor)
        past_start = max(0, ei - 128)
        past_events = evt[past_start:ei]  # events before cursor
        past_gaps = np.diff(past_events)  # gaps between consecutive past events

        if len(past_gaps) == 0:
            continue

        # Find streak length ending at the most recent gap
        recent_gap = past_gaps[-1]
        if recent_gap <= 0:
            continue

        streak_len = 1  # at least the most recent gap
        for j in range(len(past_gaps) - 2, -1, -1):
            if recent_gap > 0 and abs(past_gaps[j] - recent_gap) / recent_gap <= TOLERANCE:
                streak_len += 1
            else:
                break

        # Record that this streak exists
        for k in range(1, streak_len + 1):
            streak_present[k] += 1

        # Check if target continues the streak
        if recent_gap > 0 and abs(target_gap - recent_gap) / recent_gap <= TOLERANCE:
            for k in range(1, streak_len + 1):
                streak_continues[k] += 1

    print(f"\n{'='*70}")
    print(f"METRONOME TARGET ANALYSIS (5% tolerance)")
    print(f"{'='*70}")
    print(f"\nTotal samples: {len(samples)}")
    print(f"Non-STOP samples: {total_non_stop}")
    print(f"STOP samples: {total_stop}")

    print(f"\n--- Given a streak of length K in context, does the target continue it? ---")
    print(f"{'Streak':>7}  {'Present':>10}  {'Continues':>10}  {'Rate':>7}  {'% of all':>8}")
    for k in range(1, 65):
        if streak_present[k] == 0:
            break
        pres = streak_present[k]
        cont = streak_continues[k]
        rate = cont / pres * 100
        pct_all = pres / total_non_stop * 100
        print(f"  {k:5d}  {pres:10d}  {cont:10d}  {rate:6.1f}%  {pct_all:7.1f}%")

    # Also report: what % of ALL samples have target matching last gap (streak >= 1)?
    print(f"\n--- Summary ---")
    if streak_present[1] > 0:
        print(f"Samples with at least 1 past gap:    {streak_present[1]:,d} ({streak_present[1]/total_non_stop*100:.1f}%)")
        print(f"  ...where target matches last gap:   {streak_continues[1]:,d} ({streak_continues[1]/total_non_stop*100:.1f}%)")
    if streak_present[8] > 0:
        print(f"Samples with streak >= 8 in context: {streak_present[8]:,d} ({streak_present[8]/total_non_stop*100:.1f}%)")
        print(f"  ...where target continues streak:   {streak_continues[8]:,d} ({streak_continues[8]/total_non_stop*100:.1f}%)")
    if streak_present[16] > 0:
        print(f"Samples with streak >= 16 in context:{streak_present[16]:,d} ({streak_present[16]/total_non_stop*100:.1f}%)")
        print(f"  ...where target continues streak:   {streak_continues[16]:,d} ({streak_continues[16]/total_non_stop*100:.1f}%)")


if __name__ == "__main__":
    main()
