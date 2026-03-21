"""Analyze metronome patterns in taiko_v2 dataset.

For each chart, compute gaps between consecutive events and find
streaks of same-gap (within 5% tolerance). Reports:
- Longest streak per chart and overall
- Average streak length
- Distribution of streak lengths
- What fraction of events are in metronome streaks
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

DS_DIR = Path(__file__).parent / "datasets" / "taiko_v2"
BIN_MS = 4.988662131519274  # ms per bin


def find_streaks(gaps, tolerance=0.05):
    """Find streaks of consecutive same-gap values (within tolerance).

    Returns list of (start_idx, length, gap_value) tuples for streaks >= 2.
    """
    if len(gaps) < 2:
        return []

    streaks = []
    streak_start = 0
    streak_gap = gaps[0]
    streak_len = 1

    for i in range(1, len(gaps)):
        # Check if this gap matches the streak gap within tolerance
        if streak_gap > 0 and abs(gaps[i] - streak_gap) / streak_gap <= tolerance:
            streak_len += 1
        else:
            if streak_len >= 2:
                streaks.append((streak_start, streak_len, streak_gap))
            streak_start = i
            streak_gap = gaps[i]
            streak_len = 1

    # Final streak
    if streak_len >= 2:
        streaks.append((streak_start, streak_len, streak_gap))

    return streaks


def main():
    manifest = json.load(open(DS_DIR / "manifest.json"))
    charts = manifest["charts"]

    print(f"Analyzing {len(charts)} charts...\n")

    all_streaks = []  # (chart_id, start, length, gap_ms)
    chart_stats = []  # per-chart summary

    for chart in charts:
        try:
            events = np.load(DS_DIR / "events" / chart["event_file"])
        except FileNotFoundError:
            continue
        if len(events) < 3:
            continue

        gaps = np.diff(events)  # gaps between consecutive events in bins
        streaks = find_streaks(gaps, tolerance=0.05)

        total_events_in_streaks = sum(s[1] + 1 for s in streaks)  # +1 because streak of N gaps = N+1 events

        for start, length, gap in streaks:
            gap_ms = gap * BIN_MS
            all_streaks.append((chart["chart_id"], start, length, gap_ms))

        if streaks:
            longest = max(s[1] for s in streaks)
            chart_stats.append({
                "chart_id": chart["chart_id"],
                "total_events": len(events),
                "num_streaks": len(streaks),
                "longest_streak": longest,
                "events_in_streaks": total_events_in_streaks,
                "streak_pct": total_events_in_streaks / len(events) * 100,
            })

    # Sort all streaks by length
    all_streaks.sort(key=lambda x: x[2], reverse=True)

    # Overall stats
    streak_lengths = [s[2] for s in all_streaks]
    streak_gaps_ms = [s[3] for s in all_streaks]

    print("=" * 70)
    print("METRONOME STREAK ANALYSIS (5% tolerance)")
    print("=" * 70)

    print(f"\nTotal charts analyzed: {len(charts)}")
    print(f"Charts with streaks >= 2: {len(chart_stats)}")
    print(f"Total streaks found: {len(all_streaks)}")

    if not all_streaks:
        print("No streaks found!")
        return

    print(f"\n--- Streak Length Distribution ---")
    length_counts = Counter(streak_lengths)
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        print(f"  Length {length:3d}: {count:6d} streaks")

    print(f"\n--- Summary Stats ---")
    print(f"  Mean streak length:   {np.mean(streak_lengths):.1f} gaps")
    print(f"  Median streak length: {np.median(streak_lengths):.1f} gaps")
    print(f"  Max streak length:    {max(streak_lengths)} gaps")
    print(f"  Mean gap in streaks:  {np.mean(streak_gaps_ms):.1f} ms")
    print(f"  Median gap in streaks:{np.median(streak_gaps_ms):.1f} ms")

    # What % of events are in streaks?
    total_events_all = sum(c["total_events"] for c in chart_stats)
    total_in_streaks = sum(c["events_in_streaks"] for c in chart_stats)
    print(f"\n--- Coverage ---")
    print(f"  Total events (charts with streaks): {total_events_all}")
    print(f"  Events in streaks: {total_in_streaks} ({total_in_streaks/total_events_all*100:.1f}%)")

    # Per-chart streak coverage
    streak_pcts = [c["streak_pct"] for c in chart_stats]
    print(f"  Per-chart streak coverage: mean={np.mean(streak_pcts):.1f}%, median={np.median(streak_pcts):.1f}%, max={max(streak_pcts):.1f}%")

    # Top 20 longest streaks
    print(f"\n--- Top 20 Longest Streaks ---")
    print(f"  {'Length':>6}  {'Gap(ms)':>8}  {'Gap(BPM)':>9}  Chart")
    for chart_id, start, length, gap_ms in all_streaks[:20]:
        bpm = 60000 / gap_ms if gap_ms > 0 else 0
        short_id = chart_id[:60] + "..." if len(chart_id) > 60 else chart_id
        print(f"  {length:6d}  {gap_ms:8.1f}  {bpm:9.1f}  {short_id}")

    # Streak length buckets
    print(f"\n--- Streak Length Buckets ---")
    buckets = [(2, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 999)]
    for lo, hi in buckets:
        count = sum(1 for l in streak_lengths if lo <= l <= hi)
        events = sum(l + 1 for l in streak_lengths if lo <= l <= hi)
        if count > 0:
            print(f"  {lo:3d}-{hi:3d}: {count:6d} streaks, {events:8d} events")

    # Gap distribution for long streaks (10+)
    long_gaps = [g for s, g in zip(streak_lengths, streak_gaps_ms) if s >= 10]
    if long_gaps:
        print(f"\n--- Gap Distribution for Streaks >= 10 ---")
        # Bucket by common BPM ranges
        gap_buckets = Counter()
        for g in long_gaps:
            bpm = 60000 / g if g > 0 else 0
            bucket = round(bpm / 10) * 10  # round to nearest 10 BPM
            gap_buckets[bucket] += 1
        for bpm in sorted(gap_buckets.keys()):
            print(f"  ~{bpm:4d} BPM ({60000/bpm:.0f}ms): {gap_buckets[bpm]} streaks")

    # Charts with highest streak coverage
    chart_stats.sort(key=lambda x: x["streak_pct"], reverse=True)
    print(f"\n--- Top 10 Charts by Streak Coverage ---")
    print(f"  {'Pct':>5}  {'Longest':>7}  {'#Str':>4}  Chart")
    for c in chart_stats[:10]:
        short_id = c["chart_id"][:55] + "..." if len(c["chart_id"]) > 55 else c["chart_id"]
        print(f"  {c['streak_pct']:5.1f}%  {c['longest_streak']:7d}  {c['num_streaks']:4d}  {short_id}")


if __name__ == "__main__":
    main()
