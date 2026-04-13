"""Generate LLM evaluation prompts for chart quality assessment.

Creates a folder structure with audio files, gap sequences in 20 encodings,
and ready-to-paste prompts for manual testing with GPT-4o, Claude, and Gemini.

Usage:
    python generate_llm_eval.py --songs 53ar --output llm_eval_53ar
    python generate_llm_eval.py --songs 42ar --output llm_eval_42ar
    python generate_llm_eval.py --songs both --output llm_eval_all
"""
import os
import json
import argparse
import shutil
import numpy as np
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_MS = 5.0


def _compress_audio(src_path, dst_path, max_mb=7.0):
    """Compress audio to MP3 under max_mb using ffmpeg."""
    import subprocess

    # get duration
    result = subprocess.run(
        ["ffmpeg", "-i", src_path, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    duration_s = 180  # default estimate
    for line in result.stderr.split("\n"):
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
            try:
                duration_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except (ValueError, IndexError):
                pass

    # target bitrate to fit under max_mb
    target_bits = max_mb * 8 * 1024 * 1024
    target_kbps = int(target_bits / duration_s / 1000)
    target_kbps = max(32, min(target_kbps, 192))  # clamp to reasonable range

    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-b:a", f"{target_kbps}k",
         "-ac", "1", "-ar", "22050", dst_path],
        capture_output=True, check=True
    )


# ──────────────────────────────────────────────
#  Song/chart loading
# ──────────────────────────────────────────────

def load_csv_events_ms(csv_path):
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("time_ms"):
                continue
            parts = line.strip().split(",")
            if parts:
                try:
                    events.append(float(parts[0]))
                except ValueError:
                    continue
    return np.array(events, dtype=np.float64)


def load_53ar_songs():
    """Load 53-AR songs with all model charts."""
    exp_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_53ar")
    charts_dir = os.path.join(exp_dir, "charts")
    audio_dir = os.path.join(exp_dir, "audio")
    models = ["exp14", "exp44", "exp45", "exp53"]

    songs = []
    song_defs = [
        ("01", "arashi_five", "Arashi - Five"),
        ("02", "sakurazaka46_growing_up_train", "Sakurazaka46 - Growing Up Train"),
        ("03", "camellia_denkoh_sekka", "Camellia - Denkoh Sekka"),
        ("04", "redalice_tpazolite_xterfusion", "REDALiCE x t+pazolite - Xterfusion"),
        ("05", "courtney_barnett_stay_in_your_lane", "Courtney Barnett - Stay in Your Lane"),
    ]

    for num, stem, name in song_defs:
        # find audio
        audio_files = [f for f in os.listdir(audio_dir) if f.startswith(f"{num}_")]
        if not audio_files:
            continue
        audio_path = os.path.join(audio_dir, audio_files[0])

        chart_data = {}
        for m in models:
            csv_files = [f for f in os.listdir(charts_dir)
                         if f.startswith(f"{num}_") and f.endswith(f"_{m}.csv")]
            if csv_files:
                events_ms = load_csv_events_ms(os.path.join(charts_dir, csv_files[0]))
                chart_data[m] = events_ms

        if chart_data:
            songs.append({
                "name": name,
                "id": f"53ar_{num}",
                "audio_path": audio_path,
                "charts": chart_data,
                "models": models,
                "human_ranking": "exp45 > exp44 > exp53 > exp14 (global)",
            })

    return songs


def load_42ar_songs():
    """Load 42-AR songs with all model charts."""
    exp_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_42ar")
    charts_dir = os.path.join(exp_dir, "charts")
    audio_dir = os.path.join(exp_dir, "audio")
    models = ["exp14", "exp35c", "exp42"]

    songs = []
    exp14_dir = os.path.join(charts_dir, "exp14")
    song_names = sorted(set(
        f.replace(".csv", "") for f in os.listdir(exp14_dir) if f.endswith(".csv")
    ))[:5]  # first 5 songs

    for sname in song_names:
        audio_files = [f for f in os.listdir(audio_dir) if sname[:20] in f or f.startswith(sname[:15])]
        if not audio_files:
            # try looser match
            for af in os.listdir(audio_dir):
                if sname.split(" - ")[0][:10] in af:
                    audio_files = [af]
                    break
        if not audio_files:
            continue

        audio_path = os.path.join(audio_dir, audio_files[0])
        chart_data = {}
        for m in models:
            csv_path = os.path.join(charts_dir, m, f"{sname}.csv")
            if os.path.exists(csv_path):
                chart_data[m] = load_csv_events_ms(csv_path)

        if chart_data:
            safe = sname.replace(" ", "_")[:30]
            songs.append({
                "name": sname,
                "id": f"42ar_{safe}",
                "audio_path": audio_path,
                "charts": chart_data,
                "models": models,
                "human_ranking": "exp14 > exp42 > exp35c (global)",
            })

    return songs


# ──────────────────────────────────────────────
#  Gap sequence encodings
# ──────────────────────────────────────────────

def compute_bpm_from_gaps(gaps_ms):
    """Estimate BPM from median gap."""
    if len(gaps_ms) == 0:
        return 120
    med = np.median(gaps_ms)
    if med <= 0:
        return 120
    return 60000.0 / med


def encode_raw_ms(events_ms):
    """Encoding 1: Raw gap list in milliseconds."""
    gaps = np.diff(events_ms)
    return "Gap sequence (milliseconds between consecutive notes):\n" + \
           ", ".join(f"{g:.0f}" for g in gaps)


def encode_raw_seconds(events_ms):
    """Encoding 2: Raw onset times in seconds."""
    return "Onset times (seconds from start):\n" + \
           ", ".join(f"{t/1000:.2f}" for t in events_ms)


def encode_gap_ratios(events_ms):
    """Encoding 3: Consecutive gap ratios."""
    gaps = np.diff(events_ms)
    gaps = gaps[gaps > 0]
    if len(gaps) < 2:
        return "Too few gaps for ratio encoding."
    ratios = gaps[1:] / gaps[:-1]
    return "Gap ratios (each gap divided by previous gap, 1.0 = same speed):\n" + \
           ", ".join(f"{r:.2f}" for r in ratios)


def encode_beat_fractions(events_ms):
    """Encoding 4: Gaps as beat fractions (1/4, 1/8, etc.)."""
    gaps = np.diff(events_ms)
    bpm = compute_bpm_from_gaps(gaps)
    beat_ms = 60000.0 / bpm

    def to_fraction(gap):
        ratio = gap / beat_ms
        fracs = [(4, "whole"), (2, "half"), (1, "quarter"), (0.5, "eighth"),
                 (0.25, "sixteenth"), (0.125, "thirty-second")]
        best = min(fracs, key=lambda f: abs(ratio - f[0]))
        if abs(ratio - best[0]) / max(best[0], 0.01) < 0.15:
            return best[1]
        return f"{ratio:.2f}x quarter"

    encoded = [to_fraction(g) for g in gaps]
    return f"Estimated BPM: {bpm:.0f}\nGaps as beat fractions:\n" + ", ".join(encoded)


def encode_rle(events_ms):
    """Encoding 5: Run-length encoded gaps (gap_ms x count)."""
    gaps = np.diff(events_ms)
    runs = []
    if len(gaps) == 0:
        return "No gaps."
    current = round(gaps[0] / 5) * 5  # round to nearest 5ms
    count = 1
    for g in gaps[1:]:
        rounded = round(g / 5) * 5
        if abs(rounded - current) <= 10:  # within 10ms = same
            count += 1
        else:
            runs.append((current, count))
            current = rounded
            count = 1
    runs.append((current, count))
    return "Run-length encoded gaps (gap_ms x repeat_count):\n" + \
           ", ".join(f"{g:.0f}ms x{c}" for g, c in runs)


def encode_visual_rhythm(events_ms):
    """Encoding 6: Visual text-art rhythm bar."""
    gaps = np.diff(events_ms)
    if len(gaps) == 0:
        return "No gaps."
    # each character = 25ms, X = note, . = silence
    lines = []
    resolution_ms = 25
    total_ms = events_ms[-1] - events_ms[0]
    n_chars = min(int(total_ms / resolution_ms), 200)  # cap at 200 chars per line

    # build grid
    grid = ['.'] * (n_chars + 1)
    for t in events_ms:
        idx = int((t - events_ms[0]) / resolution_ms)
        if 0 <= idx < len(grid):
            grid[idx] = 'X'

    # split into 80-char lines
    grid_str = ''.join(grid)
    for i in range(0, len(grid_str), 80):
        time_s = (i * resolution_ms + events_ms[0]) / 1000
        lines.append(f"[{time_s:6.1f}s] {grid_str[i:i+80]}")

    return f"Visual rhythm (X=note, .=silence, each char={resolution_ms}ms):\n" + \
           "\n".join(lines[:30])  # cap at 30 lines


def encode_stats_only(events_ms):
    """Encoding 7: Statistical summary only."""
    gaps = np.diff(events_ms)
    if len(gaps) == 0:
        return "No gaps."
    gaps_pos = gaps[gaps > 0]
    bpm = compute_bpm_from_gaps(gaps_pos)

    # gap distribution
    unique, counts = np.unique(np.round(gaps_pos / 10) * 10, return_counts=True)
    top_3 = sorted(zip(unique, counts), key=lambda x: -x[1])[:3]

    # metronomic streak
    max_streak = 1
    streak = 1
    for i in range(1, len(gaps_pos)):
        if abs(gaps_pos[i] - gaps_pos[i-1]) / max(gaps_pos[i-1], 1) <= 0.05:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1

    duration_s = (events_ms[-1] - events_ms[0]) / 1000
    return (f"Chart statistics:\n"
            f"  Total events: {len(events_ms)}\n"
            f"  Duration: {duration_s:.1f}s\n"
            f"  Density: {len(events_ms)/max(duration_s,0.1):.1f} events/sec\n"
            f"  Estimated BPM: {bpm:.0f}\n"
            f"  Gap mean: {gaps_pos.mean():.0f}ms, median: {np.median(gaps_pos):.0f}ms, std: {gaps_pos.std():.0f}ms\n"
            f"  Gap CV (variety): {gaps_pos.std()/gaps_pos.mean():.3f}\n"
            f"  Top 3 gaps: {', '.join(f'{g:.0f}ms ({c/len(gaps_pos):.0%})' for g,c in top_3)}\n"
            f"  Longest same-gap streak: {max_streak} consecutive\n"
            f"  Gap range: {gaps_pos.min():.0f}ms - {gaps_pos.max():.0f}ms")


def encode_density_curve(events_ms):
    """Encoding 8: Density over time (text histogram)."""
    if len(events_ms) < 2:
        return "Too few events."
    window_ms = 5000  # 5-second windows
    start = events_ms[0]
    end = events_ms[-1]
    lines = []
    t = start
    while t < end:
        n = np.sum((events_ms >= t) & (events_ms < t + window_ms))
        density = n / (window_ms / 1000)
        bar = '#' * int(density * 2)
        lines.append(f"[{t/1000:6.1f}s] {density:5.1f}/s {bar}")
        t += window_ms
    return "Density over time (events/sec in 5-second windows):\n" + "\n".join(lines[:30])


def encode_pattern_annotated(events_ms):
    """Encoding 9: Gap sequence with annotations for metronomic/varied sections."""
    gaps = np.diff(events_ms)
    if len(gaps) < 2:
        return "Too few gaps."
    gaps_pos = gaps[gaps > 0]

    sections = []
    i = 0
    while i < len(gaps_pos):
        # detect metronomic streaks
        streak_start = i
        while i < len(gaps_pos) - 1 and abs(gaps_pos[i+1] - gaps_pos[i]) / max(gaps_pos[i], 1) <= 0.05:
            i += 1
        streak_len = i - streak_start + 1
        if streak_len >= 4:
            t_start = sum(gaps_pos[:streak_start]) / 1000
            sections.append(f"  [{t_start:.1f}s] METRONOMIC: {gaps_pos[streak_start]:.0f}ms repeated {streak_len}x")
        elif streak_len >= 2:
            t_start = sum(gaps_pos[:streak_start]) / 1000
            avg_gap = gaps_pos[streak_start:i+1].mean()
            sections.append(f"  [{t_start:.1f}s] steady: ~{avg_gap:.0f}ms for {streak_len} notes")
        else:
            t_start = sum(gaps_pos[:streak_start]) / 1000
            sections.append(f"  [{t_start:.1f}s] varied: {gaps_pos[i]:.0f}ms")
        i += 1

    return "Pattern-annotated gap sequence:\n" + "\n".join(sections[:50])


def encode_time_grid(events_ms):
    """Encoding 10: Binned time grid (X=onset, .=silence)."""
    if len(events_ms) < 2:
        return "Too few events."
    gaps = np.diff(events_ms)
    bpm = compute_bpm_from_gaps(gaps[gaps > 0])
    beat_ms = 60000.0 / bpm
    sixteenth = beat_ms / 4

    lines = []
    # show first 32 beats
    start = events_ms[0]
    for beat in range(min(32, int((events_ms[-1] - start) / beat_ms))):
        beat_start = start + beat * beat_ms
        cells = []
        for sub in range(4):
            t = beat_start + sub * sixteenth
            # check if any event within ±10ms
            hit = np.any(np.abs(events_ms - t) < 15)
            cells.append("X" if hit else ".")
        lines.append(f"Beat {beat+1:3d}: [{' '.join(cells)}]")

    return f"Time grid (BPM={bpm:.0f}, each beat = 4 sixteenth subdivisions, X=note, .=rest):\n" + \
           "\n".join(lines)


def encode_section_summary(events_ms):
    """Encoding 11: Per-10-second section summary."""
    if len(events_ms) < 2:
        return "Too few events."
    gaps = np.diff(events_ms)
    gaps_pos = gaps[gaps > 0]
    window_ms = 10000
    start = events_ms[0]
    end = events_ms[-1]

    lines = []
    t = start
    while t < end:
        mask = (events_ms >= t) & (events_ms < t + window_ms)
        section_events = events_ms[mask]
        if len(section_events) >= 2:
            sg = np.diff(section_events)
            sg = sg[sg > 0]
            if len(sg) > 0:
                density = len(section_events) / (window_ms / 1000)
                cv = sg.std() / sg.mean() if sg.mean() > 0 else 0
                # metronomic?
                max_s = 1
                s = 1
                for i in range(1, len(sg)):
                    if abs(sg[i] - sg[i-1]) / max(sg[i-1], 1) <= 0.05:
                        s += 1
                        max_s = max(max_s, s)
                    else:
                        s = 1
                metro_pct = max_s / len(sg)
                label = "METRONOMIC" if metro_pct > 0.5 else "varied" if cv > 0.3 else "steady"
                lines.append(f"  [{t/1000:.0f}-{(t+window_ms)/1000:.0f}s] "
                             f"density={density:.1f}/s, variety={cv:.2f}, "
                             f"longest_streak={max_s}, style={label}")
        t += window_ms

    return "Per-section summary (10-second windows):\n" + "\n".join(lines)


def encode_gap_histogram(events_ms):
    """Encoding 12: Gap distribution histogram."""
    gaps = np.diff(events_ms)
    gaps_pos = gaps[gaps > 0]
    if len(gaps_pos) == 0:
        return "No gaps."

    # bin to nearest 10ms
    binned = np.round(gaps_pos / 10) * 10
    unique, counts = np.unique(binned, return_counts=True)
    total = len(gaps_pos)

    lines = []
    for g, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:15]:
        bar = '#' * int(c / total * 100)
        lines.append(f"  {g:6.0f}ms: {c:4d} ({c/total:5.1%}) {bar}")

    return f"Gap distribution (top 15, total {total} gaps):\n" + "\n".join(lines)


def encode_acceleration(events_ms):
    """Encoding 13: Gap changes (acceleration/deceleration)."""
    gaps = np.diff(events_ms)
    if len(gaps) < 3:
        return "Too few gaps."
    gap_diffs = np.diff(gaps)  # positive = slowing down, negative = speeding up

    # summarize in chunks of 10
    lines = []
    chunk = 10
    for i in range(0, min(len(gap_diffs), 200), chunk):
        block = gap_diffs[i:i+chunk]
        avg_acc = block.mean()
        t = events_ms[i] / 1000
        if avg_acc > 5:
            label = "SLOWING"
        elif avg_acc < -5:
            label = "SPEEDING UP"
        else:
            label = "steady"
        lines.append(f"  [{t:.1f}s] avg change: {avg_acc:+.0f}ms ({label})")

    return "Tempo changes (average gap change per 10-note block):\n" + "\n".join(lines[:30])


def encode_musical_shorthand(events_ms):
    """Encoding 14: Musical shorthand notation."""
    gaps = np.diff(events_ms)
    bpm = compute_bpm_from_gaps(gaps[gaps > 0])
    beat_ms = 60000.0 / bpm

    symbols = []
    for g in gaps:
        ratio = g / beat_ms
        if ratio < 0.15:
            symbols.append("♬♬")  # 32nd
        elif ratio < 0.3:
            symbols.append("♬")   # 16th
        elif ratio < 0.6:
            symbols.append("♪")   # 8th
        elif ratio < 1.2:
            symbols.append("♩")   # quarter
        elif ratio < 2.4:
            symbols.append("𝅗𝅥")   # half
        else:
            symbols.append("𝅝")   # whole+

    # group into measures (4 beats)
    measure_len = 4
    lines = []
    beat_count = 0
    current_measure = []
    for s, g in zip(symbols, gaps):
        current_measure.append(s)
        beat_count += g / beat_ms
        if beat_count >= measure_len:
            lines.append(" ".join(current_measure))
            current_measure = []
            beat_count = 0
    if current_measure:
        lines.append(" ".join(current_measure))

    return (f"Musical shorthand (BPM={bpm:.0f}, ♩=quarter, ♪=eighth, ♬=sixteenth):\n" +
            "\n".join(f"  |{line}|" for line in lines[:30]))


def encode_compared_to_grid(events_ms):
    """Encoding 15: How far each note is from the nearest beat grid position."""
    gaps = np.diff(events_ms)
    bpm = compute_bpm_from_gaps(gaps[gaps > 0])
    beat_ms = 60000.0 / bpm
    eighth = beat_ms / 2

    deviations = []
    for t in events_ms:
        nearest_eighth = round(t / eighth) * eighth
        dev = t - nearest_eighth
        deviations.append(dev)

    devs = np.array(deviations)
    on_grid = np.sum(np.abs(devs) < 10)
    slightly_off = np.sum((np.abs(devs) >= 10) & (np.abs(devs) < 25))
    very_off = np.sum(np.abs(devs) >= 25)

    return (f"Grid alignment (BPM={bpm:.0f}, eighth-note grid):\n"
            f"  On grid (<10ms): {on_grid}/{len(events_ms)} ({on_grid/len(events_ms):.0%})\n"
            f"  Slightly off (10-25ms): {slightly_off}/{len(events_ms)} ({slightly_off/len(events_ms):.0%})\n"
            f"  Very off (>25ms): {very_off}/{len(events_ms)} ({very_off/len(events_ms):.0%})\n"
            f"  Mean deviation: {np.mean(np.abs(devs)):.1f}ms\n"
            f"  Deviation std: {np.std(devs):.1f}ms")


def encode_combined_report(events_ms):
    """Encoding 16: Combined structured report (the kitchen sink)."""
    parts = [
        encode_stats_only(events_ms),
        "",
        encode_density_curve(events_ms),
        "",
        encode_section_summary(events_ms),
        "",
        encode_gap_histogram(events_ms),
        "",
        encode_compared_to_grid(events_ms),
    ]
    return "\n".join(parts)


# all encodings
ENCODINGS_TEXT = [
    ("01_raw_gaps_ms", "Raw gap sequence (ms)", encode_raw_ms),
    ("02_onset_times", "Onset times (seconds)", encode_raw_seconds),
    ("03_gap_ratios", "Consecutive gap ratios", encode_gap_ratios),
    ("04_beat_fractions", "Gaps as beat fractions", encode_beat_fractions),
    ("05_run_length", "Run-length encoded gaps", encode_rle),
    ("06_visual_rhythm", "Visual text-art rhythm", encode_visual_rhythm),
    ("07_stats_only", "Statistical summary only", encode_stats_only),
    ("08_density_curve", "Density over time", encode_density_curve),
    ("09_pattern_annotated", "Pattern-annotated sequence", encode_pattern_annotated),
    ("10_time_grid", "Beat-aligned time grid", encode_time_grid),
    ("11_section_summary", "Per-section summary", encode_section_summary),
    ("12_gap_histogram", "Gap distribution histogram", encode_gap_histogram),
    ("13_acceleration", "Tempo acceleration/deceleration", encode_acceleration),
    ("14_musical_shorthand", "Musical shorthand notation", encode_musical_shorthand),
    ("15_grid_alignment", "Grid alignment analysis", encode_compared_to_grid),
    ("16_combined_report", "Combined structured report", encode_combined_report),
]


# ──────────────────────────────────────────────
#  Prompt generation
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are evaluating AI-generated rhythm game charts (osu! taiko mode).

In taiko, notes appear on a timeline synced to music. A good chart:
- Places notes that align with the music's rhythm, beats, and energy
- Has varied patterns (not just the same gap repeating endlessly)
- Matches the song's intensity (dense during chorus, sparse during verse)
- Feels fun and natural to play — not robotic or random
- Avoids "metronomic" behavior (constant same-gap repetition is bad)
- Avoids hallucinations (notes where there's no musical reason for one)

You will be given multiple charts for the SAME song.
Rank them from best to worst. Explain your reasoning briefly."""

TASK_WITH_AUDIO = """Listen to the attached audio file, then evaluate each chart below.
The charts are different AI-generated rhythm game note placements for this song.
Rank them from best (most musical, best aligned with audio) to worst.

Song: {song_name}
"""

TASK_NO_AUDIO = """Evaluate the following AI-generated rhythm game charts.
You cannot hear the audio, but you can assess: pattern variety, consistency,
appropriate density, absence of metronomic behavior, and structural quality.
Rank them from best to worst.

Song: {song_name}
"""


def generate_prompts(song, output_dir, encoding_name, encoding_label, encode_fn):
    """Generate prompt files for one song with one encoding."""
    models = song["models"]
    # shuffle labels so models aren't always in same order
    import random
    rng = random.Random(42)
    labels = list("ABCDEFGH"[:len(models)])
    label_map = dict(zip(labels, models))  # A->exp14, B->exp44, etc.

    chart_sections = []
    for label, model in label_map.items():
        if model not in song["charts"]:
            continue
        events_ms = song["charts"][model]
        encoded = encode_fn(events_ms)
        chart_sections.append(f"\n### Chart {label}\n\n{encoded}")

    charts_text = "\n".join(chart_sections)
    mapping_text = "\n".join(f"  {label} = {model}" for label, model in label_map.items())
    answer_key = f"\nANSWER KEY (do not include in prompt):\n{mapping_text}\nHuman ranking: {song['human_ranking']}\n"

    # with-audio prompt
    with_audio = (
        SYSTEM_PROMPT + "\n\n" +
        TASK_WITH_AUDIO.format(song_name=song["name"]) +
        f"\nChart encoding: {encoding_label}\n" +
        charts_text +
        f"\n\nRank the charts from best to worst (e.g., {' > '.join(reversed(labels))}). Explain briefly."
    )

    # no-audio prompt
    no_audio = (
        SYSTEM_PROMPT + "\n\n" +
        TASK_NO_AUDIO.format(song_name=song["name"]) +
        f"\nChart encoding: {encoding_label}\n" +
        charts_text +
        f"\n\nRank the charts from best to worst (e.g., {' > '.join(reversed(labels))}). Explain briefly."
    )

    enc_dir = os.path.join(output_dir, song["id"], encoding_name)
    os.makedirs(enc_dir, exist_ok=True)

    with open(os.path.join(enc_dir, "prompt_with_audio.txt"), "w", encoding="utf-8") as f:
        f.write(with_audio)
    with open(os.path.join(enc_dir, "prompt_no_audio.txt"), "w", encoding="utf-8") as f:
        f.write(no_audio)
    with open(os.path.join(enc_dir, "answer_key.txt"), "w", encoding="utf-8") as f:
        f.write(answer_key)


def main():
    parser = argparse.ArgumentParser(description="Generate LLM evaluation prompts")
    parser.add_argument("--songs", default="both", choices=["42ar", "53ar", "both"])
    parser.add_argument("--output", default="llm_eval")
    args = parser.parse_args()

    output_dir = os.path.join(SCRIPT_DIR, args.output)
    os.makedirs(output_dir, exist_ok=True)

    songs = []
    if args.songs in ("53ar", "both"):
        songs.extend(load_53ar_songs())
    if args.songs in ("42ar", "both"):
        songs.extend(load_42ar_songs())

    print(f"Loaded {len(songs)} songs")

    for song in songs:
        print(f"\n  {song['name']} ({song['id']})")
        song_dir = os.path.join(output_dir, song["id"])
        os.makedirs(song_dir, exist_ok=True)

        # compress audio to MP3 under 7MB
        if os.path.exists(song["audio_path"]):
            audio_dest = os.path.join(song_dir, "audio.mp3")
            if not os.path.exists(audio_dest):
                _compress_audio(song["audio_path"], audio_dest, max_mb=7.0)
                size_mb = os.path.getsize(audio_dest) / (1024 * 1024)
                print(f"    compressed audio: {size_mb:.1f}MB")
            else:
                size_mb = os.path.getsize(audio_dest) / (1024 * 1024)
                print(f"    audio exists ({size_mb:.1f}MB)")

        # generate all encodings
        for enc_name, enc_label, enc_fn in ENCODINGS_TEXT:
            generate_prompts(song, output_dir, enc_name, enc_label, enc_fn)
        print(f"    generated {len(ENCODINGS_TEXT)} encodings")

    # write index
    index = {
        "songs": [{
            "id": s["id"],
            "name": s["name"],
            "models": s["models"],
            "human_ranking": s["human_ranking"],
            "n_charts": len(s["charts"]),
        } for s in songs],
        "encodings": [{
            "name": n,
            "label": l,
            "for_audio_models": True,
            "for_text_models": True,
        } for n, l, _ in ENCODINGS_TEXT],
        "recommended_tests": {
            "with_audio_models": [
                "Start with 07_stats_only or 16_combined_report — gives the LLM structured info to reason about alongside audio",
                "Try 06_visual_rhythm — visual pattern the LLM can 'see'",
                "Try 01_raw_gaps_ms — raw numbers, baseline for how well the model reads numeric data",
                "Try 14_musical_shorthand — closest to symbolic music notation",
            ],
            "text_only_models": [
                "Start with 16_combined_report — most information",
                "Try 07_stats_only — can it rank from statistics alone?",
                "Try 09_pattern_annotated — highlights metronomic sections",
                "Try 11_section_summary — temporal quality variation",
            ],
            "quick_test": "Use 07_stats_only on one song with audio. If the LLM gets the ranking right, try harder encodings.",
        },
    }

    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nGenerated {len(songs)} songs x {len(ENCODINGS_TEXT)} encodings = {len(songs) * len(ENCODINGS_TEXT)} prompt sets")
    print(f"Output: {output_dir}/")
    print(f"\nQuick start:")
    print(f"  1. Pick a song folder (e.g., {songs[0]['id']}/)")
    print(f"  2. Upload audio.wav to an audio-capable model")
    print(f"  3. Paste prompt from 07_stats_only/prompt_with_audio.txt")
    print(f"  4. Compare LLM ranking to answer_key.txt")


if __name__ == "__main__":
    main()
