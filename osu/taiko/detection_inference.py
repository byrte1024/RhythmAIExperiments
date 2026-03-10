"""Run onset detection on an audio file, output a CSV like the preprocessed data.

Usage:
  python detection_inference.py --checkpoint checkpoints/taiko_v1/best.pt --audio song.mp3
  python detection_inference.py --checkpoint checkpoints/taiko_v1/best.pt --audio song.mp3 --density-mean 5.0 --density-peak 10
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm
from collections import Counter

from detection_model import OnsetDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# must match training
SAMPLE_RATE = 22050
HOP_LENGTH = 110
N_FFT = 2048
N_MELS = 80
F_MIN = 20.0
F_MAX = 8000.0
BIN_MS = HOP_LENGTH / SAMPLE_RATE * 1000  # exact: ~4.9887ms per mel frame

A_BINS = 500
B_BINS = 500
C_EVENTS = 128
N_CLASSES = 501


def load_audio_mel(audio_path, device):
    """Load audio and compute mel spectrogram on GPU."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0,
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

    wav = torch.from_numpy(y).float().to(device)
    with torch.no_grad():
        mel = amp_to_db(mel_transform(wav))  # (n_mels, T)

    return mel.cpu().numpy().astype(np.float32), len(y) / SAMPLE_RATE


def extract_mel_window(mel, cursor):
    """Extract padded mel window around cursor."""
    n_mels, total_frames = mel.shape
    start = cursor - A_BINS
    end = cursor + B_BINS

    pad_left = max(0, -start)
    pad_right = max(0, end - total_frames)
    s = max(0, start)
    e = min(total_frames, end)
    window = mel[:, s:e]

    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")

    return window


@torch.no_grad()
def run_inference(model, mel, conditioning, device, hop_bins=20, max_events=10000):
    """Autoregressive inference: predict events one at a time.

    Returns (events, run_stats) where run_stats has detailed inference metrics.
    """
    model.eval()
    total_frames = mel.shape[1]
    events = []  # list of bin positions
    cursor = 0

    # Tracking stats
    stop_count = 0
    total_calls = 0
    stop_positions = []  # cursor positions where STOP was predicted
    event_offsets = []  # raw predicted offsets (before adding cursor)
    cursor_history = []  # (call_idx, cursor_pos, prediction)

    cond_tensor = torch.tensor(conditioning, dtype=torch.float32).unsqueeze(0).to(device)
    duration_s = total_frames * BIN_MS / 1000
    pbar = tqdm(total=total_frames, desc="Inference", unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]")

    t_start = time.perf_counter()

    for _ in range(max_events):
        if cursor >= total_frames:
            break
        pbar.n = min(cursor, total_frames)
        pbar.set_postfix_str(f"{cursor*BIN_MS/1000:.1f}s/{duration_s:.1f}s, {len(events)} events, {stop_count} stops")
        pbar.refresh()

        total_calls += 1

        # mel window
        mel_window = extract_mel_window(mel, cursor)
        mel_tensor = torch.from_numpy(mel_window).unsqueeze(0).to(device)

        # past events as offsets from cursor
        if len(events) > 0:
            past = np.array(events[-C_EVENTS:], dtype=np.int64) - cursor
            n_past = len(past)
        else:
            past = np.array([], dtype=np.int64)
            n_past = 0

        evt_offsets = np.zeros(C_EVENTS, dtype=np.int64)
        evt_mask = np.ones(C_EVENTS, dtype=bool)
        if n_past > 0:
            evt_offsets[-n_past:] = past
            evt_mask[-n_past:] = False

        evt_tensor = torch.from_numpy(evt_offsets).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(evt_mask).unsqueeze(0).to(device)

        logits, _audio_logits, _context_logits = model(mel_tensor, evt_tensor, mask_tensor, cond_tensor)
        pred = logits.argmax(dim=1).item()

        cursor_history.append((total_calls, cursor, pred))

        if pred == N_CLASSES - 1:  # STOP
            stop_count += 1
            stop_positions.append(cursor)
            cursor += hop_bins
            continue

        # predicted bin offset from cursor
        event_offsets.append(pred)
        event_bin = cursor + pred
        events.append(event_bin)
        cursor = event_bin  # move cursor to this event

    t_end = time.perf_counter()
    inference_time = t_end - t_start

    pbar.n = total_frames
    pbar.set_postfix_str(f"{duration_s:.1f}s/{duration_s:.1f}s, {len(events)} events, {stop_count} stops")
    pbar.refresh()
    pbar.close()

    # --- Compute detailed stats ---
    run_stats = _compute_run_stats(
        events, event_offsets, stop_count, stop_positions,
        total_calls, cursor_history, total_frames, duration_s, inference_time
    )

    return events, run_stats


def _compute_run_stats(events, event_offsets, stop_count, stop_positions,
                       total_calls, cursor_history, total_frames, duration_s, inference_time):
    """Compute comprehensive inference statistics."""
    stats = {
        "total_events": len(events),
        "total_frames": int(total_frames),
        "duration_s": round(duration_s, 2),
        "events_per_sec": round(len(events) / max(duration_s, 0.01), 2),
        "stop_count": stop_count,
        "total_model_calls": total_calls,
        "event_ratio": round(len(events) / max(total_calls, 1), 4),
        "stop_ratio": round(stop_count / max(total_calls, 1), 4),
        "inference_time_s": round(inference_time, 2),
        "events_per_sec_realtime": round(len(events) / max(inference_time, 0.01), 1),
        "realtime_factor": round(duration_s / max(inference_time, 0.01), 2),
    }

    # Timing per call
    stats["timing"] = {
        "inference_s": round(inference_time, 3),
        "ms_per_call": round(inference_time * 1000 / max(total_calls, 1), 2),
        "ms_per_event": round(inference_time * 1000 / max(len(events), 1), 2),
        "calls_per_sec": round(total_calls / max(inference_time, 0.01), 1),
    }

    # Event offset distribution
    if event_offsets:
        offsets = np.array(event_offsets)
        stats["event_distribution"] = {
            "min_offset": int(offsets.min()),
            "max_offset": int(offsets.max()),
            "mean_offset": round(float(offsets.mean()), 2),
            "median_offset": int(np.median(offsets)),
            "std_offset": round(float(offsets.std()), 2),
            "mean_offset_ms": round(float(offsets.mean()) * BIN_MS, 2),
            "median_offset_ms": round(float(np.median(offsets)) * BIN_MS, 2),
        }

        # Offset histogram (top 15 most common)
        offset_counts = Counter(event_offsets)
        stats["event_distribution"]["top_offsets"] = [
            {"offset": k, "offset_ms": round(k * BIN_MS, 1), "count": v}
            for k, v in offset_counts.most_common(15)
        ]

        # What % of events are at offset 0 (same position = potential stutter)
        zero_offsets = sum(1 for o in event_offsets if o == 0)
        stats["event_distribution"]["zero_offset_count"] = zero_offsets
        stats["event_distribution"]["zero_offset_pct"] = round(zero_offsets / len(event_offsets) * 100, 2)
    else:
        stats["event_distribution"] = {}

    # Inter-onset intervals
    if len(events) > 1:
        event_times_ms = [e * BIN_MS for e in events]
        iois = np.diff(event_times_ms)
        stats["ioi"] = {
            "min_ms": round(float(iois.min()), 1),
            "max_ms": round(float(iois.max()), 1),
            "mean_ms": round(float(iois.mean()), 1),
            "median_ms": round(float(np.median(iois)), 1),
            "std_ms": round(float(iois.std()), 1),
        }

        # Very short IOIs (< 20ms) = potential double-triggers
        short_iois = sum(1 for i in iois if i < 20)
        stats["ioi"]["short_ioi_count"] = int(short_iois)
        stats["ioi"]["short_ioi_pct"] = round(short_iois / len(iois) * 100, 2)

        # Very long IOIs (> 2s) = potential gaps
        long_iois = sum(1 for i in iois if i > 2000)
        stats["ioi"]["long_gap_count"] = int(long_iois)

        # IOI histogram buckets (10ms resolution, top entries)
        ioi_buckets = Counter(int(round(i / 10)) * 10 for i in iois)
        stats["ioi"]["histogram"] = [
            {"ms": k, "count": v} for k, v in sorted(ioi_buckets.items()) if v >= 2
        ]

        # Estimated BPM from most common IOI
        common_iois = Counter(int(round(i / 5)) * 5 for i in iois if 100 < i < 1500)
        if common_iois:
            common_ioi_ms = common_iois.most_common(1)[0][0]
            stats["ioi"]["estimated_bpm"] = round(60000 / common_ioi_ms, 1)
    else:
        stats["ioi"] = {}

    # Density over time (1-second windows)
    if events:
        event_times_ms = [e * BIN_MS for e in events]
        density_timeline = []
        for t_start in range(0, int(duration_s * 1000), 1000):
            t_end = t_start + 1000
            count = sum(1 for t in event_times_ms if t_start <= t < t_end)
            density_timeline.append({"time_s": t_start / 1000, "density": count})

        densities = [d["density"] for d in density_timeline]
        stats["density"] = {
            "mean": round(np.mean(densities), 2),
            "peak": int(np.max(densities)),
            "std": round(np.std(densities), 2),
            "min": int(np.min(densities)),
            "cv": round(float(np.std(densities) / max(np.mean(densities), 0.01)), 3),
            "timeline": density_timeline,
        }

        # Silence regions (0 events for 2+ seconds)
        silence_stretches = []
        current_silence_start = None
        for d in density_timeline:
            if d["density"] == 0:
                if current_silence_start is None:
                    current_silence_start = d["time_s"]
            else:
                if current_silence_start is not None:
                    length = d["time_s"] - current_silence_start
                    if length >= 2:
                        silence_stretches.append({
                            "start_s": current_silence_start,
                            "end_s": d["time_s"],
                            "duration_s": round(length, 1),
                        })
                    current_silence_start = None
        stats["density"]["silence_regions"] = silence_stretches

        # Dense regions (>2x mean density for 3+ seconds)
        mean_d = np.mean(densities)
        dense_stretches = []
        current_dense_start = None
        for d in density_timeline:
            if d["density"] > mean_d * 2:
                if current_dense_start is None:
                    current_dense_start = d["time_s"]
            else:
                if current_dense_start is not None:
                    length = d["time_s"] - current_dense_start
                    if length >= 3:
                        dense_stretches.append({
                            "start_s": current_dense_start,
                            "end_s": d["time_s"],
                            "duration_s": round(length, 1),
                        })
                    current_dense_start = None
        stats["density"]["dense_regions"] = dense_stretches
    else:
        stats["density"] = {}

    # STOP pattern analysis
    if stop_positions:
        stop_times_ms = [s * BIN_MS for s in stop_positions]
        stats["stop_analysis"] = {
            "total_stops": stop_count,
            "first_stop_ms": round(stop_times_ms[0], 1),
            "last_stop_ms": round(stop_times_ms[-1], 1),
        }

        # Consecutive STOP runs
        runs = []
        current_run = 1
        for i in range(1, len(stop_positions)):
            # Check if consecutive (same cursor hop apart)
            if cursor_history:
                # Approximate: if stop positions are close together, they're a run
                gap = (stop_positions[i] - stop_positions[i-1]) * BIN_MS
                if gap < 200:  # within ~200ms = likely consecutive STOPs
                    current_run += 1
                else:
                    if current_run > 1:
                        runs.append(current_run)
                    current_run = 1
        if current_run > 1:
            runs.append(current_run)

        stats["stop_analysis"]["consecutive_runs"] = len(runs)
        stats["stop_analysis"]["longest_run"] = max(runs) if runs else 0
        stats["stop_analysis"]["avg_run_length"] = round(np.mean(runs), 1) if runs else 0

        # Average gap between stops
        if len(stop_times_ms) > 1:
            stop_gaps = np.diff(stop_times_ms)
            stats["stop_analysis"]["avg_stop_gap_ms"] = round(float(np.mean(stop_gaps)), 1)
            stats["stop_analysis"]["min_stop_gap_ms"] = round(float(np.min(stop_gaps)), 1)
            stats["stop_analysis"]["max_stop_gap_ms"] = round(float(np.max(stop_gaps)), 1)
    else:
        stats["stop_analysis"] = {}

    # Copy avg_stop_gap to top level for viewer
    stats["avg_stop_gap_ms"] = stats.get("stop_analysis", {}).get("avg_stop_gap_ms", 0)

    return stats


def events_to_csv(events, output_path, audio_name=""):
    """Write events as a CSV matching the preprocessed format."""
    with open(output_path, "w", encoding="utf-8") as f:
        if audio_name:
            f.write(f"# audio: {audio_name}\n")
        f.write("time_ms,type\n")
        for bin_idx in events:
            time_ms = int(bin_idx * BIN_MS)
            f.write(f"{time_ms},predicted\n")
    print(f"Wrote {len(events)} events to {output_path}")


def print_stats_report(stats):
    """Print a detailed human-readable stats report to console."""
    print("\n" + "=" * 70)
    print("  INFERENCE REPORT")
    print("=" * 70)

    print(f"\n  Audio duration:      {stats['duration_s']:.1f}s")
    print(f"  Total events:        {stats['total_events']}")
    print(f"  Events/sec:          {stats['events_per_sec']:.1f}")
    print(f"  Total model calls:   {stats['total_model_calls']}")
    print(f"  STOP predictions:    {stats['stop_count']} ({stats['stop_ratio']*100:.1f}% of calls)")
    print(f"  Event predictions:   {stats['total_events']} ({stats['event_ratio']*100:.1f}% of calls)")

    print(f"\n  --- Timing ---")
    t = stats["timing"]
    print(f"  Inference wall time: {t['inference_s']:.2f}s")
    print(f"  Realtime factor:     {stats['realtime_factor']:.1f}x {'(faster)' if stats['realtime_factor'] > 1 else '(slower)'}")
    print(f"  ms/model call:       {t['ms_per_call']:.2f}")
    print(f"  ms/event:            {t['ms_per_event']:.2f}")
    print(f"  Calls/sec:           {t['calls_per_sec']:.0f}")

    ed = stats.get("event_distribution", {})
    if ed:
        print(f"\n  --- Event Offset Distribution ---")
        print(f"  Range:     {ed['min_offset']}-{ed['max_offset']} bins ({ed['min_offset']*BIN_MS:.0f}-{ed['max_offset']*BIN_MS:.0f}ms)")
        print(f"  Mean:      {ed['mean_offset']:.1f} bins ({ed['mean_offset_ms']:.1f}ms)")
        print(f"  Median:    {ed['median_offset']} bins ({ed['median_offset_ms']:.1f}ms)")
        print(f"  Std:       {ed['std_offset']:.1f} bins")
        if ed.get("zero_offset_count", 0) > 0:
            print(f"  Zero-offset events: {ed['zero_offset_count']} ({ed['zero_offset_pct']:.1f}%) [potential stutter]")

        if ed.get("top_offsets"):
            print(f"  Top offsets:")
            for entry in ed["top_offsets"][:8]:
                bar = "#" * min(40, entry["count"])
                print(f"    {entry['offset']:4d} ({entry['offset_ms']:6.1f}ms): {entry['count']:4d} {bar}")

    ioi = stats.get("ioi", {})
    if ioi:
        print(f"\n  --- Inter-Onset Intervals ---")
        print(f"  Min:       {ioi['min_ms']:.1f}ms")
        print(f"  Max:       {ioi['max_ms']:.1f}ms")
        print(f"  Mean:      {ioi['mean_ms']:.1f}ms")
        print(f"  Median:    {ioi['median_ms']:.1f}ms")
        print(f"  Std:       {ioi['std_ms']:.1f}ms")
        if ioi.get("short_ioi_count", 0) > 0:
            print(f"  Short IOIs (<20ms): {ioi['short_ioi_count']} ({ioi['short_ioi_pct']:.1f}%) [double-triggers]")
        if ioi.get("long_gap_count", 0) > 0:
            print(f"  Long gaps (>2s):    {ioi['long_gap_count']}")
        if ioi.get("estimated_bpm"):
            print(f"  Est. BPM:  {ioi['estimated_bpm']:.0f}")

    d = stats.get("density", {})
    if d:
        print(f"\n  --- Density Profile ---")
        print(f"  Mean:      {d['mean']:.1f} events/s")
        print(f"  Peak:      {d['peak']} events/s")
        print(f"  Min:       {d['min']} events/s")
        print(f"  Std:       {d['std']:.1f}")
        print(f"  CV:        {d['cv']:.3f}")

        if d.get("silence_regions"):
            print(f"  Silence regions ({len(d['silence_regions'])}):")
            for sr in d["silence_regions"][:5]:
                print(f"    {sr['start_s']:.1f}s - {sr['end_s']:.1f}s ({sr['duration_s']:.1f}s)")

        if d.get("dense_regions"):
            print(f"  Dense regions ({len(d['dense_regions'])}; >2x mean for 3+s):")
            for dr in d["dense_regions"][:5]:
                print(f"    {dr['start_s']:.1f}s - {dr['end_s']:.1f}s ({dr['duration_s']:.1f}s)")

    sa = stats.get("stop_analysis", {})
    if sa:
        print(f"\n  --- STOP Pattern ---")
        print(f"  Total STOPs:       {sa['total_stops']}")
        if sa.get("avg_stop_gap_ms"):
            print(f"  Avg gap between:   {sa['avg_stop_gap_ms']:.0f}ms")
            print(f"  Min gap:           {sa['min_stop_gap_ms']:.0f}ms")
            print(f"  Max gap:           {sa['max_stop_gap_ms']:.0f}ms")
        if sa.get("consecutive_runs", 0) > 0:
            print(f"  Consecutive runs:  {sa['consecutive_runs']} (longest: {sa['longest_run']})")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run onset detection inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio", default=None, help="Path to audio file (opens file browser if not set)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: {audio_stem}_predicted.csv)")
    parser.add_argument("--density-mean", type=float, default=4.3, help="Target mean density (events/sec)")
    parser.add_argument("--density-peak", type=float, default=8.0, help="Target peak density")
    parser.add_argument("--density-std", type=float, default=1.5, help="Target density std")
    parser.add_argument("--hop-ms", type=float, default=100, help="Cursor hop on STOP prediction (ms, default 100)")
    parser.add_argument("--andlaunch", action="store_true", help="Launch viewer after inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # pick audio file
    if args.audio is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        args.audio = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.ogg *.wav *.flac *.m4a"), ("All files", "*.*")],
        )
        root.destroy()
        if not args.audio:
            print("No file selected, exiting.")
            return

    t_total_start = time.perf_counter()

    # load checkpoint
    t0 = time.perf_counter()
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]

    model = OnsetDetector(
        n_mels=N_MELS,
        d_model=ckpt_args.get("d_model", 384),
        d_event=ckpt_args.get("d_event", 128),
        enc_layers=ckpt_args.get("enc_layers", 4),
        enc_event_layers=ckpt_args.get("enc_event_layers", 2),
        audio_path_layers=ckpt_args.get("audio_path_layers", 2),
        context_path_layers=ckpt_args.get("context_path_layers", 3),
        n_heads=ckpt_args.get("n_heads", 8),
        n_classes=N_CLASSES,
        max_events=C_EVENTS,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    t_model_load = time.perf_counter() - t0
    print(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, acc={ckpt.get('val_metrics', {}).get('accuracy', 0):.3f}")

    # load audio
    t0 = time.perf_counter()
    print(f"Loading audio: {args.audio}")
    mel, duration = load_audio_mel(args.audio, args.device)
    t_audio_load = time.perf_counter() - t0
    print(f"  Duration: {duration:.1f}s, {mel.shape[1]} mel frames")

    # run inference
    conditioning = [args.density_mean, args.density_peak, args.density_std]
    print(f"  Conditioning: mean={conditioning[0]}, peak={conditioning[1]}, std={conditioning[2]}")

    hop_bins = max(1, int(args.hop_ms / BIN_MS))
    print(f"  STOP hop: {args.hop_ms}ms = {hop_bins} bins")
    events, run_stats = run_inference(model, mel, conditioning, args.device, hop_bins=hop_bins)
    print(f"  Predicted {len(events)} events ({len(events) / duration:.1f}/s)")

    # Add extra info to stats
    run_stats["checkpoint"] = os.path.basename(args.checkpoint)
    run_stats["epoch"] = ckpt["epoch"]
    run_stats["val_loss"] = round(ckpt["val_loss"], 4)
    run_stats["val_accuracy"] = round(ckpt.get("val_metrics", {}).get("accuracy", 0), 4)
    run_stats["device"] = args.device
    run_stats["audio_file"] = os.path.basename(args.audio)
    run_stats["conditioning"] = {
        "mean": args.density_mean, "peak": args.density_peak, "std": args.density_std,
    }
    run_stats["hop_ms"] = args.hop_ms
    run_stats["hop_bins"] = hop_bins
    run_stats["timing"]["model_load_s"] = round(t_model_load, 3)
    run_stats["timing"]["audio_load_s"] = round(t_audio_load, 3)
    run_stats["timing"]["total_s"] = round(time.perf_counter() - t_total_start, 3)

    # Print report
    print_stats_report(run_stats)

    # write CSV
    tmp_dir = None
    if args.output is None:
        if args.andlaunch:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix="beatdetect_")
            stem = os.path.splitext(os.path.basename(args.audio))[0]
            args.output = os.path.join(tmp_dir, f"{stem}_predicted.csv")
        else:
            stem = os.path.splitext(os.path.basename(args.audio))[0]
            args.output = os.path.join(SCRIPT_DIR, f"{stem}_predicted.csv")

    events_to_csv(events, args.output, audio_name=os.path.abspath(args.audio))

    # Write stats JSON alongside CSV
    stats_path = args.output.replace(".csv", "_stats.json")
    # Remove timeline from JSON to keep it manageable (can be large)
    json_stats = {k: v for k, v in run_stats.items()}
    if "density" in json_stats and "timeline" in json_stats["density"]:
        json_stats["density"] = {k: v for k, v in json_stats["density"].items() if k != "timeline"}
    if "ioi" in json_stats and "histogram" in json_stats["ioi"]:
        # Keep only top 20 histogram entries
        json_stats["ioi"]["histogram"] = json_stats["ioi"]["histogram"][:20]

    with open(stats_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"Wrote stats to {stats_path}")

    if args.andlaunch:
        import subprocess, sys
        viewer_path = os.path.join(SCRIPT_DIR, "viewer.py")
        print(f"\nLaunching viewer: {args.output}")
        cmd = [sys.executable, viewer_path, args.output, "--audio", args.audio,
               "--stats-json", stats_path]
        subprocess.run(cmd)

    if tmp_dir is not None:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up temp directory: {tmp_dir}")


if __name__ == "__main__":
    main()
