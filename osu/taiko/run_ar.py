"""Run AR inference on the standard 30 val songs.

Generates prediction CSVs and saves a songs.json manifest for analyze_ar.py.
Output goes to experiments/<experiment>/ar_eval/<checkpoint_name>/.

Usage:
    cd osu/taiko
    python run_ar.py experiment_62 runs/detect_experiment_62/checkpoints/best.pt
    python run_ar.py experiment_62 runs/detect_experiment_58/checkpoints/best.pt
    python run_ar.py experiment_62 runs/detect_experiment_44/checkpoints/eval_005.pt --density-mult 1.2
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")

DENSITY_REGIMES = {
    "song_density": None,
    "fixed_5.75": {"density_mean": 5.75, "density_peak": 11.1, "density_std": 1.5},
}


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


def select_30_val_songs(manifest):
    """Standard 30 val song selection. Matches all previous experiments."""
    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    val_songs = songs[:n_val]

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
            "beatmapset_id": c["beatmapset_id"],
            "artist": c["artist"],
            "title": c["title"],
            "density_mean": c["density_mean"],
            "density_peak": c["density_peak"],
            "density_std": c["density_std"],
            "duration_s": c["duration_s"],
            "event_file": c["event_file"],
            "audio_path": audio_path,
        })
    candidates.sort(key=lambda x: x["density_mean"])
    n = 30
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


def safe_name(song):
    name = f"{song['beatmapset_id']}_{song['artist'][:20]}_{song['title'][:20]}"
    name = name.replace(" ", "_").replace("/", "_")
    for ch in "*?:<>|\"":
        name = name.replace(ch, "")
    return name


def run_inference(checkpoint, song, output_csv, density_override=None, density_mult=1.0,
                  hop_ms=75, max_onsets=0):
    if density_override:
        d_mean = density_override["density_mean"]
        d_peak = density_override["density_peak"]
        d_std = density_override["density_std"]
    else:
        d_mean = song["density_mean"] * density_mult
        d_peak = song["density_peak"] * density_mult
        d_std = song["density_std"]

    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "detection_inference.py"),
        "--checkpoint", checkpoint,
        "--audio", song["audio_path"],
        "--output", output_csv,
        "--density-mean", str(d_mean),
        "--density-peak", str(d_peak),
        "--density-std", str(d_std),
        "--hop-ms", str(hop_ms),
    ]
    if max_onsets > 0:
        cmd += ["--max-onsets", str(max_onsets)]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"\n    ERROR: {result.stderr[-300:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run AR inference on 30 val songs")
    parser.add_argument("experiment", help="Output experiment (e.g. experiment_62)")
    parser.add_argument("checkpoint", help="Full path to checkpoint .pt file")
    parser.add_argument("--density-mult", type=float, default=1.0,
                        help="Density multiplier applied to song_density regime")
    parser.add_argument("--hop-ms", type=float, default=75, help="STOP hop in ms")
    parser.add_argument("--max-onsets", type=int, default=0,
                        help="Max onsets to place per step (0=all)")
    parser.add_argument("--label", default=None,
                        help="Custom label for output directory (default: auto from checkpoint path)")
    args = parser.parse_args()

    exp_name = args.experiment
    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    if args.label:
        ckpt_label = args.label
    else:
        # Derive a short name from the checkpoint path for the output directory
        ckpt_stem = os.path.splitext(os.path.basename(checkpoint))[0]
        ckpt_parent = os.path.basename(os.path.dirname(os.path.dirname(checkpoint)))
        ckpt_label = f"{ckpt_parent}_{ckpt_stem}"
        if args.max_onsets > 0:
            ckpt_label += f"_mo{args.max_onsets}"

    output_dir = os.path.join(SCRIPT_DIR, "experiments", exp_name, "ar_eval", ckpt_label)
    os.makedirs(output_dir, exist_ok=True)

    print(f"AR Inference: {exp_name}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Label: {ckpt_label}")
    if args.max_onsets > 0:
        print(f"  Max onsets: {args.max_onsets}")
    print(f"  Density mult: {args.density_mult}")
    print(f"  Output: {output_dir}")

    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    songs = select_30_val_songs(manifest)
    print(f"  Songs: {len(songs)}")
    print(f"  Regimes: {', '.join(DENSITY_REGIMES.keys())}")
    print()

    # Save songs manifest for analyze_ar.py
    songs_manifest = []
    for song in songs:
        sname = safe_name(song)
        songs_manifest.append({
            "safe_name": sname,
            "artist": song["artist"],
            "title": song["title"],
            "beatmapset_id": song["beatmapset_id"],
            "density_mean": song["density_mean"],
            "density_peak": song["density_peak"],
            "density_std": song["density_std"],
            "duration_s": song["duration_s"],
            "event_file": song["event_file"],
        })

    with open(os.path.join(output_dir, "songs.json"), "w", encoding="utf-8") as f:
        json.dump(songs_manifest, f, indent=2)

    for regime_name, density_override in DENSITY_REGIMES.items():
        d_label = f"fixed {density_override['density_mean']}" if density_override else f"per-song x{args.density_mult}"

        csv_dir = os.path.join(output_dir, "csvs", regime_name)
        os.makedirs(csv_dir, exist_ok=True)

        n_cached = sum(1 for s in songs if os.path.exists(
            os.path.join(csv_dir, f"{safe_name(s)}_predicted.csv")))
        n_todo = len(songs) - n_cached

        pbar = tqdm(songs, desc=f"{regime_name} ({d_label})",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        n_ok, n_err = 0, 0
        for song in pbar:
            sname = safe_name(song)
            csv_path = os.path.join(csv_dir, f"{sname}_predicted.csv")

            if os.path.exists(csv_path):
                n_ok += 1
                pbar.set_postfix_str(f"cached | {n_ok} done, {n_err} err")
                continue

            pbar.set_postfix_str(f"{song['artist'][:15]} - {song['title'][:15]}")
            ok = run_inference(checkpoint, song, csv_path,
                              density_override=density_override,
                              density_mult=args.density_mult, hop_ms=args.hop_ms,
                              max_onsets=args.max_onsets)
            if ok:
                n_ok += 1
            else:
                n_err += 1
            pbar.set_postfix_str(f"{n_ok} done, {n_err} err")

        pbar.close()

    print(f"\nDone. Run analysis with:")
    print(f"  python analyze_ar.py {exp_name} {ckpt_label}")


if __name__ == "__main__":
    main()
