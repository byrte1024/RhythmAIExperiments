"""Create a named dataset from osu!taiko charts.

Produces:
  datasets/{name}/
    manifest.json     - chart list with metadata, density stats, audio refs
    mels/{stem}.npy   - mel spectrogram per unique audio file (80, T) float16
    events/{id}.npy   - event bin indices per chart (N,) int32

Usage:
  python create_dataset.py my_dataset_v1
  python create_dataset.py my_dataset_v1 --workers 6
"""
import os
import sys
import json
import glob
import time
import hashlib
import argparse
import zipfile
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ── paths (relative to script) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")

# ── mel params ──
SAMPLE_RATE = 22050
HOP_LENGTH = 110        # ~5ms bins
N_FFT = 2048
N_MELS = 80
F_MIN = 20.0
F_MAX = 8000.0
BIN_MS = HOP_LENGTH / SAMPLE_RATE * 1000  # exact: ~4.9887ms per mel frame


# ─────────────────────────── parsing ───────────────────────────

def parse_osu(text):
    """Parse a single .osu file, return metadata + hit objects."""
    meta = {}
    onsets = []
    section = None

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("["):
            section = line
            continue

        if section == "[General]":
            if line.startswith("Mode:"):
                meta["mode"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("AudioFilename:"):
                meta["audio"] = line.split(":", 1)[1].strip()

        elif section == "[Metadata]":
            if line.startswith("Title:"):
                meta["title"] = line.split(":", 1)[1]
            elif line.startswith("Artist:"):
                meta["artist"] = line.split(":", 1)[1]
            elif line.startswith("Version:"):
                meta["difficulty"] = line.split(":", 1)[1]
            elif line.startswith("BeatmapID:"):
                meta["beatmap_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("BeatmapSetID:"):
                meta["beatmapset_id"] = line.split(":", 1)[1].strip()

        elif section == "[Difficulty]":
            if line.startswith("OverallDifficulty:"):
                meta["od"] = float(line.split(":", 1)[1].strip())

        elif section == "[HitObjects]":
            if not line or line.startswith("["):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            time_ms = int(parts[2])
            obj_type = int(parts[3])
            hit_sound = int(parts[4])
            if obj_type & 1:
                kind = "don" if (hit_sound & 0x0A) == 0 else "ka"
            elif obj_type & 2:
                kind = "drumroll"
            elif obj_type & 8:
                kind = "spinner"
            else:
                kind = "unknown"
            onsets.append((time_ms, kind))

    return meta, onsets


def compute_density_stats(onsets):
    """Compute density stats from onset list."""
    if len(onsets) < 2:
        return {}
    first_ms = onsets[0][0]
    last_ms = onsets[-1][0]
    duration_s = (last_ms - first_ms) / 1000.0
    if duration_s <= 0:
        return {}

    n = len(onsets)
    density_mean = n / duration_s

    # per-second buckets
    n_buckets = last_ms // 1000 + 1
    buckets = [0] * n_buckets
    for t, _ in onsets:
        buckets[t // 1000] += 1
    active = [b for b in buckets if b > 0]
    peak = max(buckets)
    avg_active = sum(active) / len(active) if active else 0
    std = (sum((v - avg_active) ** 2 for v in active) / len(active)) ** 0.5 if active else 0

    return {
        "density_mean": round(density_mean, 3),
        "density_peak": peak,
        "density_std": round(std, 3),
        "duration_s": round(duration_s, 2),
        "total_events": n,
    }


# ─────────────────────────── audio loading (worker) ───────────

def load_audio_worker(args):
    """Load + resample audio. Runs in subprocess."""
    osz_path, audio_name = args
    import librosa
    try:
        with zipfile.ZipFile(osz_path) as z:
            if audio_name not in z.namelist():
                return audio_name, None, f"not in zip"
            audio_bytes = z.read(audio_name)

        # write to temp, decode with librosa
        import tempfile
        ext = os.path.splitext(audio_name)[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        finally:
            os.unlink(tmp_path)

        return audio_name, y, None
    except Exception as e:
        return audio_name, None, str(e)


# ─────────────────────────── main pipeline ─────────────────────

def scan_all_osz():
    """Scan all .osz files, extract taiko chart info grouped by audio."""
    osz_files = sorted(glob.glob(os.path.join(CHARTS_DIR, "*.osz")))
    print(f"Scanning {len(osz_files)} .osz files...")

    # audio_key -> { osz_path, audio_name, charts: [{meta, onsets}, ...] }
    audio_groups = {}
    total_charts = 0
    errors = 0

    pbar = tqdm(osz_files, desc="Scanning .osz", unit="file")
    for osz_path in pbar:
        basename = os.path.splitext(os.path.basename(osz_path))[0]
        try:
            with zipfile.ZipFile(osz_path) as z:
                osu_files = [n for n in z.namelist() if n.endswith(".osu")]
                for osu_name in osu_files:
                    text = z.read(osu_name).decode("utf-8", errors="replace")
                    meta, onsets = parse_osu(text)

                    if meta.get("mode") != 1 or not onsets:
                        continue

                    audio_name = meta.get("audio", "")
                    if not audio_name:
                        continue

                    audio_key = f"{basename}__{audio_name}"

                    if audio_key not in audio_groups:
                        audio_groups[audio_key] = {
                            "osz_path": osz_path,
                            "audio_name": audio_name,
                            "basename": basename,
                            "charts": [],
                        }

                    diff = meta.get("difficulty", "unknown")
                    for ch in '<>:"/\\|?*':
                        diff = diff.replace(ch, "_")

                    chart_id = f"{basename} [{diff}]"
                    stats = compute_density_stats(onsets)
                    event_bins = np.array(
                        [int(t / BIN_MS) for t, _ in onsets], dtype=np.int32
                    )

                    audio_groups[audio_key]["charts"].append({
                        "chart_id": chart_id,
                        "meta": meta,
                        "stats": stats,
                        "event_bins": event_bins,
                    })
                    total_charts += 1

        except zipfile.BadZipFile:
            errors += 1
        except Exception as e:
            errors += 1

        pbar.set_postfix(charts=total_charts, audio=len(audio_groups), err=errors)

    print(f"Scan done: {total_charts} charts, {len(audio_groups)} unique audio, {errors} errors")
    return audio_groups


def create_dataset(name, workers=6):
    ds_dir = os.path.join(DATASETS_DIR, name)
    mel_dir = os.path.join(ds_dir, "mels")
    evt_dir = os.path.join(ds_dir, "events")
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(evt_dir, exist_ok=True)

    # ── phase 1: scan all charts ──
    audio_groups = scan_all_osz()

    # ── phase 2: load audio with workers, compute mels on GPU ──
    print(f"\nExtracting mels with {workers} workers + GPU...")

    import torch
    import torchaudio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0,
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

    # prepare worker tasks: (osz_path, audio_name) for each unique audio
    tasks = []
    key_order = []
    for key, group in audio_groups.items():
        mel_stem = _safe_filename(key)
        mel_path = os.path.join(mel_dir, f"{mel_stem}.npy")
        if os.path.exists(mel_path):
            group["mel_file"] = f"{mel_stem}.npy"
            group["mel_frames"] = np.load(mel_path).shape[1]
            continue
        tasks.append((group["osz_path"], group["audio_name"]))
        key_order.append(key)

    print(f"  {len(audio_groups) - len(tasks)} mels cached, {len(tasks)} to extract")

    errors = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_key = {}
        for task, key in zip(tasks, key_order):
            future_to_key[pool.submit(load_audio_worker, task)] = key

        pbar = tqdm(as_completed(future_to_key), total=len(tasks), desc="Extracting mels", unit="file")
        for future in pbar:
            key = future_to_key[future]
            group = audio_groups[key]
            audio_name, waveform, err = future.result()

            if err or waveform is None:
                tqdm.write(f"  SKIP {audio_name}: {err}")
                group["mel_file"] = None
                errors += 1
                pbar.set_postfix(err=errors)
                continue

            # GPU mel
            wav_tensor = torch.from_numpy(waveform).float().to(device)
            with torch.no_grad():
                mel = amp_to_db(mel_transform(wav_tensor))
            mel_np = mel.cpu().numpy().astype(np.float16)

            mel_stem = _safe_filename(key)
            np.save(os.path.join(mel_dir, f"{mel_stem}.npy"), mel_np)
            group["mel_file"] = f"{mel_stem}.npy"
            group["mel_frames"] = mel_np.shape[1]

    print(f"  Mels done: {len(tasks) - errors} extracted, {errors} errors")

    # ── phase 3: save events + build manifest ──
    print("\nSaving events and building manifest...\n")
    manifest = {
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_fft": N_FFT,
            "n_mels": N_MELS,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "bin_ms": BIN_MS,
        },
        "charts": [],
    }

    all_charts = [
        (key, group, chart)
        for key, group in audio_groups.items()
        if group.get("mel_file")
        for chart in group["charts"]
    ]

    for key, group, chart in tqdm(all_charts, desc="Saving events", unit="chart"):
        mel_file = group["mel_file"]
        mel_frames = group.get("mel_frames", 0)
        chart_id = chart["chart_id"]
        evt_file = f"{_safe_filename(chart_id)}.npy"
        np.save(os.path.join(evt_dir, evt_file), chart["event_bins"])

        entry = {
            "chart_id": chart_id,
            "mel_file": mel_file,
            "mel_frames": mel_frames,
            "event_file": evt_file,
            "audio_name": group["audio_name"],
            "artist": chart["meta"].get("artist", ""),
            "title": chart["meta"].get("title", ""),
            "difficulty": chart["meta"].get("difficulty", ""),
            "od": chart["meta"].get("od", 0),
            "beatmap_id": chart["meta"].get("beatmap_id", ""),
            "beatmapset_id": chart["meta"].get("beatmapset_id", ""),
            **chart["stats"],
        }
        manifest["charts"].append(entry)

    manifest["total_charts"] = len(manifest["charts"])
    manifest["total_mels"] = sum(1 for g in audio_groups.values() if g.get("mel_file"))

    manifest_path = os.path.join(ds_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nDataset '{name}' created at {ds_dir}")
    print(f"  {manifest['total_charts']} charts")
    print(f"  {manifest['total_mels']} mel spectrograms")
    print(f"  Manifest: {manifest_path}")


def _safe_filename(s, max_len=120):
    """Sanitize string for use as filename."""
    for ch in '<>:"/\\|?*\n\r':
        s = s.replace(ch, "_")
    s = s.strip(". ")
    if len(s) > max_len:
        h = hashlib.md5(s.encode()).hexdigest()[:8]
        s = s[:max_len - 9] + "_" + h
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a named osu!taiko dataset")
    parser.add_argument("name", help="Dataset name (e.g. taiko_v1)")
    parser.add_argument("--workers", type=int, default=6, help="CPU workers for audio decoding")
    args = parser.parse_args()
    create_dataset(args.name, args.workers)
