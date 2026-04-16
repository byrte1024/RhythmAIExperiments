"""Download TaikoNation's exact evaluation charts.

Downloads the 10 beatmapsets Emily Halina shared, extracts the highest
taiko difficulty from each, and saves audio + event data for evaluation.

Usage:
    cd osu/taiko
    python experiments/experiment_63b/download_taikonation_charts.py
"""

import json
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "charts")

# TaikoNation evaluation beatmaps (from Emily Halina)
# Format: (beatmapset_id, specific_beatmap_id)
TAIKONATION_CHARTS = [
    (1116903, 2333205),
    (202040, 491412),
    (304553, 682296),
    (535179, 1133589),
    (562299, 1574370),
    (565959, 1200360),
    (724269, 1529288),
    (748803, 1580130),
    (821700, 1722197),
    (821745, 1722264),
]

# osu! API
CLIENT_ID = "51590"
CLIENT_SECRET = "5Nk6wGHIwugkLNpbDEhMv1UVSfZsCyrZQkB29X9r"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"
API_BASE = "https://osu.ppy.sh/api/v2"


def get_token():
    resp = requests.post(TOKEN_URL, json={
        "client_id": int(CLIENT_ID),
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "public",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def download_osz(beatmapset_id, output_path):
    """Download .osz file from osu! mirrors."""
    # Try catboy mirror first (no auth needed)
    mirrors = [
        f"https://catboy.best/d/{beatmapset_id}",
        f"https://api.chimu.moe/v1/download/{beatmapset_id}",
    ]
    for url in mirrors:
        try:
            print(f"    Trying {url}...")
            resp = requests.get(url, timeout=60, stream=True)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                print(f"    Downloaded {len(resp.content) / 1024:.0f} KB")
                return True
        except Exception as e:
            print(f"    Failed: {e}")
    return False


def parse_osu_file(osu_path):
    """Parse a .osu file to extract metadata and hit objects."""
    metadata = {}
    timing_points = []
    hit_objects = []
    section = None

    with open(osu_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section == "General":
                if line.startswith("AudioFilename:"):
                    metadata["audio_filename"] = line.split(":", 1)[1].strip()
                elif line.startswith("Mode:"):
                    metadata["mode"] = int(line.split(":", 1)[1].strip())

            elif section == "Metadata":
                if line.startswith("Title:"):
                    metadata["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("Artist:"):
                    metadata["artist"] = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"):
                    metadata["version"] = line.split(":", 1)[1].strip()
                elif line.startswith("BeatmapID:"):
                    metadata["beatmap_id"] = line.split(":", 1)[1].strip()
                elif line.startswith("BeatmapSetID:"):
                    metadata["beatmapset_id"] = line.split(":", 1)[1].strip()

            elif section == "Difficulty":
                if line.startswith("OverallDifficulty:"):
                    metadata["od"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("HPDrainRate:"):
                    metadata["hp"] = float(line.split(":", 1)[1].strip())

            elif section == "HitObjects":
                parts = line.split(",")
                if len(parts) >= 4:
                    try:
                        time_ms = int(parts[2])
                        hit_objects.append(time_ms)
                    except ValueError:
                        pass

    metadata["hit_objects"] = sorted(hit_objects)
    metadata["n_objects"] = len(hit_objects)
    return metadata


def find_highest_taiko_difficulty(osz_path, target_beatmap_id=None):
    """Extract all taiko .osu files, return the highest difficulty (or target)."""
    with zipfile.ZipFile(osz_path, "r") as zf:
        osu_files = [n for n in zf.namelist() if n.endswith(".osu")]
        if not osu_files:
            return None, None

        tmpdir = tempfile.mkdtemp()
        zf.extractall(tmpdir)

        best = None
        best_objects = 0
        audio_filename = None

        for osu_name in osu_files:
            osu_path = os.path.join(tmpdir, osu_name)
            try:
                meta = parse_osu_file(osu_path)
            except Exception as e:
                print(f"    Error parsing {osu_name}: {e}")
                continue

            # Must be taiko mode (1)
            if meta.get("mode", 0) != 1:
                continue

            # If we have a target beatmap_id, prefer it
            if target_beatmap_id and str(meta.get("beatmap_id", "")) == str(target_beatmap_id):
                best = meta
                audio_filename = meta.get("audio_filename")
                break

            # Otherwise take highest object count (proxy for difficulty)
            if meta["n_objects"] > best_objects:
                best = meta
                best_objects = meta["n_objects"]
                audio_filename = meta.get("audio_filename")

        # Copy audio file out
        audio_path = None
        if audio_filename and best:
            src = os.path.join(tmpdir, audio_filename)
            if os.path.exists(src):
                audio_path = src
            else:
                # Try case-insensitive search
                for f in os.listdir(tmpdir):
                    if f.lower() == audio_filename.lower():
                        audio_path = os.path.join(tmpdir, f)
                        break

        return best, audio_path, tmpdir


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading TaikoNation evaluation charts...")
    print(f"Output: {OUTPUT_DIR}")
    print()

    results = []

    for beatmapset_id, beatmap_id in TAIKONATION_CHARTS:
        print(f"[{beatmapset_id}] beatmap {beatmap_id}")

        osz_path = os.path.join(OUTPUT_DIR, f"{beatmapset_id}.osz")

        # Download
        if not os.path.exists(osz_path):
            ok = download_osz(beatmapset_id, osz_path)
            if not ok:
                print(f"  FAILED to download")
                continue
        else:
            print(f"  (cached)")

        # Extract
        meta, audio_src, tmpdir = find_highest_taiko_difficulty(osz_path, target_beatmap_id=beatmap_id)
        if meta is None:
            print(f"  No taiko difficulty found!")
            shutil.rmtree(tmpdir, ignore_errors=True)
            continue

        print(f"  {meta.get('artist', '?')} - {meta.get('title', '?')} [{meta.get('version', '?')}]")
        print(f"  Objects: {meta['n_objects']}, BeatmapID: {meta.get('beatmap_id', '?')}")

        # Save audio
        chart_dir = os.path.join(OUTPUT_DIR, str(beatmapset_id))
        os.makedirs(chart_dir, exist_ok=True)

        if audio_src and os.path.exists(audio_src):
            audio_dst = os.path.join(chart_dir, os.path.basename(audio_src))
            shutil.copy2(audio_src, audio_dst)
            meta["audio_path"] = audio_dst
            print(f"  Audio: {os.path.basename(audio_dst)}")
        else:
            print(f"  WARNING: No audio file found!")
            meta["audio_path"] = None

        # Save hit object times as JSON
        events_path = os.path.join(chart_dir, "events.json")
        with open(events_path, "w") as f:
            json.dump({
                "beatmapset_id": beatmapset_id,
                "beatmap_id": beatmap_id,
                "artist": meta.get("artist", ""),
                "title": meta.get("title", ""),
                "difficulty": meta.get("version", ""),
                "od": meta.get("od", 0),
                "n_objects": meta["n_objects"],
                "hit_times_ms": meta["hit_objects"],
            }, f, indent=2)

        meta["events_path"] = events_path
        results.append(meta)

        shutil.rmtree(tmpdir, ignore_errors=True)

    # Save manifest
    manifest = []
    for meta in results:
        manifest.append({
            "beatmapset_id": meta.get("beatmapset_id", ""),
            "beatmap_id": meta.get("beatmap_id", ""),
            "artist": meta.get("artist", ""),
            "title": meta.get("title", ""),
            "difficulty": meta.get("version", ""),
            "n_objects": meta["n_objects"],
            "audio_path": meta.get("audio_path"),
            "events_path": meta.get("events_path"),
        })

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {len(results)}/{len(TAIKONATION_CHARTS)} charts downloaded")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
