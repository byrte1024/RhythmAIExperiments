"""Render all chart CSVs to mp4 videos with audio + hit sounds."""
import os
import sys
import subprocess
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)


def main():
    viewer_script = os.path.join(TAIKO_DIR, "viewer.py")
    python = sys.executable

    csvs = sorted(glob.glob(os.path.join(CHARTS_DIR, "*.csv")))
    print(f"Found {len(csvs)} chart CSVs to render")

    for csv_path in csvs:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        output_mp4 = os.path.join(VIDEOS_DIR, f"{stem}.mp4")

        if os.path.exists(output_mp4):
            print(f"  SKIP (exists): {output_mp4}")
            continue

        # find matching audio file (strip model suffix)
        # e.g. "01_arashi_five_exp14.csv" -> "01_arashi_five.wav"
        parts = stem.rsplit("_", 1)  # split off model name
        if len(parts) == 2:
            audio_stem = parts[0]
        else:
            audio_stem = stem
        audio_path = os.path.join(AUDIO_DIR, f"{audio_stem}.wav")

        if not os.path.exists(audio_path):
            print(f"  WARNING: audio not found for {stem}: {audio_path}")
            continue

        print(f"\nRendering: {stem}")
        cmd = (f'{python} "{viewer_script}" "{csv_path}" '
               f'--audio "{audio_path}" '
               f'--render "{output_mp4}" --render-fps 120')

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"  ERROR: render failed for {stem}")

    print(f"\nDone! Videos in {VIDEOS_DIR}")
    print(f"Total mp4s: {len(glob.glob(os.path.join(VIDEOS_DIR, '*.mp4')))}")


if __name__ == "__main__":
    main()
