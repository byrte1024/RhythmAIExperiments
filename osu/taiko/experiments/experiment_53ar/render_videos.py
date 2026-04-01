"""Render all chart CSVs to mp4 videos with audio + hit sounds.

Creates temporary copies of CSVs with model names stripped from filenames
so the rendered video header doesn't reveal which model generated it.
"""
import os
import sys
import subprocess
import glob
import shutil
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)

MODELS = ["exp14", "exp44", "exp45", "exp53"]


def main():
    viewer_script = os.path.join(TAIKO_DIR, "viewer.py")
    python = sys.executable

    csvs = sorted(glob.glob(os.path.join(CHARTS_DIR, "*.csv")))
    print(f"Found {len(csvs)} chart CSVs to render")

    tmp_dir = tempfile.mkdtemp(prefix="53ar_render_")

    for csv_path in csvs:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        output_mp4 = os.path.join(VIDEOS_DIR, f"{stem}.mp4")

        if os.path.exists(output_mp4):
            print(f"  SKIP (exists): {output_mp4}")
            continue

        # find matching audio file (strip model suffix)
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in MODELS:
            audio_stem = parts[0]
        else:
            audio_stem = stem
        audio_path = os.path.join(AUDIO_DIR, f"{audio_stem}.wav")

        if not os.path.exists(audio_path):
            print(f"  WARNING: audio not found for {stem}: {audio_path}")
            continue

        # copy CSV to temp with clean name (no model identifier)
        clean_name = f"{audio_stem}_predicted.csv"
        clean_csv = os.path.join(tmp_dir, clean_name)
        shutil.copy2(csv_path, clean_csv)

        gif_path = os.path.join(SCRIPT_DIR, "gawr-gura-hololive.gif")
        gif_arg = f' --gif "{gif_path}" --gif-cycles 2' if os.path.exists(gif_path) else ""

        print(f"\nRendering: {stem}")
        cmd = (f'{python} "{viewer_script}" "{clean_csv}" '
               f'--audio "{audio_path}" '
               f'--render "{output_mp4}" --render-fps 120'
               f'{gif_arg}')

        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"  ERROR: render failed for {stem}")

    # cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone! Videos in {VIDEOS_DIR}")
    print(f"Total mp4s: {len(glob.glob(os.path.join(VIDEOS_DIR, '*.mp4')))}")


if __name__ == "__main__":
    main()
