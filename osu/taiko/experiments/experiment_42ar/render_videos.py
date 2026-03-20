"""Render all chart CSVs to video for experiment 42-AR.

Usage:
    python experiments/experiment_42ar/render_videos.py
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VIEWER_SCRIPT = os.path.join(TAIKO_DIR, "viewer.py")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")

PYTHON = sys.executable

MODELS = ["exp14", "exp35c", "exp42"]


def find_audio_for_chart(chart_stem):
    """Find the matching audio file for a chart stem."""
    for f in os.listdir(AUDIO_DIR):
        if not f.endswith(".wav"):
            continue
        # try matching by checking if audio filename starts similarly
        audio_stem = os.path.splitext(f)[0]
        # exact match or close enough
        if audio_stem == chart_stem:
            return os.path.join(AUDIO_DIR, f)
    # fallback: try substring matching
    for f in os.listdir(AUDIO_DIR):
        if not f.endswith(".wav"):
            continue
        # check if key words match
        chart_lower = chart_stem.lower()
        audio_lower = f.lower()
        # match by first significant word
        chart_words = set(chart_lower.replace("-", " ").replace("_", " ").split())
        audio_words = set(audio_lower.replace("-", " ").replace("_", " ").split())
        if len(chart_words & audio_words) >= 2:
            return os.path.join(AUDIO_DIR, f)
    return None


def main():
    total = 0
    tasks = []

    for model_name in MODELS:
        chart_dir = os.path.join(CHARTS_DIR, model_name)
        video_dir = os.path.join(VIDEOS_DIR, model_name)
        os.makedirs(video_dir, exist_ok=True)

        if not os.path.exists(chart_dir):
            print(f"WARNING: chart dir not found: {chart_dir}")
            continue

        for csv_file in sorted(os.listdir(chart_dir)):
            if not csv_file.endswith(".csv"):
                continue
            stem = os.path.splitext(csv_file)[0]
            csv_path = os.path.join(chart_dir, csv_file)
            video_path = os.path.join(video_dir, f"{stem}.mp4")

            audio_path = find_audio_for_chart(stem)

            tasks.append({
                "model": model_name,
                "stem": stem,
                "csv": csv_path,
                "video": video_path,
                "audio": audio_path,
            })
            total += 1

    print(f"Found {total} charts to render across {len(MODELS)} models")
    print()

    for i, task in enumerate(tasks):
        if os.path.exists(task["video"]):
            print(f"[{i+1}/{total}] SKIP (exists): {task['model']}/{task['stem']}")
            continue

        audio_flag = ["--audio", task["audio"]] if task["audio"] else []
        if not task["audio"]:
            print(f"[{i+1}/{total}] WARNING: no audio found for {task['stem']}")

        print(f"[{i+1}/{total}] Rendering: {task['model']}/{task['stem']}")

        cmd = [
            PYTHON, VIEWER_SCRIPT,
            task["csv"],
            *audio_flag,
            "--render", task["video"],
            "--render-fps", "60",
        ]

        try:
            result = subprocess.run(cmd, cwd=TAIKO_DIR, timeout=600)
            if result.returncode != 0:
                print(f"  ERROR (return code {result.returncode})")
            elif os.path.exists(task["video"]):
                size_mb = os.path.getsize(task["video"]) / 1024 / 1024
                print(f"  OK: {size_mb:.1f} MB")
            else:
                print(f"  WARNING: video not created")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT (600s)")
        except Exception as e:
            print(f"  EXCEPTION: {e}")

    print(f"\nDone! Videos saved to: {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
