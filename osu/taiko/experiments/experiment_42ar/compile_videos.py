"""Compile per-model videos into blind A/B/C comparison videos.

For each song, creates one video that plays Alpha, Beta, Gamma versions
back-to-back with labels. The mapping of Alpha/Beta/Gamma to models is
randomized per song and saved to a .txt file alongside the video.

Usage:
    python experiments/experiment_42ar/compile_videos.py
"""
import os
import random
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
COMPILED_DIR = os.path.join(SCRIPT_DIR, "compiled")

MODELS = ["exp14", "exp35c", "exp42"]
LABELS = ["Alpha", "Beta", "Gamma"]


def find_common_songs():
    """Find songs that have videos for all 3 models."""
    songs_per_model = {}
    for model in MODELS:
        model_dir = os.path.join(VIDEOS_DIR, model)
        if not os.path.exists(model_dir):
            print(f"WARNING: {model_dir} not found")
            songs_per_model[model] = set()
            continue
        songs = set()
        for f in os.listdir(model_dir):
            if f.endswith(".mp4"):
                songs.add(f)
        songs_per_model[model] = songs

    common = songs_per_model[MODELS[0]]
    for model in MODELS[1:]:
        common = common & songs_per_model[model]

    return sorted(common)


def main():
    os.makedirs(COMPILED_DIR, exist_ok=True)
    random.seed(42)  # reproducible but shuffled

    songs = find_common_songs()
    print(f"Found {len(songs)} songs with all 3 model videos")

    if not songs:
        print("No common songs found. Run render_videos.py first.")
        return

    for song_file in songs:
        stem = os.path.splitext(song_file)[0]
        output_mp4 = os.path.join(COMPILED_DIR, f"{stem}_comparison.mp4")
        mapping_txt = os.path.join(COMPILED_DIR, f"{stem}_mapping.txt")

        if os.path.exists(output_mp4):
            print(f"SKIP (exists): {stem}")
            continue

        print(f"\nCompiling: {stem}")

        # randomize model order for this song
        shuffled_models = MODELS.copy()
        random.shuffle(shuffled_models)
        label_map = dict(zip(LABELS, shuffled_models))

        # save mapping
        with open(mapping_txt, "w", encoding="utf-8") as f:
            f.write(f"Song: {stem}\n")
            f.write(f"---\n")
            for label, model in zip(LABELS, shuffled_models):
                f.write(f"{label} = {model}\n")
        print(f"  Mapping: {' | '.join(f'{l}={m}' for l, m in zip(LABELS, shuffled_models))}")

        # for each label, add text overlay to the model's video
        labeled_parts = []
        for label, model in zip(LABELS, shuffled_models):
            src_video = os.path.join(VIDEOS_DIR, model, song_file)
            labeled_video = os.path.join(COMPILED_DIR, f"_tmp_{stem}_{label}.mp4")
            labeled_parts.append(labeled_video)

            # add label text overlay using ffmpeg drawtext
            cmd = [
                "ffmpeg", "-y",
                "-i", src_video,
                "-vf", (
                    f"drawtext=text='{label}':"
                    f"fontsize=36:fontcolor=white:borderw=3:bordercolor=black:"
                    f"x=20:y=20:"
                    f"enable='between(t,0,9999)',"
                    f"scale=640:160"
                ),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-r", "30",
                "-c:a", "aac", "-b:a", "96k",
                labeled_video,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                print(f"  ERROR labeling {label}: {result.stderr[-200:]}")
                return

        # concatenate all 3 labeled videos
        concat_list = os.path.join(COMPILED_DIR, f"_tmp_{stem}_concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for part in labeled_parts:
                f.write(f"file '{os.path.abspath(part)}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_mp4,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)

        if result.returncode != 0:
            print(f"  ERROR concatenating: {result.stderr[-300:]}")
        elif os.path.exists(output_mp4):
            size_mb = os.path.getsize(output_mp4) / 1024 / 1024
            print(f"  OK: {output_mp4} ({size_mb:.1f} MB)")
        else:
            print(f"  WARNING: output not created")

        # cleanup temp files
        for part in labeled_parts:
            try:
                os.unlink(part)
            except OSError:
                pass
        try:
            os.unlink(concat_list)
        except OSError:
            pass

    print(f"\nDone! Compiled videos in: {COMPILED_DIR}")
    print(f"Share the .mp4 files with evaluators. Keep the .txt mappings secret until results are in.")


if __name__ == "__main__":
    main()
