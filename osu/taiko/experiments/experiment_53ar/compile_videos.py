"""Create blind Alpha/Beta/Gamma/Delta comparison videos.

For each song, creates one video that plays Alpha, Beta, Gamma, Delta versions
back-to-back with labels. The mapping of Alpha/Beta/Gamma/Delta to models is
randomized per song and saved to a secret mapping file.
"""
import os
import sys
import random
import subprocess
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
COMPILED_DIR = os.path.join(SCRIPT_DIR, "compiled")
os.makedirs(COMPILED_DIR, exist_ok=True)

LABELS = ["Alpha", "Beta", "Gamma", "Delta"]
MODELS = ["exp14", "exp44", "exp45", "exp53"]

# songs identified by their number prefix
SONGS = sorted(set(
    f.rsplit("_", 1)[0]
    for f in os.listdir(VIDEOS_DIR)
    if f.endswith(".mp4")
))


def main():
    random.seed(None)  # truly random mappings

    for song_stem in SONGS:
        print(f"\n{'='*60}")
        print(f"  Song: {song_stem}")

        # find all model videos for this song
        model_videos = {}
        for model in MODELS:
            video_path = os.path.join(VIDEOS_DIR, f"{song_stem}_{model}.mp4")
            if os.path.exists(video_path):
                model_videos[model] = video_path
            else:
                print(f"  WARNING: missing {video_path}")

        if len(model_videos) < len(MODELS):
            print(f"  SKIP: only {len(model_videos)}/{len(MODELS)} models available")
            continue

        # random mapping
        shuffled_models = list(MODELS)
        random.shuffle(shuffled_models)
        mapping = dict(zip(LABELS, shuffled_models))

        # save mapping (secret!)
        mapping_txt = os.path.join(COMPILED_DIR, f"{song_stem}_mapping.txt")
        with open(mapping_txt, "w", encoding="utf-8") as f:
            for label, model in mapping.items():
                f.write(f"{label} = {model}\n")

        # build ffmpeg concat with label overlays
        concat_inputs = []
        filter_parts = []
        for i, label in enumerate(LABELS):
            model = mapping[label]
            video_path = model_videos[model]
            concat_inputs.extend(["-i", video_path])

            # add label overlay to each segment
            filter_parts.append(
                f"[{i}:v]drawtext=text='{label}':fontsize=36:fontcolor=white:"
                f"x=10:y=10:box=1:boxcolor=black@0.6:boxborderw=5:"
                f"enable='between(t,0,3)'[v{i}]"
            )

        # concat all segments
        concat_filter = ";".join(filter_parts)
        concat_streams = "".join(f"[v{i}][{i}:a]" for i in range(len(LABELS)))
        full_filter = f"{concat_filter};{concat_streams}concat=n={len(LABELS)}:v=1:a=1[outv][outa]"

        output_path = os.path.join(COMPILED_DIR, f"{song_stem}_comparison.mp4")

        cmd = ["ffmpeg", "-y"]
        cmd.extend(concat_inputs)
        cmd.extend([
            "-filter_complex", full_filter,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ])

        print(f"  Mapping: {mapping}")
        print(f"  Compiling to {output_path}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-200:]}")
        else:
            print(f"  OK: {output_path}")

    print(f"\nDone! Compiled videos in {COMPILED_DIR}")
    print(f"Share the .mp4 files with evaluators. Keep the .txt mappings secret until results are in.")


if __name__ == "__main__":
    main()
