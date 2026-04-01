"""Run all 4 models on all 10 songs for experiment 53-AR."""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # experiments/ -> taiko/
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

MODELS = {
    "exp14": os.path.join(TAIKO_DIR, "runs", "detect_experiment_14", "checkpoints", "best.pt"),
    "exp44": os.path.join(TAIKO_DIR, "runs", "detect_experiment_44", "checkpoints", "eval_014.pt"),
    "exp45": os.path.join(TAIKO_DIR, "runs", "detect_experiment_45", "checkpoints", "eval_008.pt"),
    "exp53": os.path.join(TAIKO_DIR, "runs", "detect_experiment_53", "checkpoints", "eval_014.pt"),
}

DENSITY = "--density-mean 5.75 --density-peak 11.1"

SONGS = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])

def main():
    inference_script = os.path.join(TAIKO_DIR, "detection_inference.py")
    python = sys.executable

    for model_name, ckpt_path in MODELS.items():
        if not os.path.exists(ckpt_path):
            print(f"WARNING: checkpoint not found: {ckpt_path}")
            continue

        for song_file in SONGS:
            audio_path = os.path.join(AUDIO_DIR, song_file)
            stem = os.path.splitext(song_file)[0]
            output_csv = os.path.join(CHARTS_DIR, f"{stem}_{model_name}.csv")

            if os.path.exists(output_csv):
                print(f"  SKIP (exists): {output_csv}")
                continue

            print(f"\n{'='*60}")
            print(f"  Model: {model_name}")
            print(f"  Song:  {song_file}")
            print(f"{'='*60}")

            cmd = (f'{python} "{inference_script}" '
                   f'--checkpoint "{ckpt_path}" '
                   f'--audio "{audio_path}" '
                   f'--output "{output_csv}" '
                   f'{DENSITY}')

            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"  ERROR: inference failed for {model_name} on {song_file}")

    print(f"\nDone! Charts in {CHARTS_DIR}")
    print(f"Total CSVs: {len([f for f in os.listdir(CHARTS_DIR) if f.endswith('.csv')])}")


if __name__ == "__main__":
    main()
