"""Run all 3 models on all 10 songs for experiment 42-AR.

Usage:
    python experiments/experiment_42ar/run_inference.py
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INFERENCE_SCRIPT = os.path.join(TAIKO_DIR, "detection_inference.py")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

# Python executable
PYTHON = sys.executable

# Models to compare
MODELS = {
    "exp14": os.path.join(TAIKO_DIR, "runs", "detect_experiment_14", "checkpoints", "best.pt"),
    "exp35c": os.path.join(TAIKO_DIR, "runs", "detect_experiment_35c", "checkpoints", "best.pt"),
    "exp42": os.path.join(TAIKO_DIR, "runs", "detect_experiment_42", "checkpoints", "best.pt"),
}

# Density conditioning (fixed across all models)
DENSITY_MEAN = 6.75
DENSITY_PEAK = 12.1


def main():
    # find all wav files
    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
    print(f"Found {len(audio_files)} audio files")
    print(f"Models: {list(MODELS.keys())}")
    print()

    # verify checkpoints exist
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"WARNING: checkpoint not found: {name} -> {path}")

    # create output dirs
    for model_name in MODELS:
        out_dir = os.path.join(CHARTS_DIR, model_name)
        os.makedirs(out_dir, exist_ok=True)

    total = len(audio_files) * len(MODELS)
    done = 0

    for audio_file in audio_files:
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        stem = os.path.splitext(audio_file)[0]
        # clean stem for filename
        safe_stem = stem.replace("/", "_").replace("\\", "_").replace("'", "").replace('"', '')

        for model_name, ckpt_path in MODELS.items():
            done += 1
            out_dir = os.path.join(CHARTS_DIR, model_name)
            out_csv = os.path.join(out_dir, f"{safe_stem}.csv")

            if os.path.exists(out_csv):
                print(f"[{done}/{total}] SKIP (exists): {model_name} / {safe_stem}")
                continue

            print(f"[{done}/{total}] Running: {model_name} / {safe_stem}")

            cmd = [
                PYTHON, INFERENCE_SCRIPT,
                "--checkpoint", ckpt_path,
                "--audio", audio_path,
                "--output", out_csv,
                "--density-mean", str(DENSITY_MEAN),
                "--density-peak", str(DENSITY_PEAK),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=TAIKO_DIR,
                    timeout=300,
                )
                if result.returncode != 0:
                    print(f"  ERROR (return code {result.returncode})")
                else:
                    if os.path.exists(out_csv):
                        stats_path = out_csv.replace(".csv", "_stats.json")
                        has_stats = "✓" if os.path.exists(stats_path) else "✗"
                        print(f"  OK: {out_csv} (stats: {has_stats})")
                    else:
                        print(f"  WARNING: csv not created")
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT (300s)")
            except Exception as e:
                print(f"  EXCEPTION: {e}")

    print(f"\nDone! Charts saved to: {CHARTS_DIR}")
    print(f"Structure:")
    for model_name in MODELS:
        out_dir = os.path.join(CHARTS_DIR, model_name)
        n_files = len([f for f in os.listdir(out_dir) if f.endswith(".csv")]) if os.path.exists(out_dir) else 0
        print(f"  {model_name}/: {n_files} charts")


if __name__ == "__main__":
    main()
