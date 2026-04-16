"""Run TaikoNation on its own evaluation charts.

Reuses exp63's exact model architecture and inference logic.

Must be run with the Python 3.7 venv:
    cd osu/taiko
    experiments/experiment_63/taikonation_env/venv37/Scripts/python.exe experiments/experiment_63b/run_taikonation.py
"""

import os
import sys
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CHART_DIR = os.path.join(SCRIPT_DIR, "charts")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
TN_STEP_MS = 23

# Import exp63's model building and inference
sys.path.insert(0, os.path.join(TAIKO_DIR, "experiments", "experiment_63"))
from run_taikonation import build_model, extract_features, generate_chart


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "csvs"), exist_ok=True)

    manifest_path = os.path.join(CHART_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("ERROR: Run download_taikonation_charts.py first!")
        sys.exit(1)

    with open(manifest_path) as f:
        charts = json.load(f)

    print("Charts: {}".format(len(charts)))
    print("\nBuilding TaikoNation model...")
    model = build_model()
    print("Model loaded!\n")

    for ci, chart in enumerate(charts):
        audio_path = chart.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            print("  [{}] SKIP: no audio".format(ci + 1))
            continue

        bsid = chart.get("beatmapset_id", "?")
        artist = chart.get("artist", "?")
        title = chart.get("title", "?")
        diff = chart.get("difficulty", "?")

        output_csv = os.path.join(RESULTS_DIR, "csvs", "{}_taikonation_predicted.csv".format(bsid))
        if os.path.exists(output_csv):
            print("  [{}] {} - {}: (cached)".format(ci + 1, artist, title))
            continue

        print("  [{}/{}] {} - {} [{}]".format(ci + 1, len(charts), artist, title, diff))

        try:
            mel_features = extract_features(audio_path)
            print("    Features: {} frames ({:.1f}s)".format(
                len(mel_features), len(mel_features) * TN_STEP_MS / 1000))

            np.random.seed(42)
            events_ms = generate_chart(model, mel_features)
            print("    Generated: {} events".format(len(events_ms)))

            with open(output_csv, "w", encoding="utf-8") as f:
                f.write("# TaikoNation v1 on its own eval chart\ntime_ms,type\n")
                for t in events_ms:
                    f.write("{},don\n".format(t))

            print("    Saved: {}".format(output_csv))

        except Exception as e:
            print("    FAILED: {}".format(e))
            import traceback
            traceback.print_exc()

    print("\nDone! Now run run_comparison.py to compute metrics.")


if __name__ == "__main__":
    main()
