"""Experiment 63: Run TaikoNation on our val songs.

Uses TaikoNation's exact architecture + pretrained weights to generate
charts for the same 30 val songs used in exp 59-H. Outputs onset CSVs
for comparison with our models.

Must be run with the Python 3.7 venv:
    cd osu/taiko
    experiments/experiment_63/taikonation_env/venv37/Scripts/python.exe experiments/experiment_63/run_taikonation.py
"""

import os
import sys
import json
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import librosa
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(TAIKO_DIR, "audio")
MODEL_DIR = os.path.join(TAIKO_DIR, "..", "..", "external", "TaikoNationV1", "extracted_model")
BIN_MS = 4.9887
TN_STEP_MS = 23  # TaikoNation's timestep


def build_model():
    """Rebuild TaikoNation architecture and load weights."""
    import tensorflow as tf
    import tflearn

    net = tflearn.input_data([None, 1385])
    song = tf.slice(net, [0, 0], [-1, 1280])
    song = tf.reshape(song, [-1, 16, 80])
    prev_notes = tf.slice(net, [0, 1280], [-1, 105])
    prev_notes = tf.reshape(prev_notes, [-1, 7, 15])

    song_encoder = tflearn.conv_1d(song, nb_filter=16, filter_size=3, activation='relu')
    song_encoder = tflearn.dropout(song_encoder, keep_prob=0.8)
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)

    song_encoder = tflearn.conv_1d(song, nb_filter=32, filter_size=3, activation='relu')
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)

    song_encoder = tflearn.fully_connected(song_encoder, n_units=128, activation='relu')
    song_encoder = tf.reshape(song_encoder, [-1, 8, 16])

    past_chunks = tf.slice(song_encoder, [0, 0, 0], [-1, 8, 15])
    curr_chunk = tf.slice(song_encoder, [0, 0, 15], [-1, 8, 1])

    lstm_input = tf.unstack(past_chunks, axis=1)
    lstm_input = tf.math.multiply(lstm_input, prev_notes)
    lstm_input = tf.reshape(lstm_input, [-1])
    curr_chunk = tf.math.multiply(curr_chunk, tf.ones([8, 15]))
    curr_chunk = tf.reshape(curr_chunk, [-1])
    lstm_input = tf.concat([lstm_input, curr_chunk], 0)
    lstm_input = tf.reshape(lstm_input, [-1, 16, 88])

    lstm_input = tflearn.lstm(lstm_input, 64, dropout=0.8, activation='relu')
    lstm_input = tf.reshape(lstm_input, [-1, 8, 8])
    lstm_input = tflearn.lstm(song_encoder, 64, dropout=0.8, activation='relu')
    lstm_input = tflearn.fully_connected(lstm_input, n_units=28, activation='softmax')
    lstm_input = tflearn.reshape(lstm_input, [-1, 4, 7])

    network = tflearn.regression(lstm_input, optimizer='adam', loss='categorical_crossentropy',
                                  learning_rate=0.000005, batch_size=1)
    model = tflearn.DNN(network)
    model.load(os.path.join(MODEL_DIR, 'nr_model.tfl'))
    return model


def extract_features(audio_path, step_ms=TN_STEP_MS):
    """Extract mel features matching TaikoNation's preprocessing.

    23ms windows, 80 mel bands, normalized per band to zero mean unit variance.
    Returns (n_frames, 80) array.
    """
    sr = 22050
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    hop_length = int(sr * step_ms / 1000)  # ~507 samples for 23ms
    n_fft = 2048

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=80, fmin=20, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize per band (zero mean, unit variance) — matching essentia's behavior
    mel_db = mel_db.T  # (n_frames, 80)
    mean = mel_db.mean(axis=0, keepdims=True)
    std = mel_db.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    mel_norm = (mel_db - mean) / std

    return mel_norm.astype(np.float32)


def generate_chart(model, mel_features):
    """Run TaikoNation inference on extracted features.

    Sliding window of 16 frames, predicting next 4 timestamps.
    Overlapping predictions are averaged, then sampled.

    Returns list of onset times in ms.
    """
    n_frames = len(mel_features)
    # Accumulate predictions per timestamp: sum of 7-class probability vectors
    pred_accum = [np.zeros(7, dtype=np.float64) for _ in range(n_frames + 16)]
    pred_counts = [0] * (n_frames + 16)

    # Initial note context: zeros (no previous notes)
    note_queue = [np.zeros(7, dtype=np.float32) for _ in range(15)]

    for i in tqdm(range(n_frames), desc="    Generating", leave=False, unit="frame"):
        # Build song input: last 16 frames (padded at start)
        song_input = []
        for k in range(16):
            idx = i - k
            if idx < 0:
                song_input.append(np.zeros(80, dtype=np.float32))
            else:
                song_input.append(mel_features[idx])
        song_input = np.array(song_input).flatten()  # (1280,)

        # Build note input: last 15 note predictions (7-class each)
        note_input = []
        for k in range(15):
            idx = len(note_queue) - 1 - k
            if idx >= 0:
                note_input.append(note_queue[idx])
            else:
                note_input.append(np.zeros(7, dtype=np.float32))
        note_input = np.array(note_input).flatten()  # (105,)

        input_chunk = np.concatenate([song_input, note_input]).reshape(1, 1385)

        # Predict
        output = model.predict(input_chunk)  # (1, 4, 7)
        output = np.array(output[0])  # (4, 7)

        # Accumulate predictions for timestamps i through i+3
        for j in range(4):
            t = i + j
            if t < len(pred_accum):
                pred_accum[t] += output[j]
                pred_counts[t] += 1

        # Feed the argmax of the first prediction back as context
        pred_class = np.argmax(output[0])
        note_vec = np.zeros(7, dtype=np.float32)
        note_vec[pred_class] = 1.0
        note_queue.append(note_vec)

    # Average overlapping predictions and decide per timestamp
    events_ms = []
    for t in range(n_frames):
        if pred_counts[t] > 0:
            avg_pred = pred_accum[t] / pred_counts[t]
            # Normalize to probability distribution
            avg_pred = avg_pred / (avg_pred.sum() + 1e-10)
            # Sample from distribution (as TaikoNation does)
            pred_class = np.random.choice(7, p=avg_pred)
            # Class 0 = no note, classes 1-6 = note types
            if pred_class > 0:
                time_ms = int(t * TN_STEP_MS)
                events_ms.append(time_ms)

    # Post-processing: remove double positives (notes < 23ms apart)
    filtered = []
    for t in events_ms:
        if not filtered or (t - filtered[-1]) >= TN_STEP_MS:
            filtered.append(t)

    return filtered


def get_val_songs(manifest):
    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    return songs[:n_val], song_to_charts


def find_audio_file(beatmapset_id, artist, title):
    prefix = "{} {} - {}".format(beatmapset_id, artist, title)
    for ext in [".mp3", ".ogg", ".wav", ".flac"]:
        path = os.path.join(AUDIO_DIR, prefix + ext)
        if os.path.exists(path):
            return path
    for f in os.listdir(AUDIO_DIR):
        if f.startswith(str(beatmapset_id) + " "):
            return os.path.join(AUDIO_DIR, f)
    return None


def select_songs(manifest, n=30):
    val_songs, song_to_charts = get_val_songs(manifest)
    charts = manifest["charts"]
    candidates = []
    for sid in val_songs:
        idxs = song_to_charts[sid]
        sorted_idxs = sorted(idxs, key=lambda i: charts[i]["density_mean"])
        ci = sorted_idxs[len(sorted_idxs) // 2]
        c = charts[ci]
        audio_path = find_audio_file(c["beatmapset_id"], c["artist"], c["title"])
        if audio_path is None:
            continue
        candidates.append({
            "chart_idx": ci,
            "beatmapset_id": c["beatmapset_id"],
            "artist": c["artist"],
            "title": c["title"],
            "density_mean": c["density_mean"],
            "event_file": c["event_file"],
            "duration_s": c["duration_s"],
            "audio_path": audio_path,
        })
    candidates.sort(key=lambda x: x["density_mean"])
    if len(candidates) <= n:
        return candidates
    step = len(candidates) * 1.0 / n
    return [candidates[int(i * step)] for i in range(n)]


def load_gt_events_ms(event_file):
    events = np.load(os.path.join(DATASET_DIR, "events", event_file))
    return events.astype(np.float64) * BIN_MS


def _find_closest(sorted_arr, value):
    idx = np.searchsorted(sorted_arr, value)
    best = float("inf")
    for j in [idx - 1, idx, idx + 1]:
        if 0 <= j < len(sorted_arr):
            d = abs(sorted_arr[j] - value)
            if d < best:
                best = d
    return best


def compute_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0 or len(gt_ms) == 0:
        return None
    pred_sorted = np.sort(np.array(pred_ms, dtype=np.float64))
    gt_sorted = np.sort(np.array(gt_ms, dtype=np.float64))

    gt_errors = np.array([_find_closest(pred_sorted, g) for g in gt_sorted])
    pred_errors = np.array([_find_closest(gt_sorted, p) for p in pred_sorted])

    n_pred = len(pred_sorted)
    n_gt = len(gt_sorted)

    if n_pred > 1:
        pred_density = n_pred / max((pred_sorted[-1] - pred_sorted[0]) / 1000.0, 0.1)
    else:
        pred_density = 0.0
    if n_gt > 1:
        gt_density = n_gt / max((gt_sorted[-1] - gt_sorted[0]) / 1000.0, 0.1)
    else:
        gt_density = 0.0

    # TaikoNation patterning metrics (binary at 23ms resolution)
    def _to_binary(times_ms, step_ms=TN_STEP_MS):
        if len(times_ms) == 0:
            return np.array([0], dtype=np.int32)
        max_ms = int(max(times_ms)) + step_ms
        n_steps = max_ms // step_ms + 1
        binary = np.zeros(n_steps, dtype=np.int32)
        for t in times_ms:
            idx = int(t) // step_ms
            if 0 <= idx < n_steps:
                binary[idx] = 1
        return binary

    pred_bin = _to_binary(pred_sorted)
    gt_bin = _to_binary(gt_sorted)
    max_len = max(len(pred_bin), len(gt_bin))
    pred_pad = np.zeros(max_len, dtype=np.int32)
    gt_pad = np.zeros(max_len, dtype=np.int32)
    pred_pad[:len(pred_bin)] = pred_bin
    gt_pad[:len(gt_bin)] = gt_bin

    # Over. P-Space: unique 8-step patterns as % of 256
    scale = 8
    pred_patterns = set()
    for i in range(len(pred_pad) - scale + 1):
        pred_patterns.add(tuple(pred_pad[i:i + scale]))
    gt_patterns = set()
    for i in range(len(gt_pad) - scale + 1):
        gt_patterns.add(tuple(gt_pad[i:i + scale]))
    over_pspace = (len(pred_patterns) / 2**scale) * 100 if len(pred_pad) >= scale else 0

    # HI P-Space: % of GT patterns found in pred
    hi_pspace = (len(pred_patterns & gt_patterns) / max(len(gt_patterns), 1)) * 100

    # DCHuman: direct binary match
    start = 0
    limit = min(len(pred_pad), len(gt_pad))
    for i in range(limit):
        if gt_pad[i] == 1:
            start = i
            break
    total = limit - start
    dc_human = float((pred_pad[start:limit] == gt_pad[start:limit]).sum() / max(total, 1)) * 100 if total > 0 else 0

    # DCRand
    tn_rng = np.random.RandomState(42)
    noise = tn_rng.randint(0, 2, size=len(pred_pad))
    dc_rand = float((pred_pad == noise).sum() / max(len(pred_pad), 1)) * 100

    # Gap-based metrics (for synthetic evaluator comparison)
    pred_gaps = np.diff(pred_sorted)
    pred_gaps = pred_gaps[pred_gaps > 0]
    gap_std = float(pred_gaps.std()) if len(pred_gaps) > 1 else 0
    gap_cv = float(pred_gaps.std() / pred_gaps.mean()) if len(pred_gaps) > 1 and pred_gaps.mean() > 0 else 0

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "pred_gt_ratio": n_pred / max(n_gt, 1),
        "matched_rate": float((gt_errors <= 25).mean()),
        "close_rate": float((gt_errors <= 50).mean()),
        "far_rate": float((gt_errors > 100).mean()),
        "hallucination_rate": float((pred_errors > 100).mean()),
        "gt_error_mean": float(gt_errors.mean()),
        "gt_error_median": float(np.median(gt_errors)),
        "pred_density": pred_density,
        "gt_density": gt_density,
        "density_ratio": pred_density / max(gt_density, 0.01),
        "tn_over_pspace": over_pspace,
        "tn_hi_pspace": hi_pspace,
        "tn_dc_human": dc_human,
        "tn_dc_rand": dc_rand,
        "gap_std": gap_std,
        "gap_cv": gap_cv,
    }


def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    csv_dir = os.path.join(output_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    print("Loading manifest...")
    with open(os.path.join(DATASET_DIR, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("Selecting songs...")
    songs = select_songs(manifest, n=30)
    print("Selected {} songs".format(len(songs)))

    print("\nBuilding TaikoNation model...")
    model = build_model()
    print("Model loaded!\n")

    results = []
    for si, song in enumerate(tqdm(songs, desc="Songs", unit="song")):
        print("\n[{}/{}] {} - {} (d={:.1f})".format(
            si + 1, len(songs), song["artist"], song["title"], song["density_mean"]))

        try:
            # Extract features
            mel_features = extract_features(song["audio_path"])
            print("    Features: {} frames ({:.1f}s)".format(len(mel_features), len(mel_features) * TN_STEP_MS / 1000))

            # Generate chart
            np.random.seed(42)  # deterministic sampling
            events_ms = generate_chart(model, mel_features)
            print("    Generated: {} events".format(len(events_ms)))

            # Save CSV
            safe_name = "{}_{}_{}" .format(
                song["beatmapset_id"], song["artist"][:20], song["title"][:20]
            ).replace(" ", "_").replace("/", "_")
            for ch in '*?:<>|"':
                safe_name = safe_name.replace(ch, "")
            csv_path = os.path.join(csv_dir, "{}_taikonation.csv".format(safe_name))
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("# TaikoNation v1 output\ntime_ms,type\n")
                for t in events_ms:
                    f.write("{},predicted\n".format(t))

            # Save GT CSV alongside
            gt_ms = load_gt_events_ms(song["event_file"])
            gt_csv_path = os.path.join(csv_dir, "{}_gt.csv".format(safe_name))
            with open(gt_csv_path, "w", encoding="utf-8") as f:
                f.write("time_ms,type\n")
                for t in gt_ms:
                    f.write("{},gt\n".format(int(t)))

            # Compare to GT
            metrics = compute_metrics(events_ms, gt_ms)
            if metrics:
                results.append({"song": song, "metrics": metrics})
                print("    Close: {:.1%}  Hall: {:.1%}  d_ratio: {:.2f}  err_med: {:.0f}ms  P-Space: {:.1f}%  HI: {:.1f}%  DC: {:.1f}%".format(
                    metrics["close_rate"], metrics["hallucination_rate"],
                    metrics["density_ratio"], metrics["gt_error_median"],
                    metrics["tn_over_pspace"], metrics["tn_hi_pspace"], metrics["tn_dc_human"]))

        except Exception as e:
            print("    FAILED: {}".format(e))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        def avg(k):
            return float(np.mean([r["metrics"][k] for r in results]))
        print("TaikoNation on {} songs:".format(len(results)))
        print("  Close (<50ms): {:.1%}".format(avg("close_rate")))
        print("  Far (>100ms):  {:.1%}".format(avg("far_rate")))
        print("  Hallucination: {:.1%}".format(avg("hallucination_rate")))
        print("  Density ratio: {:.2f}".format(avg("density_ratio")))
        print("  Error median:  {:.0f}ms".format(avg("gt_error_median")))
        print("  Over. P-Space: {:.1f}%".format(avg("tn_over_pspace")))
        print("  HI P-Space:    {:.1f}%".format(avg("tn_hi_pspace")))
        print("  DCHuman:       {:.1f}%".format(avg("tn_dc_human")))
        print("  DCRand:        {:.1f}%".format(avg("tn_dc_rand")))
        print("  Gap std:       {:.0f}ms".format(avg("gap_std")))
        print("  Gap CV:        {:.2f}".format(avg("gap_cv")))

    # Save results
    save_data = {"n_songs": len(results)}
    if results:
        all_metric_keys = list(results[0]["metrics"].keys())
        for k in all_metric_keys:
            vals = [r["metrics"][k] for r in results if k in r["metrics"]]
            if vals:
                save_data["avg_" + k] = float(np.mean(vals))

    with open(os.path.join(output_dir, "taikonation_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print("\nResults saved to {}".format(output_dir))


if __name__ == "__main__":
    main()
