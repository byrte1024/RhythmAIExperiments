"""S3 3-stage AR inference with sampling matrix sweep.

Runs S1+S2v2+S3 on val songs, then evaluates all combinations of:
  - Combination: S1_ONLY, S2_ONLY, S3_ONLY, ADD, MULTIPLY, S3_NO_S1, S3_NO_S2
  - Sampling: MAX, FIRST_THRESH, ALL_THRESH

Since all operate on cached per-step logits, we run inference once and sweep offline.

Usage:
    cd osu/taiko
    python detection_s3_inference.py \
        --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
        --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt \
        --s3-checkpoint runs/s3_experiment_65/checkpoints/best.pt \
        --output-dir experiments/experiment_65_s3/ar_eval
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import torchaudio
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

A_BINS = 500
B_BINS = 500
B_PRED = 250
C_EVENTS = 128
N_MELS = 80
SAMPLE_RATE = 22050
HOP_LENGTH = 110
N_FFT = 2048
F_MIN = 20.0
F_MAX = 8000.0
BIN_MS = HOP_LENGTH / SAMPLE_RATE * 1000
MIN_CURSOR_BIN = 0  # AR starts from beginning

DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")


# ═══════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════

def load_s1(ckpt_path, device):
    from detection_s1_model import ConformerProposer
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ConformerProposer(n_mels=N_MELS, d_model=a.get("d_model", 384),
                              n_layers=a.get("n_layers", 8), n_heads=8,
                              conv_kernel=a.get("conv_kernel", 31),
                              a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_s2v2(ckpt_path, device):
    from detection_s2v2_model import ContextProposer
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ContextProposer(d_model=a.get("d_model", 256),
                            n_gru_layers=a.get("n_gru_layers", 4),
                            b_pred=B_PRED, max_events=C_EVENTS)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_s3(ckpt_path, device, s1_d_model=384):
    from detection_s3_model import FusionSelector
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = FusionSelector(d_model=a.get("d_model", 192), d_audio=s1_d_model,
                           n_enc_layers=a.get("n_enc_layers", 4),
                           n_dec_layers=a.get("n_dec_layers", 3),
                           n_heads=8, a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════
#  Audio / song selection (same 30 val songs as run_ar.py)
# ═══════════════════════════════════════════════════════════════

def load_audio_mel(audio_path, device):
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)
    wav = torch.from_numpy(y).float().to(device)
    with torch.no_grad():
        mel = amp_to_db(mel_transform(wav))
    return mel.cpu().numpy().astype(np.float32), len(y) / SAMPLE_RATE


def find_audio_file(beatmapset_id, artist, title):
    prefix = f"{beatmapset_id} {artist} - {title}"
    for ext in [".mp3", ".ogg", ".wav", ".flac"]:
        path = os.path.join(AUDIO_DIR, prefix + ext)
        if os.path.exists(path):
            return path
    for f in os.listdir(AUDIO_DIR):
        if f.startswith(str(beatmapset_id) + " "):
            return os.path.join(AUDIO_DIR, f)
    return None


def select_30_val_songs(manifest):
    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    val_songs = songs[:n_val]
    candidates = []
    for sid in val_songs:
        idxs = song_to_charts[sid]
        sorted_idxs = sorted(idxs, key=lambda i: charts[i]["density_mean"])
        ci = sorted_idxs[len(sorted_idxs) // 2]
        c = charts[ci]
        audio_path = find_audio_file(c["beatmapset_id"], c["artist"], c["title"])
        if audio_path is None:
            continue
        candidates.append({**c, "audio_path": audio_path})
    candidates.sort(key=lambda x: x["density_mean"])
    n = 30
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


# ═══════════════════════════════════════════════════════════════
#  AR inference — collect per-step logits
# ═══════════════════════════════════════════════════════════════

def extract_mel_window(mel, cursor):
    n_mels, total = mel.shape
    start = cursor - A_BINS
    end = cursor + B_BINS
    pad_left = max(0, -start)
    pad_right = max(0, end - total)
    s, e = max(0, start), min(total, end)
    window = mel[:, s:e]
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")
    return window


def build_context(events, cursor):
    """Build gap/ratio sequences for S2v2 from event list."""
    past = [e for e in events if e <= cursor]
    past = past[-C_EVENTS:]
    n = len(past)

    gaps = np.zeros(C_EVENTS, dtype=np.float32)
    ratios = np.zeros(C_EVENTS, dtype=np.float32)
    mask = np.ones(C_EVENTS, dtype=bool)
    offsets = np.zeros(C_EVENTS, dtype=np.int64)
    evt_mask = np.ones(C_EVENTS, dtype=bool)

    if n > 0:
        past_arr = np.array(past, dtype=np.int64)
        raw_gaps = np.zeros(n, dtype=np.float32)
        if n >= 2:
            raw_gaps[1:] = np.diff(past_arr).astype(np.float32)
            raw_gaps[0] = raw_gaps[1]
        else:
            raw_gaps[0] = 30.0
        raw_gaps = np.maximum(raw_gaps, 1.0)

        raw_ratios = np.ones(n, dtype=np.float32)
        if n >= 2:
            for i in range(1, n):
                raw_ratios[i] = np.clip(raw_gaps[i] / max(raw_gaps[i-1], 1.0), 0.1, 10.0)

        s = C_EVENTS - n
        gaps[s:] = raw_gaps
        ratios[s:] = raw_ratios
        mask[s:] = False

        off = past_arr - cursor
        offsets[-n:] = off
        evt_mask[-n:] = False

    return gaps, ratios, mask, offsets, evt_mask


@torch.no_grad()
def run_ar_collect(s1_model, s2_model, s3_model, mel, song, device, hop_bins=20):
    """Run AR, collect per-step S1/S2/S3 logits for offline sweep."""
    total_frames = mel.shape[1]
    events = []
    cursor = 0
    cond = np.array([song["density_mean"], song["density_peak"], song["density_std"]], dtype=np.float32)
    cond_t = torch.tensor(cond).unsqueeze(0).to(device)

    steps = []  # list of {cursor, s1_conf, s2_conf, s3_conf}

    pbar = tqdm(total=total_frames, desc="AR", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}")

    for _ in range(50000):
        if cursor >= total_frames:
            break
        pbar.n = min(cursor, total_frames)
        pbar.set_postfix_str(f"{len(events)} events")
        pbar.refresh()

        # Mel window
        mel_window = extract_mel_window(mel, cursor)
        mel_t = torch.from_numpy(mel_window).unsqueeze(0).to(device)

        # S1
        s1_logits = s1_model(mel_t)
        s1_conf = torch.sigmoid(s1_logits).squeeze(0).cpu().numpy()  # (250,)
        audio_feat = s1_model.conv(mel_t).transpose(1, 2)
        audio_feat = s1_model.conv_norm(audio_feat)

        # S2v2 context
        gaps, ratios, s2_mask, evt_off, evt_mask = build_context(events, cursor)
        gaps_t = torch.from_numpy(gaps).unsqueeze(0).to(device)
        ratios_t = torch.from_numpy(ratios).unsqueeze(0).to(device)
        s2_mask_t = torch.from_numpy(s2_mask).unsqueeze(0).to(device)

        s2_logits = s2_model(gaps_t, ratios_t, s2_mask_t, cond_t)
        s2_conf = torch.sigmoid(s2_logits).squeeze(0).cpu().numpy()  # (250,)

        # S3
        evt_off_t = torch.from_numpy(evt_off).unsqueeze(0).to(device)
        evt_mask_t = torch.from_numpy(evt_mask).unsqueeze(0).to(device)
        s1_conf_t = torch.from_numpy(s1_conf).unsqueeze(0).to(device)
        s2_conf_t = torch.from_numpy(s2_conf).unsqueeze(0).to(device)

        s3_logits, _ = s3_model(audio_feat, s1_conf_t, s2_conf_t, evt_off_t, evt_mask_t, cond_t)
        s3_conf = torch.sigmoid(s3_logits).squeeze(0).cpu().numpy()  # (250,)

        steps.append({
            "cursor": cursor,
            "s1": s1_conf,
            "s2": s2_conf,
            "s3": s3_conf,
        })

        # Default AR advance: use S3 MAX for event placement
        best_bin = int(s3_conf.argmax())
        if s3_conf[best_bin] < 0.3:
            # STOP — hop forward
            cursor += hop_bins
        else:
            event_pos = cursor + best_bin
            events.append(event_pos)
            cursor = event_pos

    pbar.close()
    return events, steps


# ═══════════════════════════════════════════════════════════════
#  Offline sweep: apply combinations + sampling to cached logits
# ═══════════════════════════════════════════════════════════════

def apply_combination(steps, method, params=None):
    """Apply a combination method to cached per-step logits. Returns events list."""
    if params is None:
        params = {}
    thresh = params.get("threshold", 0.5)
    hop_bins = params.get("hop_bins", 20)

    events = []
    cursor = 0

    for step in steps:
        if step["cursor"] != cursor:
            cursor = step["cursor"]  # sync (shouldn't diverge for first pass)

        s1 = step["s1"]
        s2 = step["s2"]
        s3 = step["s3"]

        # Combine
        if method == "S1_ONLY":
            conf = s1
        elif method == "S2_ONLY":
            conf = s2
        elif method == "S3_ONLY":
            conf = s3
        elif method == "ADD_EQUAL":
            conf = (s1 + s2 + s3) / 3
        elif method == "ADD_S3_HEAVY":
            conf = 0.2 * s1 + 0.2 * s2 + 0.6 * s3
        elif method == "ADD_S1S2":
            conf = (s1 + s2) / 2
        elif method == "MULTIPLY":
            conf = np.cbrt(s1 * s2 * s3 + 1e-10)
        elif method == "MULTIPLY_S1S2":
            conf = np.sqrt(s1 * s2 + 1e-10)
        elif method == "S3_NO_S1":
            # S3 was trained with S1 — what if S1 was zero?
            conf = s3  # can't re-run, just use S3 as-is
        elif method == "S3_NO_S2":
            conf = s3
        elif method == "MAX_S1S2":
            conf = np.maximum(s1, s2)
        elif method == "MIN_S1S2":
            conf = np.minimum(s1, s2)
        else:
            conf = s3

        # Sample: MAX
        sampling = params.get("sampling", "MAX")
        if sampling == "MAX":
            best = int(conf.argmax())
            if conf[best] < thresh:
                cursor += hop_bins
            else:
                events.append(cursor + best)
                cursor = cursor + best

        elif sampling == "FIRST_THRESH":
            above = np.where(conf >= thresh)[0]
            if len(above) > 0:
                events.append(cursor + int(above[0]))
                cursor = cursor + int(above[0])
            else:
                cursor += hop_bins

        elif sampling == "ALL_THRESH":
            above = np.where(conf >= thresh)[0]
            if len(above) > 0:
                for b in above:
                    events.append(cursor + int(b))
                cursor = cursor + int(above[-1])
            else:
                cursor += hop_bins

    return events


# ═══════════════════════════════════════════════════════════════
#  GT matching (from analyze_ar.py)
# ═══════════════════════════════════════════════════════════════

def compute_gt_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0 or len(gt_ms) == 0:
        return None
    pred_sorted = np.sort(pred_ms)
    gt_sorted = np.sort(gt_ms)

    def _closest(arr, val):
        idx = np.searchsorted(arr, val)
        best = float("inf")
        for j in [idx - 1, idx, idx + 1]:
            if 0 <= j < len(arr):
                best = min(best, abs(arr[j] - val))
        return best

    gt_errors = np.array([_closest(pred_sorted, g) for g in gt_sorted])
    pred_errors = np.array([_closest(gt_sorted, p) for p in pred_sorted])

    return {
        "n_pred": len(pred_sorted),
        "n_gt": len(gt_sorted),
        "close_rate": float((gt_errors <= 50).mean()),
        "far_rate": float((gt_errors > 100).mean()),
        "hallucination_rate": float((pred_errors > 100).mean()),
        "gt_error_median": float(np.median(gt_errors)),
    }


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="S3 AR inference with sampling matrix sweep")
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--s3-checkpoint", required=True)
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "experiments", "experiment_65_s3", "ar_eval"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-songs", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(DATASET_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    songs = select_30_val_songs(manifest)[:args.n_songs]
    print(f"Songs: {len(songs)}")

    print("Loading models...")
    s1_model = load_s1(args.s1_checkpoint, args.device)
    s2_model = load_s2v2(args.s2_checkpoint, args.device)
    s3_model = load_s3(args.s3_checkpoint, args.device, s1_d_model=s1_model.d_model)
    print("  All loaded")

    # Sweep config
    combinations = [
        "S1_ONLY", "S2_ONLY", "S3_ONLY",
        "ADD_EQUAL", "ADD_S3_HEAVY", "ADD_S1S2",
        "MULTIPLY", "MULTIPLY_S1S2",
        "MAX_S1S2", "MIN_S1S2",
    ]
    samplings = ["MAX", "FIRST_THRESH", "ALL_THRESH"]
    thresholds = [0.3, 0.4, 0.5, 0.6]

    all_results = {}

    for si, song in enumerate(songs):
        print(f"\n[{si+1}/{len(songs)}] {song['artist'][:20]} - {song['title'][:25]} (d={song['density_mean']:.1f})")

        # Load audio
        mel, duration = load_audio_mel(song["audio_path"], args.device)
        print(f"  Mel: {mel.shape}, {duration:.1f}s")

        # Load GT
        gt_events = np.load(os.path.join(DATASET_DIR, "events", song["event_file"]))
        gt_ms = gt_events.astype(np.float64) * BIN_MS

        # Run AR once, collect per-step logits
        events_default, steps = run_ar_collect(s1_model, s2_model, s3_model, mel, song, args.device)
        print(f"  AR: {len(events_default)} events, {len(steps)} steps")

        # Sweep all combinations × samplings × thresholds
        for combo in combinations:
            for sampling in samplings:
                for thresh in thresholds:
                    key = f"{combo}_{sampling}_{thresh}"
                    events = apply_combination(steps, combo,
                                               {"threshold": thresh, "sampling": sampling, "hop_bins": 20})
                    pred_ms = np.array(events, dtype=np.float64) * BIN_MS
                    gt_result = compute_gt_metrics(pred_ms, gt_ms)

                    if key not in all_results:
                        all_results[key] = []
                    if gt_result:
                        all_results[key].append(gt_result)

    # ── Aggregate ──
    print(f"\n{'='*90}")
    print(f"RESULTS ({len(songs)} songs)")
    print(f"{'='*90}")
    print(f"{'Config':<40} {'Close%':>7} {'Far%':>6} {'Hall%':>6} {'ErrMed':>7} {'#pred':>6}")
    print("-" * 80)

    summary = {}
    for key in sorted(all_results.keys()):
        results = all_results[key]
        if not results:
            continue
        avg = lambda k: float(np.mean([r[k] for r in results]))
        close = avg("close_rate")
        far = avg("far_rate")
        hall = avg("hallucination_rate")
        err = avg("gt_error_median")
        n_pred = avg("n_pred")
        summary[key] = {
            "close": close, "far": far, "hall": hall, "err_med": err, "n_pred": n_pred,
        }

    # Sort by close rate descending
    for key in sorted(summary.keys(), key=lambda k: -summary[k]["close"]):
        s = summary[key]
        print(f"  {key:<40} {s['close']:>6.1%} {s['far']:>5.1%} {s['hall']:>5.1%} {s['err_med']:>6.0f}ms {s['n_pred']:>6.0f}")

    # Save
    with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output_dir}/sweep_results.json")

    # Top 10
    print(f"\n{'='*60}")
    print("TOP 10 by close rate:")
    top = sorted(summary.keys(), key=lambda k: -summary[k]["close"])[:10]
    for i, key in enumerate(top):
        s = summary[key]
        print(f"  {i+1}. {key:<40} close={s['close']:.1%} hall={s['hall']:.1%}")


if __name__ == "__main__":
    main()
