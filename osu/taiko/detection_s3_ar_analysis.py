"""S3 AR analysis with independent AR per combination.

Each combination runs its own AR loop (cursor depends on decisions).
All 3 models run at each step, but each combo makes its own placement decisions.

Usage:
    cd osu/taiko
    python detection_s3_ar_analysis.py \
        --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
        --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt \
        --s3-checkpoint runs/s3_experiment_65/checkpoints/eval_001.pt
"""

import argparse
import json
import os
import random
import sys

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

DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")


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
    past = [e for e in events if e <= cursor][-C_EVENTS:]
    n = len(past)
    gaps = np.zeros(C_EVENTS, dtype=np.float32)
    ratios = np.zeros(C_EVENTS, dtype=np.float32)
    s2_mask = np.ones(C_EVENTS, dtype=bool)
    offsets = np.zeros(C_EVENTS, dtype=np.int64)
    evt_mask = np.ones(C_EVENTS, dtype=bool)

    if n > 0:
        pa = np.array(past, dtype=np.int64)
        rg = np.zeros(n, dtype=np.float32)
        if n >= 2:
            rg[1:] = np.diff(pa).astype(np.float32)
            rg[0] = rg[1]
        else:
            rg[0] = 30.0
        rg = np.maximum(rg, 1.0)
        rr = np.ones(n, dtype=np.float32)
        if n >= 2:
            for i in range(1, n):
                rr[i] = np.clip(rg[i] / max(rg[i-1], 1.0), 0.1, 10.0)
        s = C_EVENTS - n
        gaps[s:] = rg
        ratios[s:] = rr
        s2_mask[s:] = False
        offsets[-n:] = pa - cursor
        evt_mask[-n:] = False

    return gaps, ratios, s2_mask, offsets, evt_mask


def combine(s1, s2, s3, method):
    if method == "S1_ONLY": return s1
    if method == "S2_ONLY": return s2
    if method == "S3_ONLY": return s3
    if method == "ADD_EQUAL": return (s1 + s2 + s3) / 3
    if method == "ADD_S3_HEAVY": return 0.2 * s1 + 0.2 * s2 + 0.6 * s3
    if method == "ADD_S1S2": return (s1 + s2) / 2
    if method == "MULTIPLY": return np.cbrt(s1 * s2 * s3 + 1e-10)
    if method == "MULTIPLY_S1S2": return np.sqrt(s1 * s2 + 1e-10)
    if method == "MAX_S1S2": return np.maximum(s1, s2)
    if method == "MIN_S1S2": return np.minimum(s1, s2)
    return s3


@torch.no_grad()
def run_ar(s1_model, s2_model, s3_model, mel, song, device,
           combo, threshold, sampling, hop_bins=20):
    total_frames = mel.shape[1]
    events = []
    cursor = 0
    cond = np.array([song["density_mean"], song["density_peak"], song["density_std"]], dtype=np.float32)
    cond_t = torch.tensor(cond).unsqueeze(0).to(device)

    for _ in range(50000):
        if cursor >= total_frames:
            break

        mel_w = extract_mel_window(mel, cursor)
        mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)

        s1_conf = torch.sigmoid(s1_model(mel_t)).squeeze(0).cpu().numpy()
        audio_feat = s1_model.conv_norm(s1_model.conv(mel_t).transpose(1, 2))

        g, r, sm, eo, em = build_context(events, cursor)
        s2_conf = torch.sigmoid(s2_model(
            torch.from_numpy(g).unsqueeze(0).to(device),
            torch.from_numpy(r).unsqueeze(0).to(device),
            torch.from_numpy(sm).unsqueeze(0).to(device),
            cond_t)).squeeze(0).cpu().numpy()

        s3_logits, _ = s3_model(
            audio_feat,
            torch.from_numpy(s1_conf).unsqueeze(0).to(device),
            torch.from_numpy(s2_conf).unsqueeze(0).to(device),
            torch.from_numpy(eo).unsqueeze(0).to(device),
            torch.from_numpy(em).unsqueeze(0).to(device),
            cond_t)
        s3_conf = torch.sigmoid(s3_logits).squeeze(0).cpu().numpy()

        conf = combine(s1_conf, s2_conf, s3_conf, combo)

        if sampling == "MAX":
            best = int(conf.argmax())
            if conf[best] < threshold:
                cursor += hop_bins
            else:
                events.append(cursor + best)
                cursor = cursor + best
        elif sampling == "FIRST_THRESH":
            above = np.where(conf >= threshold)[0]
            if len(above) > 0:
                events.append(cursor + int(above[0]))
                cursor = cursor + int(above[0])
            else:
                cursor += hop_bins
        elif sampling == "ALL_THRESH":
            above = np.where(conf >= threshold)[0]
            if len(above) > 0:
                for b in above:
                    events.append(cursor + int(b))
                cursor = cursor + int(above[-1])
            else:
                cursor += hop_bins

    return events


TN_STEP_MS = 23

def events_ms_to_binary(events_ms, step_ms=TN_STEP_MS):
    if len(events_ms) == 0:
        return np.array([], dtype=np.int32)
    max_time = int(max(events_ms)) + step_ms
    n_steps = max_time // step_ms + 1
    binary = np.zeros(n_steps, dtype=np.int32)
    for t in events_ms:
        idx = int(t) // step_ms
        if 0 <= idx < n_steps:
            binary[idx] = 1
    return binary

def tn_over_pspace(chart, scale=8):
    patterns = set()
    last_ind = len(chart) - scale + 1
    if last_ind <= 0:
        return 0.0
    for i in range(last_ind):
        patterns.add(tuple(chart[i:i + scale]))
    return float(len(patterns) / 2**scale * 100)

def tn_hi_pspace(ai_chart, human_chart, scale=8):
    ai_p = set()
    hu_p = set()
    for i in range(len(ai_chart) - scale + 1):
        ai_p.add(tuple(ai_chart[i:i + scale]))
    for i in range(len(human_chart) - scale + 1):
        hu_p.add(tuple(human_chart[i:i + scale]))
    if len(hu_p) == 0:
        return 0.0
    return float(len(ai_p & hu_p) / len(hu_p) * 100)

def tn_dc_human(ai_chart, human_chart):
    limit = min(len(ai_chart), len(human_chart))
    if limit == 0:
        return 0.0
    start = 0
    for i in range(limit):
        if human_chart[i] == 1:
            start = i
            break
    total = limit - start
    if total <= 0:
        return 0.0
    return float((ai_chart[start:limit] == human_chart[start:limit]).sum() / total * 100)

def compute_tn_metrics(pred_ms, gt_ms):
    pb = events_ms_to_binary(pred_ms)
    gb = events_ms_to_binary(gt_ms)
    if len(pb) < 16 or len(gb) < 16:
        return None
    ml = max(len(pb), len(gb))
    pp = np.zeros(ml, dtype=np.int32)
    gp = np.zeros(ml, dtype=np.int32)
    pp[:len(pb)] = pb
    gp[:len(gb)] = gb
    return {
        "over_pspace": tn_over_pspace(pp),
        "hi_pspace": tn_hi_pspace(pp, gp),
        "dc_human": tn_dc_human(pp, gp),
    }

def compute_gt_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0 or len(gt_ms) == 0:
        return None
    ps = np.sort(pred_ms)
    gs = np.sort(gt_ms)
    def _c(arr, v):
        i = np.searchsorted(arr, v)
        return min(abs(arr[max(0,j)] - v) for j in [i-1, i, i+1] if 0 <= j < len(arr))
    ge = np.array([_c(ps, g) for g in gs])
    pe = np.array([_c(gs, p) for p in ps])
    np_, ng = len(ps), len(gs)
    pd = np_ / max((ps[-1]-ps[0])/1000, 0.1) if np_ > 1 else 0
    gd = ng / max((gs[-1]-gs[0])/1000, 0.1) if ng > 1 else 0
    result = {
        "n_pred": np_, "n_gt": ng,
        "close_rate": float((ge <= 50).mean()),
        "far_rate": float((ge > 100).mean()),
        "hallucination_rate": float((pe > 100).mean()),
        "gt_error_median": float(np.median(ge)),
        "density_ratio": pd / max(gd, 0.01),
    }
    tn = compute_tn_metrics(pred_ms, gt_ms)
    if tn:
        result.update(tn)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--s3-checkpoint", required=True)
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "experiments", "experiment_65_s3", "ar_eval"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-songs", type=int, default=10)
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
    print("  Loaded")

    configs = [
        ("S3_ONLY", "MAX", 0.3),
        ("S3_ONLY", "MAX", 0.4),
        ("S3_ONLY", "MAX", 0.5),
        ("S3_ONLY", "FIRST_THRESH", 0.3),
        ("S3_ONLY", "FIRST_THRESH", 0.5),
        ("S1_ONLY", "MAX", 0.5),
        ("S2_ONLY", "MAX", 0.5),
        ("ADD_EQUAL", "MAX", 0.3),
        ("ADD_EQUAL", "MAX", 0.5),
        ("ADD_S3_HEAVY", "MAX", 0.3),
        ("ADD_S3_HEAVY", "MAX", 0.5),
        ("ADD_S1S2", "MAX", 0.5),
        ("MULTIPLY", "MAX", 0.3),
        ("MULTIPLY_S1S2", "MAX", 0.3),
        ("MIN_S1S2", "MAX", 0.3),
        ("S3_ONLY", "ALL_THRESH", 0.5),
        ("S3_ONLY", "ALL_THRESH", 0.6),
        ("ADD_EQUAL", "ALL_THRESH", 0.5),
    ]

    all_results = {f"{c}_{s}_{t}": [] for c, s, t in configs}

    for si, song in enumerate(songs):
        print(f"\n[{si+1}/{len(songs)}] {song['artist'][:20]} - {song['title'][:25]} (d={song['density_mean']:.1f})")

        mel, duration = load_audio_mel(song["audio_path"], args.device)
        gt_events = np.load(os.path.join(DATASET_DIR, "events", song["event_file"]))
        gt_ms = gt_events.astype(np.float64) * BIN_MS

        for combo, sampling, thresh in tqdm(configs, desc="Configs", leave=False):
            key = f"{combo}_{sampling}_{thresh}"
            events = run_ar(s1_model, s2_model, s3_model, mel, song, args.device,
                            combo, thresh, sampling)
            pred_ms = np.array(events, dtype=np.float64) * BIN_MS
            result = compute_gt_metrics(pred_ms, gt_ms)
            if result:
                all_results[key].append(result)

        # Quick print
        for key in ["S3_ONLY_MAX_0.5", "S1_ONLY_MAX_0.5", "ADD_EQUAL_MAX_0.5"]:
            if all_results[key]:
                r = all_results[key][-1]
                print(f"  {key}: close={r['close_rate']:.1%} hall={r['hallucination_rate']:.1%} d={r['density_ratio']:.2f}")

    # Aggregate
    print(f"\n{'='*90}")
    print(f"RESULTS ({len(songs)} songs)")
    print(f"{'='*90}")
    print(f"{'Config':<40} {'Close%':>7} {'Far%':>6} {'Hall%':>6} {'ErrMed':>7} {'d_ratio':>8} {'P-Space':>8} {'HI-PS':>6} {'DCHum':>6}")
    print("-" * 100)

    summary = {}
    for key in sorted(all_results.keys()):
        results = all_results[key]
        if not results:
            continue
        avg = lambda k: float(np.mean([r[k] for r in results if k in r]))
        summary[key] = {
            "close": avg("close_rate"), "far": avg("far_rate"),
            "hall": avg("hallucination_rate"), "err_med": avg("gt_error_median"),
            "d_ratio": avg("density_ratio"), "n_pred": avg("n_pred"),
            "over_pspace": avg("over_pspace"), "hi_pspace": avg("hi_pspace"),
            "dc_human": avg("dc_human"),
        }

    for key in sorted(summary.keys(), key=lambda k: -summary[k]["close"]):
        s = summary[key]
        print(f"  {key:<40} {s['close']:>6.1%} {s['far']:>5.1%} {s['hall']:>5.1%} {s['err_med']:>6.0f}ms {s['d_ratio']:>7.2f} {s['over_pspace']:>7.1f}% {s['hi_pspace']:>5.1f}% {s['dc_human']:>5.1f}%")

    with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Top 5
    print(f"\nTOP 5:")
    for i, key in enumerate(sorted(summary.keys(), key=lambda k: -summary[k]["close"])[:5]):
        s = summary[key]
        print(f"  {i+1}. {key:<35} close={s['close']:.1%} hall={s['hall']:.1%} d={s['d_ratio']:.2f}")

    print(f"\nSaved to {args.output_dir}/sweep_results.json")


if __name__ == "__main__":
    main()
