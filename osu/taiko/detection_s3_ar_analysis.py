"""S3 AR analysis — single config, detailed stats.

Runs S1+S2v2+S3 AR on val songs. Computes GT matching, TaikoNation metrics,
and detailed S1/S2/S3 agreement analysis.

Usage:
    cd osu/taiko
    python detection_s3_ar_analysis.py \
        --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
        --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt \
        --s3-checkpoint runs/s3v2_experiment_65/checkpoints/best.pt
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
TN_STEP_MS = 23

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
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("_frozen_")}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_s2v2(ckpt_path, device):
    from detection_s2v2_model import ContextProposer
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ContextProposer(d_model=a.get("d_model", 256),
                            n_gru_layers=a.get("n_gru_layers", 4),
                            b_pred=B_PRED, max_events=C_EVENTS)
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("_frozen_")}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_s3(ckpt_path, device, s1_d_model=384):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    state_keys = set(ckpt["model"].keys())

    if "input_proj.0.weight" in state_keys and "head.1.weight" in state_keys:
        from detection_s3v3_model import PureProposalFusion
        model = PureProposalFusion(
            d_model=a.get("d_model", 128),
            n_layers=a.get("enc_layers", 2) + a.get("fusion_layers", 2),
            n_heads=a.get("n_heads", 4),
            n_classes=B_PRED + 1, b_pred=B_PRED,
        )
        model._is_classifier = True
        model._is_pure_proposal = True
    elif "head_smooth.0.weight" in state_keys:
        from detection_s3v2_model import FusionClassifier
        model = FusionClassifier(
            d_model=a.get("d_model", 384), d_audio=s1_d_model,
            n_layers=a.get("enc_layers", 4) + a.get("fusion_layers", 4),
            n_heads=a.get("n_heads", 8), n_classes=B_PRED + 1,
            a_bins=A_BINS, b_bins=B_BINS,
            gap_ratios=a.get("gap_ratios", True),
        )
        model._is_classifier = True
        model._is_pure_proposal = False
    else:
        from detection_s3_model import FusionSelector
        model = FusionSelector(
            d_model=a.get("d_model", 192), d_audio=s1_d_model,
            n_enc_layers=a.get("n_enc_layers", 4),
            n_dec_layers=a.get("n_dec_layers", 3),
            n_heads=8, a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED,
        )
        model._is_classifier = False

    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("_frozen_")}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════
#  Audio / songs
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


# ═══════════════════════════════════════════════════════════════
#  AR inference with per-step stats
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_ar(s1_model, s2_model, s3_model, mel, song, device, hop_bins=20):
    total_frames = mel.shape[1]
    events = []
    cursor = 0
    cond = np.array([song["density_mean"], song["density_peak"], song["density_std"]], dtype=np.float32)
    cond_t = torch.tensor(cond).unsqueeze(0).to(device)
    is_classifier = getattr(s3_model, '_is_classifier', False)
    is_pure = getattr(s3_model, '_is_pure_proposal', False)

    # Per-step tracking
    step_stats = []

    for _ in range(50000):
        if cursor >= total_frames:
            break

        mel_w = extract_mel_window(mel, cursor)
        mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)

        # S1
        s1_logits = s1_model(mel_t)
        s1_conf = torch.sigmoid(s1_logits).squeeze(0).cpu().numpy()
        audio_feat = s1_model.conv_norm(s1_model.conv(mel_t).transpose(1, 2))

        # S2v2
        g, r, sm, eo, em = build_context(events, cursor)
        s2_conf = torch.sigmoid(s2_model(
            torch.from_numpy(g).unsqueeze(0).to(device),
            torch.from_numpy(r).unsqueeze(0).to(device),
            torch.from_numpy(sm).unsqueeze(0).to(device),
            cond_t)).squeeze(0).cpu().numpy()

        # S3
        s1_conf_t = torch.from_numpy(s1_conf).unsqueeze(0).to(device)
        s2_conf_t = torch.from_numpy(s2_conf).unsqueeze(0).to(device)

        if is_pure:
            s3_input = (s1_conf_t, s2_conf_t)
        else:
            s3_input = (
                audio_feat, s1_conf_t, s2_conf_t,
                torch.from_numpy(eo).unsqueeze(0).to(device),
                torch.from_numpy(em).unsqueeze(0).to(device),
                cond_t,
            )

        if is_classifier:
            s3_logits = s3_model(*s3_input)
            s3_probs = torch.softmax(s3_logits, dim=-1).squeeze(0).cpu().numpy()
            pred = int(np.argmax(s3_probs))
            is_stop = pred >= B_PRED
            pred_conf = s3_probs[pred]
        else:
            s3_logits, _ = s3_model(*s3_input)
            s3_conf = torch.sigmoid(s3_logits).squeeze(0).cpu().numpy()
            pred = int(s3_conf.argmax())
            is_stop = s3_conf[pred] < 0.5
            pred_conf = s3_conf[pred]

        # Track per-step stats
        if not is_stop:
            s1_at_pred = float(s1_conf[min(pred, B_PRED - 1)])
            s2_at_pred = float(s2_conf[min(pred, B_PRED - 1)])
            s1_max = float(s1_conf.max())
            s2_max = float(s2_conf.max())
            s1_argmax = int(s1_conf.argmax())
            s2_argmax = int(s2_conf.argmax())

            step_stats.append({
                "pred": pred,
                "pred_conf": float(pred_conf),
                "s1_at_pred": s1_at_pred,
                "s2_at_pred": s2_at_pred,
                "s1_max": s1_max,
                "s2_max": s2_max,
                "s1_argmax": s1_argmax,
                "s2_argmax": s2_argmax,
                "s1_agrees": abs(s1_argmax - pred) <= 3,
                "s2_agrees": abs(s2_argmax - pred) <= 3,
            })

        # Advance
        if is_stop:
            cursor += hop_bins
        else:
            events.append(cursor + pred)
            cursor = cursor + pred

    return events, step_stats


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def compute_gt_metrics(pred_ms, gt_ms):
    if len(pred_ms) == 0 or len(gt_ms) == 0:
        return None
    ps = np.sort(pred_ms)
    gs = np.sort(gt_ms)
    def _c(arr, v):
        i = np.searchsorted(arr, v)
        best = float("inf")
        for j in [i-1, i, i+1]:
            if 0 <= j < len(arr):
                best = min(best, abs(arr[j] - v))
        return best
    ge = np.array([_c(ps, g) for g in gs])
    pe = np.array([_c(gs, p) for p in ps])
    n_pred, n_gt = len(ps), len(gs)
    pd = n_pred / max((ps[-1]-ps[0])/1000, 0.1) if n_pred > 1 else 0
    gd = n_gt / max((gs[-1]-gs[0])/1000, 0.1) if n_gt > 1 else 0
    return {
        "n_pred": n_pred, "n_gt": n_gt,
        "matched_rate": float((ge <= 25).mean()),
        "close_rate": float((ge <= 50).mean()),
        "far_rate": float((ge > 100).mean()),
        "hallucination_rate": float((pe > 100).mean()),
        "gt_error_mean": float(ge.mean()),
        "gt_error_median": float(np.median(ge)),
        "density_ratio": pd / max(gd, 0.01),
    }


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


def compute_tn_metrics(pred_ms, gt_ms):
    pb = events_ms_to_binary(pred_ms)
    gb = events_ms_to_binary(gt_ms)
    if len(pb) < 16 or len(gb) < 16:
        return {}
    ml = max(len(pb), len(gb))
    pp = np.zeros(ml, dtype=np.int32)
    gp = np.zeros(ml, dtype=np.int32)
    pp[:len(pb)] = pb
    gp[:len(gb)] = gb

    # Over P-Space
    scale = 8
    patterns = set()
    for i in range(len(pp) - scale + 1):
        patterns.add(tuple(pp[i:i+scale]))
    over_ps = len(patterns) / 2**scale * 100

    # HI P-Space
    ai_p = set()
    hu_p = set()
    for i in range(len(pp) - scale + 1):
        ai_p.add(tuple(pp[i:i+scale]))
    for i in range(len(gp) - scale + 1):
        hu_p.add(tuple(gp[i:i+scale]))
    hi_ps = len(ai_p & hu_p) / max(len(hu_p), 1) * 100

    # DCHuman
    limit = min(len(pp), len(gp))
    start = 0
    for i in range(limit):
        if gp[i] == 1:
            start = i
            break
    total = limit - start
    dc_h = float((pp[start:limit] == gp[start:limit]).sum() / max(total, 1) * 100) if total > 0 else 0

    return {"over_pspace": over_ps, "hi_pspace": hi_ps, "dc_human": dc_h}


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--s3-checkpoint", required=True)
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "experiments", "experiment_65_s3v2", "ar_eval"))
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
    is_cls = getattr(s3_model, '_is_classifier', False)
    print(f"  S3 type: {'classifier (single-onset)' if is_cls else 'selector (per-bin)'}")

    all_gt = []
    all_tn = []
    all_step_stats = []

    for si, song in enumerate(songs):
        print(f"\n[{si+1}/{len(songs)}] {song['artist'][:25]} - {song['title'][:25]} (d={song['density_mean']:.1f})")

        mel, duration = load_audio_mel(song["audio_path"], args.device)
        gt_events = np.load(os.path.join(DATASET_DIR, "events", song["event_file"]))
        gt_ms = gt_events.astype(np.float64) * BIN_MS

        events, step_stats = run_ar(s1_model, s2_model, s3_model, mel, song, args.device)
        pred_ms = np.array(events, dtype=np.float64) * BIN_MS

        gt_result = compute_gt_metrics(pred_ms, gt_ms)
        tn_result = compute_tn_metrics(pred_ms, gt_ms)

        if gt_result:
            all_gt.append(gt_result)
            print(f"  close={gt_result['close_rate']:.1%} hall={gt_result['hallucination_rate']:.1%} "
                  f"d={gt_result['density_ratio']:.2f} err={gt_result['gt_error_median']:.0f}ms "
                  f"events={gt_result['n_pred']}")
        if tn_result:
            all_tn.append(tn_result)

        all_step_stats.extend(step_stats)

    # ═══════════════════════════════════════════════════════════
    #  Aggregate
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"RESULTS ({len(songs)} songs, {len(all_step_stats)} AR steps)")
    print(f"{'='*80}")

    if all_gt:
        avg = lambda k: float(np.mean([r[k] for r in all_gt]))
        print(f"\nGT Matching:")
        print(f"  Close (<50ms):    {avg('close_rate'):.1%}")
        print(f"  Matched (<25ms):  {avg('matched_rate'):.1%}")
        print(f"  Far (>100ms):     {avg('far_rate'):.1%}")
        print(f"  Hallucination:    {avg('hallucination_rate'):.1%}")
        print(f"  Density ratio:    {avg('density_ratio'):.2f}")
        print(f"  Error median:     {avg('gt_error_median'):.0f}ms")

    if all_tn:
        avg = lambda k: float(np.mean([r[k] for r in all_tn if k in r]))
        print(f"\nTaikoNation Metrics:")
        print(f"  Over. P-Space:    {avg('over_pspace'):.1f}%")
        print(f"  HI P-Space:       {avg('hi_pspace'):.1f}%")
        print(f"  DCHuman:          {avg('dc_human'):.1f}%")

    # ═══════════════════════════════════════════════════════════
    #  S1/S2/S3 agreement analysis
    # ═══════════════════════════════════════════════════════════
    if all_step_stats:
        stats = all_step_stats
        n = len(stats)
        print(f"\nS1/S2/S3 Agreement ({n} onset steps):")

        s1_at = np.array([s["s1_at_pred"] for s in stats])
        s2_at = np.array([s["s2_at_pred"] for s in stats])
        s1_max = np.array([s["s1_max"] for s in stats])
        s2_max = np.array([s["s2_max"] for s in stats])
        s1_agrees = np.array([s["s1_agrees"] for s in stats])
        s2_agrees = np.array([s["s2_agrees"] for s in stats])
        pred_conf = np.array([s["pred_conf"] for s in stats])

        print(f"  S1 conf at S3's pick:   mean={s1_at.mean():.3f}  med={np.median(s1_at):.3f}")
        print(f"  S2 conf at S3's pick:   mean={s2_at.mean():.3f}  med={np.median(s2_at):.3f}")
        print(f"  S1 max conf:            mean={s1_max.mean():.3f}")
        print(f"  S2 max conf:            mean={s2_max.mean():.3f}")
        print(f"  S3 pred conf:           mean={pred_conf.mean():.3f}")
        print(f"  S1 agrees (±3 bins):    {s1_agrees.mean():.1%}")
        print(f"  S2 agrees (±3 bins):    {s2_agrees.mean():.1%}")
        print(f"  Both agree:             {(s1_agrees & s2_agrees).mean():.1%}")
        print(f"  Neither agrees:         {(~s1_agrees & ~s2_agrees).mean():.1%}")

        # S1 high vs low at pred
        s1_high = s1_at > 0.5
        s2_high = s2_at > 0.5
        print(f"\n  S1 conf > 0.5 at pred:  {s1_high.mean():.1%}")
        print(f"  S2 conf > 0.5 at pred:  {s2_high.mean():.1%}")
        print(f"  Both > 0.5:             {(s1_high & s2_high).mean():.1%}")
        print(f"  S1 only > 0.5:          {(s1_high & ~s2_high).mean():.1%}")
        print(f"  S2 only > 0.5:          {(~s1_high & s2_high).mean():.1%}")
        print(f"  Neither > 0.5:          {(~s1_high & ~s2_high).mean():.1%}")

        # When S3 picks where S1 disagrees
        s1_disagree = ~s1_agrees
        if s1_disagree.sum() > 0:
            print(f"\n  When S1 disagrees ({s1_disagree.sum()} steps):")
            print(f"    S2 agrees:     {s2_agrees[s1_disagree].mean():.1%}")
            print(f"    S2 conf:       {s2_at[s1_disagree].mean():.3f}")
            print(f"    S3 conf:       {pred_conf[s1_disagree].mean():.3f}")

        s2_disagree = ~s2_agrees
        if s2_disagree.sum() > 0:
            print(f"  When S2 disagrees ({s2_disagree.sum()} steps):")
            print(f"    S1 agrees:     {s1_agrees[s2_disagree].mean():.1%}")
            print(f"    S1 conf:       {s1_at[s2_disagree].mean():.3f}")
            print(f"    S3 conf:       {pred_conf[s2_disagree].mean():.3f}")

    # ═══════════════════════════════════════════════════════════
    #  Comparison reference
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("Reference (previous models, song_density regime):")
    print("  exp58:  close=75.9%  hall=15.6%  d_ratio=0.92  err=8ms  P-Space=10.1%")
    print("  exp62:  close=75.0%  hall=15.9%  d_ratio=0.97  err=8ms  P-Space=12.0%")

    # Save
    results = {}
    if all_gt:
        avg = lambda k: float(np.mean([r[k] for r in all_gt]))
        results["gt"] = {k: avg(k) for k in all_gt[0]}
    if all_tn:
        avg = lambda k: float(np.mean([r[k] for r in all_tn if k in r]))
        results["tn"] = {k: avg(k) for k in all_tn[0]}
    if all_step_stats:
        results["agreement"] = {
            "s1_conf_at_pred": float(s1_at.mean()),
            "s2_conf_at_pred": float(s2_at.mean()),
            "s1_agrees": float(s1_agrees.mean()),
            "s2_agrees": float(s2_agrees.mean()),
            "both_agree": float((s1_agrees & s2_agrees).mean()),
            "neither_agree": float((~s1_agrees & ~s2_agrees).mean()),
            "s1_high_at_pred": float(s1_high.mean()),
            "s2_high_at_pred": float(s2_high.mean()),
        }

    with open(os.path.join(args.output_dir, "ar_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output_dir}/ar_results.json")


if __name__ == "__main__":
    main()
