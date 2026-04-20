"""S3 3-stage inference: audio file → onset CSV + viewer data.

Runs S1 (audio) + S2v2 (context) + S3 (fusion) in autoregressive mode.
Supports configurable combination method and sampling strategy.

Usage:
    cd osu/taiko
    python detection_s3_inference.py --audio song.mp3 --density-mean 5.0 --density-peak 10
    python detection_s3_inference.py --audio song.mp3 --method S3_ONLY --sampling FIRST_THRESH --threshold 0.5
    python detection_s3_inference.py --audio song.mp3 --andlaunch
"""

import argparse
import json
import os
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


def load_s1(ckpt_path, device):
    from detection_s1_model import ConformerProposer
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ConformerProposer(n_mels=N_MELS, d_model=a.get("d_model", 384),
                              n_layers=a.get("n_layers", 8), n_heads=8,
                              conv_kernel=a.get("conv_kernel", 31),
                              a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED)
    # Filter out frozen S1/S2 weights that got saved with the checkpoint
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
    # Filter out frozen S1/S2 weights that got saved with the checkpoint
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("_frozen_")}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_s3(ckpt_path, device, s1_d_model=384):
    """Load S3 — auto-detects FusionSelector (per-bin) vs FusionClassifier (single-onset)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    state_keys = set(ckpt["model"].keys())

    # Detect model type: FusionClassifier has head_smooth, FusionSelector has dec_self_attn
    if "head_smooth.0.weight" in state_keys:
        from detection_s3v2_model import FusionClassifier
        model = FusionClassifier(
            d_model=a.get("d_model", 384), d_audio=s1_d_model,
            n_layers=a.get("enc_layers", 4) + a.get("fusion_layers", 4),
            n_heads=a.get("n_heads", 8), n_classes=B_PRED + 1,
            a_bins=A_BINS, b_bins=B_BINS,
            gap_ratios=a.get("gap_ratios", True),
        )
        model._is_classifier = True
    else:
        from detection_s3_model import FusionSelector
        model = FusionSelector(
            d_model=a.get("d_model", 192), d_audio=s1_d_model,
            n_enc_layers=a.get("n_enc_layers", 4),
            n_dec_layers=a.get("n_dec_layers", 3),
            n_heads=8, a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED,
        )
        model._is_classifier = False

    # Filter out frozen S1/S2 weights that got saved with the checkpoint
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("_frozen_")}
    model.load_state_dict(state)
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
    return mel.cpu().numpy().astype(np.float32), len(y) / SAMPLE_RATE, y


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
    if method == "MAX_S1S2": return np.maximum(s1, s2)
    return s3


@torch.no_grad()
def run_inference(s1_model, s2_model, s3_model, mel, conditioning, device,
                  method="S3_ONLY", sampling="FIRST_THRESH", threshold=0.5, hop_bins=20):
    total_frames = mel.shape[1]
    events = []
    cursor = 0
    cond_t = torch.tensor(conditioning, dtype=torch.float32).unsqueeze(0).to(device)
    duration_s = total_frames * BIN_MS / 1000

    pbar = tqdm(total=total_frames, desc="Inference",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

    t_start = time.perf_counter()
    stop_count = 0

    # Per-step confidence maps for viewer
    step_cursors = []
    step_s1 = []
    step_s2 = []
    step_s3 = []

    for _ in range(100000):
        if cursor >= total_frames:
            break
        pbar.n = min(cursor, total_frames)
        pbar.set_postfix_str(f"{cursor*BIN_MS/1000:.1f}s/{duration_s:.1f}s, {len(events)} events, {stop_count} stops")
        pbar.refresh()

        mel_w = extract_mel_window(mel, cursor)
        mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)

        # S1
        s1_conf = torch.sigmoid(s1_model(mel_t)).squeeze(0).cpu().numpy()
        audio_feat = s1_model.conv_norm(s1_model.conv(mel_t).transpose(1, 2))

        # S2v2
        g, r, sm, eo, em = build_context(events, cursor)
        s2_conf = torch.sigmoid(s2_model(
            torch.from_numpy(g).unsqueeze(0).to(device),
            torch.from_numpy(r).unsqueeze(0).to(device),
            torch.from_numpy(sm).unsqueeze(0).to(device),
            cond_t)).squeeze(0).cpu().numpy()

        # S3 (auto-detect per-bin vs single-onset)
        is_classifier = getattr(s3_model, '_is_classifier', False)
        s3_input = (
            audio_feat,
            torch.from_numpy(s1_conf).unsqueeze(0).to(device),
            torch.from_numpy(s2_conf).unsqueeze(0).to(device),
            torch.from_numpy(eo).unsqueeze(0).to(device),
            torch.from_numpy(em).unsqueeze(0).to(device),
            cond_t,
        )
        if is_classifier:
            # FusionClassifier: returns (B, 251) logits — single onset
            s3_logits = s3_model(*s3_input)
            s3_conf = torch.softmax(s3_logits, dim=-1).squeeze(0).cpu().numpy()  # (251,)
            # For per-bin confidence maps (viewer), use softmax over onset bins
            s3_conf_bins = s3_conf[:B_PRED]  # (250,) drop STOP
        else:
            # FusionSelector: returns ((B, 250) logits, aux_list) — per-bin
            s3_logits, _ = s3_model(*s3_input)
            s3_conf = torch.sigmoid(s3_logits).squeeze(0).cpu().numpy()  # (250,)
            s3_conf_bins = s3_conf

        # Save per-step data for viewer (always 250 bins)
        step_cursors.append(cursor)
        step_s1.append(s1_conf.astype(np.float16))
        step_s2.append(s2_conf.astype(np.float16))
        step_s3.append(s3_conf_bins.astype(np.float16))

        # For classifier mode, use argmax directly (single-onset like exp58)
        if is_classifier and method == "S3_ONLY" and sampling == "MAX":
            pred = int(np.argmax(s3_conf))  # includes STOP at index 250
            if pred >= B_PRED:
                # STOP
                cursor += hop_bins
                stop_count += 1
            else:
                events.append(cursor + pred)
                cursor = cursor + pred
            continue

        # Combine (per-bin mode)
        conf = combine(s1_conf, s2_conf, s3_conf_bins, method)

        # Sample
        if sampling == "MAX":
            best = int(conf.argmax())
            if conf[best] < threshold:
                cursor += hop_bins
                stop_count += 1
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
                stop_count += 1

        elif sampling == "ALL_THRESH":
            above = np.where(conf >= threshold)[0]
            if len(above) > 0:
                for b in above:
                    events.append(cursor + int(b))
                cursor = cursor + int(above[-1])
            else:
                cursor += hop_bins
                stop_count += 1

    pbar.close()
    elapsed = time.perf_counter() - t_start

    stats = {
        "n_events": len(events),
        "n_stops": stop_count,
        "duration_s": duration_s,
        "events_per_sec": len(events) / max(duration_s, 0.1),
        "inference_time_s": elapsed,
        "method": method,
        "sampling": sampling,
        "threshold": threshold,
    }

    conf_maps = {
        "cursors": step_cursors,
        "s1": step_s1,
        "s2": step_s2,
        "s3": step_s3,
    }

    return events, stats, conf_maps


def events_to_csv(events, output_path, audio_name=""):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# S3 3-stage prediction\n")
        f.write(f"# audio: {audio_name}\n")
        f.write("time_ms,type\n")
        for e in events:
            ms = int(e * BIN_MS)
            f.write(f"{ms},don\n")


def main():
    parser = argparse.ArgumentParser(description="S3 3-stage onset detection inference")
    parser.add_argument("--audio", default=None, nargs="?", const="__picker__",
                        help="Path to audio file (opens file picker if no path given)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--s1-checkpoint", default=os.path.join(SCRIPT_DIR, "runs", "s1_experiment_65", "checkpoints", "best.pt"))
    parser.add_argument("--s2-checkpoint", default=os.path.join(SCRIPT_DIR, "runs", "s2v2_experiment_65", "checkpoints", "best.pt"))
    parser.add_argument("--s3-checkpoint", default=os.path.join(SCRIPT_DIR, "runs", "s3_experiment_65_rerun", "checkpoints", "best.pt"))
    parser.add_argument("--density-mean", type=float, default=5.0)
    parser.add_argument("--density-peak", type=float, default=10.0)
    parser.add_argument("--density-std", type=float, default=1.5)
    parser.add_argument("--method", default="S3_ONLY",
                        choices=["S1_ONLY", "S2_ONLY", "S3_ONLY", "ADD_EQUAL", "ADD_S3_HEAVY", "ADD_S1S2", "MULTIPLY", "MAX_S1S2"])
    parser.add_argument("--sampling", default="FIRST_THRESH",
                        choices=["MAX", "FIRST_THRESH", "ALL_THRESH"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hop-ms", type=float, default=100, help="STOP hop in ms")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--andlaunch", action="store_true", help="Launch viewer after inference")
    args = parser.parse_args()

    # Temp dir for viewer-only mode (no explicit output)
    tmp_dir = None
    if args.output is None and args.andlaunch:
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="s3_inference_")

    # File picker if --audio with no path
    if args.audio is None or args.audio == "__picker__":
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            args.audio = filedialog.askopenfilename(
                title="Select audio file",
                filetypes=[("Audio", "*.mp3 *.ogg *.wav *.flac"), ("All", "*.*")])
            root.destroy()
            if not args.audio:
                print("No file selected")
                sys.exit(0)
        except Exception:
            print("ERROR: --audio is required (no GUI available for file picker)")
            sys.exit(1)

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.audio))[0]
        if tmp_dir:
            args.output = os.path.join(tmp_dir, f"{stem}_s3_predicted.csv")
        else:
            args.output = os.path.join(SCRIPT_DIR, f"{stem}_s3_predicted.csv")

    print(f"S3 3-Stage Inference")
    print(f"  Audio: {args.audio}")
    print(f"  Method: {args.method}")
    print(f"  Sampling: {args.sampling} @ {args.threshold}")
    print(f"  Density: mean={args.density_mean} peak={args.density_peak} std={args.density_std}")
    print(f"  Output: {args.output}")
    print()

    # Load models
    load_pbar = tqdm(total=4, desc="Loading", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}")
    load_pbar.set_postfix_str("S1 audio model")
    s1 = load_s1(args.s1_checkpoint, args.device)
    load_pbar.update(1)
    load_pbar.set_postfix_str("S2v2 context model")
    s2 = load_s2v2(args.s2_checkpoint, args.device)
    load_pbar.update(1)
    load_pbar.set_postfix_str("S3 fusion model")
    s3 = load_s3(args.s3_checkpoint, args.device, s1_d_model=s1.d_model)
    load_pbar.update(1)
    load_pbar.set_postfix_str("Audio + mel spectrogram")
    mel, duration, y_raw = load_audio_mel(args.audio, args.device)
    load_pbar.update(1)
    load_pbar.close()
    print(f"  S1: {sum(p.numel() for p in s1.parameters()):,} | S2v2: {sum(p.numel() for p in s2.parameters()):,} | S3: {sum(p.numel() for p in s3.parameters()):,}")
    print(f"  Audio: {duration:.1f}s, mel: {mel.shape}")

    # Run inference
    hop_bins = max(1, int(args.hop_ms / BIN_MS))
    conditioning = [args.density_mean, args.density_peak, args.density_std]
    events, stats, conf_maps = run_inference(s1, s2, s3, mel, conditioning, args.device,
                                  method=args.method, sampling=args.sampling,
                                  threshold=args.threshold, hop_bins=hop_bins)

    print(f"\nPredicted {len(events)} events ({len(events)/duration:.1f}/s) in {stats['inference_time_s']:.1f}s")
    print(f"  Stops: {stats['n_stops']}")

    # Save all output files
    save_pbar = tqdm(total=5, desc="Saving", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}")

    # 1. CSV
    save_pbar.set_postfix_str("onset CSV")
    events_to_csv(events, args.output, audio_name=os.path.abspath(args.audio))
    save_pbar.update(1)

    # 2. Mel spectrogram for viewer
    save_pbar.set_postfix_str("mel spectrogram")
    mel_npy_path = args.output.replace(".csv", "_mel.npy")
    np.save(mel_npy_path, mel)
    save_pbar.update(1)

    # 3. Waveform envelope for viewer
    save_pbar.set_postfix_str("waveform envelope")
    wave_npy_path = args.output.replace(".csv", "_wave.npy")
    hop = HOP_LENGTH
    n_frames = mel.shape[1]
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        e = min(s + hop, len(y_raw))
        if s < len(y_raw):
            envelope[i] = np.max(np.abs(y_raw[s:e]))
    np.save(wave_npy_path, envelope)
    save_pbar.update(1)

    # 4. Per-step S1/S2/S3 confidence maps for viewer
    step_cursors = conf_maps["cursors"]
    save_pbar.set_postfix_str(f"confidence maps ({len(step_cursors)} steps)")
    conf_path = args.output.replace(".csv", "_s3confs.npz")
    np.savez_compressed(conf_path,
                        cursors=np.array(step_cursors, dtype=np.int32),
                        s1=np.array(conf_maps["s1"], dtype=np.float16),
                        s2=np.array(conf_maps["s2"], dtype=np.float16),
                        s3=np.array(conf_maps["s3"], dtype=np.float16))
    save_pbar.update(1)

    # 5. Stats JSON
    save_pbar.set_postfix_str("stats JSON")
    stats_path = args.output.replace(".csv", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    save_pbar.update(1)
    save_pbar.close()

    print(f"\nOutput files:")
    print(f"  CSV:    {args.output}")
    print(f"  Mel:    {mel_npy_path}")
    print(f"  Wave:   {wave_npy_path}")
    print(f"  Confs:  {conf_path}")
    print(f"  Stats:  {stats_path}")

    # Launch viewer
    if args.andlaunch:
        import subprocess
        viewer_path = os.path.join(SCRIPT_DIR, "viewer.py")
        cmd = [sys.executable, viewer_path, args.output, "--audio", args.audio,
               "--stats-json", stats_path,
               "--mel-npy", mel_npy_path, "--wave-npy", wave_npy_path,
               "--s3confs-npz", conf_path]
        print(f"\nLaunching viewer...")
        subprocess.run(cmd)

        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"Cleaned up temp dir: {tmp_dir}")


if __name__ == "__main__":
    main()
