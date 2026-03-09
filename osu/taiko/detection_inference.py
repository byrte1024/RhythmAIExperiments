"""Run onset detection on an audio file, output a CSV like the preprocessed data.

Usage:
  python detection_inference.py --checkpoint checkpoints/taiko_v1/best.pt --audio song.mp3
  python detection_inference.py --checkpoint checkpoints/taiko_v1/best.pt --audio song.mp3 --density-mean 5.0 --density-peak 10
"""
import os
import argparse
import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm

from detection_model import OnsetDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# must match training
SAMPLE_RATE = 22050
HOP_LENGTH = 110
N_FFT = 2048
N_MELS = 80
F_MIN = 20.0
F_MAX = 8000.0
BIN_MS = HOP_LENGTH / SAMPLE_RATE * 1000  # exact: ~4.9887ms per mel frame

A_BINS = 500
B_BINS = 500
C_EVENTS = 128
N_CLASSES = 501


def load_audio_mel(audio_path, device):
    """Load audio and compute mel spectrogram on GPU."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX, power=2.0,
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)

    wav = torch.from_numpy(y).float().to(device)
    with torch.no_grad():
        mel = amp_to_db(mel_transform(wav))  # (n_mels, T)

    return mel.cpu().numpy().astype(np.float32), len(y) / SAMPLE_RATE


def extract_mel_window(mel, cursor):
    """Extract padded mel window around cursor."""
    n_mels, total_frames = mel.shape
    start = cursor - A_BINS
    end = cursor + B_BINS

    pad_left = max(0, -start)
    pad_right = max(0, end - total_frames)
    s = max(0, start)
    e = min(total_frames, end)
    window = mel[:, s:e]

    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, pad_right)), mode="constant")

    return window


@torch.no_grad()
def run_inference(model, mel, conditioning, device, hop_bins=20, max_events=10000):
    """Autoregressive inference: predict events one at a time.

    hop_bins: how far to advance cursor on STOP (default 20 = 0.1s at 5ms bins).
    """
    model.eval()
    total_frames = mel.shape[1]
    events = []  # list of bin positions
    cursor = 0

    cond_tensor = torch.tensor(conditioning, dtype=torch.float32).unsqueeze(0).to(device)
    duration_s = total_frames * BIN_MS / 1000
    pbar = tqdm(total=total_frames, desc="Inference", unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]")

    for _ in range(max_events):
        if cursor >= total_frames:
            break
        pbar.n = min(cursor, total_frames)
        pbar.set_postfix_str(f"{cursor*BIN_MS/1000:.1f}s/{duration_s:.1f}s, {len(events)} events")
        pbar.refresh()

        # mel window
        mel_window = extract_mel_window(mel, cursor)
        mel_tensor = torch.from_numpy(mel_window).unsqueeze(0).to(device)

        # past events as offsets from cursor
        if len(events) > 0:
            past = np.array(events[-C_EVENTS:], dtype=np.int64) - cursor
            n_past = len(past)
        else:
            past = np.array([], dtype=np.int64)
            n_past = 0

        event_offsets = np.zeros(C_EVENTS, dtype=np.int64)
        event_mask = np.ones(C_EVENTS, dtype=bool)
        if n_past > 0:
            event_offsets[-n_past:] = past
            event_mask[-n_past:] = False

        evt_tensor = torch.from_numpy(event_offsets).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(event_mask).unsqueeze(0).to(device)

        logits, _audio_logits, _context_logits = model(mel_tensor, evt_tensor, mask_tensor, cond_tensor)
        pred = logits.argmax(dim=1).item()

        if pred == N_CLASSES - 1:  # STOP
            cursor += hop_bins
            continue

        # predicted bin offset from cursor
        event_bin = cursor + pred
        events.append(event_bin)
        cursor = event_bin  # move cursor to this event

    pbar.n = total_frames
    pbar.set_postfix_str(f"{duration_s:.1f}s/{duration_s:.1f}s, {len(events)} events")
    pbar.refresh()
    pbar.close()
    return events


def events_to_csv(events, output_path, audio_name=""):
    """Write events as a CSV matching the preprocessed format."""
    with open(output_path, "w", encoding="utf-8") as f:
        if audio_name:
            f.write(f"# audio: {audio_name}\n")
        f.write("time_ms,type\n")
        for bin_idx in events:
            time_ms = int(bin_idx * BIN_MS)
            f.write(f"{time_ms},predicted\n")
    print(f"Wrote {len(events)} events to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run onset detection inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--audio", default=None, help="Path to audio file (opens file browser if not set)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: {audio_stem}_predicted.csv)")
    parser.add_argument("--density-mean", type=float, default=4.3, help="Target mean density (events/sec)")
    parser.add_argument("--density-peak", type=float, default=8.0, help="Target peak density")
    parser.add_argument("--density-std", type=float, default=1.5, help="Target density std")
    parser.add_argument("--hop-ms", type=float, default=100, help="Cursor hop on STOP prediction (ms, default 100)")
    parser.add_argument("--andlaunch", action="store_true", help="Launch viewer after inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # pick audio file
    if args.audio is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        args.audio = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.mp3 *.ogg *.wav *.flac *.m4a"), ("All files", "*.*")],
        )
        root.destroy()
        if not args.audio:
            print("No file selected, exiting.")
            return

    # load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]

    model = OnsetDetector(
        n_mels=N_MELS,
        d_model=ckpt_args.get("d_model", 384),
        d_event=ckpt_args.get("d_event", 128),
        enc_layers=ckpt_args.get("enc_layers", 4),
        enc_event_layers=ckpt_args.get("enc_event_layers", 2),
        audio_path_layers=ckpt_args.get("audio_path_layers", 2),
        context_path_layers=ckpt_args.get("context_path_layers", 3),
        n_heads=ckpt_args.get("n_heads", 8),
        n_classes=N_CLASSES,
        max_events=C_EVENTS,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    print(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, acc={ckpt.get('val_metrics', {}).get('accuracy', 0):.3f}")

    # load audio
    print(f"Loading audio: {args.audio}")
    mel, duration = load_audio_mel(args.audio, args.device)
    print(f"  Duration: {duration:.1f}s, {mel.shape[1]} mel frames")

    # run inference
    conditioning = [args.density_mean, args.density_peak, args.density_std]
    print(f"  Conditioning: mean={conditioning[0]}, peak={conditioning[1]}, std={conditioning[2]}")

    hop_bins = max(1, int(args.hop_ms / BIN_MS))
    print(f"  STOP hop: {args.hop_ms}ms = {hop_bins} bins")
    events = run_inference(model, mel, conditioning, args.device, hop_bins=hop_bins)
    print(f"  Predicted {len(events)} events ({len(events) / duration:.1f}/s)")

    # write CSV
    tmp_dir = None
    if args.output is None:
        if args.andlaunch:
            import tempfile
            tmp_dir = tempfile.mkdtemp(prefix="beatdetect_")
            stem = os.path.splitext(os.path.basename(args.audio))[0]
            args.output = os.path.join(tmp_dir, f"{stem}_predicted.csv")
        else:
            stem = os.path.splitext(os.path.basename(args.audio))[0]
            args.output = os.path.join(SCRIPT_DIR, f"{stem}_predicted.csv")

    events_to_csv(events, args.output, audio_name=os.path.abspath(args.audio))

    if args.andlaunch:
        import subprocess, sys
        viewer_path = os.path.join(SCRIPT_DIR, "viewer.py")
        print(f"\nLaunching viewer: {args.output}")
        subprocess.run([sys.executable, viewer_path, args.output, "--audio", args.audio])

    if tmp_dir is not None:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up temp directory: {tmp_dir}")


if __name__ == "__main__":
    main()
