"""Render a 512x512 heatmap of per-sample scores across the validation set.

Each pixel represents a group of consecutive samples. Color scales from
red (score=-1) to green (score=0.67). Comparing these images across
experiments reveals whether models fail on the same or different samples.

Usage:
    python analyze_val_heatmap.py --checkpoint runs/detect_experiment_44/checkpoints/eval_019.pt --label exp44
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import math
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_train import (
    OnsetDataset, split_by_song, N_CLASSES, C_EVENTS
)
from detection_model import EventEmbeddingDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")


def compute_score(pred, target):
    """Compute per-sample score in [-1, +0.67] matching training's model_score."""
    if target >= N_CLASSES - 1:
        return 0.0  # STOP sample, neutral
    t = float(target)
    p = float(pred)
    frame_err = abs(p - t)
    abs_lr = abs(math.log(p + 1) - math.log(t + 1))
    thr = math.log(1.03)
    max_p = math.log(5.0)
    pen_range = max_p - thr
    r_at_zero = (math.log(3.0) - thr) / pen_range
    if frame_err <= 1:
        return r_at_zero
    if abs_lr <= thr:
        return (1.0 - abs_lr / thr) * r_at_zero
    else:
        return -min((abs_lr - thr) / pen_range, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default="model", help="Label for output filename")
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "experiments"))
    args = parser.parse_args()

    from PIL import Image

    with open(os.path.join(DS_DIR, "manifest.json")) as f:
        manifest = json.load(f)

    # load config from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)

    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)

    val_ds = OnsetDataset(manifest, DS_DIR, val_idx, augment=False, subsample=args.subsample,
                          multi_target=False)
    print(f"Val samples: {len(val_ds)}")

    nw = 3
    val_loader = DataLoader(val_ds, batch_size=48, shuffle=False,
                            num_workers=nw, pin_memory=False, persistent_workers=nw > 0,
                            prefetch_factor=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # auto-detect model architecture from checkpoint keys (same logic as detection_inference.py)
    from detection_model import (OnsetDetector, DualStreamOnsetDetector, InterleavedOnsetDetector,
                                  ContextFiLMDetector, FramewiseOnsetDetector, EventEmbeddingDetector,
                                  AdditiveOnsetDetector, RerankerOnsetDetector, LegacyOnsetDetector,
                                  Exp17OnsetDetector, Exp18OnsetDetector)

    state = ckpt["model"]
    state_keys = set(state.keys())

    has_event_embed = "event_presence_emb" in state_keys
    has_framewise = "onset_feedback_emb" in state_keys
    has_context_film = any("fusion_context_film." in k for k in state_keys)
    has_interleaved = any("audio_self_layers." in k for k in state_keys)
    has_cross_attn_fusion = any("cross_attn_fusion." in k for k in state_keys)
    has_fusion_layers = any("fusion_layers." in k for k in state_keys)
    has_gap_encoder = any("gap_encoder." in k for k in state_keys)
    is_legacy = any("audio_path.layers" in k for k in state_keys)

    if has_event_embed:
        event_proj_key = next((k for k in state_keys if "event_proj.0.weight" in k), None)
        has_gap_ratios = False
        if event_proj_key:
            w = state[event_proj_key]
            if w.shape[1] > ckpt_args.get("d_model", 384) * 3:
                has_gap_ratios = True
        has_stop_token = "stop_query" in state_keys
        model = EventEmbeddingDetector(
            n_mels=80, d_model=ckpt_args.get("d_model", 384),
            n_layers=ckpt_args.get("enc_layers", 4) + ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES, max_events=C_EVENTS,
            dropout=ckpt_args.get("dropout", 0.1),
            gap_ratios=has_gap_ratios,
            stop_token=has_stop_token,
        ).to(device)
    elif has_fusion_layers or has_gap_encoder:
        model = OnsetDetector(
            n_mels=80, d_model=ckpt_args.get("d_model", 384),
            n_audio_layers=ckpt_args.get("enc_layers", 4),
            n_gap_layers=ckpt_args.get("gap_enc_layers", 2),
            n_fusion_layers=ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES, max_events=C_EVENTS,
        ).to(device)
    elif is_legacy:
        model = LegacyOnsetDetector(
            n_mels=80, d_model=ckpt_args.get("d_model", 384),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES, max_events=C_EVENTS,
        ).to(device)
    else:
        print(f"ERROR: Could not detect model architecture from checkpoint keys")
        return

    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state)
    model.eval()
    print(f"Loaded model: {model.__class__.__name__}")

    # collect per-sample scores, predictions, targets
    print("Running validation...")
    all_scores = []
    all_preds = []
    all_targets = []
    from tqdm import tqdm
    import torch.nn.functional as F
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            mel, evt_off, evt_mask, cond, target = batch
            mel = mel.to(device)
            evt_off = evt_off.to(device)
            evt_mask = evt_mask.to(device)
            cond = cond.to(device)

            output = model(mel, evt_off, evt_mask, cond)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            # ensure 501 classes
            if logits.shape[-1] < N_CLASSES:
                logits = F.pad(logits, (0, N_CLASSES - logits.shape[-1]), value=-10.0)

            preds = logits.argmax(1).cpu().numpy()
            targets = target.numpy()

            all_preds.append(preds)
            all_targets.append(targets)
            for i in range(len(preds)):
                score = compute_score(preds[i], targets[i])
                all_scores.append(score)

    scores = np.array(all_scores)
    preds_all = np.concatenate(all_preds)
    targets_all = np.concatenate(all_targets)
    print(f"Collected {len(scores)} scores")
    print(f"  mean={scores.mean():.3f} median={np.median(scores):.3f}")
    print(f"  >0 (good): {(scores > 0).mean()*100:.1f}%")
    print(f"  <-0.5 (bad): {(scores < -0.5).mean()*100:.1f}%")

    # pack into 512x512 image
    n_samples = len(scores)
    size = min(512, int(np.ceil(np.sqrt(n_samples))))
    n_pixels = size * size
    samples_per_pixel = max(1, n_samples // n_pixels)

    # group samples into pixels
    n_used = min(n_samples, n_pixels * samples_per_pixel)
    pixel_scores = np.zeros(n_pixels)
    pixel_scores[:n_used // samples_per_pixel] = scores[:n_used].reshape(-1, samples_per_pixel).mean(axis=1)
    # remaining pixels stay at 0 (black/neutral)

    # color map: -1 = red, 0 = black/neutral, +0.67 = green
    img = np.zeros((n_pixels, 3), dtype=np.float32)
    for i, s in enumerate(pixel_scores):
        if s >= 0:
            t = min(s / 0.67, 1.0)
            img[i] = (0, t, 0)
        else:
            t = min(-s, 1.0)
            img[i] = (t, 0, 0)

    img = img.reshape(size, size, 3)

    # gaussian blur to smooth out the pattern
    from scipy.ndimage import gaussian_filter
    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], sigma=1.5)

    # renormalize after blur
    img_max = img.max()
    if img_max > 0:
        img = img / img_max

    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    # scale up to 512x512 for visibility
    if size < 512:
        pil_img = pil_img.resize((512, 512), Image.NEAREST)

    out_path = os.path.join(args.output_dir, f"val_heatmap_{args.label}.png")
    pil_img.save(out_path)
    print(f"Saved: {out_path} ({size}x{size} → 512x512, {samples_per_pixel} samples/pixel)")

    # save raw data for cross-model analysis
    npy_path = os.path.join(args.output_dir, f"val_scores_{args.label}.npy")
    np.save(npy_path, scores)
    print(f"Saved: {npy_path}")

    data_path = os.path.join(args.output_dir, f"val_data_{args.label}.npz")
    np.savez(data_path, scores=scores, preds=preds_all, targets=targets_all)
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
