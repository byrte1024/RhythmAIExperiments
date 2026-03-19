"""Experiment 39-E: Audio analysis of failure cases.

For each wrong prediction where the correct answer is in top-K:
- Compare mel energy at target position vs predicted position
- Measure spectral features at both positions
- Export visual mel windows for manual inspection

Usage:
    python analyze_failure_audio.py taiko_v2 --checkpoint runs/detect_experiment_35c/checkpoints/eval_008.pt --subsample 8
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from detection_train import (
    OnsetDataset, N_CLASSES, B_BINS, A_BINS, C_EVENTS, split_by_song,
)
from detection_model import OnsetDetector


def is_hit(pred, target):
    if target == N_CLASSES - 1:
        return pred == target
    frame_err = abs(pred - target)
    pct_err = abs((pred + 1) / (target + 1) - 1.0)
    return pct_err <= 0.03 or frame_err <= 1


def mel_energy_at(mel_window, bin_offset, window_size=5):
    """Mean mel energy in a window around a bin position.
    mel_window: (80, 1000), bin_offset: 0-499 from cursor at frame 500.
    """
    frame = 500 + bin_offset  # cursor is at frame 500
    lo = max(0, frame - window_size)
    hi = min(mel_window.shape[1], frame + window_size + 1)
    if lo >= hi:
        return 0.0
    return mel_window[:, lo:hi].mean().item()


def spectral_flux_at(mel_window, bin_offset):
    """Spectral flux (frame-to-frame change) at a position."""
    frame = 500 + bin_offset
    if frame < 1 or frame >= mel_window.shape[1]:
        return 0.0
    diff = mel_window[:, frame] - mel_window[:, frame - 1]
    return np.maximum(diff, 0).sum().item()  # half-wave rectified


def onset_strength_at(mel_window, bin_offset, window_size=3):
    """Local onset strength: max spectral flux in a small window."""
    frame = 500 + bin_offset
    lo = max(1, frame - window_size)
    hi = min(mel_window.shape[1], frame + window_size + 1)
    max_flux = 0.0
    for f in range(lo, hi):
        diff = mel_window[:, f] - mel_window[:, f - 1]
        flux = np.maximum(diff, 0).sum()
        max_flux = max(max_flux, flux)
    return float(max_flux)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n-visual", type=int, default=10, help="Number of visual exports")
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)
    val_ds = OnsetDataset(manifest, ds_dir, val_idx, augment=False,
                          subsample=args.subsample, multi_target=False)
    print(f"Val samples: {len(val_ds)}")

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    model = OnsetDetector(
        d_model=ckpt_args["d_model"],
        n_heads=ckpt_args["n_heads"],
        enc_layers=ckpt_args["enc_layers"],
        gap_enc_layers=ckpt_args.get("gap_enc_layers", 2),
        fusion_layers=ckpt_args.get("fusion_layers", 4),
        snippet_frames=ckpt_args.get("snippet_frames", 10),
        dropout=0.0,
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # run inference sample by sample (need mel_window for audio analysis)
    N = len(val_ds)
    stop = N_CLASSES - 1

    # accumulators
    target_energy = []
    pred_energy = []
    target_flux = []
    pred_flux = []
    target_onset_str = []
    pred_onset_str = []
    target_bins = []
    pred_bins = []
    pred_confs = []
    correct_confs = []
    is_overpred = []

    # for visual exports
    visual_samples = []

    print("Analyzing failure cases...")
    for i in tqdm(range(N)):
        mel, evt_off, evt_mask, cond, target = val_ds[i]

        if target.item() == stop:
            continue

        mel_t = mel.unsqueeze(0).to(args.device)
        eo_t = evt_off.unsqueeze(0).to(args.device)
        em_t = evt_mask.unsqueeze(0).to(args.device)
        cd_t = cond.unsqueeze(0).to(args.device)

        with torch.no_grad():
            logits = model(mel_t, eo_t, em_t, cd_t)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            topk_v, topk_i = probs.topk(args.topk, dim=1)

        pred = topk_i[0, 0].item()
        t = target.item()

        if is_hit(pred, t):
            continue  # only analyze failures

        # check if correct is in top-K
        correct_rank = -1
        correct_conf_val = 0.0
        for k in range(args.topk):
            if is_hit(topk_i[0, k].item(), t):
                correct_rank = k
                correct_conf_val = topk_v[0, k].item()
                break

        if correct_rank < 0:
            continue  # correct not in top-K, skip

        mel_np = mel.numpy()  # (80, 1000)

        # audio stats at target vs predicted position
        te = mel_energy_at(mel_np, t)
        pe = mel_energy_at(mel_np, pred)
        tf = spectral_flux_at(mel_np, t)
        pf = spectral_flux_at(mel_np, pred)
        tos = onset_strength_at(mel_np, t)
        pos = onset_strength_at(mel_np, pred)

        target_energy.append(te)
        pred_energy.append(pe)
        target_flux.append(tf)
        pred_flux.append(pf)
        target_onset_str.append(tos)
        pred_onset_str.append(pos)
        target_bins.append(t)
        pred_bins.append(pred)
        pred_confs.append(topk_v[0, 0].item())
        correct_confs.append(correct_conf_val)
        is_overpred.append(pred > t)

        # collect for visual export
        if len(visual_samples) < args.n_visual:
            visual_samples.append({
                "idx": i,
                "target": t,
                "pred": pred,
                "correct_rank": correct_rank,
                "pred_conf": topk_v[0, 0].item(),
                "correct_conf": correct_conf_val,
                "topk_ids": topk_i[0].cpu().numpy().tolist(),
                "topk_vals": topk_v[0].cpu().numpy().tolist(),
                "mel": mel_np,
            })

    # convert to arrays
    target_energy = np.array(target_energy)
    pred_energy = np.array(pred_energy)
    target_flux = np.array(target_flux)
    pred_flux = np.array(pred_flux)
    target_onset_str = np.array(target_onset_str)
    pred_onset_str = np.array(pred_onset_str)
    is_overpred = np.array(is_overpred)
    N_fail = len(target_energy)

    print(f"\n{'='*70}")
    print(f"  AUDIO ANALYSIS OF FAILURE CASES ({N_fail} samples)")
    print(f"{'='*70}")

    print(f"\n  Mel energy (mean across bands, ±5 frame window):")
    print(f"    At target position:    mean={target_energy.mean():.2f}  median={np.median(target_energy):.2f}")
    print(f"    At predicted position: mean={pred_energy.mean():.2f}  median={np.median(pred_energy):.2f}")
    energy_diff = pred_energy - target_energy
    print(f"    Pred - Target:         mean={energy_diff.mean():.2f}  median={np.median(energy_diff):.2f}")
    print(f"    Pred has MORE energy:  {(energy_diff > 0).sum()} ({(energy_diff > 0).mean():.1%})")
    print(f"    Target has MORE energy: {(energy_diff < 0).sum()} ({(energy_diff < 0).mean():.1%})")

    print(f"\n  Spectral flux (onset indicator):")
    print(f"    At target:    mean={target_flux.mean():.2f}  median={np.median(target_flux):.2f}")
    print(f"    At predicted: mean={pred_flux.mean():.2f}  median={np.median(pred_flux):.2f}")
    flux_diff = pred_flux - target_flux
    print(f"    Pred - Target: mean={flux_diff.mean():.2f}  median={np.median(flux_diff):.2f}")
    print(f"    Pred has MORE flux: {(flux_diff > 0).sum()} ({(flux_diff > 0).mean():.1%})")

    print(f"\n  Onset strength (max flux in ±3 window):")
    print(f"    At target:    mean={target_onset_str.mean():.2f}  median={np.median(target_onset_str):.2f}")
    print(f"    At predicted: mean={pred_onset_str.mean():.2f}  median={np.median(pred_onset_str):.2f}")
    os_diff = pred_onset_str - target_onset_str
    print(f"    Pred - Target: mean={os_diff.mean():.2f}  median={np.median(os_diff):.2f}")
    print(f"    Pred has MORE onset str: {(os_diff > 0).sum()} ({(os_diff > 0).mean():.1%})")

    # breakdown by overpred vs underpred
    print(f"\n  Breakdown by over/underprediction:")
    print(f"    Overpredictions: {is_overpred.sum()} ({is_overpred.mean():.1%})")
    if is_overpred.sum() > 0:
        print(f"      Pred energy > target: {(energy_diff[is_overpred] > 0).mean():.1%}")
        print(f"      Pred flux > target:   {(flux_diff[is_overpred] > 0).mean():.1%}")
        print(f"      Pred onset > target:  {(os_diff[is_overpred] > 0).mean():.1%}")
    if (~is_overpred).sum() > 0:
        print(f"    Underpredictions: {(~is_overpred).sum()} ({(~is_overpred).mean():.1%})")
        print(f"      Pred energy > target: {(energy_diff[~is_overpred] > 0).mean():.1%}")
        print(f"      Pred flux > target:   {(flux_diff[~is_overpred] > 0).mean():.1%}")

    # visual exports
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_39e")
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for vi, vs in enumerate(visual_samples):
        mel_np = vs["mel"]
        t = vs["target"]
        p = vs["pred"]
        t_frame = 500 + t
        p_frame = 500 + p

        fig, axes = plt.subplots(3, 1, figsize=(16, 10))

        # full mel window
        ax = axes[0]
        ax.imshow(mel_np, aspect="auto", origin="lower", cmap="magma",
                  extent=[0, 1000, 0, 80], vmin=-15, vmax=15)
        ax.axvline(500, color="green", linewidth=2, label="cursor")
        ax.axvline(t_frame, color="cyan", linewidth=2, linestyle="--", label=f"target (bin {t})")
        ax.axvline(p_frame, color="red", linewidth=2, linestyle="--", label=f"pred (bin {p})")
        ax.legend(fontsize=8)
        ax.set_title(f"Sample {vs['idx']}: target={t}, pred={p}, "
                     f"pred_conf={vs['pred_conf']:.3f}, correct_conf={vs['correct_conf']:.3f} (rank {vs['correct_rank']})")
        ax.set_ylabel("Mel band")

        # zoomed: 50 frames around target and pred
        ax = axes[1]
        zoom_lo = max(0, min(t_frame, p_frame) - 30)
        zoom_hi = min(1000, max(t_frame, p_frame) + 30)
        ax.imshow(mel_np[:, zoom_lo:zoom_hi], aspect="auto", origin="lower", cmap="magma",
                  extent=[zoom_lo, zoom_hi, 0, 80], vmin=-15, vmax=15)
        ax.axvline(t_frame, color="cyan", linewidth=2, linestyle="--", label=f"target")
        ax.axvline(p_frame, color="red", linewidth=2, linestyle="--", label=f"pred")
        ax.axvline(500, color="green", linewidth=1, alpha=0.5)
        ax.legend(fontsize=8)
        ax.set_title("Zoomed (±30 frames around target/pred)")
        ax.set_ylabel("Mel band")

        # mean energy profile around both positions
        ax = axes[2]
        energy_profile = mel_np.mean(axis=0)  # mean across bands
        ax.plot(energy_profile, color="gray", alpha=0.5, linewidth=0.5)
        # highlight regions
        ax.axvline(500, color="green", linewidth=2, label="cursor")
        ax.axvline(t_frame, color="cyan", linewidth=2, linestyle="--", label=f"target (bin {t})")
        ax.axvline(p_frame, color="red", linewidth=2, linestyle="--", label=f"pred (bin {p})")
        ax.set_xlim(zoom_lo - 20, zoom_hi + 20)
        ax.legend(fontsize=8)
        ax.set_title("Mean mel energy profile")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean energy")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"failure_{vi:02d}.png"), dpi=150)
        plt.close(fig)

    print(f"\n  Saved {len(visual_samples)} visual exports to {out_dir}/")

    # save stats
    results = {
        "n_failures": N_fail,
        "energy": {
            "target_mean": float(target_energy.mean()),
            "pred_mean": float(pred_energy.mean()),
            "pred_higher_pct": float((energy_diff > 0).mean()),
        },
        "flux": {
            "target_mean": float(target_flux.mean()),
            "pred_mean": float(pred_flux.mean()),
            "pred_higher_pct": float((flux_diff > 0).mean()),
        },
        "onset_strength": {
            "target_mean": float(target_onset_str.mean()),
            "pred_mean": float(pred_onset_str.mean()),
            "pred_higher_pct": float((os_diff > 0).mean()),
        },
    }
    with open(os.path.join(out_dir, "audio_analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved stats to {out_dir}/audio_analysis_results.json")


if __name__ == "__main__":
    main()
