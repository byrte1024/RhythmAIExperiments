"""Experiment 41: Deep entropy analysis.

Why is entropy higher for distant predictions? Is it because:
A) More valid onsets in the window (model correctly hedges), or
B) The model genuinely doesn't know where the onset is?

Measures correlations between entropy and:
- Target distance (bin offset)
- Number of future onsets in window
- Number of future onsets BETWEEN cursor and target
- Audio energy at target position
- Spectral flux at target position
- Context length (how many past events)
- Density conditioning values
- Whether prediction is correct (HIT/MISS)
- Whether overprediction matches a real future onset

Usage:
    python analyze_entropy.py taiko_v2 --checkpoint runs/detect_experiment_35c/checkpoints/eval_008.pt --subsample 8
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
    OnsetDataset, N_CLASSES, B_BINS, C_EVENTS, split_by_song, MAX_TARGETS,
)
from detection_model import OnsetDetector


def is_hit(pred, target):
    if target == N_CLASSES - 1:
        return pred == target
    frame_err = abs(pred - target)
    pct_err = abs((pred + 1) / (target + 1) - 1.0)
    return pct_err <= 0.03 or frame_err <= 1


def mel_energy_at(mel_np, bin_offset, window=5):
    frame = 500 + bin_offset
    lo = max(0, frame - window)
    hi = min(mel_np.shape[1], frame + window + 1)
    if lo >= hi:
        return 0.0
    return mel_np[:, lo:hi].mean()


def spectral_flux_at(mel_np, bin_offset):
    frame = 500 + bin_offset
    if frame < 1 or frame >= mel_np.shape[1]:
        return 0.0
    diff = mel_np[:, frame] - mel_np[:, frame - 1]
    return np.maximum(diff, 0).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)
    val_ds_single = OnsetDataset(manifest, ds_dir, val_idx, augment=False,
                                  subsample=args.subsample, multi_target=False)
    val_ds_multi = OnsetDataset(manifest, ds_dir, val_idx, augment=False,
                                 subsample=args.subsample, multi_target=True)
    print(f"Val samples: {len(val_ds_single)}")

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

    # collect predictions with entropy
    loader = DataLoader(val_ds_single, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    all_targets = []
    all_preds = []
    all_entropy = []
    all_top1_conf = []

    with torch.no_grad():
        for mel, evt_off, evt_mask, cond, target in tqdm(loader, desc="Running val"):
            mel = mel.to(args.device, non_blocking=True)
            evt_off = evt_off.to(args.device, non_blocking=True)
            evt_mask = evt_mask.to(args.device, non_blocking=True)
            cond = cond.to(args.device, non_blocking=True)

            logits = model(mel, evt_off, evt_mask, cond)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            ent = -(probs * (probs + 1e-10).log()).sum(dim=1)

            all_targets.append(target.numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_entropy.append(ent.cpu().numpy())
            all_top1_conf.append(probs.max(dim=1).values.cpu().numpy())

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    entropy = np.concatenate(all_entropy)
    top1_conf = np.concatenate(all_top1_conf)

    # collect per-sample features (need individual sample access)
    print("Collecting per-sample features...")
    N = len(targets)
    stop = N_CLASSES - 1

    n_future_onsets = np.zeros(N)
    n_onsets_before_target = np.zeros(N)
    ctx_length = np.zeros(N)
    density_mean = np.zeros(N)
    target_energy = np.zeros(N)
    target_flux = np.zeros(N)

    for i in tqdm(range(N), desc="Features"):
        # multi-target: all future onsets
        _, _, _, _, tp, nt = val_ds_multi[i]
        n_future_onsets[i] = nt.item()

        # count onsets between cursor and target
        t = targets[i]
        if t < stop and nt.item() > 0:
            future_bins = tp[:nt.item()].numpy()
            n_onsets_before_target[i] = np.sum(future_bins < t)

        # single-target: get mel, context info
        mel, evt_off, evt_mask, cond, _ = val_ds_single[i]
        ctx_length[i] = (~evt_mask).sum().item()
        density_mean[i] = cond[0].item()

        if t < stop:
            mel_np = mel.numpy()
            target_energy[i] = mel_energy_at(mel_np, t)
            target_flux[i] = spectral_flux_at(mel_np, t)

    # filter non-stop
    ns = targets < stop
    t_ns = targets[ns]
    p_ns = preds[ns]
    ent_ns = entropy[ns]
    conf_ns = top1_conf[ns]
    nfo_ns = n_future_onsets[ns]
    nobt_ns = n_onsets_before_target[ns]
    ctx_ns = ctx_length[ns]
    dens_ns = density_mean[ns]
    te_ns = target_energy[ns]
    tf_ns = target_flux[ns]
    N_ns = len(t_ns)

    hit_ns = np.array([is_hit(int(p_ns[i]), int(t_ns[i])) for i in range(N_ns)])

    print(f"\n{'='*70}")
    print(f"  ENTROPY CORRELATION ANALYSIS ({N_ns} non-STOP samples)")
    print(f"{'='*70}")

    # correlations
    features = {
        "target_distance": t_ns.astype(float),
        "n_future_onsets": nfo_ns,
        "n_onsets_before_target": nobt_ns,
        "context_length": ctx_ns,
        "density_mean": dens_ns,
        "target_mel_energy": te_ns,
        "target_spectral_flux": tf_ns,
        "top1_confidence": conf_ns,
        "is_hit": hit_ns.astype(float),
    }

    print(f"\n  Pearson correlations with entropy:")
    corrs = {}
    for name, vals in features.items():
        valid = ~np.isnan(vals) & ~np.isnan(ent_ns)
        if valid.sum() > 100:
            r = np.corrcoef(ent_ns[valid], vals[valid])[0, 1]
            corrs[name] = r
            print(f"    {name:30s}: r = {r:+.3f}")

    # entropy by target distance bins
    print(f"\n  Entropy by target distance:")
    dist_bins = [(0, 15), (15, 30), (30, 60), (60, 100), (100, 200), (200, 500)]
    print(f"    {'Range':>10s}  {'N':>7s}  {'Entropy':>8s}  {'Conf':>7s}  {'HIT':>6s}  {'FutOnsets':>9s}  {'Between':>8s}")
    for lo, hi in dist_bins:
        mask = (t_ns >= lo) & (t_ns < hi)
        if mask.sum() == 0:
            continue
        print(f"    {lo:3d}-{hi:3d}  {mask.sum():7d}  {ent_ns[mask].mean():8.3f}  "
              f"{conf_ns[mask].mean():7.3f}  {hit_ns[mask].mean():5.1%}  "
              f"{nfo_ns[mask].mean():9.1f}  {nobt_ns[mask].mean():8.1f}")

    # entropy by n_future_onsets
    print(f"\n  Entropy by number of future onsets in window:")
    print(f"    {'N_onsets':>8s}  {'Samples':>8s}  {'Entropy':>8s}  {'Conf':>7s}  {'HIT':>6s}  {'TargDist':>8s}")
    for n in [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]:
        if n < 30:
            mask = (nfo_ns >= n) & (nfo_ns < n + (2 if n < 5 else 5))
        else:
            mask = nfo_ns >= n
        if mask.sum() < 50:
            continue
        label = f"{n}-{n+(2 if n < 5 else 5)}" if n < 30 else f"{n}+"
        print(f"    {label:>8s}  {mask.sum():8d}  {ent_ns[mask].mean():8.3f}  "
              f"{conf_ns[mask].mean():7.3f}  {hit_ns[mask].mean():5.1%}  "
              f"{t_ns[mask].mean():8.1f}")

    # entropy by n_onsets_between cursor and target
    print(f"\n  Entropy by onsets BETWEEN cursor and target:")
    print(f"    {'Between':>8s}  {'Samples':>8s}  {'Entropy':>8s}  {'Conf':>7s}  {'HIT':>6s}  {'TargDist':>8s}")
    for n in [0, 1, 2, 3, 5, 10]:
        if n < 10:
            mask = nobt_ns == n
        else:
            mask = nobt_ns >= n
        if mask.sum() < 50:
            continue
        label = str(n) if n < 10 else f"{n}+"
        print(f"    {label:>8s}  {mask.sum():8d}  {ent_ns[mask].mean():8.3f}  "
              f"{conf_ns[mask].mean():7.3f}  {hit_ns[mask].mean():5.1%}  "
              f"{t_ns[mask].mean():8.1f}")

    # HIT vs MISS entropy breakdown
    print(f"\n  Entropy: HIT vs MISS:")
    print(f"    HIT  (n={hit_ns.sum():5.0f}): entropy={ent_ns[hit_ns].mean():.3f}  conf={conf_ns[hit_ns].mean():.3f}")
    print(f"    MISS (n={(~hit_ns).sum():5.0f}): entropy={ent_ns[~hit_ns].mean():.3f}  conf={conf_ns[~hit_ns].mean():.3f}")

    # save
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_41")
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "n_samples": int(N_ns),
        "correlations": {k: float(v) for k, v in corrs.items()},
        "mean_entropy": float(ent_ns.mean()),
        "hit_entropy": float(ent_ns[hit_ns].mean()),
        "miss_entropy": float(ent_ns[~hit_ns].mean()),
    }
    with open(os.path.join(out_dir, "entropy_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_dir}/entropy_analysis.json")


if __name__ == "__main__":
    main()
