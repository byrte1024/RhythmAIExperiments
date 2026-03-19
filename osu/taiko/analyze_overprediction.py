"""Analyze overpredictions: do they match real onsets beyond the nearest one?

For each val sample where the model overpredicts (pred > target), check if
the prediction matches ANY future onset in the window, not just the next one.
Also check how many top-K predictions match any future onset.

Usage:
    python analyze_overprediction.py taiko_v2 --checkpoint runs/detect_experiment_35c/checkpoints/eval_008.pt --subsample 8
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)

    # load both single-target and multi-target datasets (same samples, different targets)
    val_ds_single = OnsetDataset(manifest, ds_dir, val_idx, augment=False,
                                  subsample=args.subsample, multi_target=False)
    val_ds_multi = OnsetDataset(manifest, ds_dir, val_idx, augment=False,
                                 subsample=args.subsample, multi_target=True)
    print(f"Val samples: {len(val_ds_single)}")

    # load model
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
    print(f"  HIT: {ckpt['val_metrics'].get('hit_rate', 0):.1%}")

    loader_single = DataLoader(val_ds_single, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

    # run inference and collect predictions + topk
    all_targets = []      # nearest target (single-target)
    all_preds = []
    all_topk = []         # (N, K) top-K predicted bins

    with torch.no_grad():
        for mel, evt_off, evt_mask, cond, target in tqdm(loader_single, desc="Running val"):
            mel = mel.to(args.device, non_blocking=True)
            evt_off = evt_off.to(args.device, non_blocking=True)
            evt_mask = evt_mask.to(args.device, non_blocking=True)
            cond = cond.to(args.device, non_blocking=True)

            logits = model(mel, evt_off, evt_mask, cond)
            if isinstance(logits, tuple):
                logits = logits[0]

            all_targets.append(target.numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_topk.append(logits.topk(args.topk, dim=1).indices.cpu().numpy())

    targets = np.concatenate(all_targets)
    preds = np.concatenate(all_preds)
    topk = np.concatenate(all_topk)

    # now get all future onsets for each sample from multi-target dataset
    print("Loading all future onsets...")
    all_future_onsets = []  # list of arrays, each with all future onset bins
    for i in tqdm(range(len(val_ds_multi)), desc="Future onsets"):
        _, _, _, _, targets_padded, n_tgt = val_ds_multi[i]
        nt = n_tgt.item()
        if nt > 0:
            future = targets_padded[:nt].numpy()
        else:
            future = np.array([], dtype=np.int64)
        all_future_onsets.append(future)

    N = len(targets)
    stop = N_CLASSES - 1

    # analysis
    print(f"\nAnalyzing {N} samples...\n")

    # counters
    n_nonstop = 0
    n_hit = 0
    n_miss = 0
    n_overpred = 0          # pred > target (predicted further than nearest)
    n_overpred_matches_any = 0  # overprediction matches a real future onset
    n_overpred_matches_2nd = 0  # specifically matches the 2nd onset
    n_underpred = 0         # pred < target

    # topk analysis
    topk_matches_nearest = np.zeros(args.topk)   # K-th entry matches nearest target
    topk_matches_any = np.zeros(args.topk)        # K-th entry matches ANY future onset

    for i in range(N):
        t = targets[i]
        p = preds[i]
        future = all_future_onsets[i]

        if t == stop:
            continue
        n_nonstop += 1

        if is_hit(p, t):
            n_hit += 1
        else:
            n_miss += 1

            if p > t:  # overprediction
                n_overpred += 1
                # check if prediction matches any future onset
                for fo in future:
                    if is_hit(p, int(fo)):
                        n_overpred_matches_any += 1
                        # check if it's specifically the 2nd onset
                        if len(future) >= 2 and is_hit(p, int(future[1])):
                            n_overpred_matches_2nd += 1
                        break
            elif p < t:
                n_underpred += 1

        # topk analysis: for each k, does any of the top-k match nearest or any future?
        for k in range(args.topk):
            tk_pred = topk[i, k]
            if is_hit(tk_pred, t):
                topk_matches_nearest[k] += 1
            for fo in future:
                if is_hit(tk_pred, int(fo)):
                    topk_matches_any[k] += 1
                    break

    # cumulative topk
    topk_cum_nearest = np.cumsum(topk_matches_nearest) / n_nonstop
    topk_cum_any = np.cumsum(topk_matches_any) / n_nonstop

    print(f"{'='*70}")
    print(f"  OVERPREDICTION ANALYSIS")
    print(f"{'='*70}")
    print(f"  Non-STOP samples: {n_nonstop}")
    print(f"  HIT (nearest): {n_hit} ({n_hit/n_nonstop:.1%})")
    print(f"  MISS: {n_miss} ({n_miss/n_nonstop:.1%})")
    print(f"    Overpredictions (pred > target): {n_overpred} ({n_overpred/n_nonstop:.1%})")
    print(f"    Underpredictions (pred < target): {n_underpred} ({n_underpred/n_nonstop:.1%})")
    print()
    print(f"  Of {n_overpred} overpredictions:")
    print(f"    Matches ANY future onset: {n_overpred_matches_any} ({n_overpred_matches_any/max(1,n_overpred):.1%})")
    print(f"    Matches 2nd onset specifically: {n_overpred_matches_2nd} ({n_overpred_matches_2nd/max(1,n_overpred):.1%})")
    print(f"    Doesn't match any onset: {n_overpred - n_overpred_matches_any} ({(n_overpred - n_overpred_matches_any)/max(1,n_overpred):.1%})")
    print()

    # theoretical: if we counted overpredictions that match future onsets as correct
    effective_hit = n_hit + n_overpred_matches_any
    print(f"  Theoretical HIT if overpred→future counted:")
    print(f"    Current:  {n_hit/n_nonstop:.1%}")
    print(f"    Adjusted: {effective_hit/n_nonstop:.1%} (+{n_overpred_matches_any/n_nonstop:.1%})")
    print()

    print(f"  Top-K analysis (cumulative):")
    print(f"  {'K':>3s}  {'Nearest':>10s}  {'Any future':>10s}  {'Gain':>10s}")
    for k in range(args.topk):
        print(f"  {k+1:3d}  {topk_cum_nearest[k]:10.1%}  {topk_cum_any[k]:10.1%}  {topk_cum_any[k]-topk_cum_nearest[k]:+10.1%}")

    # save results
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_35c")
    results = {
        "n_nonstop": n_nonstop,
        "n_hit": n_hit,
        "n_miss": n_miss,
        "n_overpred": n_overpred,
        "n_overpred_matches_any": n_overpred_matches_any,
        "n_overpred_matches_2nd": n_overpred_matches_2nd,
        "n_underpred": n_underpred,
        "topk_cum_nearest": topk_cum_nearest.tolist(),
        "topk_cum_any": topk_cum_any.tolist(),
    }
    with open(os.path.join(out_dir, "overprediction_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_dir}/overprediction_analysis.json")


if __name__ == "__main__":
    main()
