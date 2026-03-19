"""Experiment 39-D: Top-K depth analysis.

1. How many of the top-K candidates are HITs? (not cumulative — actual count per sample)
   Do we see 1 correct answer or 3-4 near-duplicates?

2. When the model is wrong (top-1 miss) but the correct answer IS in top-K:
   How far in confidence is the correct answer from the chosen one?

Usage:
    python analyze_topk_depth.py taiko_v2 --checkpoint runs/detect_experiment_35c/checkpoints/eval_008.pt --subsample 8
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from detection_train import (
    OnsetDataset, N_CLASSES, C_EVENTS, split_by_song,
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

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    all_targets = []
    all_topk_ids = []
    all_topk_vals = []

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
            topk_v, topk_i = probs.topk(args.topk, dim=1)

            all_targets.append(target.numpy())
            all_topk_ids.append(topk_i.cpu().numpy())
            all_topk_vals.append(topk_v.cpu().numpy())

    targets = np.concatenate(all_targets)
    topk_ids = np.concatenate(all_topk_ids)
    topk_vals = np.concatenate(all_topk_vals)

    stop = N_CLASSES - 1
    ns = targets < stop
    t_ns = targets[ns]
    tk_ids = topk_ids[ns]
    tk_vals = topk_vals[ns]
    N = len(t_ns)

    K = args.topk

    # === ANALYSIS 1: How many HITs per sample in top-K? ===
    hits_per_sample = []
    for i in range(N):
        n_hits = sum(1 for k in range(K) if is_hit(int(tk_ids[i, k]), int(t_ns[i])))
        hits_per_sample.append(n_hits)
    hits_per_sample = np.array(hits_per_sample)

    hit_count_dist = Counter(hits_per_sample)

    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: HIT count per sample in top-{K}")
    print(f"{'='*70}")
    print(f"  Total non-STOP samples: {N}")
    print(f"  Mean HITs in top-{K}: {hits_per_sample.mean():.2f}")
    print(f"  Median: {np.median(hits_per_sample):.0f}")
    print()
    print(f"  Distribution:")
    for n_hits in sorted(hit_count_dist.keys()):
        count = hit_count_dist[n_hits]
        print(f"    {n_hits} HITs: {count:6d} samples ({count/N:.1%})")

    # of those with >=1 HIT, what's the avg?
    has_hit = hits_per_sample > 0
    if has_hit.any():
        print(f"\n  Of samples with ≥1 HIT in top-{K} ({has_hit.sum()}):")
        print(f"    Mean HITs: {hits_per_sample[has_hit].mean():.2f}")
        print(f"    Median: {np.median(hits_per_sample[has_hit]):.0f}")

    # === ANALYSIS 2: When wrong, how far is the correct answer? ===
    top1_wrong = np.array([not is_hit(int(tk_ids[i, 0]), int(t_ns[i])) for i in range(N)])
    correct_in_topk = np.zeros(N, dtype=bool)
    correct_rank = np.full(N, -1, dtype=int)
    correct_conf = np.full(N, 0.0)
    chosen_conf = tk_vals[:, 0]

    for i in range(N):
        for k in range(K):
            if is_hit(int(tk_ids[i, k]), int(t_ns[i])):
                correct_in_topk[i] = True
                correct_rank[i] = k
                correct_conf[i] = tk_vals[i, k]
                break

    # wrong AND correct exists in top-K
    wrong_but_present = top1_wrong & correct_in_topk
    n_wrong = top1_wrong.sum()
    n_wrong_present = wrong_but_present.sum()
    n_wrong_absent = n_wrong - n_wrong_present

    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: When top-1 is WRONG, where is the correct answer?")
    print(f"{'='*70}")
    print(f"  Total wrong (top-1 miss): {n_wrong} ({n_wrong/N:.1%})")
    print(f"  Correct in top-{K}: {n_wrong_present} ({n_wrong_present/max(1,n_wrong):.1%})")
    print(f"  Correct NOT in top-{K}: {n_wrong_absent} ({n_wrong_absent/max(1,n_wrong):.1%})")

    if n_wrong_present > 0:
        wp_ranks = correct_rank[wrong_but_present]
        wp_conf = correct_conf[wrong_but_present]
        wp_chosen_conf = chosen_conf[wrong_but_present]
        conf_gap = wp_chosen_conf - wp_conf  # positive = chosen was more confident
        conf_ratio = wp_conf / wp_chosen_conf.clip(min=1e-8)

        print(f"\n  Of {n_wrong_present} wrong predictions where correct is in top-{K}:")
        print(f"    Correct answer rank: mean={wp_ranks.mean():.1f}, median={np.median(wp_ranks):.0f}")
        rank_dist = Counter(wp_ranks)
        for r in sorted(rank_dist.keys()):
            print(f"      Rank {r}: {rank_dist[r]:5d} ({rank_dist[r]/n_wrong_present:.1%})")

        print(f"\n    Confidence of correct answer:")
        print(f"      Mean: {wp_conf.mean():.4f}")
        print(f"      Median: {np.median(wp_conf):.4f}")

        print(f"\n    Confidence of chosen (wrong) answer:")
        print(f"      Mean: {wp_chosen_conf.mean():.4f}")
        print(f"      Median: {np.median(wp_chosen_conf):.4f}")

        print(f"\n    Confidence gap (chosen - correct):")
        print(f"      Mean: {conf_gap.mean():.4f}")
        print(f"      Median: {np.median(conf_gap):.4f}")
        print(f"      Min: {conf_gap.min():.4f}")
        print(f"      Max: {conf_gap.max():.4f}")

        print(f"\n    Confidence ratio (correct / chosen):")
        print(f"      Mean: {conf_ratio.mean():.3f}")
        print(f"      Median: {np.median(conf_ratio):.3f}")
        print(f"      >0.9 (nearly equal): {(conf_ratio > 0.9).sum()} ({(conf_ratio > 0.9).mean():.1%})")
        print(f"      >0.5 (within 2x): {(conf_ratio > 0.5).sum()} ({(conf_ratio > 0.5).mean():.1%})")
        print(f"      <0.1 (model very sure of wrong): {(conf_ratio < 0.1).sum()} ({(conf_ratio < 0.1).mean():.1%})")

    # save
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_39d")
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "n_samples": int(N),
        "mean_hits_per_sample": float(hits_per_sample.mean()),
        "hit_count_distribution": {str(k): int(v) for k, v in sorted(hit_count_dist.items())},
        "n_wrong": int(n_wrong),
        "n_wrong_correct_in_topk": int(n_wrong_present),
        "n_wrong_correct_absent": int(n_wrong_absent),
    }
    if n_wrong_present > 0:
        results["correct_rank_mean"] = float(wp_ranks.mean())
        results["conf_gap_mean"] = float(conf_gap.mean())
        results["conf_ratio_mean"] = float(conf_ratio.mean())
    with open(os.path.join(out_dir, "topk_depth_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_dir}/topk_depth_results.json")


if __name__ == "__main__":
    main()
