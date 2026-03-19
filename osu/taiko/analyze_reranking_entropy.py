"""Experiment 39-C: Entropy-weighted reranking.

Adds a third weight that downscales candidates further from #1 in confidence.
When the model is confident, the original pick is preserved.
When hedging, proximity can override.

score = conf_w * confidence + pos_w * (1 - bin/500) + ent_w * (confidence / top1_confidence)

Usage:
    python analyze_reranking_entropy.py taiko_v2 --checkpoint runs/detect_experiment_35c/checkpoints/eval_008.pt --subsample 8
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
    tk_ids_ns = topk_ids[ns]
    tk_vals_ns = topk_vals[ns]
    N_ns = len(t_ns)

    # baseline
    baseline_preds = tk_ids_ns[:, 0]
    baseline_hits = np.array([is_hit(int(baseline_preds[i]), int(t_ns[i])) for i in range(N_ns)])
    baseline_hit_rate = baseline_hits.mean()
    print(f"\nBaseline: {baseline_hit_rate:.1%} HIT")

    # precompute scores
    conf_scores = tk_vals_ns  # raw confidence (N, K)
    pos_scores = 1.0 - tk_ids_ns / 500.0  # proximity (N, K)
    # confidence relative to top-1: how close is this candidate to the best?
    top1_conf = tk_vals_ns[:, 0:1].clip(min=1e-8)  # (N, 1)
    rel_conf = tk_vals_ns / top1_conf  # (N, K) — 1.0 for top-1, <1 for others

    # sweep
    conf_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pos_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
    ent_weights = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]

    total = len(conf_weights) * len(pos_weights) * len(ent_weights)
    print(f"Sweeping {total} combinations...")

    results = []
    best_hit_rate = 0
    best_params = None

    for cw in tqdm(conf_weights, desc="Sweeping"):
        for pw in pos_weights:
            for ew in ent_weights:
                scores = cw * conf_scores + pw * pos_scores + ew * rel_conf
                best_idx = scores.argmax(axis=1)
                reranked = tk_ids_ns[np.arange(N_ns), best_idx]

                hits = np.array([is_hit(int(reranked[i]), int(t_ns[i])) for i in range(N_ns)])
                hit_rate = hits.mean()

                regression = int((baseline_hits & ~hits).sum())
                improvement = int((~baseline_hits & hits).sum())

                results.append({
                    "conf_weight": cw, "pos_weight": pw, "ent_weight": ew,
                    "hit_rate": float(hit_rate),
                    "regression": regression, "improvement": improvement,
                    "net": improvement - regression,
                })

                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_params = (cw, pw, ew)

    results.sort(key=lambda r: -r["hit_rate"])

    print(f"\n{'='*90}")
    print(f"  ENTROPY-WEIGHTED RERANKING (top-{args.topk})")
    print(f"{'='*90}")
    print(f"  Baseline: {baseline_hit_rate:.1%}")
    print(f"  Best: {best_hit_rate:.1%} (conf={best_params[0]}, pos={best_params[1]}, ent={best_params[2]})")
    print(f"  Improvement: {best_hit_rate - baseline_hit_rate:+.1%}")
    print()
    print(f"  {'cw':>5s}  {'pw':>5s}  {'ew':>5s}  {'HIT':>7s}  {'Δ':>7s}  {'Impr':>6s}  {'Regr':>6s}  {'Net':>6s}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in results[:25]:
        d = r["hit_rate"] - baseline_hit_rate
        print(f"  {r['conf_weight']:5.1f}  {r['pos_weight']:5.1f}  {r['ent_weight']:5.1f}  {r['hit_rate']:6.1%}  {d:+6.1%}  {r['improvement']:6d}  {r['regression']:6d}  {r['net']:+6d}")

    # save
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_39c")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "reranking_entropy_results.json"), "w") as f:
        json.dump({
            "baseline_hit_rate": float(baseline_hit_rate),
            "best_hit_rate": float(best_hit_rate),
            "best_params": {"conf": best_params[0], "pos": best_params[1], "ent": best_params[2]},
            "sweep": results,
        }, f, indent=2)
    print(f"\n  Saved to {out_dir}/reranking_entropy_results.json")


if __name__ == "__main__":
    main()
