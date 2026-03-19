"""Experiment 39-B: Rerank top-K by confidence × position proximity.

For each prediction, take the top-K candidates and rerank them by:
  score = confidence_rank_weight * (1 - rank/K) + position_weight * (1 - bin/500)

Higher score = prefer more confident AND closer predictions.
Sweep different weight combinations to find the best HIT rate.

Also track: how many existing HITs become misses (regression) from reranking.

Usage:
    python analyze_reranking.py taiko_v2 --checkpoint runs/detect_experiment_35c/checkpoints/eval_008.pt --subsample 8
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
    OnsetDataset, N_CLASSES, B_BINS, C_EVENTS, split_by_song,
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

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # collect predictions
    all_targets = []
    all_topk_ids = []    # (N, K) bin indices
    all_topk_vals = []   # (N, K) softmax probabilities

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
    topk_ids = np.concatenate(all_topk_ids)    # (N, K)
    topk_vals = np.concatenate(all_topk_vals)  # (N, K)

    N = len(targets)
    K = args.topk
    stop = N_CLASSES - 1

    # filter non-stop
    ns = targets < stop
    t_ns = targets[ns]
    tk_ids_ns = topk_ids[ns]
    tk_vals_ns = topk_vals[ns]
    N_ns = len(t_ns)

    # baseline: argmax (rank 0)
    baseline_preds = tk_ids_ns[:, 0]
    baseline_hits = np.array([is_hit(int(baseline_preds[i]), int(t_ns[i])) for i in range(N_ns)])
    baseline_hit_rate = baseline_hits.mean()
    print(f"\nBaseline (argmax): {baseline_hit_rate:.1%} HIT ({baseline_hits.sum()}/{N_ns})")

    # sweep: confidence_weight vs position_weight
    # score(k) = conf_w * confidence[k] + pos_w * (1 - bin[k] / 500)
    # pick the candidate with highest score
    conf_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pos_weights = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    print(f"\nSweeping {len(conf_weights)} × {len(pos_weights)} = {len(conf_weights)*len(pos_weights)} weight combinations...")

    # precompute: normalized confidence and position scores for all candidates
    # confidence: topk_vals already in descending order, normalize to [0,1]
    conf_scores = tk_vals_ns / tk_vals_ns[:, 0:1].clip(min=1e-8)  # relative to top-1
    # position: prefer closer (lower bin), normalize to [0,1]
    pos_scores = 1.0 - tk_ids_ns / 500.0  # bin 0 → 1.0, bin 500 → 0.0

    results = []
    best_hit_rate = 0
    best_params = None

    for cw in tqdm(conf_weights, desc="Confidence weights"):
        for pw in pos_weights:
            scores = cw * conf_scores + pw * pos_scores  # (N_ns, K)
            best_idx = scores.argmax(axis=1)  # (N_ns,)
            reranked_preds = tk_ids_ns[np.arange(N_ns), best_idx]

            hits = np.array([is_hit(int(reranked_preds[i]), int(t_ns[i])) for i in range(N_ns)])
            hit_rate = hits.mean()

            # regression: how many baseline HITs became misses?
            regression = (baseline_hits & ~hits).sum()
            # improvement: how many baseline misses became HITs?
            improvement = (~baseline_hits & hits).sum()
            net = improvement - regression

            results.append({
                "conf_weight": cw,
                "pos_weight": pw,
                "hit_rate": float(hit_rate),
                "hits": int(hits.sum()),
                "regression": int(regression),
                "improvement": int(improvement),
                "net": int(net),
            })

            if hit_rate > best_hit_rate:
                best_hit_rate = hit_rate
                best_params = (cw, pw)

    # sort by hit rate
    results.sort(key=lambda r: -r["hit_rate"])

    print(f"\n{'='*80}")
    print(f"  RERANKING RESULTS (top-{K} candidates)")
    print(f"{'='*80}")
    print(f"  Baseline (argmax): {baseline_hit_rate:.1%}")
    print(f"  Best reranked: {best_hit_rate:.1%} (conf_w={best_params[0]}, pos_w={best_params[1]})")
    print(f"  Improvement: {best_hit_rate - baseline_hit_rate:+.1%}")
    print()
    print(f"  {'conf_w':>7s}  {'pos_w':>7s}  {'HIT':>8s}  {'Δ':>8s}  {'Improved':>8s}  {'Regressed':>9s}  {'Net':>6s}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*6}")
    for r in results[:20]:
        delta = r["hit_rate"] - baseline_hit_rate
        print(f"  {r['conf_weight']:7.1f}  {r['pos_weight']:7.1f}  {r['hit_rate']:7.1%}  {delta:+7.1%}  {r['improvement']:8d}  {r['regression']:9d}  {r['net']:+6d}")

    # also show worst results
    print(f"\n  ... worst 5:")
    for r in results[-5:]:
        delta = r["hit_rate"] - baseline_hit_rate
        print(f"  {r['conf_weight']:7.1f}  {r['pos_weight']:7.1f}  {r['hit_rate']:7.1%}  {delta:+7.1%}  {r['improvement']:8d}  {r['regression']:9d}  {r['net']:+6d}")

    # save
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_39")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "reranking_results.json"), "w") as f:
        json.dump({
            "baseline_hit_rate": float(baseline_hit_rate),
            "best_hit_rate": float(best_hit_rate),
            "best_params": {"conf_weight": best_params[0], "pos_weight": best_params[1]},
            "sweep": results,
        }, f, indent=2)
    print(f"\n  Saved to {out_dir}/reranking_results.json")


if __name__ == "__main__":
    main()
