"""Compare entropy/skip analysis across checkpoints to see if training helps.

Usage:
    python analyze_entropy_progression.py taiko_v2 --checkpoints eval_001.pt eval_004.pt eval_008.pt --subsample 8
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


def analyze_checkpoint(model, val_ds_single, val_ds_multi, device, batch_size, workers):
    stop = N_CLASSES - 1
    loader = DataLoader(val_ds_single, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)

    all_targets = []
    all_preds = []
    all_entropy = []
    all_top1_conf = []

    with torch.no_grad():
        for mel, evt_off, evt_mask, cond, target in tqdm(loader, desc="  Val", leave=False):
            mel = mel.to(device, non_blocking=True)
            evt_off = evt_off.to(device, non_blocking=True)
            evt_mask = evt_mask.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)

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

    N = len(targets)

    # get future onsets for skip analysis
    # skip=1 means: target at 75, predicted 150, and 150 is a real onset (skipped 1 ahead)
    n_skipped = np.zeros(N)
    for i in range(N):
        t = targets[i]
        p = preds[i]
        if t >= stop:
            continue
        _, _, _, _, tp, nt = val_ds_multi[i]
        if nt.item() > 0:
            future_bins = tp[:nt.item()].numpy()
            if p > t:
                onsets_beyond_target = future_bins[future_bins > t]
                n_skipped[i] = np.sum(onsets_beyond_target <= p)
            elif p < t:
                n_skipped[i] = -1  # underprediction

    ns = targets < stop
    t_ns = targets[ns]
    p_ns = preds[ns]
    ent_ns = entropy[ns]
    conf_ns = top1_conf[ns]
    skip_ns = n_skipped[ns]
    N_ns = len(t_ns)

    hit_ns = np.array([is_hit(int(p_ns[i]), int(t_ns[i])) for i in range(N_ns)])
    overpred = p_ns > t_ns
    underpred = p_ns < t_ns

    results = {
        "hit_rate": hit_ns.mean(),
        "miss_rate": 1 - hit_ns.mean(),
        "mean_entropy": ent_ns.mean(),
        "mean_conf": conf_ns.mean(),
        "overpred_rate": overpred.mean(),
        "underpred_rate": underpred[~hit_ns].sum() / N_ns,
        "hit_entropy": ent_ns[hit_ns].mean() if hit_ns.any() else 0,
        "miss_entropy": ent_ns[~hit_ns].mean() if (~hit_ns).any() else 0,
        "hit_conf": conf_ns[hit_ns].mean() if hit_ns.any() else 0,
        "miss_conf": conf_ns[~hit_ns].mean() if (~hit_ns).any() else 0,
    }

    # skip breakdown
    for n_skip in [0, 1, 2]:
        if n_skip < 2:
            mask = skip_ns == n_skip
        else:
            mask = skip_ns >= n_skip
        label = str(n_skip) if n_skip < 2 else f"{n_skip}+"
        if mask.sum() > 0:
            results[f"skip{label}_n"] = int(mask.sum())
            results[f"skip{label}_pct"] = mask.mean()
            results[f"skip{label}_hit"] = hit_ns[mask].mean()
            results[f"skip{label}_entropy"] = ent_ns[mask].mean()
            results[f"skip{label}_conf"] = conf_ns[mask].mean()

    # entropy by distance
    for lo, hi in [(0, 30), (30, 100), (100, 500)]:
        mask = (t_ns >= lo) & (t_ns < hi)
        if mask.sum() > 0:
            results[f"dist{lo}-{hi}_entropy"] = ent_ns[mask].mean()
            results[f"dist{lo}-{hi}_conf"] = conf_ns[mask].mean()
            results[f"dist{lo}-{hi}_hit"] = hit_ns[mask].mean()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--checkpoints", nargs="+", required=True)
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

    all_results = []

    for ckpt_path in args.checkpoints:
        print(f"\n{'='*60}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        ckpt_args = ckpt["args"]
        eval_step = ckpt.get("eval_step", "?")

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
        print(f"  Eval step: {eval_step}, HIT: {ckpt['val_metrics'].get('hit_rate', 0):.1%}")

        r = analyze_checkpoint(model, val_ds_single, val_ds_multi,
                               args.device, args.batch_size, args.workers)
        r["checkpoint"] = ckpt_path
        r["eval_step"] = eval_step
        all_results.append(r)

    # comparison table
    print(f"\n{'='*90}")
    print(f"  PROGRESSION COMPARISON")
    print(f"{'='*90}")

    headers = ["Metric"] + [f"eval_{r['eval_step']}" for r in all_results]
    print(f"  {'Metric':30s}" + "".join(f"  {h:>12s}" for h in headers[1:]))
    print(f"  {'-'*30}" + "".join(f"  {'-'*12}" for _ in all_results))

    metrics = [
        ("HIT rate", "hit_rate", ".1%"),
        ("Mean entropy", "mean_entropy", ".3f"),
        ("Mean confidence", "mean_conf", ".3f"),
        ("HIT entropy", "hit_entropy", ".3f"),
        ("MISS entropy", "miss_entropy", ".3f"),
        ("HIT confidence", "hit_conf", ".3f"),
        ("MISS confidence", "miss_conf", ".3f"),
        ("Overpred rate", "overpred_rate", ".1%"),
        ("Skip 0 HIT", "skip0_hit", ".1%"),
        ("Skip 0 entropy", "skip0_entropy", ".3f"),
        ("Skip 1 HIT", "skip1_hit", ".1%"),
        ("Skip 1 entropy", "skip1_entropy", ".3f"),
        ("Skip 2+ HIT", "skip2+_hit", ".1%"),
        ("Skip 2+ entropy", "skip2+_entropy", ".3f"),
        ("Dist 0-30 HIT", "dist0-30_hit", ".1%"),
        ("Dist 0-30 entropy", "dist0-30_entropy", ".3f"),
        ("Dist 30-100 HIT", "dist30-100_hit", ".1%"),
        ("Dist 30-100 entropy", "dist30-100_entropy", ".3f"),
        ("Dist 100-500 HIT", "dist100-500_hit", ".1%"),
        ("Dist 100-500 entropy", "dist100-500_entropy", ".3f"),
    ]

    for label, key, fmt in metrics:
        row = f"  {label:30s}"
        for r in all_results:
            val = r.get(key, None)
            if val is not None:
                row += f"  {val:>12{fmt}}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)

    # save
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_41")
    with open(os.path.join(out_dir, "progression_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out_dir}/progression_results.json")


if __name__ == "__main__":
    main()
