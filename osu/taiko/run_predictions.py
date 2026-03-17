"""Run model on validation set and save all predictions + context to .npz.

General-purpose script for offline analysis. Saves targets, predictions,
event offsets, event masks, and top-k logits for each sample.

Usage:
    python run_predictions.py taiko_v2 --checkpoint runs/detect_experiment_27/checkpoints/epoch_008.pt
    python run_predictions.py taiko_v2 --checkpoint runs/detect_experiment_27/checkpoints/epoch_008.pt --subsample 10
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
from detection_model import OnsetDetector, DualStreamOnsetDetector, InterleavedOnsetDetector, ContextFiLMDetector


def main():
    parser = argparse.ArgumentParser(description="Run val predictions and save to .npz")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default=None, help="Output .npz path (default: auto)")
    parser.add_argument("--subsample", type=int, default=1, help="Use 1/N of val set")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=10, help="Save top-K predictions")
    args = parser.parse_args()

    # load dataset
    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)
    val_ds = OnsetDataset(manifest, ds_dir, val_idx, augment=False, subsample=args.subsample)
    print(f"Val samples: {len(val_ds)} (subsample={args.subsample})")

    # load model
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt["args"]
    # detect model type from checkpoint
    state_keys = set(ckpt["model"].keys())
    if any("fusion_context_film." in k for k in state_keys):
        model = ContextFiLMDetector(
            d_model=ckpt_args["d_model"],
            enc_layers=ckpt_args.get("enc_layers", 4),
            gap_enc_layers=ckpt_args.get("gap_enc_layers", 2),
            fusion_layers=ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args["n_heads"],
            snippet_frames=ckpt_args.get("snippet_frames", 10),
            dropout=0.0,
        ).to(args.device)
    elif any("audio_self_layers." in k for k in state_keys):
        model = InterleavedOnsetDetector(
            d_model=ckpt_args["d_model"],
            n_blocks=ckpt_args.get("n_blocks", 4),
            n_heads=ckpt_args["n_heads"],
            snippet_frames=ckpt_args.get("snippet_frames", 10),
            dropout=0.0,
        ).to(args.device)
    elif any("cross_attn_fusion." in k for k in state_keys):
        model = DualStreamOnsetDetector(
            d_model=ckpt_args["d_model"],
            n_heads=ckpt_args["n_heads"],
            enc_layers=ckpt_args["enc_layers"],
            gap_enc_layers=ckpt_args.get("gap_enc_layers", 4),
            cross_attn_layers=ckpt_args.get("cross_attn_layers", 2),
            snippet_frames=ckpt_args.get("snippet_frames", 10),
            dropout=0.0,
        ).to(args.device)
    else:
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
    print(f"  Eval step: {ckpt.get('eval_step', '?')}, "
          f"HIT: {ckpt['val_metrics'].get('hit_rate', 0):.1%}")

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    # run val pass
    all_targets = []
    all_preds = []
    all_event_offsets = []
    all_event_masks = []
    all_topk_vals = []
    all_topk_ids = []

    with torch.no_grad():
        for mel, evt_off, evt_mask, cond, target in tqdm(val_loader, desc="Running val"):
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
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_event_offsets.append(evt_off.cpu().numpy())
            all_event_masks.append(evt_mask.cpu().numpy())
            all_topk_vals.append(topk_v.cpu().numpy())
            all_topk_ids.append(topk_i.cpu().numpy())

    data = {
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "event_offsets": np.concatenate(all_event_offsets),
        "event_masks": np.concatenate(all_event_masks),
        "topk_vals": np.concatenate(all_topk_vals),
        "topk_ids": np.concatenate(all_topk_ids),
    }

    # output path
    if args.output:
        out_path = args.output
    else:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        run_name = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint)))
        out_path = os.path.join(SCRIPT_DIR, "runs", run_name,
                                f"predictions_{ckpt_name}_sub{args.subsample}.npz")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **data)
    n = len(data["targets"])
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved {n} samples to {out_path} ({size_mb:.1f} MB)")
    print(f"  Arrays: {', '.join(f'{k}{v.shape}' for k,v in data.items())}")


if __name__ == "__main__":
    main()
