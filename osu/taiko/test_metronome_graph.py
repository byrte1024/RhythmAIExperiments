"""Quick test: load exp44 model, run val with subsample 8, generate metronome graphs.

Mirrors the training script's validation setup exactly, just with subsample 8 for speed.
"""
import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_train import (
    OnsetDataset, split_by_song, validate_and_collect, compute_metrics,
    save_eval_graphs, N_CLASSES, C_EVENTS
)
from detection_model import EventEmbeddingDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, "runs", "detect_experiment_44")
DS_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")

def main():
    # load manifest + config
    with open(os.path.join(DS_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(RUN_DIR, "config.json")) as f:
        config = json.load(f)

    # split (same seed as training)
    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)

    # val dataset with subsample 8 for speed
    val_ds = OnsetDataset(manifest, DS_DIR, val_idx, augment=False, subsample=8,
                          multi_target=False)
    print(f"Val samples (subsample=8): {len(val_ds)}")

    nw = config.get("workers", 3)
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=False, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    # load model (same params as training)
    device = torch.device(config.get("device", "cuda"))
    model = EventEmbeddingDetector(
        n_mels=80, d_model=config["d_model"],
        n_layers=config["enc_layers"] + config["fusion_layers"],
        n_heads=config["n_heads"],
        n_classes=N_CLASSES, max_events=C_EVENTS, dropout=config["dropout"],
    ).to(device)

    # find latest checkpoint
    ckpt_dir = os.path.join(RUN_DIR, "checkpoints")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpts:
        print("No checkpoints found!")
        return
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {ckpts[-1]}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state["model"] if "model" in state else state
    # handle compiled model keys
    clean_state = {}
    for k, v in model_state.items():
        clean_key = k.replace("_orig_mod.", "")
        clean_state[clean_key] = v
    model.load_state_dict(clean_state)

    # validate — same call as _run_eval in training script
    amp_enabled = config.get("amp", False)
    criterion = torch.nn.CrossEntropyLoss()
    print("Running validation...")
    val_loss, extra = validate_and_collect(
        model, val_loader, criterion, device, amp_enabled=amp_enabled,
        sigmoid_mode=False, multi_target=False, framewise=False,
    )
    targets = extra["targets"]
    preds = extra["preds"]
    metrics = compute_metrics(targets, preds)
    print(f"Val loss: {val_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"HIT rate: {metrics['hit_rate']:.4f}")

    # check top3_gaps
    t3 = extra.get("top3_gaps")
    if t3 is not None:
        print(f"top3_gaps shape: {t3.shape}")
        valid1 = np.isfinite(t3[:, 0]).sum()
        valid2 = np.isfinite(t3[:, 1]).sum()
        valid3 = np.isfinite(t3[:, 2]).sum()
        print(f"Valid top1: {valid1}, top2: {valid2}, top3: {valid3}")
    else:
        print("WARNING: top3_gaps not in extra!")

    # generate graphs to a test directory
    test_dir = os.path.join(RUN_DIR, "test_metronome")
    os.makedirs(test_dir, exist_ok=True)
    print(f"Saving graphs to {test_dir}")
    save_eval_graphs(targets, preds, metrics, 999, test_dir, extra=extra)
    print("Done! Check test_metronome/ for metronome_scatter.png and metronome_heatmap.png")


if __name__ == "__main__":
    main()
