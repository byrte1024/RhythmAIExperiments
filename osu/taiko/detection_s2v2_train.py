"""Training script for S2-v2 Context Proposer (per-bin output).

Same input as S2 (gap sequences), but per-bin sigmoid output like S1.
Focal BCE loss, same as S1 training.

Usage:
    cd osu/taiko
    python detection_s2v2_train.py taiko_v2 --run-name s2v2_experiment_65
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from detection_s2v2_model import ContextProposer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

B_PRED = 250
C_EVENTS = 128
MIN_CURSOR_BIN = 6000

# Reuse S2's dataset for gap/ratio input, S1's target format for per-bin binary
from detection_s2_train import S2Dataset


class S2v2Dataset(S2Dataset):
    """Extends S2Dataset to also return per-bin binary targets."""

    def __init__(self, manifest, ds_dir, chart_indices, **kwargs):
        super().__init__(manifest, ds_dir, chart_indices, **kwargs)

    def __getitem__(self, idx):
        # Get S2's gap/ratio/mask/cond/target
        gaps, ratios, mask, cond, target_scalar = super().__getitem__(idx)

        # Build per-bin binary targets (same as S1)
        ci, ei = self.samples[idx][:2]
        evt = self.events[ci]
        if ei == 0:
            cursor = max(0, evt[0] - B_PRED) if len(evt) > 0 else 0
        else:
            cursor = int(evt[ei - 1])

        bin_targets = np.zeros(B_PRED, dtype=np.float32)
        future_events = evt[(evt > cursor) & (evt <= cursor + B_PRED)]
        for e in future_events:
            bin_idx = int(e) - cursor - 1
            if 0 <= bin_idx < B_PRED:
                bin_targets[bin_idx] = 1.0
                if bin_idx > 0:
                    bin_targets[bin_idx - 1] = max(bin_targets[bin_idx - 1], 0.5)
                if bin_idx < B_PRED - 1:
                    bin_targets[bin_idx + 1] = max(bin_targets[bin_idx + 1], 0.5)

        return gaps, ratios, mask, cond, torch.from_numpy(bin_targets)


def compute_metrics(all_logits, all_targets, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    confs = torch.sigmoid(all_logits)
    targets_binary = (all_targets >= 0.5).float()
    m = {}

    onset_mask = targets_binary == 1
    non_onset_mask = targets_binary == 0
    if onset_mask.sum() > 0:
        m["onset_conf_mean"] = float(confs[onset_mask].mean())
    if non_onset_mask.sum() > 0:
        m["non_onset_conf_mean"] = float(confs[non_onset_mask].mean())
    if onset_mask.sum() > 0 and non_onset_mask.sum() > 0:
        m["conf_separation"] = m["onset_conf_mean"] - m["non_onset_conf_mean"]

    for thresh in thresholds:
        preds = (confs >= thresh).float()
        tp = (preds * targets_binary).sum()
        fp = (preds * (1 - targets_binary)).sum()
        fn = ((1 - preds) * targets_binary).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        m[f"precision_{thresh:.1f}"] = float(prec)
        m[f"recall_{thresh:.1f}"] = float(rec)
        m[f"f1_{thresh:.1f}"] = float(f1)
        m[f"avg_proposals_{thresh:.1f}"] = float(preds.sum(dim=-1).mean())

    return m


def save_eval_graphs(all_logits, all_targets, metrics, eval_step, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_dir = os.path.join(run_dir, "evals")
    os.makedirs(eval_dir, exist_ok=True)
    prefix = os.path.join(eval_dir, f"eval_{eval_step:03d}")

    confs = torch.sigmoid(all_logits).numpy()
    targets = all_targets.numpy()
    targets_binary = (targets >= 0.5).astype(np.float32)

    # Confidence distribution
    onset_confs = confs[targets_binary == 1]
    non_onset_confs = confs[targets_binary == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(non_onset_confs, bins=100, range=(0, 1), alpha=0.6, color="#4a90d9",
            label=f"Non-onset (n={len(non_onset_confs):,})", density=True)
    ax.hist(onset_confs, bins=100, range=(0, 1), alpha=0.6, color="#eb4528",
            label=f"Onset (n={len(onset_confs):,})", density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"S2v2 Eval {eval_step}: Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_conf_dist.png", dpi=120)
    plt.close(fig)

    # F1/P/R vs threshold
    thresholds = np.arange(0.05, 0.95, 0.05)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (confs >= t).astype(np.float32)
        tp = (preds * targets_binary).sum()
        fp = (preds * (1 - targets_binary)).sum()
        fn = ((1 - preds) * targets_binary).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
        precs.append(p)
        recs.append(r)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s, "k-", linewidth=2, label="F1")
    ax.plot(thresholds, precs, "b--", linewidth=1.5, label="Precision")
    ax.plot(thresholds, recs, "r--", linewidth=1.5, label="Recall")
    best_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_idx], color="green", linestyle=":", alpha=0.7,
               label=f"Best F1={f1s[best_idx]:.3f} @ {thresholds[best_idx]:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"S2v2 Eval {eval_step}: P/R/F1 vs Threshold")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_pr_curve.png", dpi=120)
    plt.close(fig)


def validate(model, val_loader, device, pos_weight):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_logits = []
    all_targets = []
    bce_pos = torch.tensor([pos_weight], device=device)

    with torch.no_grad():
        for gaps, ratios, mask, cond, bin_targets in val_loader:
            gaps, ratios, mask, cond, bin_targets = (
                gaps.to(device), ratios.to(device), mask.to(device),
                cond.to(device), bin_targets.to(device),
            )
            logits = model(gaps, ratios, mask, cond)
            loss = F.binary_cross_entropy_with_logits(logits, bin_targets, pos_weight=bce_pos)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_targets.append(bin_targets.cpu())

    val_loss = total_loss / max(n_batches, 1)
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    return val_loss, metrics, all_logits, all_targets


def main():
    parser = argparse.ArgumentParser(description="Train S2-v2 Context Proposer (per-bin)")
    parser.add_argument("dataset")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-gru-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    run_dir = os.path.join(SCRIPT_DIR, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evals"), exist_ok=True)

    print(f"S2-v2 Context Proposer (per-bin)")
    print(f"Dataset: {args.dataset}")
    print(f"Run: {args.run_name}")

    config_path = os.path.join(run_dir, "config.json")
    if not args.resume or not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # Split
    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    val_songs = set(songs[:n_val])
    train_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id", str(i)) not in val_songs]
    val_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id", str(i)) in val_songs]

    train_ds = S2v2Dataset(manifest, ds_dir, train_idx, augment=True, subsample=args.subsample)
    val_ds = S2v2Dataset(manifest, ds_dir, val_idx, augment=False, subsample=args.subsample)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = ContextProposer(
        d_model=args.d_model, n_gru_layers=args.n_gru_layers,
        b_pred=B_PRED, max_events=C_EVENTS,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    bce_pos = torch.tensor([args.pos_weight], device=args.device)
    focal_gamma = args.focal_gamma

    start_epoch = 0
    eval_step = 0
    history = []
    best_val_loss = float("inf")

    if args.resume:
        ckpt_path = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt["epoch"]) + 1
            eval_step = ckpt.get("eval_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            hist_path = os.path.join(run_dir, "history.json")
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    history = json.load(f)
            print(f"Resumed from epoch {start_epoch}")

    steps_per_eval = max(1, len(train_loader) // args.evals_per_epoch)
    print(f"Steps/epoch: {len(train_loader)}, evals/epoch: {args.evals_per_epoch}")
    print()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        n_steps = 0
        ema_loss = None
        ema_f1 = None
        ema_prec = None
        ema_rec = None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        for batch_idx, (gaps, ratios, mask, cond, bin_targets) in enumerate(pbar):
            gaps, ratios, mask, cond, bin_targets = (
                gaps.to(args.device), ratios.to(args.device), mask.to(args.device),
                cond.to(args.device), bin_targets.to(args.device),
            )

            logits = model(gaps, ratios, mask, cond)

            # Focal BCE
            bce = F.binary_cross_entropy_with_logits(
                logits, bin_targets, pos_weight=bce_pos, reduction="none")
            if focal_gamma > 0:
                p_t = torch.sigmoid(logits) * bin_targets + (1 - torch.sigmoid(logits)) * (1 - bin_targets)
                focal_weight = (1 - p_t) ** focal_gamma
                loss = (bce * focal_weight).mean()
            else:
                loss = bce.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_steps += 1

            alpha = 0.05
            if ema_loss is None:
                ema_loss = batch_loss
                ema_f1 = 0.0
                ema_prec = 0.0
                ema_rec = 0.0
            ema_loss = ema_loss * (1 - alpha) + batch_loss * alpha

            with torch.no_grad():
                confs = torch.sigmoid(logits)
                tgt_bin = (bin_targets >= 0.5).float()
                best_f1, best_p, best_r, best_t = 0, 0, 0, 0.5
                for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                    preds = (confs >= t).float()
                    tp = (preds * tgt_bin).sum()
                    fp = (preds * (1 - tgt_bin)).sum()
                    fn = ((1 - preds) * tgt_bin).sum()
                    p = float(tp / (tp + fp + 1e-8))
                    r = float(tp / (tp + fn + 1e-8))
                    f1 = 2 * p * r / (p + r + 1e-8)
                    if f1 > best_f1:
                        best_f1, best_p, best_r, best_t = f1, p, r, t
                ema_f1 = ema_f1 * (1 - alpha) + best_f1 * alpha
                ema_prec = ema_prec * (1 - alpha) + best_p * alpha
                ema_rec = ema_rec * (1 - alpha) + best_r * alpha

            pbar.set_postfix_str(f"loss={ema_loss:.3f} F1={ema_f1:.3f} P={ema_prec:.3f} R={ema_rec:.3f} @{best_t:.1f}")

            if (batch_idx + 1) % steps_per_eval == 0:
                eval_step += 1
                train_loss = epoch_loss / max(n_steps, 1)

                val_loss, val_metrics, val_logits, val_targets = validate(
                    model, val_loader, args.device, args.pos_weight)

                epoch_frac = epoch + (batch_idx + 1) / len(train_loader)
                best_f1_val = max(val_metrics.get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7])
                best_t_val = max([0.3, 0.4, 0.5, 0.6, 0.7],
                                 key=lambda t: val_metrics.get(f"f1_{t:.1f}", 0))
                sep = val_metrics.get("conf_separation", 0)

                print(f"\n  Eval {eval_step} (ep {epoch_frac:.2f}): "
                      f"loss={train_loss:.4f}/{val_loss:.4f} | "
                      f"F1={best_f1_val:.3f}@{best_t_val} sep={sep:.4f} "
                      f"props={val_metrics.get(f'avg_proposals_{best_t_val:.1f}', 0):.1f}")

                save_eval_graphs(val_logits, val_targets, val_metrics, eval_step, run_dir)

                entry = {
                    "eval_step": eval_step,
                    "epoch": round(epoch_frac, 4),
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "val_metrics": {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in val_metrics.items()},
                }
                history.append(entry)
                with open(os.path.join(run_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=2)

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                ckpt_data = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch_frac,
                    "eval_step": eval_step,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                }
                torch.save(ckpt_data, os.path.join(ckpt_dir, f"eval_{eval_step:03d}.pt"))
                torch.save(ckpt_data, os.path.join(ckpt_dir, "latest.pt"))
                if is_best:
                    torch.save(ckpt_data, os.path.join(ckpt_dir, "best.pt"))

                model.train()

        scheduler.step()

    print("\nDone!")
    best_entry = max(history, key=lambda e: max(
        e["val_metrics"].get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7]))
    best_f1 = max(best_entry["val_metrics"].get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7])
    print(f"Best F1: {best_f1:.3f} at eval {best_entry['eval_step']}")


if __name__ == "__main__":
    main()
