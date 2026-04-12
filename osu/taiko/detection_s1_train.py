"""Training script for S1 Conformer Proposer.

Audio-only onset detection. Per-bin binary classification.
No events, no context, no density conditioning.

Usage:
    cd osu/taiko
    python detection_s1_train.py taiko_v2 --run-name s1_experiment_65
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from detection_s1_model import ConformerProposer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

A_BINS = 500
B_BINS = 500
B_PRED = 250
MIN_CURSOR_BIN = 6000
N_MELS = 80


# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════

class S1Dataset(Dataset):
    """Yields (mel_window, bin_targets) for audio-only onset detection.

    bin_targets: binary vector of length B_PRED, 1 at onset positions.
    """

    def __init__(self, manifest, ds_dir, chart_indices, augment=False, subsample=1):
        self.mel_dir = os.path.join(ds_dir, "mels")
        self.charts = [manifest["charts"][i] for i in chart_indices]
        self.augment = augment

        self.events = []
        evt_dir = os.path.join(ds_dir, "events")
        for chart in self.charts:
            evt = np.load(os.path.join(evt_dir, chart["event_file"]))
            self.events.append(evt)

        # Build sample index (same cursor logic as main training)
        self.samples = []
        for ci, evt in enumerate(self.events):
            for ei in range(len(evt)):
                cursor = max(0, int(evt[0]) - B_PRED) if ei == 0 else int(evt[ei - 1])
                if cursor >= MIN_CURSOR_BIN:
                    self.samples.append((ci, ei, cursor))
            if len(evt) > 0 and int(evt[-1]) >= MIN_CURSOR_BIN:
                self.samples.append((ci, len(evt), int(evt[-1])))

        if subsample > 1:
            self.samples = self.samples[::subsample]

        self._mel_cache = {}

    def _get_mel(self, mel_file):
        if mel_file not in self._mel_cache:
            self._mel_cache[mel_file] = np.load(
                os.path.join(self.mel_dir, mel_file), mmap_mode="r"
            )
        return self._mel_cache[mel_file]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, ei, cursor = self.samples[idx]
        chart = self.charts[ci]
        evt = self.events[ci]

        # Mel window
        mel = self._get_mel(chart["mel_file"])
        total_frames = mel.shape[1]
        start = cursor - A_BINS
        end = cursor + B_BINS
        pad_left = max(0, -start)
        pad_right = max(0, end - total_frames)
        mel_window = mel[:, max(0, start):min(total_frames, end)].astype(np.float32)
        if pad_left > 0 or pad_right > 0:
            mel_window = np.pad(mel_window, ((0, 0), (pad_left, pad_right)), mode="constant")

        # Audio augmentation
        if self.augment:
            mel_window = self._augment_audio(mel_window)

        # Binary targets: 1 at each onset bin in B_PRED range
        bin_targets = np.zeros(B_PRED, dtype=np.float32)
        future_events = evt[(evt > cursor) & (evt <= cursor + B_PRED)]
        for e in future_events:
            bin_idx = int(e) - cursor - 1  # 0-indexed, bin 0 = cursor+1
            if 0 <= bin_idx < B_PRED:
                bin_targets[bin_idx] = 1.0
                # ±1 bin soft targets
                if bin_idx > 0:
                    bin_targets[bin_idx - 1] = max(bin_targets[bin_idx - 1], 0.5)
                if bin_idx < B_PRED - 1:
                    bin_targets[bin_idx + 1] = max(bin_targets[bin_idx + 1], 0.5)

        return torch.from_numpy(mel_window), torch.from_numpy(bin_targets)

    def _augment_audio(self, mel):
        # Gain ±2dB
        if random.random() < 0.3:
            mel = mel + np.random.uniform(-2, 2)
        # Noise
        if random.random() < 0.15:
            mel = mel + np.random.randn(*mel.shape).astype(np.float32) * np.random.uniform(0, 0.3)
        # Freq jitter
        if random.random() < 0.15:
            shift = np.random.randint(-3, 4)
            mel = np.roll(mel, shift, axis=0)
        # SpecAugment freq
        if random.random() < 0.2:
            f0 = np.random.randint(0, mel.shape[0] - 10)
            mel[f0:f0 + 10] = 0
        # SpecAugment time
        if random.random() < 0.2:
            t0 = np.random.randint(0, max(1, mel.shape[1] - 30))
            mel[:, t0:t0 + 30] = 0
        return mel


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(all_logits, all_targets, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Compute per-bin detection metrics."""
    confs = torch.sigmoid(all_logits)
    targets_binary = (all_targets >= 0.5).float()  # hard targets from soft

    m = {}

    # Confidence stats
    onset_mask = targets_binary == 1
    non_onset_mask = targets_binary == 0
    if onset_mask.sum() > 0:
        m["onset_conf_mean"] = float(confs[onset_mask].mean())
    if non_onset_mask.sum() > 0:
        m["non_onset_conf_mean"] = float(confs[non_onset_mask].mean())
    if onset_mask.sum() > 0 and non_onset_mask.sum() > 0:
        m["conf_separation"] = m["onset_conf_mean"] - m["non_onset_conf_mean"]

    # Per-threshold metrics
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

        # Average proposals per sample
        proposals_per_sample = preds.sum(dim=-1).mean()
        m[f"avg_proposals_{thresh:.1f}"] = float(proposals_per_sample)

    return m


# ═══════════════════════════════════════════════════════════════
#  Graphs
# ═══════════════════════════════════════════════════════════════

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

    # ── 1. Confidence distribution: onset vs non-onset ──
    onset_confs = confs[targets_binary == 1]
    non_onset_confs = confs[targets_binary == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(non_onset_confs, bins=100, range=(0, 1), alpha=0.6, color="#4a90d9",
            label=f"Non-onset (n={len(non_onset_confs):,})", density=True)
    ax.hist(onset_confs, bins=100, range=(0, 1), alpha=0.6, color="#eb4528",
            label=f"Onset (n={len(onset_confs):,})", density=True)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"S1 Eval {eval_step}: Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_conf_dist.png", dpi=120)
    plt.close(fig)

    # ── 2. F1 / Precision / Recall vs threshold ──
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
    best_f1_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_f1_idx], color="green", linestyle=":", alpha=0.7,
               label=f"Best F1={f1s[best_f1_idx]:.3f} @ {thresholds[best_f1_idx]:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"S1 Eval {eval_step}: Precision / Recall / F1 vs Threshold")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_pr_curve.png", dpi=120)
    plt.close(fig)

    # ── 3. Average confidence per bin position ──
    mean_conf = confs.mean(axis=0)  # (B_PRED,)
    mean_target = targets_binary.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(mean_conf, color="#e8834a", linewidth=1)
    axes[0].set_title(f"S1 Eval {eval_step}: Mean Confidence by Bin Position")
    axes[0].set_ylabel("Mean confidence")
    axes[1].plot(mean_target, color="#4a90d9", linewidth=1)
    axes[1].set_title("Mean Target Density by Bin Position")
    axes[1].set_ylabel("Mean target")
    axes[1].set_xlabel("Bin offset from cursor")
    fig.tight_layout()
    fig.savefig(f"{prefix}_bin_profile.png", dpi=120)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════

def validate(model, val_loader, device, pos_weight):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_logits = []
    all_targets = []

    bce_pos = torch.tensor([pos_weight], device=device)

    with torch.no_grad():
        for mel, targets in val_loader:
            mel, targets = mel.to(device), targets.to(device)
            logits = model(mel)
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=bce_pos)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    val_loss = total_loss / max(n_batches, 1)
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)

    return val_loss, metrics, all_logits, all_targets


def main():
    parser = argparse.ArgumentParser(description="Train S1 Conformer Proposer")
    parser.add_argument("dataset", help="Dataset name (e.g. taiko_v2)")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
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

    print(f"S1 Conformer Proposer")
    print(f"Dataset: {args.dataset} ({manifest['total_charts']} charts)")
    print(f"Run: {args.run_name}")

    config_path = os.path.join(run_dir, "config.json")
    if not args.resume or not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # Train/val split
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

    train_ds = S1Dataset(manifest, ds_dir, train_idx, augment=True, subsample=args.subsample)
    val_ds = S1Dataset(manifest, ds_dir, val_idx, augment=False, subsample=args.subsample)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Model
    model = ConformerProposer(
        n_mels=N_MELS, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=8, conv_kernel=args.conv_kernel,
        a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss: focal BCE
    bce_pos = torch.tensor([args.pos_weight], device=args.device)
    focal_gamma = args.focal_gamma

    # Resume
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
            print(f"Resumed from epoch {start_epoch}, eval {eval_step}")

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

        for batch_idx, (mel, targets) in enumerate(pbar):
            mel, targets = mel.to(args.device), targets.to(args.device)

            logits = model(mel)

            # Focal BCE
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=bce_pos, reduction="none")
            if focal_gamma > 0:
                p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
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

            # EMA for tqdm
            alpha = 0.05
            if ema_loss is None:
                ema_loss = batch_loss
                ema_f1 = 0.0
                ema_prec = 0.0
                ema_rec = 0.0
            ema_loss = ema_loss * (1 - alpha) + batch_loss * alpha

            # Quick F1/prec/rec with auto-adapting threshold
            with torch.no_grad():
                confs = torch.sigmoid(logits)
                tgt_bin = (targets >= 0.5).float()
                # Try thresholds, pick best F1
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

            # Eval
            if (batch_idx + 1) % steps_per_eval == 0:
                eval_step += 1
                train_loss = epoch_loss / max(n_steps, 1)

                val_loss, val_metrics, val_logits, val_targets = validate(
                    model, val_loader, args.device, args.pos_weight)

                epoch_frac = epoch + (batch_idx + 1) / len(train_loader)
                best_f1 = max(val_metrics.get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7])
                best_thresh = max([0.3, 0.4, 0.5, 0.6, 0.7],
                                  key=lambda t: val_metrics.get(f"f1_{t:.1f}", 0))
                sep = val_metrics.get("conf_separation", 0)

                print(f"\n  Eval {eval_step} (ep {epoch_frac:.2f}): "
                      f"loss={train_loss:.4f}/{val_loss:.4f} | "
                      f"F1={best_f1:.3f}@{best_thresh} sep={sep:.4f} "
                      f"props={val_metrics.get(f'avg_proposals_{best_thresh:.1f}', 0):.1f}")

                # Graphs
                save_eval_graphs(val_logits, val_targets, val_metrics, eval_step, run_dir)

                # History
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

                # Checkpoints
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

    print("\nTraining complete!")
    best_entry = max(history, key=lambda e: max(
        e["val_metrics"].get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7]))
    best_f1 = max(best_entry["val_metrics"].get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7])
    print(f"Best F1: {best_f1:.3f} at eval {best_entry['eval_step']}")


if __name__ == "__main__":
    main()
