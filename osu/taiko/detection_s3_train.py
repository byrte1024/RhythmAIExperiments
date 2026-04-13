"""Training script for S3 Fusion Selector.

Runs frozen S1 + S2v2 in real-time, fuses their outputs via DETR-style
encoder-decoder with auxiliary losses per decoder layer.

Usage:
    cd osu/taiko
    python detection_s3_train.py taiko_v2 --run-name s3_experiment_65 \
        --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
        --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt
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
from detection_s2v2_model import ContextProposer
from detection_s3_model import FusionSelector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

A_BINS = 500
B_BINS = 500
B_PRED = 250
C_EVENTS = 128
MIN_CURSOR_BIN = 6000
N_MELS = 80


# ═══════════════════════════════════════════════════════════════
#  Dataset — provides mel + context for S1/S2v2/S3
# ═══════════════════════════════════════════════════════════════

class S3Dataset(Dataset):
    """Yields (mel_window, gap_seq, ratio_seq, event_mask, event_offsets, cond, bin_targets)."""

    def __init__(self, manifest, ds_dir, chart_indices, augment=False, subsample=1):
        self.mel_dir = os.path.join(ds_dir, "mels")
        self.charts = [manifest["charts"][i] for i in chart_indices]
        self.augment = augment

        self.events = []
        evt_dir = os.path.join(ds_dir, "events")
        for chart in self.charts:
            evt = np.load(os.path.join(evt_dir, chart["event_file"]))
            self.events.append(evt)

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
                os.path.join(self.mel_dir, mel_file), mmap_mode="r")
        return self._mel_cache[mel_file]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, ei, cursor = self.samples[idx]
        chart = self.charts[ci]
        evt = self.events[ci]

        # ── Mel window ──
        mel = self._get_mel(chart["mel_file"])
        total_frames = mel.shape[1]
        start = cursor - A_BINS
        end = cursor + B_BINS
        pad_left = max(0, -start)
        pad_right = max(0, end - total_frames)
        mel_window = mel[:, max(0, start):min(total_frames, end)].astype(np.float32)
        if pad_left > 0 or pad_right > 0:
            mel_window = np.pad(mel_window, ((0, 0), (pad_left, pad_right)), mode="constant")

        # ── Past events as offsets from cursor ──
        if ei > 0:
            past_start = max(0, ei - C_EVENTS)
            past_bins = evt[past_start:ei].astype(np.int64).copy() - cursor
        else:
            past_bins = np.array([], dtype=np.int64)

        # ── Gaps and ratios for S2v2 ──
        if ei > 0:
            past_abs = evt[max(0, ei - C_EVENTS):ei].astype(np.int64).copy()
        else:
            past_abs = np.array([], dtype=np.int64)

        n_past = len(past_abs)
        gaps = np.zeros(C_EVENTS, dtype=np.float32)
        ratios = np.zeros(C_EVENTS, dtype=np.float32)
        s2_mask = np.ones(C_EVENTS, dtype=bool)

        if n_past > 0:
            raw_gaps = np.zeros(n_past, dtype=np.float32)
            if n_past >= 2:
                raw_gaps[1:] = np.diff(past_abs).astype(np.float32)
                raw_gaps[0] = raw_gaps[1] if n_past >= 2 else 30.0
            else:
                raw_gaps[0] = 30.0
            raw_gaps = np.maximum(raw_gaps, 1.0)

            raw_ratios = np.ones(n_past, dtype=np.float32)
            if n_past >= 2:
                for i in range(1, n_past):
                    raw_ratios[i] = np.clip(raw_gaps[i] / max(raw_gaps[i - 1], 1.0), 0.1, 10.0)

            s = C_EVENTS - n_past
            gaps[s:] = raw_gaps
            ratios[s:] = raw_ratios
            s2_mask[s:] = False

        # ── Event offsets for S3 encoder ──
        n_past_off = len(past_bins)
        event_offsets = np.zeros(C_EVENTS, dtype=np.int64)
        event_mask = np.ones(C_EVENTS, dtype=bool)
        if n_past_off > 0:
            if n_past_off > C_EVENTS:
                past_bins = past_bins[-C_EVENTS:]
                n_past_off = C_EVENTS
            event_offsets[-n_past_off:] = past_bins
            event_mask[-n_past_off:] = False

        # ── Conditioning ──
        cond = np.array([
            chart.get("density_mean", 4.0),
            chart.get("density_peak", 8),
            chart.get("density_std", 1.5),
        ], dtype=np.float32)

        # ── Augmentations ──
        if self.augment:
            mel_window = self._augment_audio(mel_window)
            # Context augmentation (light)
            if n_past_off > 0:
                jitter = np.random.randint(-1, 2, size=n_past_off)
                event_offsets[-n_past_off:] += jitter
            if n_past > 0:
                gap_jitter = np.random.randint(-1, 2, size=n_past)
                raw_gaps_j = np.maximum(raw_gaps + gap_jitter, 1.0)
                s = C_EVENTS - n_past
                gaps[s:] = raw_gaps_j
                raw_ratios_j = np.ones(n_past, dtype=np.float32)
                if n_past >= 2:
                    for i in range(1, n_past):
                        raw_ratios_j[i] = np.clip(raw_gaps_j[i] / max(raw_gaps_j[i-1], 1.0), 0.1, 10.0)
                ratios[s:] = raw_ratios_j
            # Density jitter
            if random.random() < 0.2:
                cond = cond * (1.0 + np.random.uniform(-0.05, 0.05, size=3).astype(np.float32))
            # Context truncation (rare)
            if random.random() < 0.02 and n_past_off > 32:
                keep = random.randint(32, n_past_off)
                event_offsets[:C_EVENTS - keep] = 0
                event_mask[:C_EVENTS - keep] = True

        # ── Binary targets ──
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

        return (
            torch.from_numpy(mel_window),      # for S1
            torch.from_numpy(gaps),             # for S2v2
            torch.from_numpy(ratios),           # for S2v2
            torch.from_numpy(s2_mask),          # for S2v2
            torch.from_numpy(event_offsets),    # for S3 encoder
            torch.from_numpy(event_mask),       # for S3 encoder
            torch.from_numpy(cond),             # for S2v2 + S3
            torch.from_numpy(bin_targets),      # supervision
        )

    def _augment_audio(self, mel):
        if random.random() < 0.3:
            mel = mel + np.random.uniform(-2, 2)
        if random.random() < 0.15:
            mel = mel + np.random.randn(*mel.shape).astype(np.float32) * np.random.uniform(0, 0.3)
        if random.random() < 0.15:
            mel = np.roll(mel, np.random.randint(-3, 4), axis=0)
        if random.random() < 0.2:
            f0 = np.random.randint(0, mel.shape[0] - 10)
            mel[f0:f0 + 10] = 0
        if random.random() < 0.2:
            t0 = np.random.randint(0, max(1, mel.shape[1] - 30))
            mel[:, t0:t0 + 30] = 0
        return mel


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(all_logits, all_targets, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    confs = torch.sigmoid(all_logits)
    tgt = (all_targets >= 0.5).float()
    m = {}

    onset_mask = tgt == 1
    non_onset_mask = tgt == 0
    if onset_mask.sum() > 0:
        m["onset_conf"] = float(confs[onset_mask].mean())
    if non_onset_mask.sum() > 0:
        m["non_onset_conf"] = float(confs[non_onset_mask].mean())
    if onset_mask.sum() > 0 and non_onset_mask.sum() > 0:
        m["conf_separation"] = m["onset_conf"] - m["non_onset_conf"]

    for t in thresholds:
        preds = (confs >= t).float()
        tp = (preds * tgt).sum()
        fp = (preds * (1 - tgt)).sum()
        fn = ((1 - preds) * tgt).sum()
        p = float(tp / (tp + fp + 1e-8))
        r = float(tp / (tp + fn + 1e-8))
        f1 = 2 * p * r / (p + r + 1e-8)
        m[f"f1_{t:.1f}"] = f1
        m[f"precision_{t:.1f}"] = p
        m[f"recall_{t:.1f}"] = r
        m[f"proposals_{t:.1f}"] = float(preds.sum(dim=-1).mean())

    return m


def save_eval_graphs(all_logits, all_targets, aux_logits_list, metrics, eval_step, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eval_dir = os.path.join(run_dir, "evals")
    os.makedirs(eval_dir, exist_ok=True)
    prefix = os.path.join(eval_dir, f"eval_{eval_step:03d}")

    confs = torch.sigmoid(all_logits).numpy()
    tgt = (all_targets.numpy() >= 0.5).astype(np.float32)

    # Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confs[tgt == 0], bins=100, range=(0, 1), alpha=0.6, color="#4a90d9", label="Non-onset", density=True)
    ax.hist(confs[tgt == 1], bins=100, range=(0, 1), alpha=0.6, color="#eb4528", label="Onset", density=True)
    ax.set_xlabel("Confidence")
    ax.set_title(f"S3 Eval {eval_step}: Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_conf_dist.png", dpi=120)
    plt.close(fig)

    # PR curve
    thresholds = np.arange(0.05, 0.95, 0.05)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        preds = (confs >= t).astype(np.float32)
        tp = (preds * tgt).sum()
        fp = (preds * (1 - tgt)).sum()
        fn = ((1 - preds) * tgt).sum()
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
    ax.set_title(f"S3 Eval {eval_step}: P/R/F1")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_pr_curve.png", dpi=120)
    plt.close(fig)

    # Per-decoder-layer F1
    if aux_logits_list:
        fig, ax = plt.subplots(figsize=(10, 6))
        for li, aux in enumerate(aux_logits_list):
            aux_confs = torch.sigmoid(aux).numpy()
            layer_f1s = []
            for t in thresholds:
                preds = (aux_confs >= t).astype(np.float32)
                tp = (preds * tgt).sum()
                fp = (preds * (1 - tgt)).sum()
                fn = ((1 - preds) * tgt).sum()
                p = tp / (tp + fp + 1e-8)
                r = tp / (tp + fn + 1e-8)
                layer_f1s.append(2 * p * r / (p + r + 1e-8))
            ax.plot(thresholds, layer_f1s, linewidth=2, label=f"Layer {li+1} (best={max(layer_f1s):.3f})")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_title(f"S3 Eval {eval_step}: F1 per Decoder Layer")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_layer_f1.png", dpi=120)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════

def validate(s1_model, s2_model, s3_model, val_loader, device, focal_gamma, pos_weight):
    s3_model.eval()
    total_loss = 0
    n_batches = 0
    all_logits = []
    all_targets = []
    all_aux = [[] for _ in range(3)]

    bce_pos = torch.tensor([pos_weight], device=device)

    with torch.no_grad():
        for mel, gaps, ratios, s2_mask, evt_off, evt_mask, cond, targets in val_loader:
            mel = mel.to(device)
            gaps, ratios, s2_mask, cond = gaps.to(device), ratios.to(device), s2_mask.to(device), cond.to(device)
            evt_off, evt_mask = evt_off.to(device), evt_mask.to(device)
            targets = targets.to(device)

            # Run frozen S1
            s1_logits = s1_model(mel)
            s1_conf = torch.sigmoid(s1_logits)
            # Extract audio features from S1's conv stem
            with torch.no_grad():
                audio_feat = s1_model.conv(mel).transpose(1, 2)
                audio_feat = s1_model.conv_norm(audio_feat)

            # Run frozen S2v2
            s2_logits = s2_model(gaps, ratios, s2_mask, cond)
            s2_conf = torch.sigmoid(s2_logits)

            # Run S3
            final_logits, aux_logits = s3_model(audio_feat, s1_conf, s2_conf,
                                                 evt_off, evt_mask, cond)

            # Loss (average over decoder layers)
            loss = 0
            for aux in aux_logits:
                bce = F.binary_cross_entropy_with_logits(aux, targets, pos_weight=bce_pos, reduction="none")
                if focal_gamma > 0:
                    p_t = torch.sigmoid(aux) * targets + (1 - torch.sigmoid(aux)) * (1 - targets)
                    bce = bce * ((1 - p_t) ** focal_gamma)
                loss += bce.mean()
            loss /= len(aux_logits)

            total_loss += loss.item()
            n_batches += 1
            all_logits.append(final_logits.cpu())
            all_targets.append(targets.cpu())
            for li, aux in enumerate(aux_logits):
                all_aux[li].append(aux.cpu())

    val_loss = total_loss / max(n_batches, 1)
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_aux = [torch.cat(a) for a in all_aux]
    metrics = compute_metrics(all_logits, all_targets)

    return val_loss, metrics, all_logits, all_targets, all_aux


def load_frozen_s1(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ConformerProposer(n_mels=N_MELS, d_model=a.get("d_model", 384),
                              n_layers=a.get("n_layers", 8), n_heads=8,
                              conv_kernel=a.get("conv_kernel", 31),
                              a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_frozen_s2(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ContextProposer(d_model=a.get("d_model", 256),
                            n_gru_layers=a.get("n_gru_layers", 4),
                            b_pred=B_PRED, max_events=C_EVENTS)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser(description="Train S3 Fusion Selector")
    parser.add_argument("dataset")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-enc-layers", type=int, default=4)
    parser.add_argument("--n-dec-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json")) as f:
        manifest = json.load(f)

    run_dir = os.path.join(SCRIPT_DIR, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evals"), exist_ok=True)

    print(f"S3 Fusion Selector")
    print(f"Dataset: {args.dataset}")
    print(f"Run: {args.run_name}")

    config_path = os.path.join(run_dir, "config.json")
    if not args.resume or not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # Load frozen S1 + S2v2
    print(f"Loading frozen S1: {args.s1_checkpoint}")
    s1_model = load_frozen_s1(args.s1_checkpoint, args.device)
    s1_d = sum(p.numel() for p in s1_model.parameters())
    print(f"  S1: {s1_d:,} params (frozen)")

    print(f"Loading frozen S2v2: {args.s2_checkpoint}")
    s2_model = load_frozen_s2(args.s2_checkpoint, args.device)
    s2_d = sum(p.numel() for p in s2_model.parameters())
    print(f"  S2v2: {s2_d:,} params (frozen)")

    # S1's audio feature dimension
    s1_d_model = s1_model.d_model

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

    train_ds = S3Dataset(manifest, ds_dir, train_idx, augment=True, subsample=args.subsample)
    val_ds = S3Dataset(manifest, ds_dir, val_idx, augment=False, subsample=args.subsample)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # S3 model
    s3_model = FusionSelector(
        d_model=args.d_model, d_audio=s1_d_model,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers,
        n_heads=8, a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED,
    ).to(args.device)

    s3_params = sum(p.numel() for p in s3_model.parameters() if p.requires_grad)
    print(f"  S3: {s3_params:,} params (trainable)")
    print(f"  Total system: {s1_d + s2_d + s3_params:,} params")

    optimizer = torch.optim.AdamW(s3_model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            s3_model.load_state_dict(ckpt["model"])
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
        s3_model.train()
        epoch_loss = 0
        n_steps = 0
        ema_loss = None
        ema_f1 = None

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        for batch_idx, (mel, gaps, ratios, s2_mask, evt_off, evt_mask, cond, targets) in enumerate(pbar):
            mel = mel.to(args.device)
            gaps, ratios, s2_mask, cond = gaps.to(args.device), ratios.to(args.device), s2_mask.to(args.device), cond.to(args.device)
            evt_off, evt_mask = evt_off.to(args.device), evt_mask.to(args.device)
            targets = targets.to(args.device)

            # Frozen S1 forward
            with torch.no_grad():
                s1_logits = s1_model(mel)
                s1_conf = torch.sigmoid(s1_logits)
                audio_feat = s1_model.conv(mel).transpose(1, 2)
                audio_feat = s1_model.conv_norm(audio_feat)

                s2_logits = s2_model(gaps, ratios, s2_mask, cond)
                s2_conf = torch.sigmoid(s2_logits)

            # S3 forward
            final_logits, aux_logits = s3_model(audio_feat, s1_conf, s2_conf,
                                                 evt_off, evt_mask, cond)

            # Auxiliary loss at every decoder layer
            loss = 0
            for aux in aux_logits:
                bce = F.binary_cross_entropy_with_logits(aux, targets, pos_weight=bce_pos, reduction="none")
                if focal_gamma > 0:
                    p_t = torch.sigmoid(aux) * targets + (1 - torch.sigmoid(aux)) * (1 - targets)
                    bce = bce * ((1 - p_t) ** focal_gamma)
                loss += bce.mean()
            loss /= len(aux_logits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(s3_model.parameters(), 1.0)
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_steps += 1

            alpha = 0.05
            if ema_loss is None:
                ema_loss = batch_loss
                ema_f1 = 0.0
            ema_loss = ema_loss * (1 - alpha) + batch_loss * alpha

            with torch.no_grad():
                confs = torch.sigmoid(final_logits)
                tgt_bin = (targets >= 0.5).float()
                preds = (confs >= 0.5).float()
                tp = (preds * tgt_bin).sum()
                fp = (preds * (1 - tgt_bin)).sum()
                fn = ((1 - preds) * tgt_bin).sum()
                p = float(tp / (tp + fp + 1e-8))
                r = float(tp / (tp + fn + 1e-8))
                b_f1 = 2 * p * r / (p + r + 1e-8)
                ema_f1 = ema_f1 * (1 - alpha) + b_f1 * alpha

            pbar.set_postfix_str(f"loss={ema_loss:.3f} F1={ema_f1:.3f}")

            # Eval
            if (batch_idx + 1) % steps_per_eval == 0:
                eval_step += 1
                train_loss = epoch_loss / max(n_steps, 1)

                val_loss, val_metrics, val_logits, val_targets, val_aux = validate(
                    s1_model, s2_model, s3_model, val_loader, args.device,
                    focal_gamma, args.pos_weight)

                epoch_frac = epoch + (batch_idx + 1) / len(train_loader)
                best_f1 = max(val_metrics.get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7])
                best_t = max([0.3, 0.4, 0.5, 0.6, 0.7],
                             key=lambda t: val_metrics.get(f"f1_{t:.1f}", 0))
                sep = val_metrics.get("conf_separation", 0)

                # Per-layer F1
                layer_f1s = []
                for aux in val_aux:
                    lm = compute_metrics(aux, val_targets)
                    lf1 = max(lm.get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7])
                    layer_f1s.append(lf1)

                print(f"\n  Eval {eval_step} (ep {epoch_frac:.2f}): "
                      f"loss={train_loss:.4f}/{val_loss:.4f} | "
                      f"F1={best_f1:.3f}@{best_t} sep={sep:.4f} "
                      f"layers=[{', '.join(f'{f:.3f}' for f in layer_f1s)}]")

                save_eval_graphs(val_logits, val_targets, val_aux, val_metrics, eval_step, run_dir)

                entry = {
                    "eval_step": eval_step,
                    "epoch": round(epoch_frac, 4),
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "lr": scheduler.get_last_lr()[0],
                    "val_metrics": {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in val_metrics.items()},
                    "layer_f1s": [round(f, 4) for f in layer_f1s],
                }
                history.append(entry)
                with open(os.path.join(run_dir, "history.json"), "w") as f:
                    json.dump(history, f, indent=2)

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                ckpt_data = {
                    "model": s3_model.state_dict(),
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

                s3_model.train()

        scheduler.step()

    print("\nDone!")
    best_entry = max(history, key=lambda e: max(
        e["val_metrics"].get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7]))
    best_f1 = max(best_entry["val_metrics"].get(f"f1_{t:.1f}", 0) for t in [0.3, 0.4, 0.5, 0.6, 0.7])
    print(f"Best F1: {best_f1:.3f} at eval {best_entry['eval_step']}")


if __name__ == "__main__":
    main()
