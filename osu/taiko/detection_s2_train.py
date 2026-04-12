"""Training script for S2 Context Predictor.

Trains a context-only model to predict the next onset from gap sequences.
No audio input. Separate checkpoint from the main detection model.

Usage:
    cd osu/taiko
    python detection_s2_train.py taiko_v2 --run-name s2_experiment_65
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from detection_s2_model import ContextPredictor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Window config (must match main training)
B_PRED = 250
N_CLASSES = 251
C_EVENTS = 128
MIN_CURSOR_BIN = 6000


# ═══════════════════════════════════════════════════════════════
#  Loss (reuse OnsetLoss from main training)
# ═══════════════════════════════════════════════════════════════

from detection_train import OnsetLoss


# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════

class S2Dataset(Dataset):
    """Yields (gap_sequence, ratio_sequence, event_mask, conditioning, target) samples.

    No mel spectrogram — pure event data + density conditioning.
    """

    def __init__(self, manifest, ds_dir, chart_indices, augment=False, subsample=1,
                 density_jitter_rate=0.20, density_jitter_pct=0.05):
        self.charts = [manifest["charts"][i] for i in chart_indices]
        self.augment = augment
        self._djr = density_jitter_rate
        self._djp = density_jitter_pct

        self.events = []
        evt_dir = os.path.join(ds_dir, "events")
        for chart in self.charts:
            evt = np.load(os.path.join(evt_dir, chart["event_file"]))
            self.events.append(evt)

        # Build sample index (same as main training)
        self.samples = []
        for ci, evt in enumerate(self.events):
            for ei in range(len(evt)):
                cursor = max(0, int(evt[0]) - B_PRED) if ei == 0 else int(evt[ei - 1])
                if cursor >= MIN_CURSOR_BIN:
                    self.samples.append((ci, ei))
            if len(evt) > 0 and int(evt[-1]) >= MIN_CURSOR_BIN:
                self.samples.append((ci, len(evt)))

        if subsample > 1:
            self.samples = self.samples[::subsample]

        # Class distribution for balanced sampling
        self.class_counts = np.zeros(N_CLASSES, dtype=np.int64)
        for ci, ei in self.samples:
            self.class_counts[self._get_target(ci, ei)] += 1

    def _get_target(self, ci, ei):
        evt = self.events[ci]
        if ei == 0:
            cursor = max(0, int(evt[0]) - B_PRED) if len(evt) > 0 else 0
        else:
            cursor = int(evt[ei - 1])
        if ei < len(evt):
            offset = max(0, int(evt[ei]) - cursor)
            return N_CLASSES - 1 if offset >= B_PRED else offset
        return N_CLASSES - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, ei = self.samples[idx]
        chart = self.charts[ci]
        evt = self.events[ci]

        if ei == 0:
            cursor = max(0, evt[0] - B_PRED) if len(evt) > 0 else 0
        else:
            cursor = int(evt[ei - 1])

        # Target
        if ei < len(evt):
            offset = max(0, int(evt[ei]) - cursor)
            target = N_CLASSES - 1 if offset >= B_PRED else offset
        else:
            target = N_CLASSES - 1

        # Past events
        if ei > 0:
            past_start = max(0, ei - C_EVENTS)
            past_bins = evt[past_start:ei].astype(np.int64).copy()
        else:
            past_bins = np.array([], dtype=np.int64)

        # Augmentation (very light)
        if self.augment and len(past_bins) > 0:
            # ±1 bin jitter
            jitter = np.random.randint(-1, 2, size=len(past_bins))
            past_bins = past_bins + jitter
            past_bins = np.sort(past_bins)  # maintain order

            # 2% truncation
            if random.random() < 0.02 and len(past_bins) > 32:
                keep = random.randint(32, len(past_bins))
                past_bins = past_bins[-keep:]

        # Compute gaps and ratios
        n_past = len(past_bins)
        gaps = np.zeros(C_EVENTS, dtype=np.float32)
        ratios = np.zeros(C_EVENTS, dtype=np.float32)
        mask = np.ones(C_EVENTS, dtype=bool)  # True = padding

        if n_past > 0:
            # Gaps: distance between consecutive events
            raw_gaps = np.zeros(n_past, dtype=np.float32)
            if n_past >= 2:
                raw_gaps[1:] = np.diff(past_bins).astype(np.float32)
                raw_gaps[0] = raw_gaps[1] if n_past >= 2 else 30.0  # fallback for first
            else:
                raw_gaps[0] = 30.0  # single event, use reasonable default

            raw_gaps = np.maximum(raw_gaps, 1.0)  # clamp to avoid 0

            # Ratios: gap[i] / gap[i-1]
            raw_ratios = np.ones(n_past, dtype=np.float32)
            if n_past >= 2:
                for i in range(1, n_past):
                    raw_ratios[i] = np.clip(raw_gaps[i] / max(raw_gaps[i - 1], 1.0), 0.1, 10.0)

            # Place in padded arrays (right-aligned, most recent last)
            start = C_EVENTS - n_past
            gaps[start:] = raw_gaps
            ratios[start:] = raw_ratios
            mask[start:] = False

        # Conditioning
        cond = np.array([
            chart.get("density_mean", 4.0),
            chart.get("density_peak", 8),
            chart.get("density_std", 1.5),
        ], dtype=np.float32)

        if self.augment and random.random() < self._djr:
            jitter = 1.0 + np.random.uniform(-self._djp, self._djp, size=3).astype(np.float32)
            cond = cond * jitter

        return (
            torch.from_numpy(gaps),
            torch.from_numpy(ratios),
            torch.from_numpy(mask),
            torch.from_numpy(cond),
            torch.tensor(target, dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(targets, preds):
    """Compute HIT/MISS/accuracy metrics."""
    stop = N_CLASSES - 1
    ns = targets != stop
    m = {}

    m["accuracy"] = float((preds == targets).float().mean())
    m["stop_f1"] = 0.0

    # Stop metrics
    is_stop_tgt = targets == stop
    is_stop_pred = preds == stop
    tp = (is_stop_tgt & is_stop_pred).sum().float()
    fp = (~is_stop_tgt & is_stop_pred).sum().float()
    fn = (is_stop_tgt & ~is_stop_pred).sum().float()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    m["stop_f1"] = float(2 * prec * rec / (prec + rec + 1e-8))
    m["stop_precision"] = float(prec)
    m["stop_recall"] = float(rec)
    m["stop_pred_rate"] = float(is_stop_pred.float().mean())
    m["stop_target_rate"] = float(is_stop_tgt.float().mean())

    if ns.sum() == 0:
        m["hit_rate"] = 0.0
        m["miss_rate"] = 1.0
        return m

    t_ns = targets[ns].float()
    p_ns = preds[ns].float()

    # Ratio-based
    pct_err = torch.abs(p_ns / (t_ns + 1) - 1)
    frame_err = torch.abs(p_ns - t_ns)

    hit = (pct_err <= 0.03) | (frame_err <= 1)
    miss = pct_err > 0.20

    m["hit_rate"] = float(hit.float().mean())
    m["miss_rate"] = float(miss.float().mean())
    m["good_rate"] = float(((pct_err <= 0.10) | (frame_err <= 2)).float().mean())
    m["exact_match"] = float((frame_err == 0).float().mean())
    m["frame_error_mean"] = float(frame_err.mean())
    m["frame_error_median"] = float(frame_err.median())
    m["unique_preds"] = int(p_ns.unique().numel())

    # HIT by streak length
    # (computed in eval, not here — needs access to raw gap sequences)

    return m


# ═══════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []
    all_gaps = []       # last gap before each prediction
    all_ratios = []     # last ratio before each prediction
    all_streaks = []    # streak length at each prediction

    with torch.no_grad():
        for gaps, ratios, mask, cond, target in val_loader:
            gaps, ratios, mask, cond, target = (
                gaps.to(device), ratios.to(device), mask.to(device),
                cond.to(device), target.to(device),
            )
            logits = model(gaps, ratios, mask, cond)
            loss = criterion(logits, target)
            total_loss += loss.item()
            n_batches += 1

            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(target.cpu())

            # Collect context info for analysis
            g = gaps.cpu().numpy()
            r = ratios.cpu().numpy()
            m_np = mask.cpu().numpy()
            for b in range(g.shape[0]):
                valid = ~m_np[b]
                if valid.sum() >= 2:
                    valid_gaps = g[b][valid]
                    valid_ratios = r[b][valid]
                    all_gaps.append(float(valid_gaps[-1]))
                    all_ratios.append(float(valid_ratios[-1]))
                    # Streak: count consecutive ~1.0x ratios from the end
                    streak = 0
                    for ri in range(len(valid_ratios) - 1, -1, -1):
                        if abs(valid_ratios[ri] - 1.0) <= 0.05:
                            streak += 1
                        else:
                            break
                    all_streaks.append(streak)
                else:
                    all_gaps.append(0.0)
                    all_ratios.append(1.0)
                    all_streaks.append(0)

    val_loss = total_loss / max(n_batches, 1)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(targets, preds)

    extra = {
        "last_gaps": np.array(all_gaps, dtype=np.float32),
        "last_ratios": np.array(all_ratios, dtype=np.float32),
        "streaks": np.array(all_streaks, dtype=np.int32),
    }

    return val_loss, metrics, preds, targets, extra


# ═══════════════════════════════════════════════════════════════
#  Graphs and Analysis
# ═══════════════════════════════════════════════════════════════

def save_eval_graphs(targets, preds, metrics, extra, eval_step, run_dir):
    """Generate S2-specific eval graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from matplotlib.colors import LogNorm

    eval_dir = os.path.join(run_dir, "evals")
    os.makedirs(eval_dir, exist_ok=True)
    prefix = os.path.join(eval_dir, f"eval_{eval_step:03d}")

    stop = N_CLASSES - 1
    ns = targets < stop
    t_ns = targets[ns].numpy()
    p_ns = preds[ns].numpy()
    max_bin = B_PRED

    if len(t_ns) == 0:
        return

    # ── 1. Prediction distribution ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].hist(t_ns, bins=250, range=(0, max_bin), color="#4a90d9", alpha=0.8)
    axes[0].set_title(f"S2 Eval {eval_step}: Target Distribution (non-STOP)")
    axes[0].set_xlabel("Gap bin")
    axes[1].hist(p_ns, bins=250, range=(0, max_bin), color="#e8834a", alpha=0.8)
    axes[1].set_title(f"S2 Eval {eval_step}: Predicted Distribution - {len(np.unique(p_ns))} unique")
    axes[1].set_xlabel("Gap bin")
    fig.tight_layout()
    fig.savefig(f"{prefix}_pred_dist.png", dpi=120)
    plt.close(fig)

    # ── 2. Scatter: target vs predicted ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(t_ns, p_ns, alpha=0.02, s=1, color="#4a90d9")
    ax.plot([0, max_bin], [0, max_bin], "r--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Target gap bin")
    ax.set_ylabel("Predicted gap bin")
    ax.set_title(f"S2 Eval {eval_step}: Target vs Predicted (context only)")
    ax.set_xlim(0, max_bin)
    ax.set_ylim(0, max_bin)
    fig.tight_layout()
    fig.savefig(f"{prefix}_scatter.png", dpi=120)
    plt.close(fig)

    # ── 3. Heatmap: target vs predicted ──
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    h, _, _ = np.histogram2d(t_ns, p_ns, bins=250, range=[[0, max_bin], [0, max_bin]])
    h = gaussian_filter(h.astype(np.float64), sigma=1.0)
    h[h < 0.5] = np.nan
    ax.imshow(h.T, origin="lower", aspect="auto", extent=[0, max_bin, 0, max_bin],
              norm=LogNorm(vmin=1), cmap="viridis")
    ax.plot([0, max_bin], [0, max_bin], "r--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Target gap bin", color="white")
    ax.set_ylabel("Predicted gap bin", color="white")
    ax.set_title(f"S2 Eval {eval_step}: Prediction Density (context only)", color="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(f"{prefix}_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 4. HIT by streak length ──
    streaks = extra["streaks"]
    pct_err = np.abs(p_ns.astype(np.float64) / (t_ns.astype(np.float64) + 1) - 1)
    frame_err = np.abs(p_ns.astype(np.float64) - t_ns.astype(np.float64))
    hit = (pct_err <= 0.03) | (frame_err <= 1)

    # Only non-STOP samples have streak data matching
    streaks_ns = streaks[ns.numpy()]

    streak_bins = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 999, "11+")]
    streak_hits = []
    streak_counts = []
    streak_labels = []
    for lo, hi, label in streak_bins:
        mask_s = (streaks_ns >= lo) & (streaks_ns <= hi)
        if mask_s.sum() > 0:
            streak_hits.append(float(hit[mask_s].mean()))
            streak_counts.append(int(mask_s.sum()))
        else:
            streak_hits.append(0.0)
            streak_counts.append(0)
        streak_labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(streak_labels, [h * 100 for h in streak_hits], color="#6bc46d")
    for bar, count in zip(bars, streak_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={count}", ha="center", fontsize=9)
    ax.set_xlabel("Streak length (consecutive ~1.0x ratios)")
    ax.set_ylabel("HIT%")
    ax.set_title(f"S2 Eval {eval_step}: HIT by Streak Length")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_streak_hit.png", dpi=120)
    plt.close(fig)

    # ── 5. HIT by ratio bucket ──
    last_ratios = extra["last_ratios"][ns.numpy()]
    ratio_buckets = [
        (0.0, 0.6, "~0.5x (double)"),
        (0.6, 0.85, "~0.67x"),
        (0.85, 1.15, "~1.0x (repeat)"),
        (1.15, 1.6, "~1.33x"),
        (1.6, 2.5, "~2.0x (halve)"),
        (2.5, 20.0, ">2.5x"),
    ]
    ratio_hits = []
    ratio_counts = []
    ratio_labels = []
    for lo, hi, label in ratio_buckets:
        mask_r = (last_ratios >= lo) & (last_ratios < hi)
        if mask_r.sum() > 0:
            ratio_hits.append(float(hit[mask_r].mean()))
            ratio_counts.append(int(mask_r.sum()))
        else:
            ratio_hits.append(0.0)
            ratio_counts.append(0)
        ratio_labels.append(label)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(ratio_labels, [h * 100 for h in ratio_hits], color="#4a90d9")
    for bar, count in zip(bars, ratio_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"n={count}", ha="center", fontsize=9)
    ax.set_xlabel("Last gap ratio bucket")
    ax.set_ylabel("HIT%")
    ax.set_title(f"S2 Eval {eval_step}: HIT by Last Ratio")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(f"{prefix}_ratio_hit.png", dpi=120)
    plt.close(fig)

    # ── 6. Metronome analysis ──
    # "Metronome" = target is within 5% of last gap (continue the pattern)
    last_gaps = extra["last_gaps"][ns.numpy()]
    is_metro = np.abs(t_ns.astype(np.float64) / (last_gaps + 1) - 1) <= 0.05
    is_anti = ~is_metro

    metro_hit = float(hit[is_metro].mean()) if is_metro.sum() > 0 else 0
    anti_hit = float(hit[is_anti].mean()) if is_anti.sum() > 0 else 0
    metro_n = int(is_metro.sum())
    anti_n = int(is_anti.sum())

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(["Metronome\n(continue pattern)", "Anti-metronome\n(break pattern)"],
                  [metro_hit * 100, anti_hit * 100],
                  color=["#6bc46d", "#eb4528"])
    ax.text(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height() + 0.5,
            f"n={metro_n} ({metro_n/(metro_n+anti_n)*100:.0f}%)", ha="center", fontsize=10)
    ax.text(bars[1].get_x() + bars[1].get_width() / 2, bars[1].get_height() + 0.5,
            f"n={anti_n} ({anti_n/(metro_n+anti_n)*100:.0f}%)", ha="center", fontsize=10)
    ax.set_ylabel("HIT%")
    ax.set_title(f"S2 Eval {eval_step}: Metronome vs Anti-Metronome")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_metronome.png", dpi=120)
    plt.close(fig)

    # Save analysis metrics
    analysis = {
        "streak_hit": dict(zip(streak_labels, streak_hits)),
        "streak_counts": dict(zip(streak_labels, streak_counts)),
        "ratio_hit": dict(zip(ratio_labels, ratio_hits)),
        "ratio_counts": dict(zip(ratio_labels, ratio_counts)),
        "metronome_hit": metro_hit,
        "metronome_n": metro_n,
        "anti_metronome_hit": anti_hit,
        "anti_metronome_n": anti_n,
        "metronome_pct": metro_n / max(metro_n + anti_n, 1),
    }

    with open(f"{prefix}_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"  Graphs saved. Metro HIT={metro_hit:.1%} ({metro_n}), Anti={anti_hit:.1%} ({anti_n})")


def main():
    parser = argparse.ArgumentParser(description="Train S2 Context Predictor")
    parser.add_argument("dataset", help="Dataset name (e.g. taiko_v2)")
    parser.add_argument("--run-name", required=True, help="Run directory name")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-gru-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--balanced", action="store_true", default=True)
    parser.add_argument("--ramp-alpha", type=float, default=2.5)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Run directory
    run_dir = os.path.join(SCRIPT_DIR, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evals"), exist_ok=True)

    print(f"S2 Context Predictor")
    print(f"Dataset: {args.dataset} ({manifest['total_charts']} charts)")
    print(f"Run: {args.run_name}")

    # Save config
    config_path = os.path.join(run_dir, "config.json")
    if not args.resume or not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # Train/val split (same as main training)
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

    train_ds = S2Dataset(manifest, ds_dir, train_idx, augment=True, subsample=args.subsample)
    val_ds = S2Dataset(manifest, ds_dir, val_idx, augment=False, subsample=args.subsample)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Balanced sampling
    sampler = None
    if args.balanced:
        counts = train_ds.class_counts.astype(np.float64)
        counts[counts == 0] = 1
        weights = 1.0 / np.sqrt(counts)
        sample_weights = np.array([weights[train_ds._get_target(ci, ei)] for ci, ei in train_ds.samples])
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Model
    model = ContextPredictor(
        d_model=args.d_model,
        n_gru_layers=args.n_gru_layers,
        n_classes=N_CLASSES,
        max_events=C_EVENTS,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Loss
    criterion = OnsetLoss(
        hard_alpha=0.5, good_pct=0.03, fail_pct=0.20,
        frame_tolerance=2, stop_weight=1.5,
        ramp_alpha=args.ramp_alpha,
    ).to(args.device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

    # Training loop
    steps_per_eval = len(train_loader) // args.evals_per_epoch
    if steps_per_eval < 1:
        steps_per_eval = 1

    print(f"Steps/epoch: {len(train_loader)}, evals/epoch: {args.evals_per_epoch}, steps/eval: {steps_per_eval}")
    print()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        n_steps = 0
        main._ema_loss = None  # reset EMA each epoch

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

        for batch_idx, (gaps, ratios, mask, cond, target) in enumerate(pbar):
            gaps, ratios, mask, cond, target = (
                gaps.to(args.device), ratios.to(args.device), mask.to(args.device),
                cond.to(args.device), target.to(args.device),
            )

            logits = model(gaps, ratios, mask, cond)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_steps += 1

            # Running averages
            if main._ema_loss is None:
                main._ema_loss = batch_loss
                main._ema_hit = 0.0
                main._ema_miss = 0.0
            alpha = 0.05
            main._ema_loss = main._ema_loss * (1 - alpha) + batch_loss * alpha

            # Quick batch metrics
            pred = logits.argmax(dim=-1)
            ns = target != N_CLASSES - 1
            if ns.sum() > 0:
                t_ns = target[ns].float()
                p_ns = pred[ns].float()
                pct_err = torch.abs(p_ns / (t_ns + 1) - 1)
                frame_err = torch.abs(p_ns - t_ns)
                b_hit = ((pct_err <= 0.03) | (frame_err <= 1)).float().mean().item()
                b_miss = (pct_err > 0.20).float().mean().item()
                main._ema_hit = main._ema_hit * (1 - alpha) + b_hit * alpha
                main._ema_miss = main._ema_miss * (1 - alpha) + b_miss * alpha
                pbar.set_postfix_str(
                    f"loss={main._ema_loss:.3f} HIT={main._ema_hit:.1%} MISS={main._ema_miss:.1%}")

            # Eval checkpoint
            if (batch_idx + 1) % steps_per_eval == 0:
                eval_step += 1
                train_loss = epoch_loss / max(n_steps, 1)

                val_loss, val_metrics, val_preds, val_targets, val_extra = validate(
                    model, val_loader, criterion, args.device)

                epoch_frac = epoch + (batch_idx + 1) / len(train_loader)

                hit = val_metrics.get("hit_rate", 0)
                miss = val_metrics.get("miss_rate", 1)
                acc = val_metrics.get("accuracy", 0)
                sf1 = val_metrics.get("stop_f1", 0)

                print(f"\n  Eval {eval_step} (ep {epoch_frac:.2f}): "
                      f"loss={train_loss:.4f}/{val_loss:.4f} | "
                      f"HIT={hit:.1%} MISS={miss:.1%} acc={acc:.1%} "
                      f"StpF1={sf1:.3f} uniq={val_metrics.get('unique_preds', 0)}")

                # Graphs and analysis
                save_eval_graphs(val_targets, val_preds, val_metrics, val_extra,
                                 eval_step, run_dir)

                # Save history
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

                # Save checkpoint
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
    best_entry = max(history, key=lambda e: e["val_metrics"].get("hit_rate", 0))
    print(f"Best HIT: {best_entry['val_metrics']['hit_rate']:.1%} at eval {best_entry['eval_step']}")


if __name__ == "__main__":
    main()
