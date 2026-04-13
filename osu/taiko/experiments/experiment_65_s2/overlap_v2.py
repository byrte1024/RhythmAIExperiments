"""Overlap V2: Per-bin confidence comparison between S1 and S2.

Both models output per-bin probability maps. Compare them directly:
- Per-bin F1 at various thresholds for each model
- Confidence separation (correct vs incorrect bins)
- Disagreement analysis: when S1 and S2 disagree, who's right?
- Peak detection for S2's soft distributions

Usage:
    cd osu/taiko
    python experiments/experiment_65_s2/overlap_v2.py \
        --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
        --s2-checkpoint runs/s2_experiment_65/checkpoints/best.pt
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, TAIKO_DIR)

from detection_s1_model import ConformerProposer
from detection_s1_train import S1Dataset, A_BINS, B_BINS, B_PRED, N_MELS, MIN_CURSOR_BIN
from detection_s2_model import ContextPredictor
from detection_s2_train import S2Dataset, N_CLASSES, C_EVENTS


def load_s1(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = ConformerProposer(
        n_mels=N_MELS, d_model=args.get("d_model", 384),
        n_layers=args.get("n_layers", 8), n_heads=8,
        conv_kernel=args.get("conv_kernel", 31),
        a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_s2(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = ContextPredictor(
        d_model=args.get("d_model", 256),
        n_gru_layers=args.get("n_gru_layers", 4),
        n_classes=N_CLASSES, max_events=C_EVENTS,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def s2_logits_to_bin_confs(s2_logits, smooth_sigma=2.0):
    """Convert S2's 251-class logits to a B_PRED-length confidence map.

    S2 outputs (B, 251) logits over bins 0-249 + STOP.
    We take softmax, drop STOP, and optionally smooth with gaussian.
    """
    probs = F.softmax(s2_logits, dim=-1)  # (B, 251)
    bin_probs = probs[:, :B_PRED]  # (B, 250) drop STOP

    # Gaussian smooth to spread peaks (S2 produces soft distributions)
    if smooth_sigma > 0:
        k = int(smooth_sigma * 3) * 2 + 1  # kernel size
        x = torch.arange(k, dtype=torch.float32, device=bin_probs.device) - k // 2
        gaussian = torch.exp(-0.5 * (x / smooth_sigma) ** 2)
        gaussian = gaussian / gaussian.sum()
        gaussian = gaussian.view(1, 1, -1)
        # Conv1d smooth
        bp = bin_probs.unsqueeze(1)  # (B, 1, 250)
        bp = F.pad(bp, (k // 2, k // 2), mode="reflect")
        bp = F.conv1d(bp, gaussian)
        bin_probs = bp.squeeze(1)  # (B, 250)

    # Normalize to [0, 1] range per sample
    max_val = bin_probs.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    bin_probs = bin_probs / max_val

    return bin_probs


def compute_bin_metrics(confs, targets_binary, thresholds):
    """Compute per-bin F1/precision/recall at various thresholds."""
    results = {}
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        preds = (confs >= t).float()
        tp = (preds * targets_binary).sum()
        fp = (preds * (1 - targets_binary)).sum()
        fn = ((1 - preds) * targets_binary).sum()
        p = float(tp / (tp + fp + 1e-8))
        r = float(tp / (tp + fn + 1e-8))
        f1 = 2 * p * r / (p + r + 1e-8)
        results[f"f1_{t:.2f}"] = f1
        results[f"precision_{t:.2f}"] = p
        results[f"recall_{t:.2f}"] = r
        results[f"proposals_{t:.2f}"] = float(preds.sum(dim=-1).mean())
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    results["best_f1"] = best_f1
    results["best_threshold"] = best_thresh
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Max val samples to process (0=all)")
    args = parser.parse_args()

    ds_dir = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Val split
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
    val_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id", str(i)) in val_songs]

    # Load models
    print(f"Loading S1: {args.s1_checkpoint}")
    s1_model = load_s1(args.s1_checkpoint, args.device)
    print(f"Loading S2: {args.s2_checkpoint}")
    s2_model = load_s2(args.s2_checkpoint, args.device)

    # Datasets
    s1_ds = S1Dataset(manifest, ds_dir, val_idx, augment=False)
    s2_ds = S2Dataset(manifest, ds_dir, val_idx, augment=False)
    assert len(s1_ds) == len(s2_ds), f"Dataset mismatch: S1={len(s1_ds)} S2={len(s2_ds)}"

    n_samples = len(s1_ds)
    if args.max_samples > 0:
        n_samples = min(n_samples, args.max_samples)
    print(f"Val samples: {n_samples}")

    s1_loader = DataLoader(s1_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    s2_loader = DataLoader(s2_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    # Collect per-bin confidences
    all_s1_confs = []  # (N, 250) sigmoid
    all_s2_confs = []  # (N, 250) normalized softmax
    all_targets = []   # (N, 250) binary

    n_collected = 0
    print("Running inference...")
    with torch.no_grad():
        for (mel, s1_tgt), (gaps, ratios, mask, cond, s2_tgt) in tqdm(
                zip(s1_loader, s2_loader), total=min(len(s1_loader), (n_samples + args.batch_size - 1) // args.batch_size)):

            if n_collected >= n_samples:
                break

            mel = mel.to(args.device)
            gaps, ratios, mask, cond = (
                gaps.to(args.device), ratios.to(args.device),
                mask.to(args.device), cond.to(args.device),
            )

            # S1: per-bin sigmoid
            s1_logits = s1_model(mel)
            s1_conf = torch.sigmoid(s1_logits).cpu()

            # S2: 251-class logits → per-bin confidence
            s2_logits = s2_model(gaps, ratios, mask, cond)
            s2_conf = s2_logits_to_bin_confs(s2_logits.cpu(), smooth_sigma=2.0)

            # Targets from S1 dataset (binary per-bin)
            targets_binary = (s1_tgt >= 0.5).float()

            all_s1_confs.append(s1_conf)
            all_s2_confs.append(s2_conf)
            all_targets.append(targets_binary)
            n_collected += mel.size(0)

    s1_confs = torch.cat(all_s1_confs)[:n_samples]
    s2_confs = torch.cat(all_s2_confs)[:n_samples]
    targets = torch.cat(all_targets)[:n_samples]

    print(f"Collected: {s1_confs.shape[0]} samples, {s1_confs.shape[1]} bins each")
    print(f"Onset bins: {targets.sum():.0f} ({targets.mean():.4f} density)")

    # ═══════════════════════════════════════════════════════════════
    #  Per-model F1 sweeps
    # ═══════════════════════════════════════════════════════════════
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    print(f"\n{'='*70}")
    print("S1 (Audio Conformer) per-bin metrics:")
    s1_metrics = compute_bin_metrics(s1_confs, targets, thresholds)
    print(f"  Best F1: {s1_metrics['best_f1']:.3f} @ {s1_metrics['best_threshold']:.2f}")
    for t in thresholds:
        print(f"    @{t:.2f}: F1={s1_metrics[f'f1_{t:.2f}']:.3f} P={s1_metrics[f'precision_{t:.2f}']:.3f} R={s1_metrics[f'recall_{t:.2f}']:.3f} props={s1_metrics[f'proposals_{t:.2f}']:.1f}")

    print(f"\nS2 (Context GRU) per-bin metrics:")
    s2_metrics = compute_bin_metrics(s2_confs, targets, thresholds)
    print(f"  Best F1: {s2_metrics['best_f1']:.3f} @ {s2_metrics['best_threshold']:.2f}")
    for t in thresholds:
        print(f"    @{t:.2f}: F1={s2_metrics[f'f1_{t:.2f}']:.3f} P={s2_metrics[f'precision_{t:.2f}']:.3f} R={s2_metrics[f'recall_{t:.2f}']:.3f} props={s2_metrics[f'proposals_{t:.2f}']:.1f}")

    # ═══════════════════════════════════════════════════════════════
    #  Confidence separation (correct vs incorrect)
    # ═══════════════════════════════════════════════════════════════
    onset_mask = targets == 1
    non_onset_mask = targets == 0

    print(f"\n{'='*70}")
    print("Confidence at onset vs non-onset bins:")
    s1_onset = float(s1_confs[onset_mask].mean())
    s1_non = float(s1_confs[non_onset_mask].mean())
    s2_onset = float(s2_confs[onset_mask].mean())
    s2_non = float(s2_confs[non_onset_mask].mean())
    print(f"  S1: onset={s1_onset:.4f} non-onset={s1_non:.4f} sep={s1_onset-s1_non:.4f}")
    print(f"  S2: onset={s2_onset:.4f} non-onset={s2_non:.4f} sep={s2_onset-s2_non:.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  Disagreement analysis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Disagreement analysis (per-bin):")

    # Use best threshold for each
    s1_t = s1_metrics["best_threshold"]
    s2_t = s2_metrics["best_threshold"]
    s1_pred = (s1_confs >= s1_t).float()
    s2_pred = (s2_confs >= s2_t).float()

    both_pos = (s1_pred == 1) & (s2_pred == 1)
    s1_only_pos = (s1_pred == 1) & (s2_pred == 0)
    s2_only_pos = (s1_pred == 0) & (s2_pred == 1)
    both_neg = (s1_pred == 0) & (s2_pred == 0)

    for label, mask in [("Both positive", both_pos), ("S1 only", s1_only_pos),
                        ("S2 only", s2_only_pos), ("Both negative", both_neg)]:
        n = mask.sum().item()
        if n > 0:
            correct = (targets[mask] == 1).float().mean().item()
            print(f"  {label:<16}: {n:>10,} bins ({n/(targets.numel())*100:.2f}%)  onset_rate={correct:.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  Confidence-weighted disagreement
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Confidence-weighted disagreement (continuous):")

    # Where S2 is more confident than S1
    s2_stronger = s2_confs > s1_confs  # per bin
    s1_stronger = s1_confs > s2_confs

    for label, mask in [("S2 conf > S1 conf", s2_stronger), ("S1 conf > S2 conf", s1_stronger)]:
        n = mask.sum().item()
        if n > 0:
            onset_rate = targets[mask].mean().item()
            avg_s1 = s1_confs[mask].mean().item()
            avg_s2 = s2_confs[mask].mean().item()
            print(f"  {label}: {n:>10,} bins ({n/(targets.numel())*100:.1f}%)  onset_rate={onset_rate:.4f}  s1_conf={avg_s1:.4f}  s2_conf={avg_s2:.4f}")

    # Bins where they strongly disagree (conf diff > 0.3)
    conf_diff = s2_confs - s1_confs
    for gap in [0.2, 0.3, 0.5]:
        s2_much_higher = conf_diff > gap
        s1_much_higher = conf_diff < -gap
        n_s2h = s2_much_higher.sum().item()
        n_s1h = s1_much_higher.sum().item()
        if n_s2h > 0:
            rate_s2h = targets[s2_much_higher].mean().item()
            print(f"  S2 >> S1 (gap>{gap}): {n_s2h:>8,} bins  onset_rate={rate_s2h:.4f}")
        if n_s1h > 0:
            rate_s1h = targets[s1_much_higher].mean().item()
            print(f"  S1 >> S2 (gap>{gap}): {n_s1h:>8,} bins  onset_rate={rate_s1h:.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  Combined model (simple average, product, max)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Combined model experiments:")

    for name, combined in [
        ("Average (S1+S2)/2", (s1_confs + s2_confs) / 2),
        ("Product sqrt(S1*S2)", torch.sqrt(s1_confs * s2_confs + 1e-10)),
        ("Max(S1, S2)", torch.max(s1_confs, s2_confs)),
        ("S1 * S2^0.5", s1_confs * (s2_confs ** 0.5)),
        ("S2 * S1^0.5", s2_confs * (s1_confs ** 0.5)),
        ("Weighted 0.7*S1 + 0.3*S2", 0.7 * s1_confs + 0.3 * s2_confs),
        ("Weighted 0.3*S1 + 0.7*S2", 0.3 * s1_confs + 0.7 * s2_confs),
    ]:
        m = compute_bin_metrics(combined, targets, thresholds)
        print(f"  {name:<30}: F1={m['best_f1']:.3f} @ {m['best_threshold']:.2f}")

    # ═══════════════════════════════════════════════════════════════
    #  Save results
    # ═══════════════════════════════════════════════════════════════
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "n_samples": int(s1_confs.shape[0]),
        "n_bins_per_sample": B_PRED,
        "s1_metrics": s1_metrics,
        "s2_metrics": s2_metrics,
        "s1_onset_conf": s1_onset,
        "s1_non_onset_conf": s1_non,
        "s2_onset_conf": s2_onset,
        "s2_non_onset_conf": s2_non,
    }

    with open(os.path.join(output_dir, "overlap_v2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ═══════════════════════════════════════════════════════════════
    #  Graphs
    # ═══════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. F1 vs threshold for both models + combined
    fig, ax = plt.subplots(figsize=(12, 7))
    s1_f1s = [s1_metrics[f"f1_{t:.2f}"] for t in thresholds]
    s2_f1s = [s2_metrics[f"f1_{t:.2f}"] for t in thresholds]

    combined_avg = (s1_confs + s2_confs) / 2
    combined_prod = torch.sqrt(s1_confs * s2_confs + 1e-10)
    avg_m = compute_bin_metrics(combined_avg, targets, thresholds)
    prod_m = compute_bin_metrics(combined_prod, targets, thresholds)
    avg_f1s = [avg_m[f"f1_{t:.2f}"] for t in thresholds]
    prod_f1s = [prod_m[f"f1_{t:.2f}"] for t in thresholds]

    ax.plot(thresholds, s1_f1s, "b-o", linewidth=2, label=f"S1 Audio (best={s1_metrics['best_f1']:.3f})")
    ax.plot(thresholds, s2_f1s, "r-o", linewidth=2, label=f"S2 Context (best={s2_metrics['best_f1']:.3f})")
    ax.plot(thresholds, avg_f1s, "g-o", linewidth=2, label=f"Average (best={avg_m['best_f1']:.3f})")
    ax.plot(thresholds, prod_f1s, "m-o", linewidth=2, label=f"Product (best={prod_m['best_f1']:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title("Per-Bin F1: S1 vs S2 vs Combined")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overlap_v2_f1_curves.png"), dpi=150)
    plt.close(fig)

    # 2. Confidence distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_i, (name, confs) in enumerate([("S1 Audio", s1_confs), ("S2 Context", s2_confs)]):
        ax = axes[ax_i]
        onset_c = confs[onset_mask].numpy()
        non_c = confs[non_onset_mask].numpy()
        ax.hist(non_c, bins=100, range=(0, 1), alpha=0.6, color="#4a90d9",
                label=f"Non-onset", density=True)
        ax.hist(onset_c, bins=100, range=(0, 1), alpha=0.6, color="#eb4528",
                label=f"Onset", density=True)
        ax.set_title(f"{name} confidence distribution")
        ax.set_xlabel("Confidence")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overlap_v2_conf_dist.png"), dpi=150)
    plt.close(fig)

    # 3. S1 vs S2 confidence scatter (subsample for visibility)
    n_scatter = min(500000, onset_mask.sum().item())
    onset_indices = onset_mask.nonzero(as_tuple=False)
    perm = torch.randperm(len(onset_indices))[:n_scatter]
    scatter_idx = onset_indices[perm]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    # Onset bins
    ax = axes[0]
    s1_sc = s1_confs[scatter_idx[:, 0], scatter_idx[:, 1]].numpy()
    s2_sc = s2_confs[scatter_idx[:, 0], scatter_idx[:, 1]].numpy()
    ax.scatter(s1_sc, s2_sc, alpha=0.02, s=1, color="#eb4528")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("S1 confidence")
    ax.set_ylabel("S2 confidence")
    ax.set_title(f"Onset bins (n={n_scatter:,})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Non-onset bins (subsample)
    n_neg_scatter = min(500000, non_onset_mask.sum().item())
    neg_indices = non_onset_mask.nonzero(as_tuple=False)
    perm2 = torch.randperm(len(neg_indices))[:n_neg_scatter]
    neg_idx = neg_indices[perm2]
    ax = axes[1]
    s1_neg = s1_confs[neg_idx[:, 0], neg_idx[:, 1]].numpy()
    s2_neg = s2_confs[neg_idx[:, 0], neg_idx[:, 1]].numpy()
    ax.scatter(s1_neg, s2_neg, alpha=0.02, s=1, color="#4a90d9")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("S1 confidence")
    ax.set_ylabel("S2 confidence")
    ax.set_title(f"Non-onset bins (n={n_neg_scatter:,})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("S1 vs S2 Confidence: Onset bins vs Non-onset bins")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overlap_v2_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
