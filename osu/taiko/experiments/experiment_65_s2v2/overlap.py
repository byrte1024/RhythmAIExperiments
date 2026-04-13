"""Overlap: S1 (audio, per-bin) vs S2v2 (context, per-bin).

Both models output (B, 250) per-bin sigmoid. Direct comparison.

Usage:
    cd osu/taiko
    python experiments/experiment_65_s2v2/overlap.py \
        --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
        --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt
"""

import argparse
import json
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
from detection_s1_train import S1Dataset, A_BINS, B_BINS, B_PRED, N_MELS
from detection_s2v2_model import ContextProposer
from detection_s2v2_train import S2v2Dataset, C_EVENTS


def load_s1(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ConformerProposer(n_mels=N_MELS, d_model=a.get("d_model", 384),
                              n_layers=a.get("n_layers", 8), n_heads=8,
                              conv_kernel=a.get("conv_kernel", 31),
                              a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_s2v2(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ContextProposer(d_model=a.get("d_model", 256),
                            n_gru_layers=a.get("n_gru_layers", 4),
                            b_pred=B_PRED, max_events=C_EVENTS)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def compute_f1(confs, targets, thresh):
    preds = (confs >= thresh).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    p = float(tp / (tp + fp + 1e-8))
    r = float(tp / (tp + fn + 1e-8))
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=100000)
    args = parser.parse_args()

    ds_dir = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
    with open(os.path.join(ds_dir, "manifest.json")) as f:
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

    print(f"Loading S1: {args.s1_checkpoint}")
    s1_model = load_s1(args.s1_checkpoint, args.device)
    print(f"Loading S2v2: {args.s2_checkpoint}")
    s2_model = load_s2v2(args.s2_checkpoint, args.device)

    s1_ds = S1Dataset(manifest, ds_dir, val_idx, augment=False)
    s2_ds = S2v2Dataset(manifest, ds_dir, val_idx, augment=False)
    assert len(s1_ds) == len(s2_ds)

    n_samples = min(len(s1_ds), args.max_samples) if args.max_samples > 0 else len(s1_ds)
    print(f"Val samples: {n_samples}")

    s1_loader = DataLoader(s1_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    s2_loader = DataLoader(s2_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    all_s1 = []
    all_s2 = []
    all_tgt = []
    n_collected = 0

    print("Running inference...")
    with torch.no_grad():
        for (mel, s1_tgt), (gaps, ratios, mask, cond, s2_tgt) in tqdm(
                zip(s1_loader, s2_loader), total=min(len(s1_loader), (n_samples + args.batch_size - 1) // args.batch_size)):
            if n_collected >= n_samples:
                break
            mel = mel.to(args.device)
            gaps, ratios, mask, cond = gaps.to(args.device), ratios.to(args.device), mask.to(args.device), cond.to(args.device)

            s1_conf = torch.sigmoid(s1_model(mel)).cpu()
            s2_conf = torch.sigmoid(s2_model(gaps, ratios, mask, cond)).cpu()
            tgt = (s1_tgt >= 0.5).float()

            all_s1.append(s1_conf)
            all_s2.append(s2_conf)
            all_tgt.append(tgt)
            n_collected += mel.size(0)

    s1 = torch.cat(all_s1)[:n_samples]
    s2 = torch.cat(all_s2)[:n_samples]
    tgt = torch.cat(all_tgt)[:n_samples]

    onset_mask = tgt == 1
    non_onset_mask = tgt == 0
    print(f"Collected: {s1.shape[0]} samples x {s1.shape[1]} bins")
    print(f"Onset bins: {onset_mask.sum():,} ({onset_mask.float().mean():.4f})")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    # ── Individual F1 sweeps ──
    print(f"\n{'='*70}")
    print("Individual model F1 sweeps:")
    print(f"{'Thresh':>6} | {'S1 F1':>7} {'S1 P':>6} {'S1 R':>6} | {'S2v2 F1':>7} {'S2v2 P':>6} {'S2v2 R':>6}")
    print("-" * 60)
    s1_best_f1, s1_best_t = 0, 0.5
    s2_best_f1, s2_best_t = 0, 0.5
    for t in thresholds:
        f1_1, p1, r1 = compute_f1(s1, tgt, t)
        f1_2, p2, r2 = compute_f1(s2, tgt, t)
        print(f"  {t:.2f} | {f1_1:.3f} {p1:.3f} {r1:.3f} | {f1_2:.3f} {p2:.3f} {r2:.3f}")
        if f1_1 > s1_best_f1: s1_best_f1, s1_best_t = f1_1, t
        if f1_2 > s2_best_f1: s2_best_f1, s2_best_t = f1_2, t
    print(f"\nS1 best: F1={s1_best_f1:.3f} @ {s1_best_t}")
    print(f"S2v2 best: F1={s2_best_f1:.3f} @ {s2_best_t}")

    # ── Confidence separation ──
    print(f"\n{'='*70}")
    print("Confidence at onset vs non-onset bins:")
    s1_on = float(s1[onset_mask].mean())
    s1_off = float(s1[non_onset_mask].mean())
    s2_on = float(s2[onset_mask].mean())
    s2_off = float(s2[non_onset_mask].mean())
    print(f"  S1:   onset={s1_on:.4f}  non={s1_off:.4f}  sep={s1_on-s1_off:.4f}")
    print(f"  S2v2: onset={s2_on:.4f}  non={s2_off:.4f}  sep={s2_on-s2_off:.4f}")

    # ── Disagreement at best thresholds ──
    print(f"\n{'='*70}")
    print(f"Disagreement (S1 @ {s1_best_t}, S2v2 @ {s2_best_t}):")
    s1_pred = (s1 >= s1_best_t).float()
    s2_pred = (s2 >= s2_best_t).float()

    both_pos = (s1_pred == 1) & (s2_pred == 1)
    s1_only = (s1_pred == 1) & (s2_pred == 0)
    s2_only = (s1_pred == 0) & (s2_pred == 1)
    both_neg = (s1_pred == 0) & (s2_pred == 0)

    total_bins = tgt.numel()
    for label, m in [("Both positive", both_pos), ("S1 only positive", s1_only),
                     ("S2v2 only positive", s2_only), ("Both negative", both_neg)]:
        n = m.sum().item()
        onset_rate = tgt[m].mean().item() if n > 0 else 0
        print(f"  {label:<22}: {n:>10,} ({n/total_bins*100:>5.2f}%)  onset_rate={onset_rate:.4f}")

    # ── Confidence-weighted disagreement ──
    print(f"\n{'='*70}")
    print("Where they disagree most (confidence gap):")
    diff = s2 - s1
    for gap in [0.1, 0.2, 0.3, 0.5]:
        s2_higher = diff > gap
        s1_higher = diff < -gap
        for label, m in [(f"S2v2 >> S1 (gap>{gap})", s2_higher), (f"S1 >> S2v2 (gap>{gap})", s1_higher)]:
            n = m.sum().item()
            if n > 0:
                rate = tgt[m].mean().item()
                print(f"  {label}: {n:>10,} bins  onset_rate={rate:.4f}")

    # ── Combined models ──
    print(f"\n{'='*70}")
    print("Combined model experiments:")
    combos = [
        ("S1 only", s1),
        ("S2v2 only", s2),
        ("Average", (s1 + s2) / 2),
        ("Product sqrt(S1*S2)", torch.sqrt(s1 * s2 + 1e-10)),
        ("Max(S1, S2v2)", torch.max(s1, s2)),
        ("Min(S1, S2v2)", torch.min(s1, s2)),
        ("S1*S2v2^0.5", s1 * s2 ** 0.5),
        ("S2v2*S1^0.5", s2 * s1 ** 0.5),
        ("0.7*S1 + 0.3*S2v2", 0.7 * s1 + 0.3 * s2),
        ("0.5*S1 + 0.5*S2v2", 0.5 * s1 + 0.5 * s2),
        ("0.3*S1 + 0.7*S2v2", 0.3 * s1 + 0.7 * s2),
    ]
    best_combo_f1 = 0
    best_combo_name = ""
    for name, combined in combos:
        best_f1, best_t = 0, 0.5
        for t in thresholds:
            f1, _, _ = compute_f1(combined, tgt, t)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        marker = ""
        if best_f1 > best_combo_f1:
            best_combo_f1 = best_f1
            best_combo_name = name
            marker = " <-- BEST"
        print(f"  {name:<25}: F1={best_f1:.3f} @ {best_t}{marker}")

    print(f"\nBest combination: {best_combo_name} F1={best_combo_f1:.3f}")
    print(f"vs S1 alone: {s1_best_f1:.3f} (+{best_combo_f1-s1_best_f1:.3f})")
    print(f"vs S2v2 alone: {s2_best_f1:.3f} (+{best_combo_f1-s2_best_f1:.3f})")

    # ── Save ──
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "s1_best_f1": s1_best_f1, "s1_best_threshold": s1_best_t,
        "s2v2_best_f1": s2_best_f1, "s2v2_best_threshold": s2_best_t,
        "best_combo": best_combo_name, "best_combo_f1": best_combo_f1,
        "s1_onset_conf": s1_on, "s1_non_conf": s1_off,
        "s2v2_onset_conf": s2_on, "s2v2_non_conf": s2_off,
    }
    with open(os.path.join(output_dir, "overlap_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Graphs ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # F1 curves
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, combined, color, ls in [
        ("S1 Audio", s1, "#4a90d9", "-"),
        ("S2v2 Context", s2, "#eb4528", "-"),
        ("Average", (s1+s2)/2, "#6bc46d", "--"),
        ("Product", torch.sqrt(s1*s2+1e-10), "#c76dba", "--"),
        ("Max", torch.max(s1, s2), "#e6a817", "--"),
    ]:
        f1s = [compute_f1(combined, tgt, t)[0] for t in thresholds]
        best = max(f1s)
        ax.plot(thresholds, f1s, color=color, linestyle=ls, linewidth=2, label=f"{name} (best={best:.3f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title("Per-Bin F1: S1 vs S2v2 vs Combined")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_curves.png"), dpi=150)
    plt.close(fig)

    # Confidence scatter (onset bins)
    n_sc = min(200000, onset_mask.sum().item())
    idx = onset_mask.nonzero(as_tuple=False)
    perm = torch.randperm(len(idx))[:n_sc]
    si = idx[perm]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    ax.scatter(s1[si[:,0], si[:,1]].numpy(), s2[si[:,0], si[:,1]].numpy(), alpha=0.03, s=1, color="#eb4528")
    ax.plot([0,1],[0,1],"k--",alpha=0.5)
    ax.set_xlabel("S1 confidence"); ax.set_ylabel("S2v2 confidence")
    ax.set_title(f"Onset bins (n={n_sc:,})")
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    n_neg = min(200000, non_onset_mask.sum().item())
    nidx = non_onset_mask.nonzero(as_tuple=False)
    perm2 = torch.randperm(len(nidx))[:n_neg]
    ni = nidx[perm2]
    ax = axes[1]
    ax.scatter(s1[ni[:,0], ni[:,1]].numpy(), s2[ni[:,0], ni[:,1]].numpy(), alpha=0.03, s=1, color="#4a90d9")
    ax.plot([0,1],[0,1],"k--",alpha=0.5)
    ax.set_xlabel("S1 confidence"); ax.set_ylabel("S2v2 confidence")
    ax.set_title(f"Non-onset bins (n={n_neg:,})")
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    fig.suptitle("S1 vs S2v2 Confidence")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
