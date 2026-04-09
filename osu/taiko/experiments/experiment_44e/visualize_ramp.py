"""Visualize 3-component loss landscape.

Loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE + ramp_scale * |log(pred/target)|^ramp_exp

Component 1: Trapezoid soft CE (existing, unchanged)
Component 2: Hard CE (existing, unchanged)
Component 3: Distance ramp in log-ratio space (NEW)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from detection_train import OnsetLoss

N_CLASSES = 251
OUT_DIR = os.path.dirname(__file__)


def compute_3part_loss(target_bin, ramp_scale=0.0, ramp_exp=1.0, hard_alpha=0.5):
    """Compute all 3 loss components for each prediction."""
    criterion = OnsetLoss(good_pct=0.03, fail_pct=0.20, hard_alpha=hard_alpha,
                          frame_tolerance=2, stop_weight=1.5)
    target = torch.tensor([target_bin], dtype=torch.long)
    n = N_CLASSES - 1  # exclude STOP
    hard_losses = np.zeros(n)
    soft_losses = np.zeros(n)
    ramp_losses = np.zeros(n)
    total_losses = np.zeros(n)

    for pred_bin in range(n):
        logits = torch.full((1, N_CLASSES), -10.0)
        logits[0, pred_bin] = 10.0

        log_probs = F.log_softmax(logits, dim=-1).clamp(min=-100)
        hard_ce = F.cross_entropy(logits, target, reduction='none').item()
        soft_targets = criterion._make_soft_targets(target, N_CLASSES)
        soft_ce = (-(soft_targets * log_probs).sum(dim=-1)).item()

        ramp = 0.0
        if ramp_scale > 0 and target_bin > 0:
            dist = abs(math.log((pred_bin + 1) / (target_bin + 1)))
            ramp = ramp_scale * (dist ** ramp_exp)

        hard_losses[pred_bin] = hard_ce
        soft_losses[pred_bin] = soft_ce
        ramp_losses[pred_bin] = ramp
        total_losses[pred_bin] = hard_alpha * hard_ce + (1 - hard_alpha) * soft_ce + ramp

    return hard_losses, soft_losses, ramp_losses, total_losses


target = 50

# ── Figure 1: Component breakdown for a few configs ──
configs = [
    (0.0, 1.0, "Baseline (no ramp)"),
    (2.0, 1.0, "ramp s=2 e=1"),
    (5.0, 1.0, "ramp s=5 e=1"),
    (2.0, 2.0, "ramp s=2 e=2"),
]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for ax_idx, (rs, re, label) in enumerate(configs):
    ax = axes[ax_idx // 2][ax_idx % 2]
    hard, soft, ramp, total = compute_3part_loss(target, ramp_scale=rs, ramp_exp=re)
    ax.fill_between(range(len(total)), 0, 0.5 * np.array(hard), alpha=0.3, color='#eb4528', label='0.5 * hard_CE')
    ax.fill_between(range(len(total)), 0.5 * np.array(hard), 0.5 * np.array(hard) + 0.5 * np.array(soft), alpha=0.3, color='#4a90d9', label='0.5 * soft_CE')
    if rs > 0:
        ax.fill_between(range(len(total)), 0.5 * np.array(hard) + 0.5 * np.array(soft), total, alpha=0.4, color='#6bc46d', label='ramp')
    ax.plot(range(len(total)), total, color='#333333', linewidth=2, label='Total')
    ax.axvline(target, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_title(label, fontsize=12)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 35)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("Loss")
fig.suptitle(f"3-Component Loss Breakdown (target={target}): hard + soft + ramp", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "loss_3part_breakdown.png"), dpi=150)
plt.close(fig)
print("Saved breakdown")

# ── Figure 2: Scale sweep (exp=1.0) ──
ramp_scales = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
scale_colors = ['#333333', '#4a90d9', '#6bc46d', '#e6a817', '#eb4528', '#c76dba']

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax_i, (xlim, ylim, title) in enumerate([
    ((0, 200), (0, 40), "Full range"),
    ((20, 120), (0, 30), "Zoomed near target"),
]):
    ax = axes[ax_i]
    for rs, color in zip(ramp_scales, scale_colors):
        _, _, _, total = compute_3part_loss(target, ramp_scale=rs, ramp_exp=1.0)
        ax.plot(range(len(total)), total, color=color, linewidth=2, label=f"scale={rs}")
    ax.axvline(target, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("Total Loss")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
fig.suptitle(f"Ramp scale sweep (exp=1.0) at target={target}", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramp_scale_sweep.png"), dpi=150)
plt.close(fig)
print("Saved scale sweep")

# ── Figure 3: Exp sweep (scale=2.0) ──
ramp_exps = [0.5, 1.0, 1.5, 2.0, 3.0]
exp_colors = ['#4a90d9', '#333333', '#6bc46d', '#e6a817', '#eb4528']

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
for ax_i, (xlim, ylim, title) in enumerate([
    ((0, 200), (0, 40), "Full range"),
    ((20, 120), (0, 30), "Zoomed near target"),
]):
    ax = axes[ax_i]
    for re, color in zip(ramp_exps, exp_colors):
        _, _, _, total = compute_3part_loss(target, ramp_scale=2.0, ramp_exp=re)
        ax.plot(range(len(total)), total, color=color, linewidth=2, label=f"exp={re}")
    ax.axvline(target, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_xlabel("Predicted bin")
    ax.set_ylabel("Total Loss")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
fig.suptitle(f"Ramp exp sweep (scale=2.0) at target={target}", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramp_exp_sweep.png"), dpi=150)
plt.close(fig)
print("Saved exp sweep")

# ── Figure 4: Grid scale x exp ──
grid_scales = [0.0, 1.0, 2.0, 5.0]
grid_exps = [0.5, 1.0, 1.5, 2.0]

fig, axes = plt.subplots(len(grid_scales), len(grid_exps), figsize=(20, 18))
for ri, rs in enumerate(grid_scales):
    for ci, re in enumerate(grid_exps):
        ax = axes[ri][ci]
        _, _, _, total = compute_3part_loss(target, ramp_scale=rs, ramp_exp=re)
        ax.plot(range(len(total)), total, color='#333333', linewidth=2)
        ax.axvline(target, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_title(f"scale={rs} exp={re}", fontsize=10)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 40)
        ax.grid(True, alpha=0.2)
        if ri == len(grid_scales) - 1:
            ax.set_xlabel("Predicted bin")
        if ci == 0:
            ax.set_ylabel("Loss")
fig.suptitle(f"Loss grid at target={target} (scale x exp)", fontsize=15, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramp_grid.png"), dpi=150)
plt.close(fig)
print("Saved grid")

# ── Figure 5: Heatmaps ──
heatmap_configs = [
    (0.0, 1.0, "Baseline"),
    (2.0, 1.0, "s=2 e=1"),
    (2.0, 1.5, "s=2 e=1.5"),
    (5.0, 1.0, "s=5 e=1"),
]
max_bin = 200
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for ax_idx, (rs, re, label) in enumerate(heatmap_configs):
    ax = axes[ax_idx]
    heatmap = np.zeros((max_bin, max_bin))
    for t in range(1, max_bin):
        _, _, _, total = compute_3part_loss(t, ramp_scale=rs, ramp_exp=re)
        heatmap[:, t] = total[:max_bin]
    im = ax.imshow(heatmap, origin='lower', aspect='auto', cmap='inferno', vmin=0, vmax=30)
    ax.plot([0, max_bin], [0, max_bin], 'w--', alpha=0.5, linewidth=1)
    ax.set_xlabel("Target bin")
    ax.set_ylabel("Predicted bin")
    ax.set_title(label, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Loss heatmap: 0.5*hard + 0.5*soft + ramp", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramp_heatmaps.png"), dpi=150)
plt.close(fig)
print("Saved heatmaps")

# ── Figure 6: Multiple targets ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
targets_show = [20, 50, 100, 200]
cfgs = [(0.0, 1.0, '#333333', 'no ramp'), (2.0, 1.0, '#4a90d9', 's=2 e=1'),
        (2.0, 1.5, '#e6a817', 's=2 e=1.5'), (5.0, 1.0, '#eb4528', 's=5 e=1')]
for ax_idx, tgt in enumerate(targets_show):
    ax = axes[ax_idx // 2][ax_idx % 2]
    for rs, re, color, label in cfgs:
        _, _, _, total = compute_3part_loss(tgt, ramp_scale=rs, ramp_exp=re)
        ax.plot(range(len(total)), total, color=color, linewidth=2, label=label)
    ax.axvline(tgt, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_title(f"Target = {tgt}", fontsize=13)
    ax.set_xlim(max(0, tgt - 100), min(250, tgt + 100))
    ax.set_ylim(0, 35)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
fig.suptitle("3-component loss at various targets", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ramp_multi_targets.png"), dpi=150)
plt.close(fig)
print("Saved multi targets")
print("All done!")
