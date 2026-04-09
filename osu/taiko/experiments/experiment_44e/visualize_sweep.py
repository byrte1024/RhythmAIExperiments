"""Sweep hard_alpha x ramp_alpha (ramp_exp=1.0 fixed)."""
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


def compute_loss(target_bin, hard_alpha=0.5, ramp_alpha=0.0, ramp_exp=1.0):
    criterion = OnsetLoss(good_pct=0.03, fail_pct=0.20, hard_alpha=hard_alpha,
                          frame_tolerance=2, stop_weight=1.5)
    target = torch.tensor([target_bin], dtype=torch.long)
    n = N_CLASSES - 1
    losses = np.zeros(n)

    for pred_bin in range(n):
        logits = torch.full((1, N_CLASSES), -10.0)
        logits[0, pred_bin] = 10.0

        log_probs = F.log_softmax(logits, dim=-1).clamp(min=-100)
        hard_ce = F.cross_entropy(logits, target, reduction='none').item()
        soft_targets = criterion._make_soft_targets(target, N_CLASSES)
        soft_ce = (-(soft_targets * log_probs).sum(dim=-1)).item()

        ramp = 0.0
        if ramp_alpha > 0 and target_bin > 0:
            dist = abs(math.log((pred_bin + 1) / (target_bin + 1)))
            ramp = ramp_alpha * (dist ** ramp_exp)

        losses[pred_bin] = hard_alpha * hard_ce + (1 - hard_alpha) * soft_ce + ramp

    return losses


hard_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
ramp_alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
target = 50

# ── Figure 1: Grid of line plots (hard_alpha rows x ramp_alpha cols) ──
fig, axes = plt.subplots(len(hard_alphas), len(ramp_alphas), figsize=(25, 22))

for ri, ha in enumerate(hard_alphas):
    for ci, ra in enumerate(ramp_alphas):
        ax = axes[ri][ci]
        losses = compute_loss(target, hard_alpha=ha, ramp_alpha=ra)
        ax.plot(range(len(losses)), losses, color='#333333', linewidth=2)
        ax.axvline(target, color='green', linestyle='--', alpha=0.7, linewidth=1.5)

        # Mark 25 and 75 for asymmetry check
        l25 = losses[25]
        l75 = losses[75]
        ax.plot(25, l25, 'v', color='blue', markersize=6)
        ax.plot(75, l75, '^', color='red', markersize=6)

        ax.set_title(f"ha={ha} ra={ra}  (25:{l25:.1f} 75:{l75:.1f})", fontsize=8)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 35)
        ax.grid(True, alpha=0.2)
        if ri == len(hard_alphas) - 1:
            ax.set_xlabel("Pred bin", fontsize=8)
        if ci == 0:
            ax.set_ylabel(f"ha={ha}", fontsize=9)
        if ri == 0:
            ax.xaxis.set_label_position('top')
            ax.set_xlabel(f"ra={ra}", fontsize=9)

fig.suptitle(f"Loss Sweep: hard_alpha (rows) x ramp_alpha (cols), target={target}, ramp_exp=1.0",
             fontsize=15, y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sweep_grid.png"), dpi=150)
plt.close(fig)
print("Saved sweep grid")

# ── Figure 2: Heatmap of loss at pred=75 vs pred=25 ratio (over/under asymmetry) ──
ha_fine = np.linspace(0, 1, 20)
ra_fine = np.linspace(0, 5, 20)

# Metric: loss at 2x over / loss at 0.5x under
ratio_map = np.zeros((len(ra_fine), len(ha_fine)))
loss_at_2x = np.zeros((len(ra_fine), len(ha_fine)))
loss_at_target = np.zeros((len(ra_fine), len(ha_fine)))
loss_at_4x = np.zeros((len(ra_fine), len(ha_fine)))

for ri, ra in enumerate(ra_fine):
    for ci, ha in enumerate(ha_fine):
        losses = compute_loss(target, hard_alpha=ha, ramp_alpha=ra)
        l_under = losses[25]   # 0.5x
        l_over = losses[100]   # 2x
        l_4x = losses[200]     # 4x
        ratio_map[ri, ci] = l_over / max(l_under, 0.01)
        loss_at_2x[ri, ci] = l_over
        loss_at_target[ri, ci] = losses[target]
        loss_at_4x[ri, ci] = l_4x

fig, axes = plt.subplots(1, 4, figsize=(24, 5))

titles = [
    "Loss at target (lower = sharper dip)",
    "Loss at 2x over (pred=100)",
    "Loss at 4x over (pred=200)",
    "Ratio: loss(2x) / loss(0.5x)",
]
data = [loss_at_target, loss_at_2x, loss_at_4x, ratio_map]
cmaps = ['viridis_r', 'inferno', 'inferno', 'RdBu_r']
vmins = [0, 15, 15, 0.8]
vmaxs = [20, 30, 40, 1.2]

for ax_i, (ax, title, d, cmap, vmin, vmax) in enumerate(
        zip(axes, titles, data, cmaps, vmins, vmaxs)):
    im = ax.imshow(d, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, 1, 0, 5])
    ax.set_xlabel("hard_alpha")
    ax.set_ylabel("ramp_alpha")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f"Loss landscape metrics across hard_alpha x ramp_alpha (target={target})", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sweep_metrics.png"), dpi=150)
plt.close(fig)
print("Saved sweep metrics")

# ── Figure 3: Overlay - fix ramp_alpha, compare hard_alphas ──
fig, axes = plt.subplots(1, len(ramp_alphas), figsize=(25, 5))
ha_colors = ['#e6a817', '#6bc46d', '#333333', '#4a90d9', '#eb4528']

for ci, ra in enumerate(ramp_alphas):
    ax = axes[ci]
    for ha, color in zip(hard_alphas, ha_colors):
        losses = compute_loss(target, hard_alpha=ha, ramp_alpha=ra)
        ax.plot(range(len(losses)), losses, color=color, linewidth=2, label=f"ha={ha}")
    ax.axvline(target, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_title(f"ramp_alpha={ra}", fontsize=11)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 35)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Predicted bin")
    if ci == 0:
        ax.set_ylabel("Loss")

fig.suptitle(f"Hard alpha comparison at each ramp level (target={target})", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sweep_ha_overlay.png"), dpi=150)
plt.close(fig)
print("Saved ha overlay")

# ── Figure 4: Overlay - fix hard_alpha, compare ramp_alphas ──
fig, axes = plt.subplots(1, len(hard_alphas), figsize=(25, 5))
ra_colors = ['#333333', '#4a90d9', '#6bc46d', '#e6a817', '#eb4528']

for ci, ha in enumerate(hard_alphas):
    ax = axes[ci]
    for ra, color in zip(ramp_alphas, ra_colors):
        losses = compute_loss(target, hard_alpha=ha, ramp_alpha=ra)
        ax.plot(range(len(losses)), losses, color=color, linewidth=2, label=f"ra={ra}")
    ax.axvline(target, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_title(f"hard_alpha={ha}", fontsize=11)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 35)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Predicted bin")
    if ci == 0:
        ax.set_ylabel("Loss")

fig.suptitle(f"Ramp alpha comparison at each hard_alpha level (target={target})", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sweep_ra_overlay.png"), dpi=150)
plt.close(fig)
print("Saved ra overlay")

print("All done!")
