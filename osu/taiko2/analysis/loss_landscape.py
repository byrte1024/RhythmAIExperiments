"""Visualize candidate loss surfaces.

Each heatmap: x = target bin `t`, y = predicted bin `p`, color = loss
value when the model "predicts" `p` for target `t`. Prediction is
modelled as a Gaussian distribution `P_i ∝ exp(-0.5·((i-p)/σ)²)` so
distribution-based losses (CE, EMD) are well-defined even at
p = exact-bin (no log(0) singularities).

Each subplot uses its own color scale (auto-normalized) — losses have
wildly different magnitudes; the SHAPE of the surface is what we
care about, not absolute numbers. Specifically we look for:

  - Diagonal sharpness: how sharply does the loss minimize at p = t?
  - Octave / triplet structure: are there extra-bright bands at
    p = 2t, p = t/2, p = 3t, p = t/3 that PUNISH ridge-style errors,
    or extra-dark bands that REWARD them?
  - Symmetry in log-ratio space: does the loss treat p = 2t and
    p = t/2 with equal magnitude (perception-correct) or asymmetric
    (perception-wrong)?
  - Entropy-floor signature: does the loss have a "plateau" that the
    model can sit at without paying full price for being away from
    the diagonal?

Usage:

    osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.analysis.loss_landscape
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────── distribution helpers ────────────────────

def gaussian_pred(p: int, n_bins: int, sigma: float) -> np.ndarray:
    """Sharp-but-finite predicted distribution, Gaussian at p."""
    bins = np.arange(n_bins, dtype=np.float64)
    w = np.exp(-0.5 * ((bins - p) / sigma) ** 2)
    return w / w.sum()


def trapezoid_soft_target(t: int, n_bins: int) -> np.ndarray:
    """#002's trapezoid soft target — log-ratio plateau + ±2-frame floor."""
    bins = np.arange(n_bins, dtype=np.float64)
    log_good = np.log(1.03)
    log_fail = np.log(1.20)
    abs_log_ratio = np.abs(np.log((bins + 1) / (t + 1)))
    ratio_w = np.clip(
        (log_fail - abs_log_ratio) / (log_fail - log_good), 0.0, 1.0,
    )
    frame_dist = np.abs(bins - t)
    frame_w = np.clip((2 + 1 - frame_dist) / (2 + 1), 0.0, 1.0)
    w = np.maximum(ratio_w, frame_w)
    return w / w.sum()


def gaussian_soft_target(t: int, n_bins: int, sigma: float = 2.0) -> np.ndarray:
    """#005's Gaussian soft target."""
    bins = np.arange(n_bins, dtype=np.float64)
    w = np.exp(-0.5 * ((bins - t) / sigma) ** 2)
    return w / w.sum()


# ─────────────────────────── loss functions ──────────────────────────

def loss_l1(t: int, p: int, n_bins: int) -> float:
    return float(abs(t - p))


def loss_l2(t: int, p: int, n_bins: int) -> float:
    return float((t - p) ** 2)


def loss_log_l1(t: int, p: int, n_bins: int) -> float:
    return float(abs(np.log((p + 1) / (t + 1))))


def loss_log_l2(t: int, p: int, n_bins: int) -> float:
    return float(np.log((p + 1) / (t + 1)) ** 2)


def loss_trapezoid_soft_ce(t: int, p: int, n_bins: int, sigma: float = 0.5) -> float:
    """#002's loss: soft_CE term only (excluding hard CE mix)."""
    P = gaussian_pred(p, n_bins, sigma)
    soft_t = trapezoid_soft_target(t, n_bins)
    return float(-(soft_t * np.log(P + 1e-12)).sum())


def loss_gaussian_soft_ce(t: int, p: int, n_bins: int, sigma_pred: float = 0.5,
                          sigma_target: float = 2.0) -> float:
    """#005's loss: softmax CE with a Gaussian soft target."""
    P = gaussian_pred(p, n_bins, sigma_pred)
    soft_t = gaussian_soft_target(t, n_bins, sigma=sigma_target)
    return float(-(soft_t * np.log(P + 1e-12)).sum())


def loss_hard_ce(t: int, p: int, n_bins: int, sigma: float = 0.5) -> float:
    """Cross-entropy with a hard (one-hot) target — the dominant term
    in #002's mix."""
    P = gaussian_pred(p, n_bins, sigma)
    return float(-np.log(P[t] + 1e-12))


def loss_emd_l1(t: int, p: int, n_bins: int, sigma: float = 0.5) -> float:
    """Earth-mover distance with linear bin metric."""
    P = gaussian_pred(p, n_bins, sigma)
    bins = np.arange(n_bins, dtype=np.float64)
    return float((P * np.abs(bins - t)).sum())


def loss_emd_log_l1(t: int, p: int, n_bins: int, sigma: float = 0.5) -> float:
    """Earth-mover with log-ratio metric — the perception-correct
    candidate. Punishes mass at any bin proportionally to log-distance
    from the target; octave (2t / t/2) costs ≈ log 2 = 0.69 each
    direction."""
    P = gaussian_pred(p, n_bins, sigma)
    bins = np.arange(n_bins, dtype=np.float64)
    log_dist = np.abs(np.log((bins + 1) / (t + 1)))
    return float((P * log_dist).sum())


def loss_emd_log_l2(t: int, p: int, n_bins: int, sigma: float = 0.5) -> float:
    """Quadratic log-ratio EMD — accelerates large log-distance errors."""
    P = gaussian_pred(p, n_bins, sigma)
    bins = np.arange(n_bins, dtype=np.float64)
    log_dist = np.log((bins + 1) / (t + 1)) ** 2
    return float((P * log_dist).sum())


def loss_octave_penalty(t: int, p: int, n_bins: int, sigma: float = 0.5,
                        lam: float = 5.0) -> float:
    """Hard-CE plus a direct penalty for mass at octave / triplet
    multiples of the target."""
    P = gaussian_pred(p, n_bins, sigma)
    base = -np.log(P[t] + 1e-12)
    octave_idxs = []
    for ratio in (0.5, 2.0, 1/3, 3.0):
        idx = int(round(ratio * t))
        if 0 <= idx < n_bins:
            octave_idxs.append(idx)
    octave_mass = sum(P[i] for i in octave_idxs)
    return float(base + lam * octave_mass)


# ─────────────────────────── plot driver ─────────────────────────────

LOSSES = [
    ("L1\n|t - p|",                            loss_l1),
    ("L2\n(t - p)^2",                          loss_l2),
    ("log-L1 (perception)\n|log((p+1)/(t+1))|", loss_log_l1),
    ("log-L2 (perception^2)\n(log((p+1)/(t+1)))^2", loss_log_l2),
    ("Hard CE\n-log P_t",                      loss_hard_ce),
    ("Trapezoid soft CE (#002)",               loss_trapezoid_soft_ce),
    ("Gaussian soft CE (#005)\nsigma_target=2.0", loss_gaussian_soft_ce),
    ("EMD linear\nsum P_i |i - t|",            loss_emd_l1),
    ("EMD log-ratio (perception)\nsum P_i |log((i+1)/(t+1))|", loss_emd_log_l1),
    ("EMD log-ratio squared\nsum P_i (log((i+1)/(t+1)))^2", loss_emd_log_l2),
    ("Hard CE + octave penalty\nbase + 5 * (P_{2t} + P_{t/2} + P_{3t} + P_{t/3})",
     loss_octave_penalty),
]


def compute_grid(loss_fn, n_bins: int) -> np.ndarray:
    """Return an (n_bins, n_bins) array where row p, col t is loss(t, p)."""
    grid = np.empty((n_bins, n_bins), dtype=np.float64)
    for p in range(n_bins):
        for t in range(n_bins):
            grid[p, t] = loss_fn(t, p, n_bins)
    return grid


def render_panel(ax, grid: np.ndarray, title: str, n_bins: int) -> None:
    # Logarithmic color scale highlights structure where dynamic range
    # spans many orders of magnitude. Add a tiny epsilon so log(0) on
    # the diagonal doesn't blow up.
    g = np.clip(grid, 0.0, None)
    vmin = max(g.min(), 1e-6)
    vmax = g.max() if g.max() > vmin else vmin * 10
    im = ax.imshow(
        g, origin="lower", cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        aspect="equal", extent=(0, n_bins, 0, n_bins),
    )
    # Reference lines: diagonal + octave / triplet bands.
    bins = np.arange(n_bins)
    ax.plot(bins, bins, "w--", alpha=0.35, lw=0.6, label="p = t")
    ax.plot(bins, 2 * bins, "r:", alpha=0.4, lw=0.6, label="p = 2t")
    ax.plot(bins, bins / 2, "r:", alpha=0.4, lw=0.6)
    ax.plot(bins, 3 * bins, "y:", alpha=0.3, lw=0.5, label="p = 3t")
    ax.plot(bins, bins / 3, "y:", alpha=0.3, lw=0.5)
    ax.set_xlim(0, n_bins)
    ax.set_ylim(0, n_bins)
    ax.set_xlabel("target t")
    ax.set_ylabel("pred p")
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main(n_bins: int = 100, out_dir: Path | None = None) -> None:
    out_dir = out_dir or Path(__file__).parent / "loss_landscapes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # One combined figure for the at-a-glance comparison.
    n_losses = len(LOSSES)
    cols = 4
    rows = (n_losses + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax in axes[n_losses:]:
        ax.axis("off")
    for i, (title, fn) in enumerate(LOSSES):
        print(f"  computing {title.splitlines()[0]} ...")
        grid = compute_grid(fn, n_bins)
        render_panel(axes[i], grid, title, n_bins)
        # Per-loss high-res standalone PNG for closer inspection.
        fig_one, ax_one = plt.subplots(figsize=(6, 5.5))
        render_panel(ax_one, grid, title, n_bins)
        fig_one.tight_layout()
        slug = title.splitlines()[0].lower().replace(" ", "_").replace("/", "")
        slug = "".join(ch for ch in slug if ch.isalnum() or ch in "_-")
        fig_one.savefig(out_dir / f"{i:02d}_{slug}.png", dpi=140)
        plt.close(fig_one)
    fig.suptitle(
        f"Loss landscapes — diagonal = correct, red dotted = octave (p = 2t / t/2), "
        f"yellow dotted = triplet. Predicted distribution = Gaussian(σ=0.5) at p (near-delta). "
        f"n_bins = {n_bins}.",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "00_combined.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {n_losses + 1} graphs to {out_dir}")


if __name__ == "__main__":
    main()
