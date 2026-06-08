"""Per-eval artifacts for the typing model: confusion matrices,
confidence distributions, calibration curves, entropy histograms,
and threshold sweep plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.metrics import MetricInput
from ..domain.typing import TypingOutput, TypingTarget
from .metrics_typing import (
    STRENGTH_THRESHOLDS,
    TYPE_THRESHOLDS,
    _binary_entropy,
    _binary_prf,
)


class TypingConfusionArtifact:
    """Accumulates raw predictions across eval, saves 14 plots + 2 npz."""

    def __init__(self) -> None:
        self._type_probs: list[np.ndarray] = []
        self._type_targets: list[np.ndarray] = []
        self._str_probs: list[np.ndarray] = []
        self._str_targets: list[np.ndarray] = []

    def reset(self) -> None:
        self._type_probs.clear()
        self._type_targets.clear()
        self._str_probs.clear()
        self._str_targets.clear()

    def update(self, batch: MetricInput) -> None:
        out: TypingOutput = batch.output  # type: ignore[assignment]
        tgt: TypingTarget = batch.target  # type: ignore[assignment]
        self._type_probs.append(out.type_logit.detach().sigmoid().cpu().numpy())
        self._type_targets.append(tgt.type_target.detach().cpu().numpy())
        self._str_probs.append(out.strength_logit.detach().sigmoid().cpu().numpy())
        self._str_targets.append(tgt.strength_target.detach().cpu().numpy())

    def save(self, eval_dir: Path, step: int = 0) -> None:
        if not self._type_probs:
            return

        tp = np.concatenate(self._type_probs)
        tt = np.concatenate(self._type_targets).astype(np.int64)
        sp = np.concatenate(self._str_probs)
        st = np.concatenate(self._str_targets).astype(np.int64)

        out_dir = Path(eval_dir) / "typing"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save raw data
        np.savez(out_dir / "type_predictions.npz",
                 probs=tp, targets=tt, preds=(tp > 0.5).astype(np.int64))
        np.savez(out_dir / "strength_predictions.npz",
                 probs=sp, targets=st, preds=(sp > 0.5).astype(np.int64))

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        self._plot_confusion(tp, tt, "type", ["K", "D"], out_dir, plt)
        self._plot_confusion(sp, st, "strength", ["NORMAL", "BIG"], out_dir, plt)
        self._plot_combined_confusion(tp, tt, sp, st, out_dir, plt)
        self._plot_confidence_dist(tp, tt, "type", "P(D)", out_dir, plt)
        self._plot_confidence_dist(sp, st, "strength", "P(BIG)", out_dir, plt)
        self._plot_calibration(tp, tt, "type", "P(D)", out_dir, plt)
        self._plot_calibration(sp, st, "strength", "P(BIG)", out_dir, plt)
        self._plot_entropy_dist(tp, tt, "type", out_dir, plt)
        self._plot_entropy_dist(sp, st, "strength", out_dir, plt)
        self._plot_conf_vs_acc(tp, tt, "type", out_dir, plt)
        self._plot_conf_vs_acc(sp, st, "strength", out_dir, plt)
        self._plot_threshold_sweep(tp, tt, "type", TYPE_THRESHOLDS, out_dir, plt)
        self._plot_threshold_sweep(sp, st, "strength", STRENGTH_THRESHOLDS, out_dir, plt)

    def _plot_confusion(
        self, probs: np.ndarray, targets: np.ndarray,
        head: str, labels: list[str], out_dir: Path, plt: object,
    ) -> None:
        pred = (probs > 0.5).astype(np.int64)
        mat = np.zeros((2, 2), dtype=np.int64)
        for p, g in zip(pred, targets):
            mat[g, p] += 1

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))  # type: ignore
        total = mat.sum()
        im = ax.imshow(mat, cmap="Blues")  # type: ignore
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Pred {l}" for l in labels])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"GT {l}" for l in labels])
        for i in range(2):
            for j in range(2):
                pct = mat[i, j] / total * 100 if total else 0
                ax.text(j, i, f"{mat[i, j]:,}\n({pct:.1f}%)",
                        ha="center", va="center", fontsize=11)
        ax.set_title(f"{head} confusion (n={total:,})")
        plt.colorbar(im, ax=ax, fraction=0.046)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(out_dir / f"{head}_confusion.png", dpi=150)  # type: ignore
        plt.close()  # type: ignore

    def _plot_combined_confusion(
        self, tp: np.ndarray, tt: np.ndarray,
        sp: np.ndarray, st: np.ndarray,
        out_dir: Path, plt: object,
    ) -> None:
        pred_4 = (tp > 0.5).astype(np.int64) + (sp > 0.5).astype(np.int64) * 2
        gt_4 = tt.astype(np.int64) + st.astype(np.int64) * 2
        labels = ["DON", "KA", "BDON", "BKA"]
        mat = np.zeros((4, 4), dtype=np.int64)
        for p, g in zip(pred_4, gt_4):
            mat[g, p] += 1
        total = mat.sum()

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))  # type: ignore
        im = ax.imshow(mat, cmap="Blues")  # type: ignore
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"P:{l}" for l in labels], fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"GT:{l}" for l in labels], fontsize=9)
        for i in range(4):
            for j in range(4):
                pct = mat[i, j] / total * 100 if total else 0
                ax.text(j, i, f"{mat[i, j]:,}\n{pct:.1f}%",
                        ha="center", va="center", fontsize=8)
        ax.set_title(f"Combined 4-class confusion (n={total:,})")
        plt.colorbar(im, ax=ax, fraction=0.046)  # type: ignore
        plt.tight_layout()  # type: ignore
        plt.savefig(out_dir / "combined_confusion.png", dpi=150)  # type: ignore
        plt.close()  # type: ignore

    def _plot_confidence_dist(
        self, probs: np.ndarray, targets: np.ndarray,
        head: str, xlabel: str, out_dir: Path, plt: object,
    ) -> None:
        pred = (probs > 0.5).astype(np.int64)
        correct = pred == targets
        max_conf = np.maximum(probs, 1 - probs)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # type: ignore
        bins = np.linspace(0, 1, 51)
        ax.hist(probs[correct], bins=bins, alpha=0.6, color="green",
                label=f"Correct (n={int(correct.sum()):,})", density=True)
        ax.hist(probs[~correct], bins=bins, alpha=0.6, color="red",
                label=f"Wrong (n={int((~correct).sum()):,})", density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        decisive = float(np.mean(max_conf > 0.9)) * 100
        conflicted = float(np.mean((max_conf >= 0.4) & (max_conf <= 0.6))) * 100
        ax.set_title(
            f"{head} confidence distribution\n"
            f"decisive (>0.9): {decisive:.1f}%  |  "
            f"conflicted (0.4-0.6): {conflicted:.1f}%  |  "
            f"entropy: {float(np.mean(_binary_entropy(probs))):.4f}/0.693"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()  # type: ignore
        plt.savefig(out_dir / f"{head}_confidence_dist.png", dpi=150)  # type: ignore
        plt.close()  # type: ignore

    def _plot_calibration(
        self, probs: np.ndarray, targets: np.ndarray,
        head: str, xlabel: str, out_dir: Path, plt: object,
    ) -> None:
        n_bins = 20
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accs = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for b in range(n_bins):
            mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
            if b == n_bins - 1:
                mask |= probs == bin_edges[b + 1]
            bin_counts[b] = float(mask.sum())
            if bin_counts[b] > 0:
                bin_accs[b] = float(targets[mask].mean())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),  # type: ignore
                                        gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax1.bar(bin_centers, bin_accs, width=1 / n_bins * 0.8,
                alpha=0.7, color="steelblue", label="Actual positive rate")
        ax1.set_ylabel("Actual positive rate")
        ax1.set_title(f"{head} calibration ({xlabel})")
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.grid(True, alpha=0.3)

        ax2.bar(bin_centers, bin_counts, width=1 / n_bins * 0.8,
                alpha=0.7, color="gray")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Count")
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()  # type: ignore
        plt.savefig(out_dir / f"{head}_calibration.png", dpi=150)  # type: ignore
        plt.close()  # type: ignore

    def _plot_entropy_dist(
        self, probs: np.ndarray, targets: np.ndarray,
        head: str, out_dir: Path, plt: object,
    ) -> None:
        pred = (probs > 0.5).astype(np.int64)
        correct = pred == targets
        ent = _binary_entropy(probs)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # type: ignore
        bins = np.linspace(0, 0.7, 36)
        ax.hist(ent[correct], bins=bins, alpha=0.6, color="green",
                label="Correct", density=True)
        ax.hist(ent[~correct], bins=bins, alpha=0.6, color="red",
                label="Wrong", density=True)
        ax.set_xlabel("Binary entropy H(p)")
        ax.set_ylabel("Density")
        ax.axvline(0.693, color="gray", linestyle="--", alpha=0.5, label="max (coin flip)")
        ax.set_title(f"{head} entropy: correct vs wrong")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()  # type: ignore
        plt.savefig(out_dir / f"{head}_entropy_dist.png", dpi=150)  # type: ignore
        plt.close()  # type: ignore

    def _plot_conf_vs_acc(
        self, probs: np.ndarray, targets: np.ndarray,
        head: str, out_dir: Path, plt: object,
    ) -> None:
        pred = (probs > 0.5).astype(np.int64)
        max_conf = np.maximum(probs, 1 - probs)
        correct = pred == targets

        n_bins = 10
        bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_acc = np.zeros(n_bins)
        bin_count = np.zeros(n_bins)

        for b in range(n_bins):
            mask = (max_conf >= bin_edges[b]) & (max_conf < bin_edges[b + 1])
            if b == n_bins - 1:
                mask |= max_conf == bin_edges[b + 1]
            bin_count[b] = float(mask.sum())
            if bin_count[b] > 0:
                bin_acc[b] = float(correct[mask].mean())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),  # type: ignore
                                        gridspec_kw={"height_ratios": [3, 1]})
        ax1.bar(bin_centers, bin_acc, width=0.04, alpha=0.7, color="steelblue")
        ax1.set_ylabel("Accuracy")
        ax1.set_title(f"{head}: accuracy vs confidence")
        ax1.set_xlim(0.5, 1.0)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        ax2.bar(bin_centers, bin_count, width=0.04, alpha=0.7, color="gray")
        ax2.set_xlabel("Confidence (max(p, 1-p))")
        ax2.set_ylabel("Count")
        ax2.set_xlim(0.5, 1.0)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()  # type: ignore
        plt.savefig(out_dir / f"{head}_conf_vs_acc.png", dpi=150)  # type: ignore
        plt.close()  # type: ignore

    def _plot_threshold_sweep(
        self, probs: np.ndarray, targets: np.ndarray,
        head: str, thresholds: tuple[float, ...],
        out_dir: Path, plt: object,
    ) -> None:
        gt = targets.astype(np.int64)
        accs, f1_pos, f1_neg = [], [], []

        for thr in thresholds:
            pred = (probs > thr).astype(np.int64)
            accs.append(float(np.mean(pred == gt)))
            _, _, f1p = _binary_prf(pred, gt)
            _, _, f1n = _binary_prf(1 - pred, 1 - gt)
            f1_pos.append(f1p)
            f1_neg.append(f1n)

        best_idx = int(np.argmax(f1_pos))
        pos_label = "F1_D" if head == "type" else "F1_BIG"
        neg_label = "F1_K" if head == "type" else "F1_NORMAL"

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # type: ignore
        ax.plot(thresholds, accs, "o-", label="Accuracy", color="gray")
        ax.plot(thresholds, f1_pos, "s-", label=pos_label, color="tab:red")
        ax.plot(thresholds, f1_neg, "^-", label=neg_label, color="tab:blue")
        ax.axvline(thresholds[best_idx], color="green", linestyle="--",
                    alpha=0.7, label=f"Best {pos_label} @ {thresholds[best_idx]:.2f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{head} threshold sweep")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()  # type: ignore
        plt.savefig(out_dir / f"{head}_threshold_sweep.png", dpi=150)  # type: ignore
        plt.close()  # type: ignore
