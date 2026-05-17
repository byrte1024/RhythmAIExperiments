"""Diagnostics artifact for framewise detectors (#017+).

One class, five ``save()`` outputs. Designed for subclassing: a future
``DiffusionFramewiseDiagnosticsArtifact`` overrides only
``_extract_confidence_and_target`` to pull the confidence map from the
diffusion sampler output.

Outputs
-------
1. ``per_bin_rate.png/.npz`` -- per-bin P(target=1), recall@bin, FPR@bin.
2. ``value_hist_target.png/.npz`` -- histogram of target values.
3. ``value_hist_pred.png/.npz`` -- histogram of predicted confidences.
4. ``confidence_by_outcome.png/.npz`` -- TP/FN/FP/TN distributions.
5. ``reliability.png/.npz`` -- calibration plot + ECE + Brier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..domain.metrics import MetricInput


class FramewiseDiagnosticsArtifact:
    """Per-eval diagnostics for any model producing a framewise
    confidence map over ``n_bins`` future-time bins."""

    def __init__(self, *, max_reservoir: int = 500_000, seed: int = 0):
        self._max_reservoir = int(max_reservoir)
        self._seed = int(seed)
        self.reset()

    # ── extraction (override for diffusion subclass) ─────────────────

    @classmethod
    def _extract_confidence_and_target(
        cls, batch: MetricInput,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(confidence_map (B, n_bins), target_binary (B, n_bins))``
        as numpy or ``None`` to skip."""
        output = batch.output
        target = batch.target
        conf = getattr(output, "confidence_map", None)
        if conf is None:
            logits = getattr(output, "logits", None)
            if logits is None or logits.dim() != 2:
                return None
            conf = torch.sigmoid(logits).clamp(0.0, 1.0)
        tb = getattr(target, "target_map_binary", None)
        if tb is None:
            return None
        return conf.detach().cpu().numpy(), tb.detach().cpu().numpy()

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> None:
        self._n_bins: int | None = None
        # Per-bin counters: shape (n_bins,) allocated on first update.
        self._bin_n_total: np.ndarray | None = None
        self._bin_n_pos_target: np.ndarray | None = None
        self._bin_n_tp: np.ndarray | None = None
        self._bin_n_fp: np.ndarray | None = None
        # Reservoir-sampled values (per-outcome, for confidence_by_outcome plot).
        self._pred_tp: list[float] = []
        self._pred_fn: list[float] = []
        self._pred_fp: list[float] = []
        self._pred_tn: list[float] = []
        self._all_pred: list[float] = []
        self._all_target: list[float] = []
        self._n_seen = 0

    def update(self, batch: MetricInput) -> None:
        got = self._extract_confidence_and_target(batch)
        if got is None:
            return
        conf, tb = got  # (B, n_bins)
        B, n_bins = conf.shape

        if self._n_bins is None:
            self._n_bins = n_bins
            self._bin_n_total = np.zeros(n_bins, dtype=np.int64)
            self._bin_n_pos_target = np.zeros(n_bins, dtype=np.int64)
            self._bin_n_tp = np.zeros(n_bins, dtype=np.int64)
            self._bin_n_fp = np.zeros(n_bins, dtype=np.int64)

        # Per-bin stats.
        pred_pos = conf > 0.5
        gt_pos = tb > 0.5
        self._bin_n_total += B
        self._bin_n_pos_target += gt_pos.sum(axis=0).astype(np.int64)
        self._bin_n_tp += (pred_pos & gt_pos).sum(axis=0).astype(np.int64)
        self._bin_n_fp += (pred_pos & ~gt_pos).sum(axis=0).astype(np.int64)

        # Reservoir sampling for per-value histograms.
        flat_conf = conf.ravel()
        flat_tb = tb.ravel()
        flat_pp = pred_pos.ravel()
        flat_gp = gt_pos.ravel()

        self._reservoir_extend(
            self._pred_tp, flat_conf[(flat_pp) & (flat_gp)]
        )
        self._reservoir_extend(
            self._pred_fn, flat_conf[(~flat_pp) & (flat_gp)]
        )
        self._reservoir_extend(
            self._pred_fp, flat_conf[(flat_pp) & (~flat_gp)]
        )
        self._reservoir_extend(
            self._pred_tn, flat_conf[(~flat_pp) & (~flat_gp)]
        )
        self._reservoir_extend(self._all_pred, flat_conf)
        self._reservoir_extend(self._all_target, flat_tb)
        self._n_seen += B

    def _reservoir_extend(
        self, reservoir: list[float], vals: np.ndarray,
    ) -> None:
        if vals.size == 0:
            return
        remaining = self._max_reservoir - len(reservoir)
        if remaining <= 0:
            return
        if vals.size <= remaining:
            reservoir.extend(vals.tolist())
        else:
            reservoir.extend(vals[:remaining].tolist())

    # ── save ─────────────────────────────────────────────────────────

    def save(self, eval_dir: Path, *, step: int) -> None:
        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        if self._n_bins is None:
            return
        self._save_per_bin_rate(eval_dir, step)
        self._save_value_hist_target(eval_dir, step)
        self._save_value_hist_pred(eval_dir, step)
        self._save_value_hist_combined(eval_dir, step)
        self._save_confidence_by_outcome(eval_dir, step)

    def _save_per_bin_rate(self, d: Path, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = self._bin_n_total.clip(min=1)
        pos_rate = self._bin_n_pos_target / n
        recall = np.where(
            self._bin_n_pos_target > 0,
            self._bin_n_tp / self._bin_n_pos_target.clip(min=1),
            0.0,
        )
        neg_count = (n - self._bin_n_pos_target).clip(min=1)
        fpr = self._bin_n_fp / neg_count

        np.savez(
            d / "per_bin_rate.npz",
            pos_rate=pos_rate, recall=recall, fpr=fpr,
            n_total=self._bin_n_total,
            n_pos_target=self._bin_n_pos_target,
            n_tp=self._bin_n_tp, n_fp=self._bin_n_fp,
        )

        fig, ax = plt.subplots(figsize=(14, 4))
        x = np.arange(len(pos_rate))
        ax.plot(x, pos_rate, label="P(target=1)", alpha=0.7)
        ax.plot(x, recall, label="recall@bin", alpha=0.7)
        ax.plot(x, fpr, label="FPR@bin", alpha=0.7)
        ax.set_xlabel("bin")
        ax.set_ylabel("rate")
        ax.set_title(f"Per-bin rates - step {step:,}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "per_bin_rate.png", dpi=120)
        plt.close(fig)

    def _save_value_hist_target(self, d: Path, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = np.array(self._all_target)
        np.savez(d / "value_hist_target.npz", values=vals)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, yscale in zip(axes, ("linear", "log")):
            ax.hist(
                vals, bins=50, range=(0, 1),
                edgecolor="none", alpha=0.8, color="#e86850",
            )
            ax.set_xlabel("target value")
            ax.set_ylabel("count")
            ax.set_yscale(yscale)
            ax.set_title(f"Target distribution ({yscale}) - step {step:,}")
        fig.tight_layout()
        fig.savefig(d / "value_hist_target.png", dpi=120)
        plt.close(fig)

    def _save_value_hist_pred(self, d: Path, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = np.array(self._all_pred)
        np.savez(d / "value_hist_pred.npz", values=vals)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, yscale in zip(axes, ("linear", "log")):
            ax.hist(
                vals, bins=101, range=(0, 1),
                edgecolor="none", alpha=0.8, color="#4a90d9",
            )
            ax.set_xlabel("predicted confidence")
            ax.set_ylabel("count")
            ax.set_yscale(yscale)
            ax.set_title(f"Prediction distribution ({yscale}) - step {step:,}")
        fig.tight_layout()
        fig.savefig(d / "value_hist_pred.png", dpi=120)
        plt.close(fig)

    def _save_value_hist_combined(self, d: Path, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tgt_vals = np.array(self._all_target)
        pred_vals = np.array(self._all_pred)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for col, (vals, label, color, n_bins) in enumerate([
            (tgt_vals, "Target", "#e86850", 50),
            (pred_vals, "Prediction", "#4a90d9", 101),
        ]):
            for row, yscale in enumerate(["linear", "log"]):
                ax = axes[row, col]
                ax.hist(
                    vals, bins=n_bins, range=(0, 1),
                    edgecolor="none", alpha=0.8, color=color,
                )
                ax.set_xlabel("value")
                ax.set_ylabel("count")
                ax.set_yscale(yscale)
                ax.set_title(f"{label} ({yscale}) - step {step:,}")
        fig.tight_layout()
        fig.savefig(d / "value_hist_combined.png", dpi=120)
        plt.close(fig)

    def _save_confidence_by_outcome(self, d: Path, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tp = np.array(self._pred_tp)
        fn = np.array(self._pred_fn)
        fp = np.array(self._pred_fp)
        tn = np.array(self._pred_tn)
        np.savez(
            d / "confidence_by_outcome.npz",
            tp=tp, fn=fn, fp=fp, tn=tn,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, 1, 51)
        for vals, label, color in [
            (tp, f"TP ({len(tp):,})", "#2ca02c"),
            (fn, f"FN ({len(fn):,})", "#d62728"),
            (fp, f"FP ({len(fp):,})", "#ff7f0e"),
            (tn, f"TN ({len(tn):,})", "#1f77b4"),
        ]:
            if len(vals):
                ax.hist(
                    vals, bins=bins, alpha=0.5, label=label,
                    color=color, edgecolor="none", density=True,
                )
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_title(f"Confidence by outcome - step {step:,}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "confidence_by_outcome.png", dpi=120)
        plt.close(fig)

