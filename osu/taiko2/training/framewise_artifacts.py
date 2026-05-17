"""Per-eval artifacts for the #016 framewise diffusion stack.

Each artifact follows the same ``reset / update(batch: MetricInput) /
save(eval_dir, *, step)`` lifecycle the existing artifacts use, so the
training loop can install them uniformly.

Concretes:

- ``FramewiseHeatmapArtifact`` — accumulates ``M_0_hat`` rows at the
  sampled t (leaky — uses the value the loss already computed). Renders
  a ``(n_samples_shown, n_bins)`` heatmap with GT positions overlaid as
  red marks.
- ``FramewiseDistributionArtifact`` — histogram of predicted activation
  values at GT-positive vs GT-negative bins.
- ``FramewiseTrainingHeatmapArtifact`` — same renderer as
  ``FramewiseHeatmapArtifact`` but named differently so the loop can
  install it for the train_noaug pass (the file stem differs from the
  val one).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.metrics import MetricInput


def _extract_pred_target(batch: MetricInput) -> tuple[np.ndarray, np.ndarray] | None:
    """Pull ``(confidence_map (B, n_bins), target_binary (B, n_bins))``
    from a framewise batch's output. Returns ``None`` if the batch is
    not a framewise-shape output.

    Prefers ``output.confidence_map`` (already in ``[0, 1]``); falls
    back to ``sigmoid(output.logits)`` for BCE models where the raw
    logits are pre-sigmoid.
    """
    import torch as _torch
    output = batch.output
    target = batch.target
    pred = getattr(output, "confidence_map", None)
    if pred is None:
        logits = getattr(output, "logits", None)
        if logits is None or logits.dim() != 2:
            return None
        pred = _torch.sigmoid(logits).clamp(0.0, 1.0)
    if pred.dim() != 2:
        return None
    tgt = getattr(target, "target_map_binary", None)
    if tgt is None:
        return None
    return pred.detach().cpu().numpy(), tgt.detach().cpu().numpy()


# ─────────────────────────── Heatmap (val) ───────────────────────────


class FramewiseHeatmapArtifact:
    """Heatmap of predicted M_0_hat rows with GT positions overlaid.

    Streams up to ``max_rows`` random sample rows (seeded for
    determinism) and renders them as a ``(rows_shown, n_bins)`` image.
    GT positions are drawn as red ticks above the matching row.

    Saves:
      - ``{eval_dir}/{name}.png``
      - ``{eval_dir}/{name}.npz``  (``pred`` + ``target``)
    """
    name: str = "framewise_heatmap"

    def __init__(self, *, max_rows: int = 64, seed: int = 0):
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1 (got {max_rows})")
        self._max_rows = int(max_rows)
        self._seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._pred_rows: list[np.ndarray] = []
        self._target_rows: list[np.ndarray] = []
        self._n_seen: int = 0

    def update(self, batch: MetricInput) -> None:
        got = _extract_pred_target(batch)
        if got is None:
            return
        pred, tgt = got
        B = pred.shape[0]
        for i in range(B):
            self._n_seen += 1
            # Reservoir sampling — keep first ``max_rows`` rows, then
            # randomly replace with probability max_rows / n_seen.
            if len(self._pred_rows) < self._max_rows:
                self._pred_rows.append(pred[i])
                self._target_rows.append(tgt[i])
            else:
                # cheap stdlib RNG — numpy.random has flaky init on this box.
                import random
                rng = random.Random(self._seed + self._n_seen)
                j = rng.randrange(self._n_seen)
                if j < self._max_rows:
                    self._pred_rows[j] = pred[i]
                    self._target_rows[j] = tgt[i]

    def save(self, eval_dir: Path, *, step: int) -> None:
        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        if not self._pred_rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pred_mat = np.stack(self._pred_rows, axis=0)   # (R, n_bins)
        tgt_mat = np.stack(self._target_rows, axis=0)  # (R, n_bins)
        R, N = pred_mat.shape
        fig, ax = plt.subplots(figsize=(12, max(3, R * 0.15)))
        im = ax.imshow(
            pred_mat, origin="lower", aspect="auto",
            cmap="viridis", vmin=0.0, vmax=1.0,
        )
        # Overlay GT as red dots.
        rows, cols = np.where(tgt_mat > 0.5)
        if rows.size:
            ax.scatter(cols, rows, s=8, c="red", marker="|", alpha=0.8)
        ax.set_xlabel("bin")
        ax.set_ylabel("sample")
        ax.set_title(
            f"Framewise heatmap - step {step:,}  ({R} samples, n_bins={N})"
        )
        fig.colorbar(im, ax=ax, label="M_0_hat")
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=120)
        plt.close(fig)
        np.savez(
            eval_dir / f"{self.name}.npz",
            pred=pred_mat, target=tgt_mat,
        )


# ─────────────────────────── Distribution ────────────────────────────


class FramewiseDistributionArtifact:
    """Histogram of predicted activation at GT-positive vs negative bins.

    Saves:
      - ``{eval_dir}/{name}.png``
      - ``{eval_dir}/{name}.npz``  (``pos_vals``, ``neg_vals``)
    """
    name: str = "framewise_distribution"

    def __init__(self, *, max_points_per_class: int = 200_000, seed: int = 0):
        if max_points_per_class < 1:
            raise ValueError(
                f"max_points_per_class must be >= 1 (got {max_points_per_class})"
            )
        self._max = int(max_points_per_class)
        self._seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self._pos: list[np.ndarray] = []
        self._neg: list[np.ndarray] = []
        self._n_pos = 0
        self._n_neg = 0

    def update(self, batch: MetricInput) -> None:
        got = _extract_pred_target(batch)
        if got is None:
            return
        pred, tgt = got
        flat_pred = pred.reshape(-1)
        flat_tgt = tgt.reshape(-1)
        pos_mask = flat_tgt > 0.5
        pos = flat_pred[pos_mask]
        neg = flat_pred[~pos_mask]
        self._pos.append(pos)
        self._neg.append(neg)
        self._n_pos += int(pos.size)
        self._n_neg += int(neg.size)

    def _concat_with_cap(
        self, arrs: list[np.ndarray], n_total: int,
    ) -> np.ndarray:
        cat = np.concatenate(arrs) if arrs else np.empty(0, dtype=np.float32)
        if cat.size <= self._max:
            return cat
        import random
        rng = random.Random(self._seed)
        idx = sorted(rng.sample(range(cat.size), self._max))
        return cat[np.asarray(idx, dtype=np.int64)]

    def save(self, eval_dir: Path, *, step: int) -> None:
        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        pos_vals = self._concat_with_cap(self._pos, self._n_pos)
        neg_vals = self._concat_with_cap(self._neg, self._n_neg)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0.0, 1.0, 51)
        if neg_vals.size:
            ax.hist(
                neg_vals, bins=bins, alpha=0.6, color="#4a90d9",
                edgecolor="none", label=f"negative ({self._n_neg:,})",
                density=True,
            )
        if pos_vals.size:
            ax.hist(
                pos_vals, bins=bins, alpha=0.6, color="#e86850",
                edgecolor="none", label=f"positive ({self._n_pos:,})",
                density=True,
            )
        ax.set_xlabel("M_0_hat")
        ax.set_ylabel("density")
        ax.set_yscale("log")
        ax.set_title(
            f"Framewise activation distribution - step {step:,}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=120)
        plt.close(fig)
        np.savez(
            eval_dir / f"{self.name}.npz",
            pos_vals=pos_vals, neg_vals=neg_vals,
        )


# ─────────────────────────── Training heatmap (train_noaug) ──────────


class FramewiseTrainingHeatmapArtifact(FramewiseHeatmapArtifact):
    """Same renderer as ``FramewiseHeatmapArtifact``, different filename
    stem. Use under train_noaug for the overfit diagnostic."""
    name: str = "framewise_train_heatmap"
