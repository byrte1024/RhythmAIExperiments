"""Per-eval artifacts — things that produce files on disk, not scalar
metrics.

These aren't `Metric` subclasses because the `Metric` ABC's contract
is ``compute() -> dict[str, float]``; artifacts yield PNGs and raw
arrays. They share the reset/update lifecycle though, so a training
loop can treat them uniformly.

Contract for any artifact:
  - ``name: str`` class-level, used as a filename stem.
  - ``reset()``                        — clear accumulators at eval start.
  - ``update(batch: MetricInput)``     — fold one batch in.
  - ``save(eval_dir: Path, *, step: int)`` — write PNG + raw data.

Concretes:
  - `PredictionScatterArtifact`   — 2-D (target, pred) heatmap.
  - `DistributionArtifact`        — per-class count histograms for
                                     target vs predicted.
  - `RatioErrorScatterArtifact`   — log-scale pred/target ratio scatter.
  - `ErrorHistogramArtifact`      — linear (pred - target) distribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.metrics import MetricInput


# ─────────────────────────── Scatter heatmap ──────────────────────────

class PredictionScatterArtifact:
    """2-D histogram of ``(target_bin, predicted_bin)``. Log-scaled heatmap.

    Saves:
      - ``{eval_dir}/scatter.png``
      - ``{eval_dir}/scatter.npy``  (raw count matrix)
    """
    name = "scatter"

    def __init__(self, b_pred: int):
        if b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {b_pred}")
        self._size = b_pred + 1
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros((self._size, self._size), dtype=np.int64)

    def update(self, batch: MetricInput) -> None:
        pred = batch.output.logits.argmax(dim=-1).detach().cpu().numpy()
        target = batch.target.target_bin.detach().cpu().numpy()
        t = np.clip(target, 0, self._size - 1).astype(np.int64)
        p = np.clip(pred, 0, self._size - 1).astype(np.int64)
        np.add.at(self._hist, (t, p), 1)

    @property
    def histogram(self) -> np.ndarray:
        return self._hist

    def save(self, eval_dir: Path, *, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        display = np.log1p(self._hist)
        im = ax.imshow(display, origin="lower", aspect="equal", cmap="magma")
        last = self._size - 1
        ax.plot([0, last], [0, last], color="white",
                linestyle="--", alpha=0.35, linewidth=1, label="y = x")
        ax.axhline(last, color="cyan", alpha=0.3, linewidth=1)
        ax.axvline(last, color="cyan", alpha=0.3, linewidth=1)
        ax.set_xlabel("Predicted bin")
        ax.set_ylabel("Target bin")
        ax.set_title(
            f"Prediction scatter - step {step:,}  "
            f"({int(self._hist.sum()):,} samples, b_pred={last})"
        )
        fig.colorbar(im, ax=ax, label="log(1 + count)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=150)
        plt.close(fig)
        np.save(eval_dir / f"{self.name}.npy", self._hist)


# ─────────────────────────── Distributions ────────────────────────────

class DistributionArtifact:
    """Per-class count histograms for target vs predicted, overlaid.

    Shape differences make biases obvious: oversampling short offsets,
    missing the long tail, over-STOPping, etc.

    Saves:
      - ``{eval_dir}/distributions.png``
      - ``{eval_dir}/distributions.npz``  (``targets`` + ``preds``)
    """
    name = "distributions"

    def __init__(self, b_pred: int):
        if b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {b_pred}")
        self._size = b_pred + 1
        self.reset()

    def reset(self) -> None:
        self._targets = np.zeros(self._size, dtype=np.int64)
        self._preds = np.zeros(self._size, dtype=np.int64)

    def update(self, batch: MetricInput) -> None:
        pred = batch.output.logits.argmax(dim=-1).detach().cpu().numpy()
        target = batch.target.target_bin.detach().cpu().numpy()
        t = np.clip(target, 0, self._size - 1).astype(np.int64)
        p = np.clip(pred, 0, self._size - 1).astype(np.int64)
        self._targets += np.bincount(t, minlength=self._size)
        self._preds += np.bincount(p, minlength=self._size)

    def save(self, eval_dir: Path, *, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        last = self._size - 1
        x = np.arange(self._size)
        ax.bar(x, self._targets, width=1.0, alpha=0.55, color="#4a90d9",
               edgecolor="none", label="target")
        ax.bar(x, self._preds, width=1.0, alpha=0.55, color="#e86850",
               edgecolor="none", label="predicted")
        ax.axvline(last, color="gray", linestyle=":", alpha=0.6,
                   label=f"STOP (bin {last})")
        ax.set_xlabel("Bin offset (STOP = last)")
        ax.set_ylabel("Count (log)")
        ax.set_yscale("log")
        ax.set_title(
            f"Target vs predicted distribution - step {step:,}   "
            f"({int(self._targets.sum()):,} samples)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=150)
        plt.close(fig)
        np.savez(eval_dir / f"{self.name}.npz",
                 targets=self._targets, preds=self._preds)


# ─────────────────────────── Ratio-error scatter ──────────────────────

class RatioErrorScatterArtifact:
    """Ratio-space error scatter.

    Each point is one non-STOP prediction:
      - X: ``target + 1``, log scale.
      - Y: ``(pred + 1) / (target + 1)``, log scale.
    y=1 is perfect; ±3 % and ±10 % bands mark R-HIT / R-GOOD windows.

    To keep long evals tractable, reservoir-samples down to
    ``max_points`` during `update`. Raw arrays stored in the .npz for
    offline re-plotting.

    Saves:
      - ``{eval_dir}/ratio_error.png``
      - ``{eval_dir}/ratio_error.npz``  (``target`` + ``pred``)
    """
    name = "ratio_error"

    def __init__(self, b_pred: int, *, max_points: int = 100_000):
        if b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {b_pred}")
        if max_points <= 0:
            raise ValueError(f"max_points must be > 0, got {max_points}")
        self._stop_idx = b_pred
        self._max_points = max_points
        self.reset()

    def reset(self) -> None:
        self._targets = np.empty(0, dtype=np.int64)
        self._preds = np.empty(0, dtype=np.int64)

    def update(self, batch: MetricInput) -> None:
        pred = batch.output.logits.argmax(dim=-1).detach().cpu().numpy()
        target = batch.target.target_bin.detach().cpu().numpy()
        mask = (target != self._stop_idx) & (pred != self._stop_idx)
        if not mask.any():
            return
        self._targets = np.concatenate([self._targets, target[mask].astype(np.int64)])
        self._preds = np.concatenate([self._preds, pred[mask].astype(np.int64)])
        if self._targets.size > self._max_points * 2:
            self._downsample()

    def _downsample(self) -> None:
        if self._targets.size <= self._max_points:
            return
        rng = np.random.default_rng(42)
        idx = rng.choice(self._targets.size, size=self._max_points, replace=False)
        self._targets = self._targets[idx]
        self._preds = self._preds[idx]

    def save(self, eval_dir: Path, *, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)
        self._downsample()

        fig, ax = plt.subplots(figsize=(10, 6))
        if self._targets.size:
            x = self._targets + 1  # +1 so target=0 plots on log axis
            y = (self._preds + 1) / (self._targets + 1)
            ax.scatter(x, y, s=3, alpha=0.12, color="#4a90d9", edgecolor="none")
            ax.axhline(1.0, color="black", linestyle="--", alpha=0.6,
                       label="perfect (y = 1)")
            for lo, hi, label, color in (
                (100 / 103, 103 / 100, "+/- 3 %", "#6bc46d"),
                (100 / 110, 110 / 100, "+/- 10 %", "#fcb71e"),
            ):
                ax.axhspan(lo, hi, alpha=0.10, color=color, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Target bin + 1")
        ax.set_ylabel("(pred + 1) / (target + 1)")
        ax.set_title(
            f"Ratio error scatter - step {step:,}   "
            f"({self._targets.size:,} non-STOP points)"
        )
        ax.legend(loc="upper right")
        ax.grid(True, which="both", alpha=0.15)
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=150)
        plt.close(fig)
        np.savez(eval_dir / f"{self.name}.npz",
                 target=self._targets, pred=self._preds)


# ─────────────────────────── Signed error histogram ───────────────────

class ErrorHistogramArtifact:
    """Signed ``(pred - target)`` distribution on non-STOP samples.

    Linear bins, symmetric around 0. Directional bias and long tails
    jump out; complements the ratio scatter (which is multiplicative).

    Saves:
      - ``{eval_dir}/error_hist.png``
      - ``{eval_dir}/error_hist.npz``  (``errors``)
    """
    name = "error_hist"

    def __init__(self, b_pred: int, *, max_points: int = 200_000):
        if b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {b_pred}")
        self._stop_idx = b_pred
        self._max_points = max_points
        self.reset()

    def reset(self) -> None:
        self._errors = np.empty(0, dtype=np.int64)

    def update(self, batch: MetricInput) -> None:
        pred = batch.output.logits.argmax(dim=-1).detach().cpu().numpy()
        target = batch.target.target_bin.detach().cpu().numpy()
        mask = (target != self._stop_idx) & (pred != self._stop_idx)
        if not mask.any():
            return
        new = (pred[mask] - target[mask]).astype(np.int64)
        self._errors = np.concatenate([self._errors, new])
        if self._errors.size > self._max_points * 2:
            rng = np.random.default_rng(42)
            idx = rng.choice(self._errors.size, size=self._max_points, replace=False)
            self._errors = self._errors[idx]

    def save(self, eval_dir: Path, *, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        if self._errors.size:
            lim = max(1, int(np.quantile(np.abs(self._errors), 0.995)) + 1)
            bins = np.arange(-lim, lim + 1)
            ax.hist(self._errors, bins=bins, color="#c76dba", edgecolor="none")
            ax.axvline(0, color="black", linestyle="--", alpha=0.6,
                       label="no error")
            med = float(np.median(self._errors))
            ax.axvline(med, color="red", linestyle=":", alpha=0.8,
                       label=f"median = {med:+.1f}")
            ax.legend()
        ax.set_xlabel("pred - target  (bins)")
        ax.set_ylabel("Count (log)")
        ax.set_yscale("log")
        ax.set_title(
            f"Signed error distribution - step {step:,}   "
            f"({self._errors.size:,} non-STOP samples)"
        )
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=150)
        plt.close(fig)
        np.savez(eval_dir / f"{self.name}.npz", errors=self._errors)
