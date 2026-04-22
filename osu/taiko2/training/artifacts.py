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
  - `PredictionHeatmapArtifact`   — 2-D (target, pred) log-count heatmap.
  - `DistributionArtifact`        — per-class count histograms for
                                     target vs predicted.
  - `RatioErrorHeatmapArtifact`   — 2-D log-count heatmap over
                                     (target, pred/target ratio) in
                                     log-log space.
  - `ErrorHistogramArtifact`      — linear (pred - target) distribution.
  - `RatioHitArtifact`            — HIT rate bucketed by
                                     target / prev_gap ratio.
  - `MetronomeHitArtifact`        — HIT rate split metronome vs
                                     anti-metronome (target ≈ prev_gap
                                     vs target meaningfully different).

Rhythmic ratio guides (1/1, 1/2, 2/1, 1/3, 3/1, 1/4, 4/1, 1/6, 6/1) are
overlaid on both heatmaps so typical quantization errors (doubling,
halving, triplets, sextuplets) are visible at a glance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.metrics import MetricInput


# Rhythmic ratio guides drawn on both heatmaps. Each entry is
# (label, numeric ratio = pred / target). 1/1 is shown as a solid
# line; the others dashed to keep the overlay legible.
_RATIO_GUIDES: tuple[tuple[str, float], ...] = (
    ("1/6", 1 / 6),
    ("1/4", 1 / 4),
    ("1/3", 1 / 3),
    ("1/2", 1 / 2),
    ("1/1", 1.0),
    ("2/1", 2.0),
    ("3/1", 3.0),
    ("4/1", 4.0),
    ("6/1", 6.0),
)


# ─────────────────────────── Prediction heatmap ───────────────────────

class PredictionHeatmapArtifact:
    """2-D log-count heatmap over ``(target_bin, predicted_bin)``.

    Rhythmic ratio guides (``pred / target`` ∈ 1/6..6/1) drawn on top
    so quantization errors — halving, doubling, triplet confusion —
    jump out.

    Saves:
      - ``{eval_dir}/heatmap.png``
      - ``{eval_dir}/heatmap.npy``  (raw count matrix)
    """
    name = "heatmap"

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
        # hist is indexed [target, pred] — transpose so X=target, Y=pred.
        display = np.log1p(self._hist).T
        im = ax.imshow(display, origin="lower", aspect="equal", cmap="magma")
        last = self._size - 1

        # Ratio guides: pred = ratio * target. Clipped at [0, last] so
        # the line stays inside the imshow extent.
        xs = np.arange(0, last + 1, dtype=np.float64)
        for label, ratio in _RATIO_GUIDES:
            ys = np.clip(ratio * xs, 0, last)
            style = "-" if ratio == 1.0 else "--"
            ax.plot(
                xs, ys, color="white", linestyle=style,
                alpha=0.45 if ratio == 1.0 else 0.28,
                linewidth=1.0,
                label=f"pred = {label} * target",
            )
        ax.axhline(last, color="cyan", alpha=0.3, linewidth=1)
        ax.axvline(last, color="cyan", alpha=0.3, linewidth=1)
        ax.set_xlim(-0.5, last + 0.5)
        ax.set_ylim(-0.5, last + 0.5)
        ax.set_xlabel("Target bin")
        ax.set_ylabel("Predicted bin")
        ax.set_title(
            f"Prediction heatmap - step {step:,}  "
            f"({int(self._hist.sum()):,} samples, b_pred={last})"
        )
        fig.colorbar(im, ax=ax, label="log(1 + count)")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.85)
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


# ─────────────────────────── Ratio-error heatmap ──────────────────────

class RatioErrorHeatmapArtifact:
    """2-D log-count heatmap over ``(log(target+1), log((pred+1)/(target+1)))``.

    Streaming 2-D histogram — no per-point storage, no reservoir
    sampling. X axis is log target (linear in log-bins over
    ``[0, log(b_pred+1)]``); Y axis is log ratio, symmetric around 0
    over ``[-log(y_ratio_max), +log(y_ratio_max)]``. Rhythmic ratio
    guides (1/6..6/1) drawn as horizontal lines.

    Saves:
      - ``{eval_dir}/ratio_error.png``
      - ``{eval_dir}/ratio_error.npz``  (``hist`` + ``x_edges`` + ``y_edges``)
    """
    name = "ratio_error"

    def __init__(
        self,
        b_pred: int,
        *,
        x_bins: int = 120,
        y_bins: int = 120,
        y_ratio_max: float = 8.0,
    ):
        if b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {b_pred}")
        if x_bins <= 1 or y_bins <= 1:
            raise ValueError(
                f"x_bins and y_bins must be > 1, got {x_bins}, {y_bins}"
            )
        if y_ratio_max <= 1.0:
            raise ValueError(
                f"y_ratio_max must be > 1, got {y_ratio_max}"
            )
        self._stop_idx = b_pred
        self._x_edges = np.linspace(0.0, float(np.log(b_pred + 1)), x_bins + 1)
        y_extent = float(np.log(y_ratio_max))
        self._y_edges = np.linspace(-y_extent, y_extent, y_bins + 1)
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros(
            (len(self._x_edges) - 1, len(self._y_edges) - 1),
            dtype=np.int64,
        )
        self._n_seen = 0
        self._n_oob = 0   # samples whose log-ratio fell outside y range

    def update(self, batch: MetricInput) -> None:
        pred = batch.output.logits.argmax(dim=-1).detach().cpu().numpy()
        target = batch.target.target_bin.detach().cpu().numpy()
        mask = (target != self._stop_idx) & (pred != self._stop_idx)
        if not mask.any():
            return
        t = target[mask].astype(np.float64)
        p = pred[mask].astype(np.float64)
        log_t = np.log(t + 1.0)
        log_r = np.log((p + 1.0) / (t + 1.0))
        self._n_seen += int(t.size)
        y_lo, y_hi = self._y_edges[0], self._y_edges[-1]
        inside_y = (log_r >= y_lo) & (log_r <= y_hi)
        self._n_oob += int((~inside_y).sum())
        if inside_y.any():
            h, _, _ = np.histogram2d(
                log_t[inside_y], log_r[inside_y],
                bins=(self._x_edges, self._y_edges),
            )
            self._hist += h.astype(np.int64)

    def save(self, eval_dir: Path, *, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        # hist indexed [x_bin, y_bin] — transpose to (y, x) for imshow.
        display = np.log1p(self._hist).T
        x0, x1 = self._x_edges[0], self._x_edges[-1]
        y0, y1 = self._y_edges[0], self._y_edges[-1]
        im = ax.imshow(
            display, origin="lower", aspect="auto", cmap="magma",
            extent=(x0, x1, y0, y1),
        )

        # R-HIT / R-GOOD tolerance bands.
        for pct, label, color in (
            (3.0, "+/- 3 %", "#6bc46d"),
            (10.0, "+/- 10 %", "#fcb71e"),
        ):
            half = np.log(100.0 / (100.0 - pct))
            ax.axhspan(-half, half, alpha=0.10, color=color, label=label)

        # Rhythmic ratio guides at log(r).
        for label, ratio in _RATIO_GUIDES:
            ly = float(np.log(ratio))
            if not (y0 <= ly <= y1):
                continue
            style = "-" if ratio == 1.0 else "--"
            ax.axhline(
                ly, color="white", linestyle=style,
                alpha=0.5 if ratio == 1.0 else 0.30,
                linewidth=1.0,
                label=f"pred = {label} * target",
            )

        ax.set_xlabel("log(target + 1)")
        ax.set_ylabel("log((pred + 1) / (target + 1))")
        ax.set_title(
            f"Ratio-error heatmap - step {step:,}   "
            f"({self._n_seen:,} non-STOP, {self._n_oob:,} out-of-range)"
        )
        fig.colorbar(im, ax=ax, label="log(1 + count)")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        ax.grid(True, which="both", alpha=0.15)
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=150)
        plt.close(fig)
        np.savez(
            eval_dir / f"{self.name}.npz",
            hist=self._hist,
            x_edges=self._x_edges,
            y_edges=self._y_edges,
            n_seen=np.int64(self._n_seen),
            n_oob=np.int64(self._n_oob),
        )


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


# ─────────────────────────── Context-conditional HIT ──────────────────
# All three artifacts below share the same HIT definition as
# `OnsetMetric`: HIT = FHIT (|pred-target| <= fhit_frames)  OR  RHIT
# (|log((pred+1)/(target+1))| < log(100 / (100 - rhit_pct))).
# STOP-target or STOP-pred rows are excluded — the context artifacts
# diagnose non-STOP prediction quality only.


def _hit_mask(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    stop_idx: int,
    fhit_frames: int,
    log_rhit: float,
) -> np.ndarray:
    valid = (target != stop_idx) & (pred != stop_idx)
    p = pred.astype(np.int64)
    t = target.astype(np.int64)
    diff = np.abs(p - t)
    fhit = diff <= fhit_frames
    log_ratio = np.abs(
        np.log((p + 1.0) / np.maximum(t + 1.0, 1.0))
    )
    rhit = log_ratio < log_rhit
    return valid & (fhit | rhit)


def _prev_gap(batch: MetricInput) -> np.ndarray:
    """Bins between the two most-recent real past onsets, per sample.

    The sampler places the cursor *on* the most-recent past onset
    (``cursor_offset == 0`` for that slot), so "cursor → last onset"
    is useless as a rhythmic gap. The useful quantity is the gap
    between the last two past onsets: ``offset[-1] - offset[-2]`` with
    both offsets <= 0, which gives a positive bin count.

    Returns ``-1`` for samples with fewer than two real past events
    (ratio / metronome artifacts exclude those rows).
    """
    inp = batch.input
    if inp is None or not hasattr(inp, "event_offsets"):
        raise ValueError(
            "context artifacts require MetricInput.input with "
            "event_offsets (EventEmbeddingInput)"
        )
    offsets = inp.event_offsets.detach().cpu().numpy()   # (B, C), <= 0
    mask = inp.event_mask.detach().cpu().numpy()         # True = padded
    real = ~mask
    B, C = offsets.shape
    prev_gap = np.full(B, -1, dtype=np.int64)

    # For each row take the two largest offsets (closest to zero) among
    # real slots. Padded slots get -inf so they never win the argsort.
    masked_vals = np.where(real, offsets, -(10 ** 18))
    order = np.argsort(masked_vals, axis=1)              # ascending
    # Last two columns of `order` are the indices of the two largest.
    last_idx = order[:, -1]
    second_idx = order[:, -2] if C >= 2 else order[:, -1]
    last_off = np.take_along_axis(offsets, last_idx[:, None], axis=1).squeeze(1)
    second_off = np.take_along_axis(offsets, second_idx[:, None], axis=1).squeeze(1)
    last_real = np.take_along_axis(real, last_idx[:, None], axis=1).squeeze(1)
    second_real = np.take_along_axis(real, second_idx[:, None], axis=1).squeeze(1)

    valid = last_real & second_real & (C >= 2)
    if valid.any():
        gap = last_off - second_off                      # >= 0
        prev_gap[valid] = np.maximum(gap[valid], 0)
    return prev_gap


class RatioHitArtifact:
    """HIT rate bucketed by ``target / prev_gap`` ratio.

    Buckets log-centered around common rhythmic ratios:
    ``~0.5x``, ``~0.67x``, ``~1.0x``, ``~1.33x``, ``~2.0x``, ``>2.5x``.
    Samples without a prior event are excluded (ratio undefined).

    Saves:
      - ``{eval_dir}/ratio_hit.png``
      - ``{eval_dir}/ratio_hit.npz``  (``bucket_totals`` + ``bucket_hits``)
    """
    name = "ratio_hit"
    _BUCKETS = (
        ("~0.5x",  0.40,  0.60),
        ("~0.67x", 0.60,  0.83),
        ("~1.0x",  0.83,  1.20),
        ("~1.33x", 1.20,  1.66),
        ("~2.0x",  1.66,  2.50),
        (">2.5x",  2.50,  float("inf")),
    )

    def __init__(
        self, b_pred: int, *,
        fhit_frames: int = 2, rhit_pct: float = 3.0,
    ):
        if b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {b_pred}")
        import math
        self._stop_idx = b_pred
        self._fhit_frames = fhit_frames
        self._log_rhit = math.log(100.0 / (100.0 - rhit_pct))
        self.reset()

    def reset(self) -> None:
        self._totals = np.zeros(len(self._BUCKETS), dtype=np.int64)
        self._hits = np.zeros(len(self._BUCKETS), dtype=np.int64)

    def update(self, batch: MetricInput) -> None:
        pred = batch.output.logits.argmax(dim=-1).detach().cpu().numpy()
        target = batch.target.target_bin.detach().cpu().numpy()
        prev_gap = _prev_gap(batch)
        valid = (
            (target != self._stop_idx)
            & (pred != self._stop_idx)
            & (prev_gap > 0)
        )
        if not valid.any():
            return
        hit = _hit_mask(
            pred, target,
            stop_idx=self._stop_idx,
            fhit_frames=self._fhit_frames,
            log_rhit=self._log_rhit,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (target + 1).astype(np.float64) / np.maximum(
                prev_gap, 1,
            ).astype(np.float64)
        for i, (_, lo, hi) in enumerate(self._BUCKETS):
            sel = valid & (ratio >= lo) & (ratio < hi)
            self._totals[i] += int(sel.sum())
            self._hits[i] += int((sel & hit).sum())

    def save(self, eval_dir: Path, *, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        labels = [b[0] for b in self._BUCKETS]
        rates = np.where(
            self._totals > 0,
            self._hits / np.maximum(self._totals, 1),
            0.0,
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(labels))
        bars = ax.bar(x, rates, color="#6bc46d", edgecolor="none")
        for bar, n in zip(bars, self._totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"n={int(n):,}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("target / prev_gap (ratio bucket)")
        ax.set_ylabel("HIT rate (FHIT or RHIT)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"HIT rate by tempo ratio - step {step:,}   "
            f"({int(self._totals.sum()):,} non-STOP samples with prior event)"
        )
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=150)
        plt.close(fig)
        np.savez(
            eval_dir / f"{self.name}.npz",
            bucket_totals=self._totals, bucket_hits=self._hits,
        )


class MetronomeHitArtifact:
    """HIT rate split metronome vs anti-metronome.

    A sample is "metronome" when the target gap repeats the previous
    gap within ``metronome_pct`` (log-ratio). Anti-metronome is any
    other non-STOP pair. Catches the trivial failure mode where the
    model just parrots the last gap — metronome rises near 1.0 while
    anti-metronome stays near chance.

    Saves:
      - ``{eval_dir}/metronome.png``
      - ``{eval_dir}/metronome.npz``  (``bucket_totals`` + ``bucket_hits``)
    """
    name = "metronome"

    def __init__(
        self, b_pred: int, *,
        fhit_frames: int = 2, rhit_pct: float = 3.0,
        metronome_pct: float = 3.0,
    ):
        if b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {b_pred}")
        import math
        self._stop_idx = b_pred
        self._fhit_frames = fhit_frames
        self._log_rhit = math.log(100.0 / (100.0 - rhit_pct))
        self._log_metronome = math.log(100.0 / (100.0 - metronome_pct))
        self.reset()

    def reset(self) -> None:
        self._totals = np.zeros(2, dtype=np.int64)
        self._hits = np.zeros(2, dtype=np.int64)

    def update(self, batch: MetricInput) -> None:
        pred = batch.output.logits.argmax(dim=-1).detach().cpu().numpy()
        target = batch.target.target_bin.detach().cpu().numpy()
        prev_gap = _prev_gap(batch)
        valid = (
            (target != self._stop_idx)
            & (pred != self._stop_idx)
            & (prev_gap > 0)
        )
        if not valid.any():
            return
        hit = _hit_mask(
            pred, target,
            stop_idx=self._stop_idx,
            fhit_frames=self._fhit_frames,
            log_rhit=self._log_rhit,
        )
        log_ratio = np.abs(np.log(
            (target + 1.0) / np.maximum(prev_gap + 1.0, 1.0)
        ))
        is_metronome = valid & (log_ratio < self._log_metronome)
        is_anti = valid & (~is_metronome)
        for i, sel in enumerate((is_anti, is_metronome)):
            self._totals[i] += int(sel.sum())
            self._hits[i] += int((sel & hit).sum())

    def save(self, eval_dir: Path, *, step: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eval_dir = Path(eval_dir)
        eval_dir.mkdir(parents=True, exist_ok=True)

        rates = np.where(
            self._totals > 0,
            self._hits / np.maximum(self._totals, 1),
            0.0,
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(2)
        labels = ("anti-metronome", "metronome")
        bars = ax.bar(x, rates, color=("#e86850", "#4a90d9"), edgecolor="none")
        for bar, n in zip(bars, self._totals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"n={int(n):,}",
                ha="center", va="bottom", fontsize=10,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("HIT rate (FHIT or RHIT)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"Metronome vs anti-metronome HIT - step {step:,}   "
            f"(total {int(self._totals.sum()):,} non-STOP w/ prior event)"
        )
        ax.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(eval_dir / f"{self.name}.png", dpi=150)
        plt.close(fig)
        np.savez(
            eval_dir / f"{self.name}.npz",
            bucket_totals=self._totals, bucket_hits=self._hits,
        )
