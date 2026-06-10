"""Per-eval AR chart-level evaluation for the typing model.

Runs the typing model autoregressively over full val charts using the
shared `inference.typing_pass.type_chart` function, then compares
predicted D/K + BIG sequences against GT.

Installed as a pre_hook so its scalars merge into val_metrics before
MetricLoggerHook / MetricCurvesHook capture them.
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data_samplers.typing import (
    IDX_BDON,
    IDX_BKA,
    IDX_DON,
    IDX_KA,
    TypingSampler,
)
from ..domain.beatmap import OnsetBinned, OnsetKind
from ..domain.chart import Chart
from ..domain.training import RunSpec, TrainerHook, TrainingState
from ..inference.typing_pass import type_chart


def _kind_to_dk(kind_id: int) -> int:
    return 0 if kind_id in (IDX_DON, IDX_BDON) else 1


def _kind_to_big(kind_id: int) -> int:
    return 1 if kind_id in (IDX_BDON, IDX_BKA) else 0


# ─────────────────────────── comparison metrics ──────────────────────

def _ngram_dist(seq: np.ndarray, n: int) -> Counter:
    c: Counter = Counter()
    for i in range(len(seq) - n + 1):
        c[tuple(int(x) for x in seq[i:i + n])] += 1
    return c


def _tvd(c1: Counter, c2: Counter) -> float:
    all_keys = set(c1) | set(c2)
    t1 = sum(c1.values()) or 1
    t2 = sum(c2.values()) or 1
    return 0.5 * sum(abs(c1[k] / t1 - c2[k] / t2) for k in all_keys)


def _alternation_rate(seq: np.ndarray) -> float:
    if len(seq) < 2:
        return 0.0
    return int(np.sum(seq[:-1] != seq[1:])) / (len(seq) - 1)


def _run_lengths(seq: np.ndarray) -> list[int]:
    if len(seq) == 0:
        return []
    runs = []
    current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return runs


def _binary_prf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((pred == 1) & (gt == 1)))
    fp = float(np.sum((pred == 1) & (gt == 0)))
    fn = float(np.sum((pred == 0) & (gt == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _pattern_match(pred: np.ndarray, gt: np.ndarray, w: int) -> float:
    if len(pred) < w:
        return 0.0
    n_windows = len(pred) - w + 1
    matches = 0
    for i in range(n_windows):
        p = pred[i:i + w]
        g = gt[i:i + w]
        if np.array_equal(p, g) or np.array_equal(1 - p, g):
            matches += 1
    return matches / n_windows


def _transition_probs(seq: np.ndarray) -> np.ndarray:
    mat = np.zeros((2, 2), dtype=np.float64)
    for i in range(len(seq) - 1):
        mat[int(seq[i]), int(seq[i + 1])] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums


def compare_chart_types(
    pred_dk: np.ndarray, pred_big: np.ndarray,
    gt_dk: np.ndarray, gt_big: np.ndarray,
) -> dict[str, float]:
    n = len(pred_dk)
    if n == 0:
        return {}

    raw_acc = float(np.mean(pred_dk == gt_dk))
    flipped_acc = float(np.mean((1 - pred_dk) == gt_dk))
    sym_acc = max(raw_acc, flipped_acc)

    pm4 = _pattern_match(pred_dk, gt_dk, 4)
    pm8 = _pattern_match(pred_dk, gt_dk, 8)
    ng2_tvd = _tvd(_ngram_dist(pred_dk, 2), _ngram_dist(gt_dk, 2))
    ng4_tvd = _tvd(_ngram_dist(pred_dk, 4), _ngram_dist(gt_dk, 4))

    pred_trans = _transition_probs(pred_dk)
    gt_trans = _transition_probs(gt_dk)
    trans_tvd = 0.5 * float(np.sum(np.abs(pred_trans - gt_trans)))

    alt_pred = _alternation_rate(pred_dk)
    alt_gt = _alternation_rate(gt_dk)

    pred_runs = Counter(_run_lengths(pred_dk))
    gt_runs = Counter(_run_lengths(gt_dk))
    rl_tvd = _tvd(pred_runs, gt_runs)

    _, _, f1_d = _binary_prf(pred_dk, gt_dk)
    _, _, f1_k = _binary_prf(1 - pred_dk, 1 - gt_dk)

    str_acc = float(np.mean(pred_big == gt_big))
    str_prec, str_rec, str_f1 = _binary_prf(pred_big, gt_big)

    return {
        "type_accuracy": raw_acc,
        "type_accuracy_sym": sym_acc,
        "type_f1_D": f1_d,
        "type_f1_K": f1_k,
        "type_pattern_match_4": pm4,
        "type_pattern_match_8": pm8,
        "type_ngram_tvd_2": ng2_tvd,
        "type_ngram_tvd_4": ng4_tvd,
        "type_transition_tvd": trans_tvd,
        "type_alternation_rate_pred": alt_pred,
        "type_alternation_rate_gt": alt_gt,
        "type_alternation_rate_delta": abs(alt_pred - alt_gt),
        "type_run_length_tvd": rl_tvd,
        "strength_accuracy": str_acc,
        "strength_precision_BIG": str_prec,
        "strength_recall_BIG": str_rec,
        "strength_f1_BIG": str_f1,
        "big_ratio_pred": float(np.mean(pred_big)),
        "big_ratio_gt": float(np.mean(gt_big)),
        "big_ratio_delta": abs(float(np.mean(pred_big)) - float(np.mean(gt_big))),
        "n_events": n,
    }


# ─────────────────────────── helper ──────────────────────────────────

def _dist_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.array(values)
    return {
        "mean": round(float(np.mean(a)), 4),
        "std": round(float(np.std(a)), 4),
        "p5": round(float(np.percentile(a, 5)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
    }


def _build_stub_chart(
    bins: np.ndarray, kind_ids: np.ndarray,
) -> Chart:
    """Build a Chart from raw bins + kinds for the typing pass."""
    from ..domain.beatmap import AudioRef, Difficulty, Track
    from ..parsing.osu import compute_density

    onsets = tuple(
        OnsetBinned(time_ms=int(round(b * 5.0)), kind=OnsetKind.DON, bin=int(b))
        for b in bins
    )
    density = compute_density(onsets)
    track = Track(
        beatmap_id="0", beatmapset_id="0", artist="", title="",
        difficulty=Difficulty(version="", overall_difficulty=0.0),
        audio=AudioRef(filename="", format=""),
        onsets=onsets, density=density,
    )
    return Chart(track=track, audio=None)


# ─────────────────────────── hook ────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TypingARHookConfig:
    n_charts: int = 100
    every_n_evals: int = 2
    seed: int = 42
    strength_threshold: float = 0.8


class TypingARHook(TrainerHook):
    """Runs AR typing evaluation on val charts after every N evals.

    Uses `inference.typing_pass.type_chart` for the AR loop — the same
    code path that `cli/infer.py` and `inference/corpus.py` use, so
    training-time AR eval matches inference-time behavior exactly.
    """

    def __init__(
        self,
        *,
        config: TypingARHookConfig,
        spec: RunSpec,
        model: torch.nn.Module,
        val_sampler: TypingSampler,
        device: torch.device | str = "cpu",
    ):
        self._config = config
        self._spec = spec
        self._model = model
        self._sampler = val_sampler
        self._device = torch.device(device) if isinstance(device, str) else device
        self._eval_count = 0

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        self._eval_count += 1
        if self._eval_count % self._config.every_n_evals != 0:
            return

        cfg = self._config
        out_dir = self._spec.run_dir / "typing_ar" / f"eval_{state.step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_charts = min(cfg.n_charts, len(self._sampler._chart_ids))
        rng = np.random.default_rng(cfg.seed)
        chart_indices = rng.choice(
            len(self._sampler._chart_ids), size=n_charts, replace=False,
        )

        self._model.eval()
        t0 = time.time()

        str_thr = val_metrics.get(
            "typing/strength/best_threshold", cfg.strength_threshold,
        )

        per_chart: list[dict[str, Any]] = []
        try:
            from tqdm.auto import tqdm
            chart_iter = tqdm(chart_indices, desc="AR typing", unit="chart")
        except ImportError:
            chart_iter = chart_indices

        for ci in chart_iter:
            ci = int(ci)
            bins = self._sampler._event_bins[ci]
            kind_ids = self._sampler._event_kind_ids[ci]
            features = self._sampler._features[ci]

            if len(bins) < 4:
                continue

            gt_dk = np.array([_kind_to_dk(int(k)) for k in kind_ids])
            gt_big = np.array([_kind_to_big(int(k)) for k in kind_ids])

            # Build a stub chart with DON-only onsets, then type it
            stub = _build_stub_chart(bins, kind_ids)
            typed = type_chart(
                self._model, stub, features,
                device=self._device,
                strength_threshold=str_thr,
            )

            # Extract predicted D/K and BIG from typed chart
            pred_dk = np.array([
                0 if o.kind in (OnsetKind.DON, OnsetKind.BIG_DON) else 1
                for o in typed.track.onsets
            ])
            pred_big = np.array([
                1 if o.kind in (OnsetKind.BIG_DON, OnsetKind.BIG_KA) else 0
                for o in typed.track.onsets
            ])

            metrics = compare_chart_types(pred_dk, pred_big, gt_dk, gt_big)
            metrics["chart_id"] = self._sampler._chart_ids[ci]
            per_chart.append(metrics)

        elapsed = time.time() - t0

        if not per_chart:
            return

        # Aggregate
        scalar_keys = [
            k for k in per_chart[0]
            if isinstance(per_chart[0][k], (int, float))
        ]
        agg: dict[str, Any] = {
            "n_charts": len(per_chart),
            "elapsed_s": round(elapsed, 1),
            "strength_threshold_used": str_thr,
        }
        for key in scalar_keys:
            vals = [r[key] for r in per_chart if key in r]
            agg[key] = _dist_stats(vals)

        # Save
        with open(out_dir / "ar_real_summary.json", "w") as f:
            json.dump(agg, f, indent=2)

        csv_path = out_dir / "ar_real_per_chart.csv"
        fieldnames = list(per_chart[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_chart)

        # Merge into val_metrics
        for key in scalar_keys:
            if key == "n_events":
                continue
            stats = agg[key]
            if isinstance(stats, dict) and "mean" in stats:
                val_metrics[f"ar/{key}_mean"] = stats["mean"]

        self._plot_results(per_chart, out_dir)

    def _plot_results(
        self, per_chart: list[dict], out_dir: Path,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Type accuracy histogram
        accs = [r["type_accuracy_sym"] for r in per_chart]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(accs, bins=30, alpha=0.7, color="steelblue")
        ax.axvline(np.mean(accs), color="red", linestyle="--",
                    label=f"mean={np.mean(accs):.3f}")
        ax.axvline(0.62, color="gray", linestyle=":",
                    label="alternation baseline=0.62")
        ax.set_xlabel("Type accuracy (sym)")
        ax.set_ylabel("Charts")
        ax.set_title(f"AR type accuracy (n={len(accs)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "ar_type_accuracy_hist.png", dpi=150)
        plt.close()

        # Alternation scatter
        alt_p = [r["type_alternation_rate_pred"] for r in per_chart]
        alt_g = [r["type_alternation_rate_gt"] for r in per_chart]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(alt_g, alt_p, alpha=0.4, s=15)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("GT alternation rate")
        ax.set_ylabel("Predicted alternation rate")
        ax.set_title("AR alternation: pred vs GT")
        ax.set_xlim(0.3, 0.9)
        ax.set_ylim(0.3, 0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "ar_alternation_scatter.png", dpi=150)
        plt.close()

        # Strength F1 histogram
        f1s = [r["strength_f1_BIG"] for r in per_chart]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(f1s, bins=30, alpha=0.7, color="tab:orange")
        ax.axvline(np.mean(f1s), color="red", linestyle="--",
                    label=f"mean={np.mean(f1s):.3f}")
        ax.set_xlabel("Strength F1 (BIG)")
        ax.set_ylabel("Charts")
        ax.set_title("AR strength F1 distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "ar_strength_f1_hist.png", dpi=150)
        plt.close()

        # N-gram TVD histogram
        tvd4 = [r["type_ngram_tvd_4"] for r in per_chart]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(tvd4, bins=30, alpha=0.7, color="steelblue")
        ax.axvline(np.mean(tvd4), color="red", linestyle="--",
                    label=f"mean={np.mean(tvd4):.3f}")
        ax.set_xlabel("4-gram TVD (pred vs GT)")
        ax.set_ylabel("Charts")
        ax.set_title("AR 4-gram distribution distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "ar_ngram_tvd_hist.png", dpi=150)
        plt.close()
