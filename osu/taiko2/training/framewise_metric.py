"""Eval-pass metric for framewise detectors (#017+).

Accumulates across batches during a single ``_run_eval`` pass and
emits per-threshold x per-tolerance mini-chart comparison metrics
(via ``gt_match_metrics`` from ``domain.chart``), plus frame-level
F1 / AUC / confidence-commitment scalars.

The mini-chart comparison converts predicted and GT bin offsets to
milliseconds and feeds them to the same matching function the AR
corpus uses, so the numbers are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..domain.chart import gt_match_metrics
from ..domain.metrics import Metric, MetricConfig, MetricInput
from .framewise_curve_metrics import (
    compute_auc_pr,
    compute_auc_roc,
    compute_frame_f1_at_tolerance,
    compute_separation,
)


_DEFAULT_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
_DEFAULT_TOLERANCES_MS: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 100.0)
_NMS_KERNEL: int = 3


@dataclass(frozen=True, slots=True)
class FramewiseMetricConfig(MetricConfig):
    bins_to_ms: float = 5.0
    canonical_threshold: float = 0.5
    canonical_tolerance_frames: int = 2
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS
    tolerances_ms: tuple[float, ...] = _DEFAULT_TOLERANCES_MS
    nms_kernel: int = _NMS_KERNEL


class FramewiseMetric(Metric):
    """Accumulates framewise + mini-chart metrics across an eval pass."""

    def __init__(self, config: FramewiseMetricConfig):
        self.config = config
        self.reset()

    @property
    def name(self) -> str:
        return "frame"

    # ── accumulators ──────────────────────────────────────────────────

    def reset(self) -> None:
        self._n_samples = 0
        # Frame-level TP/FP/FN/TN at canonical threshold+tolerance.
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        # Confidence-by-outcome reservoir (capped).
        self._conf_tp: list[float] = []
        self._conf_fn: list[float] = []
        self._conf_fp: list[float] = []
        self._conf_tn: list[float] = []
        self._max_reservoir = 50_000
        # Running sums for AUC/separation.
        self._auc_pr_sum = 0.0
        self._auc_roc_sum = 0.0
        self._sep_sum = 0.0
        self._mean_pos_sum = 0.0
        self._mean_neg_sum = 0.0
        self._n_auc_batches = 0
        # Hedging / Brier.
        self._hedge_sum = 0.0
        self._brier_sum = 0.0
        self._n_bins_total = 0
        # Mini-chart: per (threshold, tolerance) running sums.
        cfg = self.config
        self._mc_sums: dict[str, float] = {}
        self._mc_counts: dict[str, int] = {}
        for tau in cfg.thresholds:
            tau_key = _tau_key(tau)
            for metric_name in _MC_METRIC_NAMES:
                k = f"mini/τ{tau_key}/{metric_name}"
                self._mc_sums[k] = 0.0
                self._mc_counts[k] = 0

    # ── update ────────────────────────────────────────────────────────

    def _extract(
        self, batch: MetricInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return ``(confidence_map, target_binary, gt_bins_padded)``
        or ``None`` if the batch isn't framewise."""
        output = batch.output
        target = batch.target
        conf = getattr(output, "confidence_map", None)
        if conf is None:
            logits = getattr(output, "logits", None)
            if logits is None or logits.dim() != 2:
                return None
            conf = torch.sigmoid(logits).clamp(0.0, 1.0)
        tb = getattr(target, "target_map_binary", None)
        gt_bins = getattr(target, "gt_bins_padded", None)
        if tb is None or gt_bins is None:
            return None
        return conf.detach(), tb.detach(), gt_bins.detach()

    def update(self, batch: MetricInput) -> None:
        got = self._extract(batch)
        if got is None:
            return
        conf, tb, gt_bins = got
        cfg = self.config
        B, n_bins = conf.shape
        self._n_samples += B

        # ── frame-level at canonical threshold+tolerance ─────────────
        pred_pos = conf > cfg.canonical_threshold
        tol = cfg.canonical_tolerance_frames
        if tol > 0:
            gt_dilated = F.max_pool1d(
                tb.unsqueeze(1),
                kernel_size=2 * tol + 1, stride=1, padding=tol,
            ).squeeze(1) > 0.5
            pred_dilated = F.max_pool1d(
                pred_pos.float().unsqueeze(1),
                kernel_size=2 * tol + 1, stride=1, padding=tol,
            ).squeeze(1) > 0.5
        else:
            gt_dilated = tb > 0.5
            pred_dilated = pred_pos

        gt_mask = tb > 0.5
        self._tp += int((pred_pos & gt_dilated).sum().item())
        self._fp += int((pred_pos & ~gt_dilated).sum().item())
        self._fn += int((~pred_dilated & gt_mask).sum().item())
        self._tn += int((~pred_pos & ~gt_mask).sum().item())

        # ── confidence by outcome (reservoir) ────────────────────────
        tp_vals = conf[pred_pos & gt_dilated]
        fn_vals = conf[~pred_dilated & gt_mask]
        fp_vals = conf[pred_pos & ~gt_dilated]
        tn_vals = conf[~pred_pos & ~gt_mask]
        self._reservoir_extend(self._conf_tp, tp_vals)
        self._reservoir_extend(self._conf_fn, fn_vals)
        self._reservoir_extend(self._conf_fp, fp_vals)
        self._reservoir_extend(self._conf_tn, tn_vals)

        # ── AUC / separation (per-batch, averaged later) ────────────
        self._auc_pr_sum += compute_auc_pr(conf, tb)
        self._auc_roc_sum += compute_auc_roc(conf, tb)
        mean_pos, mean_neg, sep = compute_separation(conf, tb)
        self._sep_sum += sep
        self._mean_pos_sum += mean_pos
        self._mean_neg_sum += mean_neg
        self._n_auc_batches += 1

        # ── hedging / Brier ──────────────────────────────────────────
        n_bins_batch = B * n_bins
        self._hedge_sum += float(
            ((conf > 0.2) & (conf < 0.8)).float().sum().item()
        )
        self._brier_sum += float(((conf - tb) ** 2).sum().item())
        self._n_bins_total += n_bins_batch

        # ── mini-chart comparison at each threshold ──────────────────
        for tau in cfg.thresholds:
            self._mini_chart_at_threshold(conf, gt_bins, tau, n_bins)

    def _reservoir_extend(
        self, reservoir: list[float], vals: torch.Tensor,
    ) -> None:
        if vals.numel() == 0:
            return
        remaining = self._max_reservoir - len(reservoir)
        if remaining <= 0:
            return
        v = vals.cpu().numpy().ravel()
        if len(v) <= remaining:
            reservoir.extend(v.tolist())
        else:
            reservoir.extend(v[:remaining].tolist())

    def _mini_chart_at_threshold(
        self,
        conf: torch.Tensor,
        gt_bins_padded: torch.Tensor,
        tau: float,
        n_bins: int,
    ) -> None:
        cfg = self.config
        tau_key = _tau_key(tau)
        B = conf.size(0)
        nms_k = cfg.nms_kernel

        for i in range(B):
            scores = conf[i]
            # NMS.
            if nms_k > 1:
                pooled = F.max_pool1d(
                    scores.view(1, 1, -1),
                    kernel_size=nms_k, stride=1, padding=nms_k // 2,
                ).view(-1)
                local_max = scores >= pooled - 1e-9
            else:
                local_max = torch.ones_like(scores, dtype=torch.bool)
            pred_bins = (scores > tau) & local_max
            pred_idx = pred_bins.nonzero(as_tuple=True)[0].cpu().numpy()

            gt_valid = gt_bins_padded[i]
            gt_idx = gt_valid[gt_valid >= 0].cpu().numpy().astype(np.float64)

            pred_ms = pred_idx.astype(np.float64) * cfg.bins_to_ms
            gt_ms = gt_idx * cfg.bins_to_ms

            mc = gt_match_metrics(
                pred_ms, gt_ms, tolerances_ms=cfg.tolerances_ms,
            )
            for metric_name in _MC_METRIC_NAMES:
                k = f"mini/τ{tau_key}/{metric_name}"
                val = mc.get(metric_name, 0.0)
                self._mc_sums[k] = self._mc_sums.get(k, 0.0) + val
                self._mc_counts[k] = self._mc_counts.get(k, 0) + 1

    # ── compute ───────────────────────────────────────────────────────

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        cfg = self.config
        tau_pct = int(round(cfg.canonical_threshold * 100))
        tol = cfg.canonical_tolerance_frames

        # Frame precision / recall / F1 at canonical.
        tp, fp, fn = self._tp, self._fp, self._fn
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[f"frame/precision_τ_{tau_pct}_tol_{tol}"] = prec
        out[f"frame/recall_τ_{tau_pct}_tol_{tol}"] = rec
        out[f"frame/f1_τ_{tau_pct}_tol_{tol}"] = f1

        # AUC / separation (batch-averaged).
        n_b = max(self._n_auc_batches, 1)
        out["frame/auc_pr"] = self._auc_pr_sum / n_b
        out["frame/auc_roc"] = self._auc_roc_sum / n_b
        out["frame/separation"] = self._sep_sum / n_b
        out["frame/mean_act_pos"] = self._mean_pos_sum / n_b
        out["frame/mean_act_neg"] = self._mean_neg_sum / n_b

        # Hedging / Brier.
        n_total = max(self._n_bins_total, 1)
        out["frame/pred_hedge_frac"] = self._hedge_sum / n_total
        out["frame/brier"] = self._brier_sum / n_total

        # ECE (10-bin calibration from the outcome reservoirs).
        out["frame/ece"] = self._compute_ece()

        # Confidence by outcome.
        out["frame/conf_tp_median"] = _np_median(self._conf_tp)
        out["frame/conf_fn_median"] = _np_median(self._conf_fn)
        out["frame/conf_fp_median"] = _np_median(self._conf_fp)
        out["frame/conf_tn_median"] = _np_median(self._conf_tn)
        out["frame/conf_tp_p10"] = _np_percentile(self._conf_tp, 10)
        out["frame/conf_fp_p90"] = _np_percentile(self._conf_fp, 90)

        # Mini-chart.
        for k, s in self._mc_sums.items():
            c = max(self._mc_counts.get(k, 1), 1)
            out[f"frame/{k}"] = s / c

        return out

    def _compute_ece(self) -> float:
        all_conf = self._conf_tp + self._conf_fn + self._conf_fp + self._conf_tn
        all_correct = (
            [1.0] * len(self._conf_tp)
            + [0.0] * len(self._conf_fn)
            + [0.0] * len(self._conf_fp)
            + [1.0] * len(self._conf_tn)
        )
        if not all_conf:
            return 0.0
        conf_arr = np.array(all_conf)
        corr_arr = np.array(all_correct)
        n_bins_cal = 10
        ece = 0.0
        total = len(conf_arr)
        for b in range(n_bins_cal):
            lo = b / n_bins_cal
            hi = (b + 1) / n_bins_cal
            mask = (conf_arr >= lo) & (conf_arr < hi)
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            avg_conf = float(conf_arr[mask].mean())
            avg_acc = float(corr_arr[mask].mean())
            ece += (n_in / total) * abs(avg_conf - avg_acc)
        return ece


# ─────────────────────────── helpers ─────────────────────────────────


_MC_METRIC_NAMES: tuple[str, ...] = (
    "matched_rate", "close_rate", "far_rate", "hallucination_rate",
    "error_mean_ms", "error_median_ms", "density_ratio",
    "matched_rate_at_tol_5", "halluc_rate_at_tol_5",
    "matched_rate_at_tol_10", "halluc_rate_at_tol_10",
    "matched_rate_at_tol_25", "halluc_rate_at_tol_25",
    "matched_rate_at_tol_50", "halluc_rate_at_tol_50",
    "matched_rate_at_tol_100", "halluc_rate_at_tol_100",
)


def _tau_key(tau: float) -> str:
    return str(int(round(tau * 100)))


def _np_median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return float(np.median(vals))


def _np_percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(vals, pct))
