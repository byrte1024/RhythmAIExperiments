"""Curve / threshold-free metrics for framewise diffusion (#016).

Pure functions + ``CurvesResult`` dataclass. No ``nn.Module``. All
PyTorch, vectorized.

Operates on ``(B, n_bins)`` predicted activation maps in ``[0, 1]``
and ``(B, n_bins)`` binary GT maps in ``{0, 1}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────── CurvesResult ─────────────────────────────


@dataclass(frozen=True, slots=True)
class CurvesResult:
    """Sweep curves at 101 thresholds × 5 tolerances + scalar summaries."""
    thresholds: torch.Tensor          # (101,) in [0, 1]
    tolerances_frames: torch.Tensor   # (5,) [1, 2, 4, 8, 20]

    # framewise (no tolerance)
    precision_curve: torch.Tensor     # (101,)
    recall_curve: torch.Tensor        # (101,)
    f1_curve: torch.Tensor            # (101,)
    pos_rate_pred_curve: torch.Tensor # (101,)

    # framewise + tolerance
    precision_tol_curve: torch.Tensor # (5, 101)
    recall_tol_curve: torch.Tensor    # (5, 101)
    f1_tol_curve: torch.Tensor        # (5, 101)

    # threshold-free
    auc_pr: float
    auc_roc: float
    mean_act_pos: float
    mean_act_neg: float
    separation: float


# ─────────────────────────── defaults ─────────────────────────────────


def default_thresholds(
    n: int = 101, device: torch.device | None = None,
) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n, device=device)


def default_tolerances_frames(
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor([1, 2, 4, 8, 20], dtype=torch.long, device=device)


# ─────────────────────────── per-bin curves ───────────────────────────


def compute_per_bin_curves(
    M_0_hat: torch.Tensor,
    target_binary: torch.Tensor,
    thresholds: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-threshold framewise precision / recall / F1 / predicted-positive rate.

    No tolerance: a TP requires ``pred[b] > τ AND target[b] == 1`` for
    the same bin. Returns dict with shape (T,) per curve, where T is
    ``thresholds.numel()``.
    """
    if thresholds is None:
        thresholds = default_thresholds(device=M_0_hat.device)
    # Flatten over batch + bins.
    pred = M_0_hat.reshape(-1)                       # (B * n_bins,)
    gt = target_binary.reshape(-1).bool()            # (B * n_bins,)
    n_pos = int(gt.sum().item())
    n_total = int(gt.numel())

    T = thresholds.numel()
    precision = torch.zeros(T, device=M_0_hat.device)
    recall = torch.zeros(T, device=M_0_hat.device)
    f1 = torch.zeros(T, device=M_0_hat.device)
    pos_rate_pred = torch.zeros(T, device=M_0_hat.device)

    for i, thr in enumerate(thresholds.tolist()):
        pred_pos = pred > thr
        n_pred = int(pred_pos.sum().item())
        tp = int((pred_pos & gt).sum().item())
        fp = n_pred - tp
        fn = n_pos - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[i] = p
        recall[i] = r
        f1[i] = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
        pos_rate_pred[i] = n_pred / max(1, n_total)

    return {
        "precision_curve": precision,
        "recall_curve": recall,
        "f1_curve": f1,
        "pos_rate_pred_curve": pos_rate_pred,
    }


# ─────────────────────────── tolerance-aware ────────────────────────


def _dilate(target_binary: torch.Tensor, k: int) -> torch.Tensor:
    """Dilate (B, n_bins) binary target by ±k frames via max-pool."""
    if k <= 0:
        return target_binary.float()
    return F.max_pool1d(
        target_binary.float().unsqueeze(1),
        kernel_size=2 * k + 1, stride=1, padding=k,
    ).squeeze(1)


def compute_per_bin_curves_at_tolerance(
    M_0_hat: torch.Tensor,
    target_binary: torch.Tensor,
    tolerances_frames: torch.Tensor | None = None,
    thresholds: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Tolerance-aware framewise P/R/F1 curves over (5, 101) grid.

    For each (τ, k):
      - precision = (# predicted positives with a GT within ±k) / (# predicted positives)
      - recall    = (# GT positives with a prediction within ±k) / (# GT positives)
      - f1        = harmonic mean
    """
    if thresholds is None:
        thresholds = default_thresholds(device=M_0_hat.device)
    if tolerances_frames is None:
        tolerances_frames = default_tolerances_frames(device=M_0_hat.device)

    K = tolerances_frames.numel()
    T = thresholds.numel()
    precision = torch.zeros(K, T, device=M_0_hat.device)
    recall = torch.zeros(K, T, device=M_0_hat.device)
    f1 = torch.zeros(K, T, device=M_0_hat.device)

    gt = target_binary.float()
    n_pos = float(gt.sum().item())

    for ki, k in enumerate(tolerances_frames.tolist()):
        dilated_gt = _dilate(target_binary, int(k))            # (B, n_bins)
        for ti, thr in enumerate(thresholds.tolist()):
            pred_pos = (M_0_hat > thr).float()                  # (B, n_bins)
            dilated_pred = _dilate(pred_pos, int(k))            # (B, n_bins)
            n_pred = float(pred_pos.sum().item())
            # Precision: of predicted positives, fraction with GT within ±k.
            tp_p = float((pred_pos * dilated_gt).sum().item())
            # Recall: of GT positives, fraction with predicted positive within ±k.
            tp_r = float((gt * dilated_pred).sum().item())
            p = tp_p / n_pred if n_pred > 0 else 0.0
            r = tp_r / n_pos if n_pos > 0 else 0.0
            precision[ki, ti] = p
            recall[ki, ti] = r
            f1[ki, ti] = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

    return {
        "precision_tol_curve": precision,
        "recall_tol_curve": recall,
        "f1_tol_curve": f1,
    }


# ─────────────────────────── threshold-free AUCs ─────────────────────


def compute_auc_pr(
    M_0_hat: torch.Tensor, target_binary: torch.Tensor,
) -> float:
    """Average-precision (AUC-PR) computed via descending-sorted ranks.

    Trapezoidal between consecutive (recall, precision) points.
    Returns 0.0 when there are no positives.
    """
    pred = M_0_hat.reshape(-1).detach().float()
    gt = target_binary.reshape(-1).detach().float()
    n_pos = float(gt.sum().item())
    if n_pos == 0.0:
        return 0.0
    order = torch.argsort(pred, descending=True)
    gt_sorted = gt[order]
    tp_cum = torch.cumsum(gt_sorted, dim=0)
    fp_cum = torch.cumsum(1.0 - gt_sorted, dim=0)
    precision = tp_cum / (tp_cum + fp_cum).clamp(min=1.0)
    recall = tp_cum / n_pos
    # Trapezoidal AUC over recall axis. Prepend (0, precision[0]).
    recall = torch.cat([torch.zeros(1, device=recall.device), recall])
    precision = torch.cat([precision[:1], precision])
    delta = recall[1:] - recall[:-1]
    avg_p = ((precision[1:] + precision[:-1]) * 0.5 * delta).sum()
    return float(avg_p.item())


def compute_auc_roc(
    M_0_hat: torch.Tensor, target_binary: torch.Tensor,
) -> float:
    """ROC-AUC via descending-sorted ranks."""
    pred = M_0_hat.reshape(-1).detach().float()
    gt = target_binary.reshape(-1).detach().float()
    n_pos = float(gt.sum().item())
    n_neg = float((1.0 - gt).sum().item())
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.0
    order = torch.argsort(pred, descending=True)
    gt_sorted = gt[order]
    tp_cum = torch.cumsum(gt_sorted, dim=0)
    fp_cum = torch.cumsum(1.0 - gt_sorted, dim=0)
    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg
    tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
    fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])
    delta = fpr[1:] - fpr[:-1]
    auc = ((tpr[1:] + tpr[:-1]) * 0.5 * delta).sum()
    return float(auc.item())


# ─────────────────────────── single-point F1 ─────────────────────────


def compute_frame_f1_at_tolerance(
    M_0_hat: torch.Tensor,
    target_binary: torch.Tensor,
    threshold: float,
    tolerance_frames: int,
) -> dict[str, float]:
    """Canonical-op-point P/R/F1 used by the loss for scalar reporting."""
    pred_pos = (M_0_hat > threshold).float()
    gt = target_binary.float()
    dilated_gt = _dilate(target_binary, int(tolerance_frames))
    dilated_pred = _dilate(pred_pos, int(tolerance_frames))
    n_pred = float(pred_pos.sum().item())
    n_pos = float(gt.sum().item())
    tp_p = float((pred_pos * dilated_gt).sum().item())
    tp_r = float((gt * dilated_pred).sum().item())
    p = tp_p / n_pred if n_pred > 0 else 0.0
    r = tp_r / n_pos if n_pos > 0 else 0.0
    f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


# ─────────────────────────── separation ──────────────────────────────


def compute_separation(
    M_0_hat: torch.Tensor, target_binary: torch.Tensor,
) -> tuple[float, float, float]:
    """``(mean_act_pos, mean_act_neg, separation)``.

    ``mean_act_pos``: average activation over GT-positive bins.
    ``mean_act_neg``: average activation over GT-negative bins.
    ``separation``: pos - neg. Larger = better-separated.
    """
    pred = M_0_hat.detach().float()
    gt = target_binary.detach().float()
    pos_mask = gt > 0.5
    neg_mask = ~pos_mask
    mean_pos = float(pred[pos_mask].mean().item()) if bool(pos_mask.any()) else 0.0
    mean_neg = float(pred[neg_mask].mean().item()) if bool(neg_mask.any()) else 0.0
    return mean_pos, mean_neg, mean_pos - mean_neg


# ─────────────────────────── aggregation ─────────────────────────────


def _stacked_mean(items: list[torch.Tensor]) -> torch.Tensor:
    if not items:
        raise ValueError("aggregate_curves: empty list")
    return torch.stack(items, dim=0).mean(dim=0)


def aggregate_curves(curves_list: list[CurvesResult]) -> CurvesResult:
    """Average curves across a list (e.g. per-eval-batch aggregation).

    Tolerances / thresholds must match across all entries. Scalar
    summaries (auc_pr, auc_roc, …) are averaged.
    """
    if not curves_list:
        raise ValueError("aggregate_curves: empty list")
    first = curves_list[0]
    return CurvesResult(
        thresholds=first.thresholds,
        tolerances_frames=first.tolerances_frames,
        precision_curve=_stacked_mean([c.precision_curve for c in curves_list]),
        recall_curve=_stacked_mean([c.recall_curve for c in curves_list]),
        f1_curve=_stacked_mean([c.f1_curve for c in curves_list]),
        pos_rate_pred_curve=_stacked_mean(
            [c.pos_rate_pred_curve for c in curves_list],
        ),
        precision_tol_curve=_stacked_mean(
            [c.precision_tol_curve for c in curves_list],
        ),
        recall_tol_curve=_stacked_mean(
            [c.recall_tol_curve for c in curves_list],
        ),
        f1_tol_curve=_stacked_mean([c.f1_tol_curve for c in curves_list]),
        auc_pr=sum(c.auc_pr for c in curves_list) / len(curves_list),
        auc_roc=sum(c.auc_roc for c in curves_list) / len(curves_list),
        mean_act_pos=sum(c.mean_act_pos for c in curves_list) / len(curves_list),
        mean_act_neg=sum(c.mean_act_neg for c in curves_list) / len(curves_list),
        separation=sum(c.separation for c in curves_list) / len(curves_list),
    )


# ─────────────────────────── npz IO ──────────────────────────────────


def save_curves_npz(path: Path | str, **curves: CurvesResult) -> None:
    """Save one or more ``CurvesResult`` to an .npz file.

    Each curves arg is stored under a prefix matching the kwarg name
    (e.g. ``eval1``, ``noaug1``). Scalar fields are stored as 0-d arrays.
    """
    path = Path(path)
    blob: dict[str, np.ndarray] = {}
    for prefix, c in curves.items():
        blob[f"{prefix}/thresholds"] = c.thresholds.cpu().numpy()
        blob[f"{prefix}/tolerances_frames"] = c.tolerances_frames.cpu().numpy()
        blob[f"{prefix}/precision_curve"] = c.precision_curve.cpu().numpy()
        blob[f"{prefix}/recall_curve"] = c.recall_curve.cpu().numpy()
        blob[f"{prefix}/f1_curve"] = c.f1_curve.cpu().numpy()
        blob[f"{prefix}/pos_rate_pred_curve"] = c.pos_rate_pred_curve.cpu().numpy()
        blob[f"{prefix}/precision_tol_curve"] = c.precision_tol_curve.cpu().numpy()
        blob[f"{prefix}/recall_tol_curve"] = c.recall_tol_curve.cpu().numpy()
        blob[f"{prefix}/f1_tol_curve"] = c.f1_tol_curve.cpu().numpy()
        blob[f"{prefix}/auc_pr"] = np.array(c.auc_pr, dtype=np.float64)
        blob[f"{prefix}/auc_roc"] = np.array(c.auc_roc, dtype=np.float64)
        blob[f"{prefix}/mean_act_pos"] = np.array(c.mean_act_pos, dtype=np.float64)
        blob[f"{prefix}/mean_act_neg"] = np.array(c.mean_act_neg, dtype=np.float64)
        blob[f"{prefix}/separation"] = np.array(c.separation, dtype=np.float64)
    np.savez(path, **blob)


def build_curves_from_batch(
    M_0_hat: torch.Tensor,
    target_binary: torch.Tensor,
) -> CurvesResult:
    """Build a full ``CurvesResult`` over a single batch.

    Convenience entry point — wires the per-bin curves, tolerance
    curves, AUCs, and separation into one struct.
    """
    thresholds = default_thresholds(device=M_0_hat.device)
    tolerances = default_tolerances_frames(device=M_0_hat.device)
    pb = compute_per_bin_curves(M_0_hat, target_binary, thresholds)
    pbt = compute_per_bin_curves_at_tolerance(
        M_0_hat, target_binary, tolerances, thresholds,
    )
    auc_pr = compute_auc_pr(M_0_hat, target_binary)
    auc_roc = compute_auc_roc(M_0_hat, target_binary)
    mean_pos, mean_neg, sep = compute_separation(M_0_hat, target_binary)
    return CurvesResult(
        thresholds=thresholds,
        tolerances_frames=tolerances,
        precision_curve=pb["precision_curve"],
        recall_curve=pb["recall_curve"],
        f1_curve=pb["f1_curve"],
        pos_rate_pred_curve=pb["pos_rate_pred_curve"],
        precision_tol_curve=pbt["precision_tol_curve"],
        recall_tol_curve=pbt["recall_tol_curve"],
        f1_tol_curve=pbt["f1_tol_curve"],
        auc_pr=auc_pr,
        auc_roc=auc_roc,
        mean_act_pos=mean_pos,
        mean_act_neg=mean_neg,
        separation=sep,
    )
