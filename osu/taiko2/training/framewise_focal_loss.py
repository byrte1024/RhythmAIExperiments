"""Focal loss for the framewise detector (#017b).

Focal modulation ``(1 - p_t)^gamma`` on top of BCE down-weights
easy-to-classify bins (the vast majority of true negatives near 0.0)
and focuses gradient on the hard cases: metronomic audio beats that
the model confidently predicts as onsets but that aren't in the GT
chart.

Positive-class upweighting (``pos_weight``) is applied independently
of the focal term to handle the ~3 % class imbalance. The two
mechanisms are complementary: ``pos_weight`` addresses class count
imbalance, focal addresses confidence-calibration imbalance.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.framewise import FramewiseTarget
from ..domain.loss import Loss, LossConfig, LossResult
from ..models.framewise_detector import FramewiseDetectorOutput
from .framewise_curve_metrics import (
    compute_auc_pr,
    compute_auc_roc,
    compute_frame_f1_at_tolerance,
    compute_separation,
)


@dataclass(frozen=True, slots=True)
class FramewiseFocalLossConfig(LossConfig):
    gamma: float = 2.0
    pos_weight_clamp_min: float = 10.0
    pos_weight_clamp_max: float = 200.0
    canonical_threshold: float = 0.5
    canonical_tolerance_frames: int = 2

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError(f"gamma must be >= 0 (got {self.gamma})")
        if self.pos_weight_clamp_min < 0.0:
            raise ValueError(
                f"pos_weight_clamp_min must be >= 0 "
                f"(got {self.pos_weight_clamp_min})"
            )
        if self.pos_weight_clamp_max < self.pos_weight_clamp_min:
            raise ValueError(
                f"pos_weight_clamp_max ({self.pos_weight_clamp_max}) "
                f"< pos_weight_clamp_min ({self.pos_weight_clamp_min})"
            )


class FramewiseFocalLoss(
    Loss[FramewiseFocalLossConfig, FramewiseDetectorOutput, FramewiseTarget],
):
    """Focal loss with positive-class upweighting."""

    def __init__(self, config: FramewiseFocalLossConfig):
        super().__init__(config)

    def forward(
        self,
        output: FramewiseDetectorOutput,
        target: FramewiseTarget,
    ) -> LossResult:
        cfg = self.config
        logits = output.logits
        target_binary = target.target_map_binary
        n_bins = logits.size(-1)

        # Per-sample pos_weight.
        n_pos = target.n_gt.float().clamp(min=1.0)
        n_neg = float(n_bins) - target.n_gt.float()
        pos_w = (n_neg / n_pos).clamp(
            min=cfg.pos_weight_clamp_min,
            max=cfg.pos_weight_clamp_max,
        )

        pos_bin = target_binary > 0.5
        weight = torch.where(
            pos_bin,
            pos_w.view(-1, 1).expand_as(logits),
            torch.ones_like(logits),
        )

        # BCE per bin (no reduction).
        bce = F.binary_cross_entropy_with_logits(
            logits, target_binary, reduction="none",
        )

        # Focal modulation: (1 - p_t)^gamma where p_t is the
        # probability assigned to the correct class.
        p = torch.sigmoid(logits)
        p_t = p * target_binary + (1.0 - p) * (1.0 - target_binary)
        focal_weight = (1.0 - p_t) ** cfg.gamma

        per_bin = focal_weight * weight * bce
        loss = per_bin.mean()

        # ── diagnostics ──────────────────────────────────────────────
        metrics: dict[str, float] = {"loss": float(loss.detach())}

        # Pos / neg decomposition (unweighted, no focal).
        unw = F.binary_cross_entropy_with_logits(
            logits, target_binary, reduction="none",
        )
        pos_loss = unw[pos_bin]
        neg_loss = unw[~pos_bin]
        pos_mean = float(pos_loss.mean().detach()) if pos_loss.numel() > 0 else 0.0
        neg_mean = float(neg_loss.mean().detach()) if neg_loss.numel() > 0 else 0.0
        metrics["loss/pos_only"] = pos_mean
        metrics["loss/neg_only"] = neg_mean
        metrics["loss/pos_neg_ratio"] = pos_mean / neg_mean if neg_mean > 0 else 0.0

        # Mean focal weight by class — diagnostic for gamma tuning.
        metrics["loss/focal_weight_pos"] = float(
            focal_weight[pos_bin].mean().detach()
        ) if pos_bin.any() else 0.0
        metrics["loss/focal_weight_neg"] = float(
            focal_weight[~pos_bin].mean().detach()
        ) if (~pos_bin).any() else 0.0

        # Frame metrics on confidence_map.
        conf = output.confidence_map.detach()
        tb = target_binary.detach()

        f1_dict = compute_frame_f1_at_tolerance(
            conf, tb,
            threshold=cfg.canonical_threshold,
            tolerance_frames=cfg.canonical_tolerance_frames,
        )
        tau_pct = int(round(cfg.canonical_threshold * 100))
        tol = cfg.canonical_tolerance_frames
        metrics[f"frame/precision_τ_{tau_pct}_tol_{tol}"] = f1_dict["precision"]
        metrics[f"frame/recall_τ_{tau_pct}_tol_{tol}"] = f1_dict["recall"]
        metrics[f"frame/f1_τ_{tau_pct}_tol_{tol}"] = f1_dict["f1"]

        metrics["frame/auc_pr"] = compute_auc_pr(conf, tb)
        metrics["frame/auc_roc"] = compute_auc_roc(conf, tb)
        mean_pos_act, mean_neg_act, sep = compute_separation(conf, tb)
        metrics["frame/mean_act_pos"] = mean_pos_act
        metrics["frame/mean_act_neg"] = mean_neg_act
        metrics["frame/separation"] = sep

        pos_rate = float(((conf > cfg.canonical_threshold).float().mean()).item())
        metrics["frame/pos_rate_pred_50"] = pos_rate
        metrics["frame/pos_rate_target"] = float(tb.float().mean().item())

        hedge = ((conf > 0.2) & (conf < 0.8)).float().mean().item()
        metrics["frame/pred_hedge_frac"] = float(hedge)

        brier = float(((conf - tb) ** 2).mean().item())
        metrics["frame/brier"] = brier

        pred_pos = conf > cfg.canonical_threshold
        tp_mask = pred_pos & (tb > 0.5)
        fn_mask = (~pred_pos) & (tb > 0.5)
        fp_mask = pred_pos & (tb < 0.5)
        tn_mask = (~pred_pos) & (tb < 0.5)
        metrics["frame/conf_tp_median"] = _median(conf[tp_mask])
        metrics["frame/conf_fn_median"] = _median(conf[fn_mask])
        metrics["frame/conf_fp_median"] = _median(conf[fp_mask])
        metrics["frame/conf_tn_median"] = _median(conf[tn_mask])

        return LossResult(loss=loss, metrics=metrics)


def _median(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    return float(t.median().item())
