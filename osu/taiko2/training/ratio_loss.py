"""Ratio-decomposed loss for the RatioDetector (taiko1 exp 67 design).

Three loss components:
  1. Divisor CE — fixed GT target (dominant gap from past events).
  2. Offset CE — fixed GT target (cursor distance from last event).
  3. Ratio CE — DYNAMIC target computed from the model's own div/off
     soft expectations: ``ratio_target = (target_bin + off_val) / div_val``.
     Mapped to the nearest log-spaced ratio bin. STOP targets map to
     the STOP class (last ratio bin).

Total: ``ratio_loss + α · div_ce + α · off_ce`` where α defaults to
0.1 (auxiliary weight). Divisor and offset gradients are stop-
gradiented from the ratio loss path (handled in the model via detach).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.loss import Loss, LossConfig, LossResult
from ..models.event_embedding import EventEmbeddingOutput, EventEmbeddingTarget
from ..models.ratio_detector import build_ratio_bin_centers


@dataclass(frozen=True, slots=True)
class RatioLossConfig(LossConfig):
    """Config for the ratio-decomposed loss."""
    divisor_bins: int = 500
    offset_bins: int = 100
    ratio_bins: int = 255
    aux_weight: float = 0.1       # weight on divisor + offset CE
    stop_weight: float = 1.5      # per-sample multiplier on STOP
    # Ratio-space RHIT/RGOOD thresholds (in log-ratio units).
    # Same definition as onset RHIT/RGOOD but applied to
    # predicted_ratio vs true_ratio.
    ratio_rhit_log: float = 0.030918  # log(100/97) ≈ 3%
    ratio_rgood_log: float = 0.10536  # log(100/90) ≈ 10%
    # Warmup: freeze ratio head for N evals (let div/off stabilize
    # first). 0 = no freeze. The training loop translates this to
    # steps via set_freeze_limit().
    ratio_freeze_evals: int = 1
    # When True, freeze the divisor/offset heads (and their val
    # embeddings) at the warmup boundary so the ratio head trains
    # against fixed aux outputs. Used by #010e.
    freeze_aux_after_warmup: bool = False


class RatioLoss(Loss[RatioLossConfig, EventEmbeddingOutput, EventEmbeddingTarget]):
    """Three-head ratio decomposition loss."""

    def __init__(self, config: RatioLossConfig):
        super().__init__(config)
        self._ratio_centers = build_ratio_bin_centers(config.ratio_bins)
        self._log_centers = torch.log(self._ratio_centers)
        self._fwd_step = 0
        self._freeze_step_limit = 0  # set by CLI via set_freeze_limit()

    def set_freeze_limit(self, eval_every: int) -> None:
        """Translate ``ratio_freeze_evals`` to a step count.

        Called by the training CLI after ``eval_every`` is known.
        """
        self._freeze_step_limit = (
            self.config.ratio_freeze_evals * eval_every
        )

    def _find_ratio_bin(self, ratio_float: torch.Tensor) -> torch.Tensor:
        """Map continuous ratio values to nearest log-spaced bin index.

        Args:
            ratio_float: (M,) positive floats.
        Returns:
            (M,) int64 bin indices in [0, ratio_bins-1].
        """
        log_r = torch.log(ratio_float.clamp(min=1e-6))
        log_c = self._log_centers.to(log_r.device)
        # (M, 1) vs (R,) → (M, R) → argmin per row.
        dists = (log_r.unsqueeze(-1) - log_c.unsqueeze(0)).abs()
        return dists.argmin(dim=-1)

    def forward(
        self,
        output: EventEmbeddingOutput,
        target: EventEmbeddingTarget,
    ) -> LossResult:
        cfg = self.config
        D = cfg.divisor_bins
        O = cfg.offset_bins
        R = cfg.ratio_bins
        raw = output.logits                                             # (B, D+O+R+1)
        targets = target.target_bin                                     # (B,)
        B = raw.size(0)
        stop_idx_ratio = R  # last class in ratio head

        # Unpack.
        div_logits = raw[:, :D]                                         # (B, D)
        off_logits = raw[:, D:D + O]                                    # (B, O)
        ratio_logits = raw[:, D + O:]                                   # (B, R+1)

        # ── Divisor + offset CE (fixed GT targets, masked by validity) ─
        div_target = target.divisor_target                              # (B,) int
        off_target = target.offset_target                               # (B,) int
        div_valid = target.divisor_valid                                # (B,) bool
        off_valid = target.offset_valid                                 # (B,) bool
        assert div_target is not None and off_target is not None, (
            "RatioLoss requires divisor_target and offset_target on the "
            "target. Use the ratio-mode adapter."
        )
        div_ce_raw = F.cross_entropy(div_logits, div_target.clamp(0, D - 1),
                                     reduction="none")
        off_ce_raw = F.cross_entropy(off_logits, off_target.clamp(0, O - 1),
                                     reduction="none")
        # Zero CE on samples where GT is unreliable.
        div_ce = div_ce_raw * div_valid.float() if div_valid is not None else div_ce_raw
        off_ce = off_ce_raw * off_valid.float() if off_valid is not None else off_ce_raw

        # ── Ratio target (DYNAMIC from model's own predictions) ──────
        is_stop = targets >= cfg.divisor_bins  # target_bin >= b_pred → STOP
        is_bin = ~is_stop

        # Soft expectations from div/off (detached in model, but we
        # recompute here from logits for the target derivation).
        div_probs = torch.softmax(div_logits.detach(), dim=-1)
        off_probs = torch.softmax(off_logits.detach(), dim=-1)
        div_bins_t = torch.arange(D, device=raw.device, dtype=torch.float32)
        off_bins_t = torch.arange(O, device=raw.device, dtype=torch.float32)
        div_val = (div_probs * div_bins_t).sum(-1)                      # (B,)
        off_val = (off_probs * off_bins_t).sum(-1)                      # (B,)

        # ratio = (target_bin + offset) / divisor
        ratio_target = torch.full((B,), stop_idx_ratio, dtype=torch.long,
                                  device=raw.device)
        if is_bin.any():
            t_bin = targets[is_bin].float()
            d_val = div_val[is_bin].clamp(min=1.0)
            o_val = off_val[is_bin]
            ratio_float = (t_bin + o_val) / d_val
            ratio_target[is_bin] = self._find_ratio_bin(ratio_float)

        # ── Ratio CE ─────────────────────────────────────────────────
        ratio_frozen = (
            self._freeze_step_limit > 0
            and self._fwd_step < self._freeze_step_limit
        )
        self._fwd_step += 1

        if ratio_frozen:
            ratio_ce = torch.zeros(B, device=raw.device)
        else:
            ratio_ce = F.cross_entropy(
                ratio_logits, ratio_target, reduction="none",
            )

        # ── STOP weighting ───────────────────────────────────────────
        per_sample = ratio_ce + cfg.aux_weight * (div_ce + off_ce)
        if cfg.stop_weight != 1.0:
            multiplier = torch.where(
                is_stop,
                torch.tensor(cfg.stop_weight, device=per_sample.device,
                             dtype=per_sample.dtype),
                torch.tensor(1.0, device=per_sample.device,
                             dtype=per_sample.dtype),
            )
            per_sample = per_sample * multiplier

        loss = per_sample.mean()

        # ── Ratio-space HIT/GOOD/MISS metrics ───────────────────────
        # Skipped while ratio head is frozen — argmax on a zero-filled
        # ratio block (model returns zeros during warmup to save
        # compute) produces meaningless metrics.
        r_rhit = 0.0
        r_rgood = 0.0
        r_rmiss = 0.0
        n_ratio_eval = 0
        if not ratio_frozen and is_bin.any():
            ratio_pred = ratio_logits[:, :R].argmax(dim=-1)             # (B,)
            centers = self._ratio_centers.to(raw.device)
            pred_ratio_val = centers[ratio_pred[is_bin]]
            true_ratio_val = centers[ratio_target[is_bin].clamp(0, R - 1)]
            log_err = (
                torch.log(pred_ratio_val.clamp(min=1e-6))
                - torch.log(true_ratio_val.clamp(min=1e-6))
            ).abs()
            n_eval = int(is_bin.sum().item())
            n_ratio_eval = n_eval
            r_rhit = float((log_err < cfg.ratio_rhit_log).float().sum()) / max(n_eval, 1)
            r_rgood = float((log_err < cfg.ratio_rgood_log).float().sum()) / max(n_eval, 1)
            r_rmiss = 1.0 - r_rgood

        # ── Divisor / offset accuracy ────────────────────────────────
        div_pred = div_logits.argmax(dim=-1)
        off_pred = off_logits.argmax(dim=-1)
        div_acc = float((div_pred == div_target).float().mean())
        div_acc_3 = float(((div_pred - div_target).abs() <= 3).float().mean())
        off_acc = float((off_pred == off_target.clamp(0, O - 1)).float().mean())

        return LossResult(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "ratio_ce": float(ratio_ce.mean().detach()),
                "div_ce": float(div_ce.mean().detach()),
                "off_ce": float(off_ce.mean().detach()),
                "stop_rate": float(is_stop.float().mean()),
                "ratio/rhit": r_rhit,
                "ratio/rgood": r_rgood,
                "ratio/rmiss": r_rmiss,
                "ratio/n_eval": float(n_ratio_eval),
                "ratio/div_acc": div_acc,
                "ratio/div_acc_3": div_acc_3,
                "ratio/off_acc": off_acc,
                "ratio/frozen": 1.0 if ratio_frozen else 0.0,
            },
        )
