"""Concrete losses for taiko2.

Starting point: `OnsetLoss`, the mixed hard-CE + trapezoid-soft-CE used
by the exp 45 port. Trapezoid is in log-ratio space: misses are
punished proportionally to the ratio between predicted and target bin,
not to the absolute frame distance. A ±N-frame floor prevents
near-zero targets (t=1, 2) from collapsing the plateau.

Extensions left for future experiments:
  - Focal modulation (exp 28, 37).
  - Class weights (exp 7-ish).
  - Distance ramp (exp 44e+).
  - Ratio head (exp 55+).
Each lands as its own `Loss` subclass (or a new config subclass here)
rather than as optional flags on this one — keeps exp 45's loss honest
about what it uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.loss import Loss, LossConfig, LossResult
from ..models.event_embedding import EventEmbeddingOutput, EventEmbeddingTarget


@dataclass(frozen=True, slots=True)
class OnsetLossConfig(LossConfig):
    """Trapezoid soft-target CE hyperparameters.

    Defaults match exp 45. Trapezoid shape lives in log-ratio space:
      - distance ≤ log(1 + good_pct)  → full credit (plateau).
      - log(1 + good_pct) < d ≤ log(1 + fail_pct) → linear ramp to 0.
      - d > log(1 + fail_pct) → 0.

    Frame-tolerance floor gives ±N-frame bins full credit regardless of
    ratio distance — prevents targets with very small ``t`` (where a
    3 % ratio window would be sub-bin wide) from having no plateau.

    STOP samples (target == ``n_classes - 1``) get a single one-hot
    soft target and the full ``stop_weight`` multiplier.
    """
    hard_alpha: float = 0.5       # weight on hard CE in the mix
    good_pct: float = 0.03        # plateau width in ratio space
    fail_pct: float = 0.20        # ramp-to-zero cutoff in ratio space
    frame_tolerance: int = 2      # ±frames guaranteed some credit
    stop_weight: float = 1.5      # per-sample multiplier when target is STOP


class OnsetLoss(Loss[OnsetLossConfig, EventEmbeddingOutput, EventEmbeddingTarget]):
    """Mixed hard + trapezoid-soft CE for onset-offset classification.

    ``loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE``,
    then per-sample weighted by ``stop_weight`` where the target is
    STOP. Sub-component means are emitted via `LossResult.metrics` so
    the training log carries `hard_ce` / `soft_ce` / `stop_rate`
    alongside the total.
    """

    def __init__(self, config: OnsetLossConfig):
        super().__init__(config)
        if not 0.0 <= config.hard_alpha <= 1.0:
            raise ValueError(
                f"hard_alpha must be in [0, 1], got {config.hard_alpha}"
            )
        if config.good_pct < 0 or config.fail_pct < config.good_pct:
            raise ValueError(
                f"require 0 ≤ good_pct ≤ fail_pct; "
                f"got good_pct={config.good_pct}, fail_pct={config.fail_pct}"
            )
        if config.frame_tolerance < 0:
            raise ValueError(
                f"frame_tolerance must be ≥ 0, got {config.frame_tolerance}"
            )
        self._log_good = math.log(1 + config.good_pct)
        self._log_fail = math.log(1 + config.fail_pct)

    # ── soft-target builder ──────────────────────────────────────────

    def _make_soft_targets(
        self, targets: torch.Tensor, n_classes: int,
    ) -> torch.Tensor:
        """Convert integer targets to trapezoid soft distributions.

        Returns (B, n_classes) — rows sum to 1. STOP targets get a
        pure one-hot on the last class; bin targets get a trapezoid
        over classes 0..n_classes-2 with a frame-tolerance floor.
        """
        device = targets.device
        B = targets.size(0)
        soft = torch.zeros(B, n_classes, device=device)

        stop_idx = n_classes - 1
        is_stop = targets == stop_idx
        is_bin = ~is_stop

        if is_stop.any():
            soft[is_stop, stop_idx] = 1.0

        if is_bin.any():
            bin_targets = targets[is_bin].float()                       # (M,)
            bins = torch.arange(stop_idx, device=device, dtype=torch.float32)

            # Proportional distance in log-ratio space: d_ij = |log((i+1)/(t_j+1))|.
            abs_log_ratio = torch.abs(
                torch.log((bins + 1).unsqueeze(0) / (bin_targets + 1).unsqueeze(1))
            )

            # Trapezoid: 1 inside `good`, linear ramp to 0 at `fail`.
            ramp_width = self._log_fail - self._log_good
            if ramp_width <= 0:
                ratio_weights = (abs_log_ratio <= self._log_good).float()
            else:
                ratio_weights = (
                    (self._log_fail - abs_log_ratio) / ramp_width
                ).clamp(0.0, 1.0)

            # Frame-tolerance floor: ±(N+1) range has ≥0 credit.
            frame_dist = torch.abs(bins.unsqueeze(0) - bin_targets.unsqueeze(1))
            tol = self.config.frame_tolerance
            frame_weights = (
                (tol + 1 - frame_dist) / (tol + 1)
            ).clamp(0.0, 1.0)

            # Take the max so either mechanism can grant credit.
            weights = torch.max(ratio_weights, frame_weights)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            soft[is_bin, :stop_idx] = weights

        return soft

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        output: EventEmbeddingOutput,
        target: EventEmbeddingTarget,
    ) -> LossResult:
        logits = output.logits                                          # (B, n_classes)
        targets = target.target_bin                                     # (B,)
        n_classes = logits.size(1)

        log_probs = F.log_softmax(logits, dim=-1).clamp(min=-100)

        # Hard CE per sample.
        hard_ce = F.cross_entropy(logits, targets, reduction="none")    # (B,)

        # Soft CE per sample.
        soft_targets = self._make_soft_targets(targets, n_classes)
        soft_ce = -(soft_targets * log_probs).sum(dim=-1)               # (B,)

        # Mix.
        cfg = self.config
        ce = cfg.hard_alpha * hard_ce + (1.0 - cfg.hard_alpha) * soft_ce

        # STOP weighting.
        is_stop = targets == (n_classes - 1)
        if cfg.stop_weight != 1.0:
            stop_multiplier = torch.where(
                is_stop,
                torch.tensor(cfg.stop_weight, device=ce.device, dtype=ce.dtype),
                torch.tensor(1.0, device=ce.device, dtype=ce.dtype),
            )
            ce = ce * stop_multiplier

        loss = ce.mean()

        return LossResult(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "hard_ce": float(hard_ce.mean().detach()),
                "soft_ce": float(soft_ce.mean().detach()),
                "stop_rate": float(is_stop.float().mean().detach()),
            },
        )


@dataclass(frozen=True, slots=True)
class GaussianCELossConfig(LossConfig):
    """Gaussian soft CE with STOP treated as a separate binary task.

    The model head is unchanged — still a single ``(B, n_classes)``
    logit tensor. The loss routes those logits two ways:
      1. ``stop_bce`` — binary cross-entropy over the STOP logit as a
         sigmoid, target = 1 iff the sample's target is STOP.
      2. ``bin_ce`` — softmax CE over the non-STOP bin logits with a
         Gaussian soft target of width ``sigma_bins``, computed only
         on non-STOP samples.

    Splitting the loss (not the head) means partial-credit smearing
    happens only between adjacent bins — STOP is a hard binary
    decision that cannot steal or donate soft mass to/from any bin.
    """
    sigma_bins: float = 2.0


class GaussianCELoss(Loss[GaussianCELossConfig, EventEmbeddingOutput, EventEmbeddingTarget]):
    """Binary-STOP BCE + bin-only Gaussian softmax CE.

    ``loss = stop_bce_mean + bin_ce_mean``. ``bin_ce_mean`` is the mean
    over non-STOP samples; empty-bin-set batches contribute 0 so the
    STOP signal keeps flowing. Sub-components are emitted via
    ``LossResult.metrics`` for logging.
    """

    def __init__(self, config: GaussianCELossConfig):
        super().__init__(config)
        if config.sigma_bins <= 0:
            raise ValueError(
                f"sigma_bins must be > 0, got {config.sigma_bins}"
            )

    def _gaussian_bin_targets(
        self, bin_targets: torch.Tensor, n_bins: int,
    ) -> torch.Tensor:
        device = bin_targets.device
        bins = torch.arange(n_bins, device=device, dtype=torch.float32)
        d = bins.unsqueeze(0) - bin_targets.float().unsqueeze(1)        # (M, n_bins)
        w = torch.exp(-0.5 * (d / self.config.sigma_bins) ** 2)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return w

    def forward(
        self,
        output: EventEmbeddingOutput,
        target: EventEmbeddingTarget,
    ) -> LossResult:
        logits = output.logits                                          # (B, n_classes)
        targets = target.target_bin                                     # (B,)
        n_classes = logits.size(1)
        stop_idx = n_classes - 1
        n_bins = stop_idx

        stop_logit = logits[:, stop_idx]                                # (B,)
        bin_logits = logits[:, :n_bins]                                 # (B, n_bins)

        is_stop = targets == stop_idx
        stop_target = is_stop.float()
        stop_bce = F.binary_cross_entropy_with_logits(
            stop_logit, stop_target, reduction="mean",
        )

        is_bin = ~is_stop
        if is_bin.any():
            bin_logits_sel = bin_logits[is_bin]                         # (M, n_bins)
            bin_targets_sel = targets[is_bin]                           # (M,)
            log_probs = F.log_softmax(bin_logits_sel, dim=-1).clamp(min=-100)
            soft = self._gaussian_bin_targets(bin_targets_sel, n_bins)
            bin_ce = -(soft * log_probs).sum(dim=-1).mean()
        else:
            bin_ce = torch.zeros((), device=logits.device, dtype=logits.dtype)

        loss = stop_bce + bin_ce

        return LossResult(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "stop_bce": float(stop_bce.detach()),
                "bin_ce": float(bin_ce.detach()),
                "stop_rate": float(is_stop.float().mean().detach()),
            },
        )
