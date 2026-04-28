"""Concrete losses for taiko2.

Starting point: `OnsetLoss`, the mixed hard-CE + trapezoid-soft-CE used
by the exp 45 port. Trapezoid is in log-ratio space: misses are
punished proportionally to the ratio between predicted and target bin,
not to the absolute frame distance. A ±N-frame floor prevents
near-zero targets (t=1, 2) from collapsing the plateau.

Extensions:
  - `GaussianCELoss` (exp 005): Gaussian soft CE + binary STOP head.
  - `LogEmdLoss` (exp 008): hard CE + log-ratio EMD.
  - `MdnLoss` (exp 009): Mixture Density Network with K Gaussian
    components. Embraces multi-modality instead of fighting it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.loss import Loss, LossConfig, LossResult
from ..models.event_embedding import EventEmbeddingOutput, EventEmbeddingTarget


# ─────────────────────────── MDN helpers ─────────────────────────────

_LOG_2PI = math.log(2.0 * math.pi)


def parse_mdn_params(
    raw: torch.Tensor, n_components: int, b_pred: int,
    max_sigma: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse raw MDN output into stop_logit, mu, sigma, pi.

    Args:
        raw: (B, K*3 + 1) raw model output.
        n_components: K.
        b_pred: maximum bin index (exclusive); mu is scaled to [0, b_pred].
        max_sigma: if > 0, clamp sigma to [1.0, max_sigma]. 0 = no cap.

    Returns:
        stop_logit: (B,)
        mu:         (B, K) in [0, b_pred]
        sigma:      (B, K) >= 1.0
        pi:         (B, K) sums to 1
    """
    B = raw.size(0)
    K = n_components
    stop_logit = raw[:, 0]                                              # (B,)
    params = raw[:, 1:].reshape(B, K, 3)                                # (B, K, 3)
    mu = torch.sigmoid(params[:, :, 0]) * b_pred                        # (B, K)
    sigma = F.softplus(params[:, :, 1]) + 1.0                           # (B, K) >= 1.0
    if max_sigma > 0:
        sigma = sigma.clamp(max=max_sigma)
    log_pi = params[:, :, 2]                                            # (B, K)
    pi = F.softmax(log_pi, dim=-1)                                      # (B, K)
    return stop_logit, mu, sigma, pi


def gaussian_log_prob(
    t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """Log N(t | mu, sigma) for each component.

    t: (M,) or (M, 1) — broadcast-ready against (M, K).
    mu, sigma: (M, K).
    Returns: (M, K).
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)                                             # (M, 1)
    var = sigma * sigma
    return -0.5 * (_LOG_2PI + var.log() + (t - mu).pow(2) / var)


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


@dataclass(frozen=True, slots=True)
class LogEmdLossConfig(LossConfig):
    """Hard CE + log-ratio Earth-Mover Distance over bin probabilities.

    Designed for #008: the loss-shape attack on the ratio-banding
    ridges that #002 / #005 / #007 all left intact. Combines #002's
    hard-CE peak signal with a log-ratio EMD term that has NO entropy
    floor and is symmetric in log-space (perception-correct: a
    halving and a doubling of the target both cost ``log 2`` in the
    EMD term, regardless of absolute target magnitude).

    Total loss per sample:

        l = hard_alpha * hard_CE + (1 - hard_alpha) * log_emd

    where:

        hard_CE = F.cross_entropy(logits, target)
        log_emd = sum_{i in bins} P_i · |log((i+1)/(t+1))|^exponent
                 (set to 0 on STOP-target samples — log-EMD is
                  defined only for bin targets)

    P is softmax over the full (n_classes,) logits including STOP, so
    mass on STOP for a bin-target sample is punished implicitly via
    hard_CE (does not contribute to log_emd directly).

    STOP samples get the same ``stop_weight`` multiplier as #002's
    OnsetLoss to keep the unified-softmax STOP-vs-bins balance that
    #002 / #007 had — #005 showed splitting STOP off broke the decode-
    time scale linkage.
    """
    hard_alpha: float = 0.5         # mix weight on hard CE
    exponent: int = 1               # 1 = linear log-EMD, 2 = squared
    stop_weight: float = 1.5        # per-sample multiplier on STOP samples


class LogEmdLoss(Loss[LogEmdLossConfig, EventEmbeddingOutput, EventEmbeddingTarget]):
    """Hard CE + log-ratio EMD loss.

    Implements the #008 hypothesis: the trapezoid soft CE in #002
    has an entropy floor that prevents the model from receiving any
    gradient outside its support, which is why the ratio-banding
    ridges never compress. log-ratio EMD has no such floor — the
    minimum is delta-at-target, and mass anywhere else costs
    proportionally to its log-ratio distance from the target.
    """

    def __init__(self, config: LogEmdLossConfig):
        super().__init__(config)
        if not 0.0 <= config.hard_alpha <= 1.0:
            raise ValueError(
                f"hard_alpha must be in [0, 1], got {config.hard_alpha}"
            )
        if config.exponent not in (1, 2):
            raise ValueError(
                f"exponent must be 1 or 2, got {config.exponent}"
            )
        if config.stop_weight <= 0:
            raise ValueError(
                f"stop_weight must be > 0, got {config.stop_weight}"
            )

    def forward(
        self,
        output: EventEmbeddingOutput,
        target: EventEmbeddingTarget,
    ) -> LossResult:
        logits = output.logits                                          # (B, n_classes)
        targets = target.target_bin                                     # (B,)
        n_classes = logits.size(1)
        stop_idx = n_classes - 1

        # Hard CE over the full (n_classes) softmax including STOP —
        # punishes mass on STOP for bin targets and mass on bins for
        # STOP targets without us touching the bin/STOP partition.
        hard_ce = F.cross_entropy(logits, targets, reduction="none")    # (B,)

        # log-EMD over bin probabilities only. Defined as 0 for
        # STOP-target samples (no bin target ⇒ no log-ratio distance).
        log_emd = torch.zeros_like(hard_ce)
        is_bin = targets != stop_idx
        if is_bin.any():
            bin_logits_sel = logits[is_bin, :stop_idx]                  # (M, stop_idx)
            bin_targets_sel = targets[is_bin].float()                   # (M,)
            P = F.softmax(logits[is_bin], dim=-1)[:, :stop_idx]         # (M, stop_idx)
            bins = torch.arange(
                stop_idx, device=logits.device, dtype=torch.float32,
            )
            log_dist = torch.abs(
                torch.log((bins.unsqueeze(0) + 1.0)
                          / (bin_targets_sel.unsqueeze(1) + 1.0))
            )                                                           # (M, stop_idx)
            if self.config.exponent == 2:
                log_dist = log_dist * log_dist
            log_emd[is_bin] = (P * log_dist).sum(dim=-1)                # (M,)

        cfg = self.config
        per_sample = cfg.hard_alpha * hard_ce + (1.0 - cfg.hard_alpha) * log_emd

        is_stop = targets == stop_idx
        if cfg.stop_weight != 1.0:
            multiplier = torch.where(
                is_stop,
                torch.tensor(cfg.stop_weight, device=per_sample.device, dtype=per_sample.dtype),
                torch.tensor(1.0, device=per_sample.device, dtype=per_sample.dtype),
            )
            per_sample = per_sample * multiplier

        loss = per_sample.mean()

        return LossResult(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "hard_ce": float(hard_ce.mean().detach()),
                "log_emd": float(log_emd.mean().detach()),
                "stop_rate": float(is_stop.float().mean().detach()),
            },
        )


@dataclass(frozen=True, slots=True)
class MdnLossConfig(LossConfig):
    """Mixture Density Network loss.

    The model outputs ``(B, K*3+1)`` raw params parsed into:
      - ``stop_logit`` (1 scalar) → sigmoid-gated STOP.
      - ``K`` Gaussian components (μ, σ, π) over bin space [0, b_pred).

    Loss for bin targets:
      ``-log((1-P_stop) · Σ_k π_k · N(t | μ_k, σ_k))``
    Loss for STOP targets:
      ``-log(P_stop) · stop_weight``

    Embraces multi-modality: the model can output multiple smooth peaks
    (one per component) without any peak being penalized for existing.
    Only requirement: SOME component covers the target.
    """
    n_components: int = 3
    b_pred: int = 500
    stop_weight: float = 1.5
    # Sigma range (bins). Floor prevents collapse; cap prevents inflation.
    # max_sigma=0 means no cap (free sigma). max_sigma=3 forces sharp peaks.
    min_sigma: float = 1.0
    max_sigma: float = 0.0


class MdnLoss(Loss[MdnLossConfig, EventEmbeddingOutput, EventEmbeddingTarget]):
    """Mixture Density Network negative log-likelihood.

    Multi-modal by design: the loss is low as long as ONE component
    covers the target. Other components can sit at octave multiples
    or anywhere else — they contribute to the mixture likelihood
    only positively. No penalty for secondary peaks.
    """

    def __init__(self, config: MdnLossConfig):
        super().__init__(config)
        if config.n_components < 1:
            raise ValueError(
                f"n_components must be >= 1, got {config.n_components}"
            )

    def forward(
        self,
        output: EventEmbeddingOutput,
        target: EventEmbeddingTarget,
    ) -> LossResult:
        raw = output.logits                                             # (B, K*3+1)
        targets = target.target_bin                                     # (B,)
        cfg = self.config
        stop_idx = cfg.b_pred

        stop_logit, mu, sigma, pi = parse_mdn_params(
            raw, cfg.n_components, cfg.b_pred,
            max_sigma=cfg.max_sigma,
        )
        p_stop = torch.sigmoid(stop_logit)                              # (B,)

        is_stop = targets == stop_idx
        is_bin = ~is_stop

        # ── STOP loss: -log(P_stop) ──────────────────────────────────
        stop_bce = F.binary_cross_entropy_with_logits(
            stop_logit,
            is_stop.float(),
            reduction="none",
        )                                                               # (B,)

        # ── Bin loss: -log((1-P_stop) · mixture(t)) ─────────────────
        mixture_nll = torch.zeros_like(stop_bce)
        # Diagnostic accumulators (detached).
        coverage_2 = 0.0
        coverage_5 = 0.0
        dominant_w = 0.0
        n_active = 0.0
        mean_sigma_val = float(sigma.mean().detach())
        correct_comp_w = 0.0

        if is_bin.any():
            mu_sel = mu[is_bin]                                         # (M, K)
            sigma_sel = sigma[is_bin]                                   # (M, K)
            pi_sel = pi[is_bin]                                         # (M, K)
            t_sel = targets[is_bin].float()                             # (M,)
            p_stop_sel = p_stop[is_bin]                                 # (M,)

            # Log-likelihood of each component.
            log_comp = gaussian_log_prob(t_sel, mu_sel, sigma_sel)       # (M, K)
            # Log-sum-exp over components weighted by π.
            log_mixture = torch.logsumexp(
                log_comp + pi_sel.clamp(min=1e-10).log(), dim=-1,
            )                                                           # (M,)
            # Add log(1 - P_stop) for the bin branch.
            log_p_bin = torch.log((1.0 - p_stop_sel).clamp(min=1e-10))
            mixture_nll[is_bin] = -(log_p_bin + log_mixture)

            # ── Diagnostics (detached) ───────────────────────────────
            with torch.no_grad():
                M = int(is_bin.sum().item())
                # Distance of each component's μ from target.
                dist = (mu_sel - t_sel.unsqueeze(-1)).abs()              # (M, K)
                min_dist = dist.min(dim=-1).values                      # (M,)
                coverage_2 = float((min_dist <= 2.0).float().mean())
                coverage_5 = float((min_dist <= 5.0).float().mean())
                # Dominant weight.
                dominant_w = float(pi_sel.max(dim=-1).values.mean())
                # Active components (π > 0.1).
                n_active = float(
                    (pi_sel > 0.1).float().sum(dim=-1).mean()
                )
                # Weight of the component closest to target.
                closest_k = dist.argmin(dim=-1)                         # (M,)
                correct_comp_w = float(
                    pi_sel.gather(1, closest_k.unsqueeze(-1)).mean()
                )

        # ── Total loss ───────────────────────────────────────────────
        per_sample = stop_bce + mixture_nll
        if cfg.stop_weight != 1.0:
            multiplier = torch.where(
                is_stop,
                torch.tensor(cfg.stop_weight, device=per_sample.device, dtype=per_sample.dtype),
                torch.tensor(1.0, device=per_sample.device, dtype=per_sample.dtype),
            )
            per_sample = per_sample * multiplier

        loss = per_sample.mean()

        return LossResult(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "mixture_nll": float(mixture_nll.mean().detach()),
                "stop_bce": float(stop_bce.mean().detach()),
                "stop_rate": float(is_stop.float().mean().detach()),
                "mdn/coverage_2bin": coverage_2,
                "mdn/coverage_5bin": coverage_5,
                "mdn/dominant_weight": dominant_w,
                "mdn/n_active_components": n_active,
                "mdn/mean_sigma": mean_sigma_val,
                "mdn/correct_component_weight": correct_comp_w,
            },
        )
