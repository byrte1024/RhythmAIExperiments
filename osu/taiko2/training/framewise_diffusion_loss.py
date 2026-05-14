"""Framewise diffusion loss for ``FramewiseDiffusionDetector`` (#016).

Consumes a structured ``FramewiseModelOutput`` (the model has already
done the q_sample + denoiser forward) and computes a weighted MSE/Huber
distance between the predicted M_0 and the target smoothed activation
map, with optional Min-SNR weighting and per-bin positive-class
upweighting.

Reports diagnostic metrics for the framewise dashboard:

- ``loss``                          — headline scalar.
- ``loss/snr_weighted``             — always-on diagnostic.
- ``loss/pos_only``, ``loss/neg_only``, ``loss/pos_neg_ratio``
- ``loss/per_t_q{0..3}``            — per-t-quartile loss.
- ``frame/precision_τ_50_tol_2``, ``frame/recall_τ_50_tol_2``,
  ``frame/f1_τ_50_tol_2`` — canonical operating point.
- ``frame/auc_pr``, ``frame/auc_roc``
- ``frame/mean_act_pos``, ``frame/mean_act_neg``, ``frame/separation``
- ``frame/pos_rate_pred_50``, ``frame/pos_rate_target``
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.framewise import FramewiseTarget
from ..domain.loss import Loss, LossConfig, LossResult
from ..models.framewise_diffusion_detector import FramewiseModelOutput
from .framewise_curve_metrics import (
    compute_auc_pr,
    compute_auc_roc,
    compute_frame_f1_at_tolerance,
    compute_separation,
)


@dataclass(frozen=True, slots=True)
class FramewiseDiffusionLossConfig(LossConfig):
    """Hyperparameters for ``FramewiseDiffusionLoss``.

    - ``loss_type`` — ``"mse"`` or ``"huber"``.
    - ``snr_weighting`` — Min-SNR (γ=5 from #015) applied to per-sample
      loss. Default ON.
    - ``snr_gamma`` — Min-SNR γ.
    - ``pos_weight_clamp_min/max`` — per-sample positive-class
      upweighting is clamped to this range. n_neg / max(n_pos, 1) for
      typical 500-bin / ~5-event windows is around 100×; clamp keeps
      both sparse and dense charts within a sane window.
    - ``n_t_buckets`` — number of equal-width t buckets for per-t loss.
    - ``canonical_threshold`` / ``canonical_tolerance_frames`` — single
      operating point reported as a scalar metric (τ=0.5, tol=±2 by
      default).
    """
    loss_type: str = "mse"
    snr_weighting: bool = True
    snr_gamma: float = 5.0
    pos_weight_clamp_min: float = 10.0
    pos_weight_clamp_max: float = 200.0
    n_t_buckets: int = 4
    canonical_threshold: float = 0.5
    canonical_tolerance_frames: int = 2

    def __post_init__(self) -> None:
        if self.loss_type not in {"mse", "huber"}:
            raise ValueError(
                f"loss_type must be 'mse' or 'huber' (got {self.loss_type!r})"
            )
        if self.snr_gamma <= 0.0:
            raise ValueError(f"snr_gamma must be > 0 (got {self.snr_gamma})")
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
        if self.n_t_buckets < 1:
            raise ValueError(
                f"n_t_buckets must be >= 1 (got {self.n_t_buckets})"
            )
        if not 0.0 <= self.canonical_threshold <= 1.0:
            raise ValueError(
                f"canonical_threshold must be in [0, 1] "
                f"(got {self.canonical_threshold})"
            )
        if self.canonical_tolerance_frames < 0:
            raise ValueError(
                f"canonical_tolerance_frames must be >= 0 "
                f"(got {self.canonical_tolerance_frames})"
            )


class FramewiseDiffusionLoss(
    Loss[
        FramewiseDiffusionLossConfig,
        FramewiseModelOutput,
        FramewiseTarget,
    ],
):
    """Weighted MSE/Huber on the framewise activation map."""

    def __init__(self, config: FramewiseDiffusionLossConfig):
        super().__init__(config)
        self._alphas_cumprod: torch.Tensor | None = None
        self._process = None
        self._denoiser = None
        self._n_steps: int | None = None

    def bind_schedule(self, alphas_cumprod: torch.Tensor) -> None:
        self._alphas_cumprod = alphas_cumprod.detach()

    def bind_model(self, model) -> None:  # type: ignore[no-untyped-def]
        """Stash process/denoiser/schedule so the loss can run the
        diffusion forward inline when handed an inference-shape output."""
        self._process = model.process
        self._denoiser = model.denoiser
        self._n_steps = model.schedule.n_steps
        self.bind_schedule(model.schedule.alphas_cumprod())

    # ── helpers ──────────────────────────────────────────────────────

    def _per_bin_distance(
        self, model_out: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-bin distance (B, n_bins)."""
        if self.config.loss_type == "mse":
            return (model_out - target) ** 2
        if self.config.loss_type == "huber":
            return F.huber_loss(model_out, target, reduction="none")
        raise ValueError(f"unknown loss_type {self.config.loss_type!r}")

    def _snr_weights(self, t: torch.Tensor) -> torch.Tensor:
        if self._alphas_cumprod is None:
            return torch.ones_like(t, dtype=torch.float32)
        ab = self._alphas_cumprod.to(t.device).gather(0, t.long())
        snr = ab / (1.0 - ab).clamp(min=1e-8)
        weight = (
            torch.minimum(snr, torch.full_like(snr, self.config.snr_gamma))
            / snr.clamp(min=1e-8)
        )
        return weight

    # ── inline diffusion forward (for inference-shape outputs) ───────

    def _inline_forward(
        self,
        output: FramewiseModelOutput,
        target: FramewiseTarget,
    ) -> None:
        if self._process is None or self._denoiser is None or self._n_steps is None:
            raise RuntimeError(
                "FramewiseDiffusionLoss got an inference-shape "
                "FramewiseModelOutput but no model is bound. "
                "Call loss.bind_model(model) once after construction."
            )
        cursor_token = output.cursor_token
        audio_features = output.audio_features
        if audio_features is None:
            raise RuntimeError(
                "FramewiseDiffusionLoss inline forward requires "
                "audio_features on the model output (was None)."
            )
        target_map = target.target_map_smoothed
        B = cursor_token.size(0)
        x_0 = self._process.encode_x0(target_map)
        t = torch.randint(
            0, self._n_steps, (B,),
            device=cursor_token.device, dtype=torch.long,
        )
        noise = torch.randn_like(x_0)
        x_t = self._process.q_sample(x_0, t, noise=noise)

        self_cond = bool(
            getattr(self._denoiser.config, "self_cond", False),
        )
        prev_x0_hat: torch.Tensor | None = None
        if self_cond and self._denoiser.training:
            mask = torch.rand(B, device=cursor_token.device) < 0.5
            if bool(mask.any()):
                with torch.no_grad():
                    first_pass = self._denoiser(
                        cursor_token, x_t, t,
                        prev_x0_hat=None,
                        audio_features=audio_features,
                    )
                    sc_x0 = self._process.predict_x0(
                        first_pass, x_t, t,
                    ).detach()
                prev_x0_hat = torch.where(
                    mask.view(-1, 1), sc_x0, torch.zeros_like(sc_x0),
                )

        model_out = self._denoiser(
            cursor_token, x_t, t,
            prev_x0_hat=prev_x0_hat if self_cond else None,
            audio_features=audio_features,
        )
        loss_target = self._process.loss_target(x_0, x_t, t, noise=noise)
        x0_hat = self._process.predict_x0(model_out, x_t, t)
        logits = self._process.decode_to_logits(x0_hat)
        object.__setattr__(output, "model_out", model_out)
        object.__setattr__(output, "loss_target", loss_target)
        object.__setattr__(output, "t", t)
        object.__setattr__(output, "x_t", x_t)
        object.__setattr__(output, "logits", logits)

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        output: FramewiseModelOutput,
        target: FramewiseTarget,
    ) -> LossResult:
        if output.model_out is None or output.loss_target is None or output.t is None:
            self._inline_forward(output, target)

        cfg = self.config
        n_bins = output.model_out.size(-1)
        target_binary = target.target_map_binary       # (B, n_bins)
        target_smoothed = output.loss_target            # (B, n_bins) — what denoiser fits

        # Per-bin loss vs the smoothed target (the denoiser's training signal).
        per_bin = self._per_bin_distance(output.model_out, target_smoothed)  # (B, n_bins)

        # Per-sample positive-class weights:
        #   pos_w = clamp(n_neg / max(n_pos, 1), [clamp_min, clamp_max]).
        # n_pos comes from the target's n_gt (count of in-window onsets).
        n_pos = target.n_gt.float().clamp(min=1.0)
        n_neg = float(n_bins) - target.n_gt.float()
        pos_w_per_sample = (n_neg / n_pos).clamp(
            min=cfg.pos_weight_clamp_min,
            max=cfg.pos_weight_clamp_max,
        )                                              # (B,)

        # Per-bin weight: pos_w on binary-positive bins, 1.0 elsewhere.
        pos_bin = target_binary > 0.5                  # (B, n_bins) bool
        bin_w = torch.where(
            pos_bin,
            pos_w_per_sample.view(-1, 1).expand_as(per_bin),
            torch.ones_like(per_bin),
        )
        weighted = per_bin * bin_w
        per_sample = weighted.mean(dim=-1)             # (B,)

        # SNR weighting (always computed for diagnostic).
        snr_w = self._snr_weights(output.t)
        snr_weighted = (per_sample * snr_w).mean()

        if cfg.snr_weighting:
            per_sample_for_loss = per_sample * snr_w
        else:
            per_sample_for_loss = per_sample
        loss = per_sample_for_loss.mean()

        # ── diagnostics ──────────────────────────────────────────────
        metrics: dict[str, float] = {
            "loss": float(loss.detach()),
            "loss/snr_weighted": float(snr_weighted.detach()),
        }

        # pos / neg / ratio over per-bin distance, no positive upweighting.
        pos_loss = per_bin[pos_bin]
        neg_loss = per_bin[~pos_bin]
        pos_mean = float(pos_loss.mean().detach()) if pos_loss.numel() > 0 else 0.0
        neg_mean = float(neg_loss.mean().detach()) if neg_loss.numel() > 0 else 0.0
        metrics["loss/pos_only"] = pos_mean
        metrics["loss/neg_only"] = neg_mean
        metrics["loss/pos_neg_ratio"] = pos_mean / neg_mean if neg_mean > 0 else 0.0

        # Per-t-quartile loss (unweighted per_sample).
        n_buckets = cfg.n_t_buckets
        t_max = float(output.t.max().detach()) if output.t.numel() else 1.0
        t_max = max(t_max, 1.0)
        bucket = (
            output.t.float() / (t_max + 1e-8) * n_buckets
        ).long().clamp(max=n_buckets - 1)
        for b in range(n_buckets):
            mask = bucket == b
            if mask.any():
                metrics[f"loss/per_t_q{b}"] = float(
                    per_sample[mask].mean().detach(),
                )
            else:
                metrics[f"loss/per_t_q{b}"] = 0.0

        # Frame-level metrics on the predicted activation map (clipped
        # via decode_to_logits if process is bound; fall back to clamp).
        m_hat = output.model_out.detach().clamp(0.0, 1.0)
        tb = target_binary.detach()
        f1_dict = compute_frame_f1_at_tolerance(
            m_hat, tb,
            threshold=cfg.canonical_threshold,
            tolerance_frames=cfg.canonical_tolerance_frames,
        )
        tau_pct = int(round(cfg.canonical_threshold * 100))
        tol = cfg.canonical_tolerance_frames
        metrics[f"frame/precision_τ_{tau_pct}_tol_{tol}"] = f1_dict["precision"]
        metrics[f"frame/recall_τ_{tau_pct}_tol_{tol}"] = f1_dict["recall"]
        metrics[f"frame/f1_τ_{tau_pct}_tol_{tol}"] = f1_dict["f1"]

        metrics["frame/auc_pr"] = compute_auc_pr(m_hat, tb)
        metrics["frame/auc_roc"] = compute_auc_roc(m_hat, tb)
        mean_pos_act, mean_neg_act, sep = compute_separation(m_hat, tb)
        metrics["frame/mean_act_pos"] = mean_pos_act
        metrics["frame/mean_act_neg"] = mean_neg_act
        metrics["frame/separation"] = sep
        pos_rate_pred_50 = float(
            ((m_hat > cfg.canonical_threshold).float().mean()).item(),
        )
        metrics["frame/pos_rate_pred_50"] = pos_rate_pred_50
        metrics["frame/pos_rate_target"] = float(tb.float().mean().item())

        return LossResult(loss=loss, metrics=metrics)
