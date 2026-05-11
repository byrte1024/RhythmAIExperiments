"""Diffusion loss for ``DiffusionDetector``.

Consumes a structured ``DiffusionModelOutput`` (the model has already
done the q_sample + denoiser forward) and computes a simple distance
between the denoiser's output and the loss target the
``DiffusionProcess`` produced.

Reports diagnostic metrics:

- ``loss``                 — the headline scalar.
- ``loss/per_t_q{0..3}``   — loss bucketed by timestep quartile.
  Reveals whether the model is good at the easy (low-t) or hard
  (high-t) end. A model that's well-trained at low-t but bad at
  high-t still has a long way to go in terms of clean denoising.
- ``loss/snr_weighted``    — alternative weighting by signal-to-
  noise ratio at the sampled t (Min-SNR weighting from Hang et al.
  2023). Reported as a metric even when not used as the loss, so we
  can compare.
- ``stop_rate``            — fraction of training samples whose target
  is the STOP class. Reported for parity with other taiko2 losses.
- ``argmax_match``         — fraction of samples where the model's
  predicted x_0 (decoded to logits) argmaxes to the GT bin. The
  diffusion-equivalent of "did the model commit correctly at the
  sampled t" — diagnostic, not a loss component.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.loss import Loss, LossConfig, LossResult
from ..models.diffusion_detector import DiffusionModelOutput
from ..models.event_embedding import EventEmbeddingTarget


@dataclass(frozen=True, slots=True)
class DiffusionLossConfig(LossConfig):
    """Diffusion loss hyperparameters.

    - ``loss_type``: distance between ``model_out`` and ``loss_target``.
      ``"mse"`` (default) and ``"huber"`` (less sensitive to outliers).
    - ``snr_weighting``: scale per-sample loss by Min-SNR weighting.
      Improves training stability on the noise/v parameterizations.
      For x0-parameterization this is a no-op (already
      uniformly-weighted).
    - ``snr_gamma``: cap for Min-SNR (Hang et al. 2023 use γ=5 typical).
    - ``stop_weight``: per-sample multiplier when the target is STOP.
      Matches the existing ``OnsetLoss`` convention.
    - ``n_t_buckets``: number of equal-width timestep buckets for the
      ``loss/per_t_q*`` diagnostic metrics. 4 = quartiles.
    """
    loss_type: str = "mse"
    snr_weighting: bool = False
    snr_gamma: float = 5.0
    stop_weight: float = 1.5
    n_t_buckets: int = 4

    def __post_init__(self) -> None:
        if self.loss_type not in {"mse", "huber"}:
            raise ValueError(
                f"loss_type must be 'mse' or 'huber' (got {self.loss_type!r})"
            )
        if self.snr_gamma <= 0.0:
            raise ValueError(f"snr_gamma must be > 0 (got {self.snr_gamma})")
        if self.stop_weight < 0.0:
            raise ValueError(
                f"stop_weight must be >= 0 (got {self.stop_weight})"
            )
        if self.n_t_buckets < 1:
            raise ValueError(
                f"n_t_buckets must be >= 1 (got {self.n_t_buckets})"
            )


class DiffusionLoss(
    Loss[DiffusionLossConfig, DiffusionModelOutput, EventEmbeddingTarget],
):
    """MSE / Huber on the structured ``DiffusionModelOutput``."""

    def __init__(self, config: DiffusionLossConfig):
        super().__init__(config)
        # Schedule reference is set lazily via ``bind_schedule`` so the
        # SNR-weighting metric can be computed without the loss
        # constructing its own schedule. ``bind_schedule`` is called
        # by the train CLI after model + loss are both built.
        self._alphas_cumprod: torch.Tensor | None = None
        # Model components — bound by ``bind_model`` so the loss can do
        # the diffusion forward (q_sample + denoiser) inline when given
        # an inference-shape ``DiffusionModelOutput`` (cursor_token only).
        # The taiko2 train loop calls ``model.predict(inp)`` generically;
        # for diffusion that means model_out / loss_target / t arrive
        # None, and the loss has to fill them.
        self._process = None
        self._denoiser = None
        self._n_steps: int | None = None

    def bind_schedule(self, alphas_cumprod: torch.Tensor) -> None:
        """Provide the schedule's cached ``alphas_cumprod`` for SNR
        weighting + diagnostics. Optional — if not called, the
        ``loss/snr_weighted`` metric is reported as 0.0 and the
        ``snr_weighting`` knob has no effect.
        """
        self._alphas_cumprod = alphas_cumprod.detach()

    def bind_model(self, model: "DiffusionDetector") -> None:  # type: ignore[name-defined]
        """Stash the model's ``process``, ``denoiser``, and schedule so
        ``forward`` can do the diffusion forward inline when handed an
        inference-shape output.

        Call this once after model + loss are both built; the train CLI
        does so automatically. ``bind_schedule`` is also invoked here.
        """
        self._process = model.process
        self._denoiser = model.denoiser
        self._n_steps = model.schedule.n_steps
        self.bind_schedule(model.schedule.alphas_cumprod())

    # ── distance functions ───────────────────────────────────────────

    def _distance(
        self, model_out: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample distance reduced over the bin axis.

        Returns ``(B,)`` float tensor.
        """
        if self.config.loss_type == "mse":
            return ((model_out - target) ** 2).mean(dim=-1)
        if self.config.loss_type == "huber":
            return F.huber_loss(model_out, target, reduction="none").mean(dim=-1)
        raise ValueError(f"unknown loss_type {self.config.loss_type!r}")

    # ── SNR weighting ────────────────────────────────────────────────

    def _snr_weights(self, t: torch.Tensor) -> torch.Tensor:
        """Min-SNR weights at the sampled timesteps.

        ``snr(t) = alpha_bar_t / (1 - alpha_bar_t)``.
        ``min_snr_weight = min(snr(t), gamma) / snr(t)``.
        Returns 1.0 if no schedule is bound.
        """
        if self._alphas_cumprod is None:
            return torch.ones_like(t, dtype=torch.float32)
        ab = self._alphas_cumprod.to(t.device).gather(0, t.long())
        snr = ab / (1.0 - ab).clamp(min=1e-8)
        weight = torch.minimum(snr, torch.full_like(snr, self.config.snr_gamma)) / snr.clamp(min=1e-8)
        return weight

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        output: DiffusionModelOutput,
        target: EventEmbeddingTarget,
    ) -> LossResult:
        if output.model_out is None or output.loss_target is None or output.t is None:
            # Inference-shape output (cursor_token only). Run the
            # diffusion forward inline using bound model components.
            # Mirrors ``DiffusionDetector.forward_diffusion`` including
            # the optional Analog-Bits two-pass self-conditioning.
            if self._process is None or self._denoiser is None or self._n_steps is None:
                raise RuntimeError(
                    "DiffusionLoss got an inference-shape "
                    "DiffusionModelOutput but no model is bound. "
                    "Call loss.bind_model(model) once after construction."
                )
            cursor_token = output.cursor_token
            target_bin = target.target_bin
            B = cursor_token.size(0)
            x_0 = self._process.encode_x0(target_bin)
            t = torch.randint(
                0, self._n_steps, (B,),
                device=cursor_token.device, dtype=torch.long,
            )
            noise = torch.randn_like(x_0)
            x_t = self._process.q_sample(x_0, t, noise=noise)

            self_cond = bool(
                getattr(self._denoiser.config, "self_cond", False)
            )
            prev_x0_hat: torch.Tensor | None = None
            if self_cond and self._denoiser.training:
                mask = torch.rand(B, device=cursor_token.device) < 0.5
                if bool(mask.any()):
                    with torch.no_grad():
                        first_pass = self._denoiser(
                            cursor_token, x_t, t, prev_x0_hat=None,
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
            )
            loss_target = self._process.loss_target(x_0, x_t, t, noise=noise)
            x0_hat = self._process.predict_x0(model_out, x_t, t)
            logits = self._process.decode_to_logits(x0_hat)
            # Mutate frozen-dataclass output in place so downstream
            # metrics (OnsetMetric) see real argmax-able logits at
            # sampled t. Loss is a privileged consumer here.
            object.__setattr__(output, "model_out", model_out)
            object.__setattr__(output, "loss_target", loss_target)
            object.__setattr__(output, "t", t)
            object.__setattr__(output, "x_t", x_t)
            object.__setattr__(output, "logits", logits)

        per_sample = self._distance(output.model_out, output.loss_target)  # (B,)

        # SNR-weighted variant (always computed for diagnostics).
        snr_w = self._snr_weights(output.t)
        snr_weighted = (per_sample * snr_w).mean()

        # STOP weighting (matches OnsetLoss convention).
        stop_idx = output.loss_target.size(-1) - 1
        is_stop = target.target_bin == stop_idx
        stop_multiplier = torch.where(
            is_stop,
            torch.full_like(per_sample, self.config.stop_weight),
            torch.full_like(per_sample, 1.0),
        )
        per_sample_weighted = per_sample * stop_multiplier
        if self.config.snr_weighting:
            per_sample_weighted = per_sample_weighted * snr_w

        loss = per_sample_weighted.mean()

        # ── diagnostic metrics ───────────────────────────────────────
        metrics: dict[str, float] = {
            "loss": float(loss.detach()),
            "loss/snr_weighted": float(snr_weighted.detach()),
            "stop_rate": float(is_stop.float().mean().detach()),
        }

        # Per-t-bucket loss
        n_buckets = self.config.n_t_buckets
        t_max = float(output.t.max().detach()) if output.t.numel() else 1.0
        t_max = max(t_max, 1.0)            # avoid div-by-zero
        bucket = (output.t.float() / (t_max + 1e-8) * n_buckets).long().clamp(max=n_buckets - 1)
        for b in range(n_buckets):
            mask = bucket == b
            if mask.any():
                metrics[f"loss/per_t_q{b}"] = float(
                    per_sample[mask].mean().detach(),
                )
            else:
                metrics[f"loss/per_t_q{b}"] = 0.0

        # Per-step argmax match — diagnostic, not a loss component.
        if output.logits is not None:
            pred = output.logits.argmax(-1)
            metrics["argmax_match"] = float(
                (pred == target.target_bin).float().mean().detach(),
            )

        return LossResult(loss=loss, metrics=metrics)
