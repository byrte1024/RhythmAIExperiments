"""Concrete diffusion processes.

Each process implements the forward (q) and reverse-step machinery for
a particular noise space. Currently:

- ``GaussianContinuousProcess``: standard DDPM operating on a
  continuous representation of the bin distribution. ``x_0`` is a
  ``(B, n_bins)`` float32 tensor (a softened one-hot or scaled one-hot);
  noising is additive Gaussian; reverse step uses the closed-form
  posterior under the Gaussian assumption.

Future processes (D3PM absorbing, GMM, …) live in this module under
the same ABCs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..domain.diffusion import (
    DiffusionProcess,
    DiffusionProcessConfig,
    NoiseSchedule,
)


# ─────────────────────────── GaussianContinuous ───────────────────────


@dataclass(frozen=True, slots=True)
class GaussianContinuousProcessConfig(DiffusionProcessConfig):
    """Configuration for ``GaussianContinuousProcess``.

    ``x0_scale`` controls the magnitude of the encoded ``x_0``. With
    one-hot ``x_0 ∈ {0, 1}^n_bins`` the per-bin variance is tiny
    (~1/n_bins after normalization) and Gaussian noise can swamp
    it. Scaling ``x_0`` by a constant (typically 1.0–10.0) keeps
    SNR balanced. Standard 2.0 is a reasonable default for n_bins
    around 500.
    """
    x0_scale: float = 2.0

    def __post_init__(self) -> None:
        DiffusionProcessConfig.__post_init__(self)
        if self.x0_scale <= 0.0:
            raise ValueError(f"x0_scale must be > 0 (got {self.x0_scale})")


class GaussianContinuousProcess(DiffusionProcess):
    """Standard DDPM in continuous space over the ``n_bins``-dim
    distribution.

    ``x_t`` is a ``(B, n_bins)`` float32 tensor. The forward process
    is ``q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) I)``.

    Three parameterizations supported (via ``config.parameterization``):

    - ``"x0"``     — denoiser predicts the clean ``x_0`` directly.
    - ``"noise"``  — denoiser predicts the additive noise ``ε``.
    - ``"v"``      — denoiser predicts ``v = sqrt(alpha_bar) * ε -
                      sqrt(1 - alpha_bar) * x_0`` (Salimans & Ho 2022).
    """

    config: GaussianContinuousProcessConfig

    def __init__(
        self,
        config: GaussianContinuousProcessConfig,
        schedule: NoiseSchedule,
    ):
        super().__init__(config, schedule)
        # Cache schedule tensors. Float64 internally for numerical
        # stability of the cumprod; cast at use time.
        betas = schedule.betas().double()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Pad a 1.0 at the front so we can index "alpha_bar at t-1".
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, dtype=alphas_cumprod.dtype),
            alphas_cumprod[:-1],
        ])
        self._betas = betas.float()
        self._alphas = alphas.float()
        self._alphas_cumprod = alphas_cumprod.float()
        self._alphas_cumprod_prev = alphas_cumprod_prev.float()
        self._sqrt_alphas_cumprod = alphas_cumprod.sqrt().float()
        self._sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt().float()

    # ── helpers ──────────────────────────────────────────────────────

    def _gather(self, table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Pick scheduler values per-sample, broadcastable with x shape."""
        out = table.to(t.device).gather(0, t.long())               # (B,)
        return out.view(-1, 1)                                     # (B, 1)

    # ── forward noising ──────────────────────────────────────────────

    def encode_x0(self, target_bin: torch.Tensor) -> torch.Tensor:
        """One-hot in ``(B, n_bins)``, scaled by ``x0_scale``."""
        oh = F.one_hot(target_bin.long(), num_classes=self.config.n_bins).float()
        return oh * self.config.x0_scale

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self._gather(self._sqrt_alphas_cumprod, t)
        sqrt_om = self._gather(self._sqrt_one_minus_alphas_cumprod, t)
        return sqrt_ab * x_0 + sqrt_om * noise

    # ── parameterization conversions ─────────────────────────────────

    def loss_target(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        param = self.config.parameterization
        if param == "x0":
            return x_0
        if param == "noise":
            if noise is None:
                # Recover noise from (x_t, x_0) deterministically
                # x_t = sqrt(ab) * x_0 + sqrt(1-ab) * eps  =>
                # eps = (x_t - sqrt(ab) * x_0) / sqrt(1-ab)
                sqrt_ab = self._gather(self._sqrt_alphas_cumprod, t)
                sqrt_om = self._gather(self._sqrt_one_minus_alphas_cumprod, t)
                return (x_t - sqrt_ab * x_0) / sqrt_om.clamp(min=1e-8)
            return noise
        if param == "v":
            # v = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * x_0
            sqrt_ab = self._gather(self._sqrt_alphas_cumprod, t)
            sqrt_om = self._gather(self._sqrt_one_minus_alphas_cumprod, t)
            if noise is None:
                eps = (x_t - sqrt_ab * x_0) / sqrt_om.clamp(min=1e-8)
            else:
                eps = noise
            return sqrt_ab * eps - sqrt_om * x_0
        raise ValueError(f"unknown parameterization {param!r}")

    def predict_x0(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        param = self.config.parameterization
        if param == "x0":
            return model_out
        sqrt_ab = self._gather(self._sqrt_alphas_cumprod, t)
        sqrt_om = self._gather(self._sqrt_one_minus_alphas_cumprod, t)
        if param == "noise":
            # x_0 = (x_t - sqrt(1 - ab) * eps) / sqrt(ab)
            return (x_t - sqrt_om * model_out) / sqrt_ab.clamp(min=1e-8)
        if param == "v":
            # v = sqrt(ab) * eps - sqrt(1-ab) * x_0
            # eps = (x_t - sqrt(ab) * x_0) / sqrt(1-ab)
            # Sub: v = sqrt(ab) * (x_t - sqrt(ab) * x_0) / sqrt(1-ab) - sqrt(1-ab) * x_0
            #         => x_0 = sqrt(ab) * x_t - sqrt(1-ab) * v
            return sqrt_ab * x_t - sqrt_om * model_out
        raise ValueError(f"unknown parameterization {param!r}")

    # ── reverse process ──────────────────────────────────────────────

    def reverse_step(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM-style mean of ``q(x_{t_prev} | x_t, x_0_hat)``.

        Standard formula:
            mean = (sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)) * x_0_hat
                 + (sqrt(alpha_t)        * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * x_t
        For ``t_prev = t - 1`` exactly. For ``t_prev != t - 1`` (DDIM
        skipping) we use the DDIM closed form, which is also produced
        here using ``alpha_bar`` at ``t_prev`` directly.

        This method is deterministic. Stochasticity (the noise term)
        is added by the sampler.
        """
        x_0_hat = self.predict_x0(model_out, x_t, t)
        sqrt_ab_t = self._gather(self._sqrt_alphas_cumprod, t)
        # When t_prev < 0 we treat it as the clean step (x_0).
        ab_prev = torch.where(
            t_prev >= 0,
            self._alphas_cumprod.to(t.device).gather(0, t_prev.clamp(min=0).long()),
            torch.ones_like(t_prev, dtype=self._alphas_cumprod.dtype),
        ).view(-1, 1)
        sqrt_ab_prev = ab_prev.sqrt()
        sqrt_om_prev = (1.0 - ab_prev).clamp(min=0.0).sqrt()

        # DDIM-style direction-to-x_t term:
        # x_{t_prev} = sqrt(ab_prev) * x_0_hat + sqrt(1 - ab_prev) * eps_hat
        # where eps_hat is recovered from x_0_hat and x_t
        sqrt_om_t = self._gather(self._sqrt_one_minus_alphas_cumprod, t)
        eps_hat = (x_t - sqrt_ab_t * x_0_hat) / sqrt_om_t.clamp(min=1e-8)
        return sqrt_ab_prev * x_0_hat + sqrt_om_prev * eps_hat

    # ── prior + decoding ─────────────────────────────────────────────

    def sample_prior(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.randn(
            batch_size, self.config.n_bins, device=device, dtype=dtype,
        )

    def decode_to_logits(self, x_0_hat: torch.Tensor) -> torch.Tensor:
        # Return as-is; the caller argmaxes to get the bin index.
        # Dividing by x0_scale doesn't change the argmax, but produces
        # something closer to a one-hot for inspection.
        return x_0_hat / self.config.x0_scale
