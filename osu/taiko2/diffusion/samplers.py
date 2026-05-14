"""Concrete diffusion samplers.

Each sampler runs the denoiser through a sequence of timesteps from
the noisy prior ``x_T`` to the clean output ``x_0``.

- ``DDPMSampler``: full Markov chain reverse process (Ho et al. 2020).
  ``n_inference_steps`` must equal ``schedule.n_steps``; visits every
  timestep with stochastic noise injection.
- ``DDIMSampler``: deterministic non-Markovian sampler (Song et al.
  2021). Subsamples timesteps; ``eta=0`` is fully deterministic,
  ``eta=1`` recovers DDPM. Standard for accelerated inference.

Future variants (ancestral, consistency, multistep solvers) live in
this module under the same ABC.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain.diffusion import (
    DenoiserHead,
    DiffusionProcess,
    DiffusionSampler,
    DiffusionSamplerConfig,
)


# ─────────────────────────── DDPMSampler ──────────────────────────────


class DDPMSampler(DiffusionSampler):
    """Standard Markov-chain reverse process.

    ``n_inference_steps`` must equal ``schedule.n_steps``. Visits
    timesteps ``[T-1, T-2, …, 0]``, applying the deterministic mean
    from ``process.reverse_step`` plus a noise injection sized to the
    posterior variance ``sigma_t``.

    ``config.eta`` controls the noise scale relative to the standard
    posterior variance: ``eta=1.0`` is the canonical DDPM, ``eta=0.0``
    drops the noise term (deterministic — but identical to a
    ``DDIMSampler`` only at full schedule length).
    """

    def __init__(
        self,
        config: DiffusionSamplerConfig,
        process: DiffusionProcess,
        denoiser: DenoiserHead,
    ):
        if config.n_inference_steps != process.schedule.n_steps:
            raise ValueError(
                "DDPMSampler requires n_inference_steps == "
                "schedule.n_steps; use DDIMSampler for accelerated "
                "sampling. Got "
                f"n_inference_steps={config.n_inference_steps}, "
                f"schedule.n_steps={process.schedule.n_steps}"
            )
        super().__init__(config, process, denoiser)

    def timesteps(self) -> torch.Tensor:
        n = self.process.schedule.n_steps
        return torch.arange(n - 1, -1, -1, dtype=torch.long)

    @torch.no_grad()
    def sample(
        self,
        cursor_token: torch.Tensor,
        x_T: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = cursor_token.device
        B = cursor_token.size(0)
        process = self.process
        denoiser = self.denoiser
        self_cond = bool(getattr(denoiser.config, "self_cond", False))

        if x_T is None:
            x_T = process.sample_prior(B, device, cursor_token.dtype)
        x_t = x_T
        prev_x0_hat: torch.Tensor | None = None

        ts = self.timesteps().to(device)
        for i, t_int in enumerate(ts):
            t = t_int.repeat(B)
            t_prev = (t_int - 1).repeat(B)
            model_out = denoiser(
                cursor_token, x_t, t,
                prev_x0_hat=prev_x0_hat if self_cond else None,
                audio_features=audio_features,
            )
            if self_cond:
                # Compute the predicted x_0 for the next step's
                # self-conditioning input. ``process.predict_x0``
                # handles the parameterization (x0 / noise / v).
                prev_x0_hat = process.predict_x0(model_out, x_t, t).detach()
            x_prev_mean = process.reverse_step(model_out, x_t, t, t_prev)
            if t_int > 0 and self.config.eta > 0.0:
                # DDPM-style noise injection scaled by eta.
                # We approximate the posterior std by recomputing it
                # from alphas_cumprod for a clean implementation; a
                # process-specific override could provide an exact
                # variance.
                noise = torch.randn_like(x_t) * self.config.eta
                x_t = x_prev_mean + self._posterior_std(t_int) * noise
            else:
                x_t = x_prev_mean
        return process.decode_to_logits(x_t)

    def _posterior_std(self, t_int: torch.Tensor) -> torch.Tensor:
        """Standard deviation of ``q(x_{t-1} | x_t, x_0)`` at step
        ``t_int``. Returns a scalar tensor.

        Formula: ``sqrt((1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t)``.
        """
        process = self.process
        if not hasattr(process, "_alphas_cumprod") or not hasattr(process, "_betas"):
            # Fallback: assume schedule-derived values are accessible
            # via a method on the process. Concrete processes can
            # override _posterior_std on the sampler if they know
            # better.
            return torch.tensor(0.0, device=t_int.device)
        ab_t = process._alphas_cumprod[t_int]
        ab_prev = (
            process._alphas_cumprod[t_int - 1]
            if int(t_int) > 0
            else torch.tensor(1.0, device=t_int.device)
        )
        beta = process._betas[t_int]
        var = (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8) * beta
        return var.clamp(min=0.0).sqrt()


# ─────────────────────────── DDIMSampler ──────────────────────────────


@dataclass(frozen=True, slots=True)
class DDIMSamplerConfig(DiffusionSamplerConfig):
    """``DDIMSampler`` config.

    ``timestep_spacing`` decides how the ``n_inference_steps``
    timesteps are subsampled from the schedule's ``n_steps``:
    - ``"linspace"`` — evenly spaced from 0 to T-1 (DDIM paper default).
    - ``"trailing"`` — evenly spaced from 1 to T (Stable Diffusion).
    - ``"leading"``  — evenly spaced from 0 to T-1 with leading bias.

    For ``n_inference_steps == n_steps`` all three coincide.

    ``time_offset`` enables Analog-Bits-style asymmetric time
    intervals (Chen, Zhang, Hinton 2022). At sampling time, the
    timestep handed to the denoiser is ``min(t + time_offset, T-1)``
    while the reverse-process transition still goes ``t → t_prev``.
    The offset partly compensates for the few-step DDIM truncation
    error in mid-trajectory states by querying the denoiser as if
    the input were slightly noisier than it actually is. ``0.0``
    (default) matches DDIM's standard behavior. Values in 0.5–2.0
    range typically help when ``n_inference_steps`` is small (≤ 8).
    """
    timestep_spacing: str = "linspace"
    time_offset: float = 0.0

    def __post_init__(self) -> None:
        DiffusionSamplerConfig.__post_init__(self)
        if self.timestep_spacing not in {"linspace", "trailing", "leading"}:
            raise ValueError(
                f"timestep_spacing must be one of "
                f"linspace/trailing/leading "
                f"(got {self.timestep_spacing!r})"
            )
        if self.time_offset < 0.0:
            raise ValueError(
                f"time_offset must be >= 0 (got {self.time_offset})"
            )


class DDIMSampler(DiffusionSampler):
    """Deterministic / partially-stochastic non-Markovian sampler
    (Song et al. 2021).

    Operates by selecting a sub-sequence of timesteps and applying
    ``process.reverse_step`` (already DDIM-style for arbitrary
    ``t_prev`` in our process). With ``eta=0.0`` this is fully
    deterministic.
    """

    config: DDIMSamplerConfig

    def __init__(
        self,
        config: DDIMSamplerConfig,
        process: DiffusionProcess,
        denoiser: DenoiserHead,
    ):
        super().__init__(config, process, denoiser)

    def timesteps(self) -> torch.Tensor:
        T = self.process.schedule.n_steps
        n = self.config.n_inference_steps
        spacing = self.config.timestep_spacing
        if spacing == "linspace":
            ts = torch.linspace(0, T - 1, n, dtype=torch.float32).round().long()
        elif spacing == "trailing":
            ts = torch.linspace(1, T, n, dtype=torch.float32).round().long() - 1
            ts = ts.clamp(min=0, max=T - 1)
        else:  # "leading"
            step = T // n
            ts = torch.arange(0, n, dtype=torch.long) * step
        # Descending order for reverse process.
        ts = torch.flip(torch.unique(ts, sorted=True), dims=(0,))
        return ts

    @torch.no_grad()
    def sample(
        self,
        cursor_token: torch.Tensor,
        x_T: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = cursor_token.device
        B = cursor_token.size(0)
        process = self.process
        denoiser = self.denoiser
        self_cond = bool(getattr(denoiser.config, "self_cond", False))
        time_offset = float(self.config.time_offset)
        T_max = int(self.process.schedule.n_steps) - 1

        if x_T is None:
            x_T = process.sample_prior(B, device, cursor_token.dtype)
        x_t = x_T
        prev_x0_hat: torch.Tensor | None = None

        ts = self.timesteps().to(device)
        for i, t_int in enumerate(ts):
            t = t_int.repeat(B)
            t_prev_int = ts[i + 1] if i + 1 < len(ts) else torch.tensor(-1, device=device)
            t_prev = t_prev_int.repeat(B)
            # Asymmetric time intervals: query the denoiser as if the
            # input were slightly noisier (t + offset), but still
            # transition along t → t_prev. Clamp to schedule range.
            if time_offset > 0.0:
                t_query = (t.float() + time_offset).round().long().clamp(
                    min=0, max=T_max,
                )
            else:
                t_query = t
            model_out = denoiser(
                cursor_token, x_t, t_query,
                prev_x0_hat=prev_x0_hat if self_cond else None,
                audio_features=audio_features,
            )
            if self_cond:
                # Use the predicted x_0 implied by this step's denoiser
                # output as the self-conditioning input for the next
                # step. ``predict_x0`` handles the parameterization.
                # We pass ``t`` (not ``t_query``) so the conversion is
                # consistent with the actual noise level of ``x_t``.
                prev_x0_hat = process.predict_x0(model_out, x_t, t).detach()
            x_t = process.reverse_step(model_out, x_t, t, t_prev)
            if self.config.eta > 0.0 and i + 1 < len(ts):
                # Scaled stochastic component (DDIM eta > 0 case).
                # The full DDIM formula has a specific eta-dependent
                # variance; for now we use an approximate posterior
                # std consistent with the DDPM formula.
                noise = torch.randn_like(x_t) * self.config.eta
                if hasattr(process, "_alphas_cumprod") and hasattr(process, "_betas"):
                    ab_t = process._alphas_cumprod[t_int]
                    ab_prev = (
                        process._alphas_cumprod[t_prev_int]
                        if int(t_prev_int) >= 0
                        else torch.tensor(1.0, device=device)
                    )
                    var = (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8) * (1.0 - ab_t / ab_prev.clamp(min=1e-8))
                    std = var.clamp(min=0.0).sqrt()
                    x_t = x_t + std * noise
        return process.decode_to_logits(x_t)

    @torch.no_grad()
    def sample_with_intermediates(
        self,
        cursor_token: torch.Tensor,
        x_T: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the DDIM rollout and return per-step predicted x0 maps.

        Returns ``{"final": (B, n_bins), "per_step": (T_inf, B, n_bins)}``
        where ``per_step[k]`` is the predicted x0 (clipped to logits via
        ``process.decode_to_logits``) at step ``k`` (k=0 is the noisiest
        predicted x0; k=T_inf-1 is the final).
        """
        device = cursor_token.device
        B = cursor_token.size(0)
        process = self.process
        denoiser = self.denoiser
        self_cond = bool(getattr(denoiser.config, "self_cond", False))
        time_offset = float(self.config.time_offset)
        T_max = int(self.process.schedule.n_steps) - 1

        if x_T is None:
            x_T = process.sample_prior(B, device, cursor_token.dtype)
        x_t = x_T
        prev_x0_hat: torch.Tensor | None = None

        ts = self.timesteps().to(device)
        per_step: list[torch.Tensor] = []
        for i, t_int in enumerate(ts):
            t = t_int.repeat(B)
            t_prev_int = ts[i + 1] if i + 1 < len(ts) else torch.tensor(-1, device=device)
            t_prev = t_prev_int.repeat(B)
            if time_offset > 0.0:
                t_query = (t.float() + time_offset).round().long().clamp(
                    min=0, max=T_max,
                )
            else:
                t_query = t
            model_out = denoiser(
                cursor_token, x_t, t_query,
                prev_x0_hat=prev_x0_hat if self_cond else None,
                audio_features=audio_features,
            )
            x0_hat = process.predict_x0(model_out, x_t, t)
            per_step.append(process.decode_to_logits(x0_hat).detach().clamp(0.0, 1.0))
            if self_cond:
                prev_x0_hat = x0_hat.detach()
            x_t = process.reverse_step(model_out, x_t, t, t_prev)
            if self.config.eta > 0.0 and i + 1 < len(ts):
                noise = torch.randn_like(x_t) * self.config.eta
                if hasattr(process, "_alphas_cumprod") and hasattr(process, "_betas"):
                    ab_t = process._alphas_cumprod[t_int]
                    ab_prev = (
                        process._alphas_cumprod[t_prev_int]
                        if int(t_prev_int) >= 0
                        else torch.tensor(1.0, device=device)
                    )
                    var = (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8) * (1.0 - ab_t / ab_prev.clamp(min=1e-8))
                    std = var.clamp(min=0.0).sqrt()
                    x_t = x_t + std * noise
        final = process.decode_to_logits(x_t).clamp(0.0, 1.0)
        return {
            "final": final,
            "per_step": torch.stack(per_step, dim=0),
        }
