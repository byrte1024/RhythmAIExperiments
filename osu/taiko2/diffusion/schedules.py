"""Concrete noise schedules.

Each schedule produces a ``(n_steps,)`` float32 tensor of ``betas``
that the paired ``DiffusionProcess`` interprets. Schedules are pure
math — no learnable parameters, no gradients.

Conventions:
- ``t = 0`` is the cleanest end (lowest beta).
- ``t = n_steps - 1`` is the noisiest end (highest beta).
- All betas are in ``(0, 1)``.

References:
- Linear: Ho et al., "Denoising Diffusion Probabilistic Models" (2020).
- Cosine: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic
  Models" (2021).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..domain.diffusion import NoiseSchedule, NoiseScheduleConfig


# ─────────────────────────── LinearSchedule ───────────────────────────


@dataclass(frozen=True, slots=True)
class LinearScheduleConfig(NoiseScheduleConfig):
    """DDPM-original linear schedule.

    Default ``beta_start=1e-4`` and ``beta_end=2e-2`` from Ho et al. 2020,
    which works well for n_steps=1000. For shorter schedules
    (n_steps=64), the same range is too gentle — ``alpha_bar`` only
    decays to ~0.6 instead of ~0. Use ``CosineSchedule`` or scaled
    betas for short schedules.
    """
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        # slots=True + frozen dataclass inheritance breaks super() in
        # __post_init__ on this CPython version; call the parent
        # method explicitly. Same workaround used in
        # OnsetAugmentedConfig / ConformerDetectorConfig.
        NoiseScheduleConfig.__post_init__(self)
        if not 0.0 < self.beta_start < 1.0:
            raise ValueError(
                f"beta_start must be in (0, 1) (got {self.beta_start})"
            )
        if not 0.0 < self.beta_end < 1.0:
            raise ValueError(
                f"beta_end must be in (0, 1) (got {self.beta_end})"
            )
        if self.beta_start >= self.beta_end:
            raise ValueError(
                f"beta_start ({self.beta_start}) must be < "
                f"beta_end ({self.beta_end})"
            )


class LinearSchedule(NoiseSchedule):
    config: LinearScheduleConfig

    def __init__(self, config: LinearScheduleConfig):
        super().__init__(config)
        self._betas = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.n_steps,
            dtype=torch.float32,
        )

    def betas(self) -> torch.Tensor:
        return self._betas


# ─────────────────────────── CosineSchedule ───────────────────────────


@dataclass(frozen=True, slots=True)
class CosineScheduleConfig(NoiseScheduleConfig):
    """Improved-DDPM cosine schedule (Nichol & Dhariwal 2021).

    Decays ``alpha_bar`` along a half-cosine curve so that the
    schedule devotes more steps to the high-signal end (where the
    denoising task is easier and finer detail matters) and rushes
    through the noisy end. Better than linear at small n_steps.

    ``s`` is a small offset preventing ``beta_0`` from being too
    close to zero. Standard value 0.008 from the paper.
    """
    s: float = 0.008
    max_beta: float = 0.999            # clamp to avoid degenerate steps

    def __post_init__(self) -> None:
        NoiseScheduleConfig.__post_init__(self)
        if self.s < 0.0:
            raise ValueError(f"s must be >= 0 (got {self.s})")
        if not 0.0 < self.max_beta < 1.0:
            raise ValueError(
                f"max_beta must be in (0, 1) (got {self.max_beta})"
            )


class CosineSchedule(NoiseSchedule):
    config: CosineScheduleConfig

    def __init__(self, config: CosineScheduleConfig):
        super().__init__(config)
        T = config.n_steps
        s = config.s
        # alpha_bar(t) = cos((t/T + s) / (1 + s) * π/2)^2
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
        alpha_bar = f / f[0]                                # normalize to 1.0 at t=0
        # beta_t = 1 - alpha_bar(t) / alpha_bar(t-1)
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(min=1e-8, max=config.max_beta)
        self._betas = betas.float()

    def betas(self) -> torch.Tensor:
        return self._betas
