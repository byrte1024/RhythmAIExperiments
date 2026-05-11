"""Concrete denoiser heads.

Each denoiser is an ``nn.Module`` that maps
``(cursor_token, x_t, t) → model_out``. The output's interpretation
(predicted x_0, predicted noise, predicted v) is decided by the
paired ``DiffusionProcess.config.parameterization`` — the denoiser
itself is parameterization-agnostic.

Currently:
- ``MLPDenoiser`` — simple MLP over ``concat([cursor_token,
  time_embed(t), x_t])``. Cheapest baseline.

Future variants (conv-on-bin-axis, transformer-on-bin-axis, AdaLN-
based, …) live in this module under the same ABC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..domain.diffusion import DenoiserConfig, DenoiserHead


# ─────────────────────────── Sinusoidal time embedding ────────────────


def sinusoidal_time_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10_000.0,
) -> torch.Tensor:
    """Standard transformer-style sinusoidal embedding for an int
    timestep ``t``.

    ``t``: ``(B,)`` int64 or float32. ``dim``: must be even.
    Returns ``(B, dim)`` float32.
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal embedding dim must be even (got {dim})")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ─────────────────────────── MLPDenoiser ──────────────────────────────


@dataclass(frozen=True, slots=True)
class MLPDenoiserConfig(DenoiserConfig):
    """``MLPDenoiser`` hyperparameters.

    The denoiser is a 3-layer MLP:
        Linear(d_in, hidden_dim) → Swish → Dropout →
        Linear(hidden_dim, hidden_dim) → Swish → Dropout →
        Linear(hidden_dim, n_bins)
    where ``d_in = d_model + time_embed_dim + n_bins`` (concatenated
    conditioning + time embed + noised state).

    ``time_embed_proj_dim`` re-projects the raw sinusoidal time embed
    into a fixed-size vector (so ``time_embed_dim`` can stay small in
    the config and the projection learns the timestep representation).
    Set to 0 to skip the projection (use raw sinusoidal directly).
    """
    hidden_dim: int = 1536
    time_embed_proj_dim: int = 256
    n_layers: int = 3                   # number of hidden Linear+Swish blocks

    def __post_init__(self) -> None:
        DenoiserConfig.__post_init__(self)
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1 (got {self.hidden_dim})")
        if self.time_embed_proj_dim < 0:
            raise ValueError(
                f"time_embed_proj_dim must be >= 0 "
                f"(got {self.time_embed_proj_dim})"
            )
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1 (got {self.n_layers})")


class MLPDenoiser(DenoiserHead):
    """Cheapest viable denoiser. Concat-and-MLP architecture.

    Forward (no self-conditioning, ``self_cond=False``):
        time_emb = sinusoidal_time_embedding(t, time_embed_dim)
        time_emb = Linear(time_embed_dim, time_embed_proj_dim)(time_emb)
        h = concat([cursor_token, time_emb, x_t])
        h = MLP(h)  # n_layers Linear+Swish blocks
        return Linear(hidden_dim, n_bins)(h)

    With ``self_cond=True`` the input concat also includes a
    ``prev_x0_hat`` channel of shape ``(B, n_bins)``. Callers pass
    ``None`` (= zeros, matches no-prior-estimate) on the first pass
    or the previous step's predicted ``x_0`` on subsequent passes.
    """

    config: MLPDenoiserConfig

    def __init__(self, config: MLPDenoiserConfig):
        super().__init__(config)
        c = config
        proj_dim = c.time_embed_proj_dim if c.time_embed_proj_dim > 0 else c.time_embed_dim
        self._proj_dim = proj_dim
        if c.time_embed_proj_dim > 0:
            self.time_proj = nn.Sequential(
                nn.Linear(c.time_embed_dim, proj_dim),
                nn.SiLU(),
                nn.Linear(proj_dim, proj_dim),
            )
        else:
            self.time_proj = nn.Identity()

        # Self-conditioning adds one extra ``n_bins`` channel at the
        # input. When disabled, the first Linear is identical to the
        # #014 version — checkpoint shapes match.
        extra_in = c.n_bins if c.self_cond else 0
        d_in = c.d_model + proj_dim + c.n_bins + extra_in
        layers: list[nn.Module] = [
            nn.Linear(d_in, c.hidden_dim),
            nn.SiLU(),
            nn.Dropout(c.dropout),
        ]
        for _ in range(c.n_layers - 1):
            layers += [
                nn.Linear(c.hidden_dim, c.hidden_dim),
                nn.SiLU(),
                nn.Dropout(c.dropout),
            ]
        layers += [nn.Linear(c.hidden_dim, c.n_bins)]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        cursor_token: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prev_x0_hat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_emb_raw = sinusoidal_time_embedding(t, self.config.time_embed_dim)
        time_emb = self.time_proj(time_emb_raw)
        if self.config.self_cond:
            if prev_x0_hat is None:
                prev_x0_hat = torch.zeros_like(x_t)
            h = torch.cat([cursor_token, time_emb, x_t, prev_x0_hat], dim=-1)
        else:
            h = torch.cat([cursor_token, time_emb, x_t], dim=-1)
        return self.mlp(h)
