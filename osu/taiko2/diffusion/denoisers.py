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
import torch.nn.functional as F

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


def sinusoidal_pos_embed(
    n_positions: int, dim: int, max_period: float = 10_000.0,
) -> torch.Tensor:
    """Static sinusoidal positional embedding over ``n_positions``.

    Returns a ``(n_positions, dim)`` float32 tensor on CPU. Mirrors
    :func:`sinusoidal_time_embedding` but indexed by an integer
    position rather than a timestep — both end up using the same cos
    || sin layout so concatenating them with other channels behaves
    consistently.
    """
    if dim % 2 != 0:
        raise ValueError(f"pos embed dim must be even (got {dim})")
    half = dim // 2
    positions = torch.arange(n_positions, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32)
        / half
    )
    args = positions.unsqueeze(-1) * freqs.unsqueeze(0)            # (N, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (N, dim)


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
        audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``audio_features`` is part of the ABC for #016 framewise
        # diffusion. The MLP denoiser is the cheapest baseline and
        # ignores it (it conditions on ``cursor_token`` only).
        del audio_features
        time_emb_raw = sinusoidal_time_embedding(t, self.config.time_embed_dim)
        time_emb = self.time_proj(time_emb_raw)
        if self.config.self_cond:
            if prev_x0_hat is None:
                prev_x0_hat = torch.zeros_like(x_t)
            h = torch.cat([cursor_token, time_emb, x_t, prev_x0_hat], dim=-1)
        else:
            h = torch.cat([cursor_token, time_emb, x_t], dim=-1)
        return self.mlp(h)


# ─────────────────────────── Conv1DDenoiser ───────────────────────────


@dataclass(frozen=True, slots=True)
class Conv1DDenoiserConfig(DenoiserConfig):
    """1D conv-on-bin-axis denoiser for framewise diffusion (#016).

    Per-bin input channels (concatenated along ``dim=1`` of a
    ``(B, C, n_bins)`` tensor):

      x_t                 (1 ch)
      prev_x0_hat         (1 ch, if ``self_cond=True``)
      positional_embed    (``pos_embed_dim`` ch, sinusoidal over the
                          bin axis)
      audio_features      (``audio_feature_dim`` ch, linearly
                          upsampled from ``audio_token_count`` to
                          ``n_bins``)
      cursor_token        (``cursor_proj_dim`` ch, broadcast)
      time_embed          (``time_proj_dim`` ch, broadcast)

    Followed by ``len(conv_kernels)`` ``Conv1d`` blocks with FiLM
    conditioning (gamma/beta from ``cursor_token + time_embed``,
    zero-initialised so the stack is identity at init).
    """
    audio_feature_dim: int = 384       # input dim of audio_features
    audio_token_count: int = 125       # T_audio of audio_features
    pos_embed_dim: int = 32
    cursor_proj_dim: int = 32
    time_proj_dim: int = 32
    conv_channels: int = 256
    conv_kernels: tuple[int, ...] = (31, 15, 15)
    # n_bins inherited (=501 default; framewise uses 500 typically).
    # self_cond inherited.

    def __post_init__(self) -> None:
        DenoiserConfig.__post_init__(self)
        if self.audio_feature_dim < 1:
            raise ValueError(
                f"audio_feature_dim must be >= 1 (got {self.audio_feature_dim})"
            )
        if self.audio_token_count < 1:
            raise ValueError(
                f"audio_token_count must be >= 1 (got {self.audio_token_count})"
            )
        if self.pos_embed_dim < 2 or self.pos_embed_dim % 2 != 0:
            raise ValueError(
                f"pos_embed_dim must be even and >= 2 (got {self.pos_embed_dim})"
            )
        if self.cursor_proj_dim < 1:
            raise ValueError(
                f"cursor_proj_dim must be >= 1 (got {self.cursor_proj_dim})"
            )
        if self.time_proj_dim < 1:
            raise ValueError(
                f"time_proj_dim must be >= 1 (got {self.time_proj_dim})"
            )
        if self.conv_channels < 1:
            raise ValueError(
                f"conv_channels must be >= 1 (got {self.conv_channels})"
            )
        if len(self.conv_kernels) < 1:
            raise ValueError("conv_kernels must be non-empty")
        if any(k % 2 == 0 for k in self.conv_kernels):
            raise ValueError(
                f"conv_kernels must all be odd (got {self.conv_kernels})"
            )
        if self.conv_channels % 8 != 0:
            raise ValueError(
                f"conv_channels must be a multiple of 8 for GroupNorm "
                f"(got {self.conv_channels})"
            )


class Conv1DDenoiser(DenoiserHead):
    """Conv-on-bin-axis denoiser with FiLM and audio context.

    Computes per-bin features by stacking conditioning channels and
    running a small Conv1d stack along the bin axis. Designed for
    framewise diffusion (#016) where the output is an activation
    map ``(B, n_bins)`` and the bins correspond to future-time
    locations covered by ``audio_features``.

    Audio features are linearly interpolated from
    ``audio_token_count`` to ``n_bins`` along the time axis — the
    upsampling assumption is that the audio trunk's temporal
    resolution is coarser than the bin axis but covers the same
    window.
    """

    config: Conv1DDenoiserConfig

    def __init__(self, config: Conv1DDenoiserConfig):
        super().__init__(config)
        c = config

        # Time embedding projection.
        self.time_proj = nn.Sequential(
            nn.Linear(c.time_embed_dim, c.time_proj_dim),
            nn.SiLU(),
            nn.Linear(c.time_proj_dim, c.time_proj_dim),
        )
        # Cursor projection.
        self.cursor_proj = nn.Linear(c.d_model, c.cursor_proj_dim)

        # Positional embedding over bin axis, precomputed buffer.
        pos = sinusoidal_pos_embed(c.n_bins, c.pos_embed_dim)         # (n_bins, pos_embed_dim)
        self.register_buffer("pos_embed", pos)

        # Input channel count.
        in_ch = (
            1
            + (1 if c.self_cond else 0)
            + c.pos_embed_dim
            + c.audio_feature_dim
            + c.cursor_proj_dim
            + c.time_proj_dim
        )

        # FiLM layers (one per conv block). Zero-init so the stack
        # is identity at init.
        film_in = c.cursor_proj_dim + c.time_proj_dim
        self.film_layers = nn.ModuleList([
            nn.Linear(film_in, 2 * c.conv_channels)
            for _ in range(len(c.conv_kernels))
        ])
        for film in self.film_layers:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        # Conv stack.
        convs: list[nn.Module] = []
        cur_in = in_ch
        for k in c.conv_kernels:
            convs.append(
                nn.Conv1d(cur_in, c.conv_channels, kernel_size=k, padding=k // 2)
            )
            cur_in = c.conv_channels
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, c.conv_channels) for _ in c.conv_kernels
        ])
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(c.dropout)
        self.out_proj = nn.Conv1d(c.conv_channels, 1, kernel_size=1)

    def forward(
        self,
        cursor_token: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        prev_x0_hat: torch.Tensor | None = None,
        audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio_features is None:
            raise ValueError(
                "Conv1DDenoiser requires audio_features "
                "(B, T_audio, audio_feature_dim); got None."
            )
        c = self.config
        B = x_t.size(0)
        n_bins = c.n_bins

        # Time embed + projection.
        time_emb_raw = sinusoidal_time_embedding(t, c.time_embed_dim)
        time_emb = self.time_proj(time_emb_raw)                       # (B, time_proj_dim)
        # Cursor projection.
        cursor_emb = self.cursor_proj(cursor_token)                   # (B, cursor_proj_dim)

        # Build per-bin channels.
        per_bin: list[torch.Tensor] = [x_t.unsqueeze(1)]              # (B, 1, n_bins)
        if c.self_cond:
            if prev_x0_hat is None:
                prev_x0_hat = torch.zeros_like(x_t)
            per_bin.append(prev_x0_hat.unsqueeze(1))                  # (B, 1, n_bins)

        # Positional embed: (n_bins, pos_dim) -> (B, pos_dim, n_bins).
        pos = self.pos_embed.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        per_bin.append(pos)

        # Audio features: (B, T_audio, audio_feature_dim) -> (B, audio_feature_dim, n_bins).
        audio = audio_features.transpose(1, 2)                        # (B, audio_feat_dim, T_audio)
        audio = F.interpolate(audio, size=n_bins, mode="linear", align_corners=False)
        per_bin.append(audio)

        # Cursor broadcast.
        cur_brd = cursor_emb.unsqueeze(-1).expand(-1, -1, n_bins)
        per_bin.append(cur_brd)
        # Time broadcast.
        time_brd = time_emb.unsqueeze(-1).expand(-1, -1, n_bins)
        per_bin.append(time_brd)

        h = torch.cat(per_bin, dim=1)                                 # (B, in_ch, n_bins)

        # FiLM input.
        film_in = torch.cat([cursor_emb, time_emb], dim=-1)

        for conv, norm, film in zip(self.convs, self.norms, self.film_layers):
            h = conv(h)
            h = norm(h)
            gamma_beta = film(film_in)                                # (B, 2 * conv_channels)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            h = self.act(h)
            h = self.dropout(h)
        out = self.out_proj(h).squeeze(1)                             # (B, n_bins)
        return out
