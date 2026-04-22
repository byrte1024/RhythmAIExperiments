"""Shared nn.Module primitives for taiko2 models.

These are concrete PyTorch modules (not ABCs). They're factored out so
any future model that needs them — sinusoidal position embeddings,
FiLM conditioning, or the 4× conv stem over mel — reuses a single
implementation instead of re-rolling its own.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Classic sin/cos positional encoding over integer positions.

    Accepts positions of arbitrary shape; the output adds a trailing
    `d_model` axis. For a batch shape `(B, T)` this returns `(B, T, D)`.
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        scale = math.log(10000.0) / (half - 1)
        freqs = torch.exp(
            torch.arange(half, device=positions.device, dtype=torch.float32)
            * -scale
        )
        shape = positions.shape
        flat = positions.float().reshape(-1, 1)
        emb = flat * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.reshape(*shape, self.d_model)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation.

    Predicts `(γ, β)` from a conditioning vector and applies
    `x * (1 + γ) + β` broadcasting across the sequence axis.
    Weight + bias initialized to zero so the module starts as identity;
    this keeps early-training stable regardless of conditioning scale.
    """

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feat_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D), cond: (B, cond_dim) → (B, T, D)."""
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)


class AudioConvStem(nn.Module):
    """mel (B, n_mels, T) → (B, T/4, d_model) via two stride-2 convs.

    4× temporal downsample. Uses a GroupNorm between the two convs and
    a LayerNorm on the output. Caller is responsible for adding any
    positional encoding afterward.
    """

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) → (B, T/4, d_model)."""
        x = self.conv(mel).transpose(1, 2)
        return self.norm(x)
