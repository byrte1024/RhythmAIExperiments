"""Onset typing transformer — predicts D/K type and normal/big strength.

Small bidirectional transformer over a window of onset tokens. Past
tokens carry known D/K and big/normal labels; target and future tokens
carry UNK. The model attends to the full window and predicts the
target onset's type and strength from the center token.

Optional temporal attention bias: Gaussian decay on pairwise IOI
distance, injected directly into scaled dot-product attention via a
custom transformer layer (not through PyTorch's src_mask, which
interacts badly with src_key_padding_mask).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..domain.model import Model
from ..domain.typing import (
    TypingInput,
    TypingModelConfig,
    TypingOutput,
)


# ─────────────────────────── custom attention ────────────────────────

class TemporalMultiheadAttention(nn.Module):
    """Multi-head attention with additive temporal bias injected
    directly into the QK^T scores before softmax.

    The bias is a (B, W, W) matrix of Gaussian-decayed pairwise
    onset distances, broadcast across heads. Padding is handled by
    setting masked positions to -inf in the attention scores.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                          # (B, W, d_model)
        temporal_bias: torch.Tensor | None = None, # (B, W, W)
        padding_mask: torch.Tensor | None = None,  # (B, W) bool, True=pad
    ) -> torch.Tensor:
        B, W, _ = x.shape
        H, D = self.n_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, W, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, W, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, W, W)

        if temporal_bias is not None:
            attn = attn + temporal_bias.unsqueeze(1)  # broadcast over heads

        if padding_mask is not None:
            # (B, W) -> (B, 1, 1, W): mask columns (keys) that are padded
            mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, W, self.d_model)
        return self.out_proj(out)


class TemporalTransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer using TemporalMultiheadAttention."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TemporalMultiheadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), temporal_bias, padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class TemporalTransformerEncoder(nn.Module):
    """Stack of TemporalTransformerLayers."""

    def __init__(
        self, n_layers: int, d_model: int, n_heads: int,
        d_ff: int, dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        temporal_bias: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, temporal_bias, padding_mask)
        return x


# ─────────────────────────── main model ──────────────────────────────

class TypingTransformer(Model[TypingModelConfig, TypingInput, TypingOutput]):

    def __init__(self, config: TypingModelConfig):
        super().__init__(config)
        c = config

        self.mel_proj = nn.Sequential(
            nn.Linear(c.n_mels * c.mel_patch, c.d_mel),
            nn.ReLU(),
        )
        self.ioi_proj = nn.Sequential(
            nn.Linear(3, c.d_ioi),
            nn.ReLU(),
        )
        self.kind_emb = nn.Embedding(3, c.d_kind)
        self.big_emb = nn.Embedding(3, c.d_kind)
        self.pos_emb = nn.Embedding(c.window, c.d_pos)

        feat_dim = c.d_mel + c.d_ioi + c.d_kind + c.d_kind + c.d_pos
        self.input_proj = nn.Linear(feat_dim, c.d_model)

        if c.temporal_bias:
            self.transformer = TemporalTransformerEncoder(
                n_layers=c.n_layers, d_model=c.d_model,
                n_heads=c.n_heads, d_ff=c.d_ff, dropout=c.dropout,
            )
            self.temporal_log_sigma = nn.Parameter(
                torch.tensor(math.log(c.temporal_sigma)),
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.n_heads,
                dim_feedforward=c.d_ff,
                dropout=c.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=c.n_layers,
            )

        self.type_head = nn.Linear(c.d_model, 1)
        self.strength_head = nn.Linear(c.d_model, 1)

    def _build_temporal_bias(self, onset_bins: torch.Tensor) -> torch.Tensor:
        """Gaussian decay: (B, W, W) additive bias for attention."""
        diff = onset_bins.unsqueeze(2) - onset_bins.unsqueeze(1)
        dist_sq = diff.float() ** 2
        sigma = torch.exp(self.temporal_log_sigma).clamp(min=1.0, max=500.0)
        return -dist_sq / (2 * sigma ** 2)

    def predict(self, x: TypingInput) -> TypingOutput:
        return self.forward(
            x.mel_patches, x.ioi_features, x.kind_labels,
            x.big_labels, x.positions, x.mask, x.onset_bins,
        )

    def forward(
        self,
        mel_patches: torch.Tensor,
        ioi_features: torch.Tensor,
        kind_labels: torch.Tensor,
        big_labels: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
        onset_bins: torch.Tensor,
    ) -> TypingOutput:
        mel_feat = self.mel_proj(mel_patches)
        ioi_feat = self.ioi_proj(ioi_features)
        kind_feat = self.kind_emb(kind_labels)
        big_feat = self.big_emb(big_labels)
        pos_feat = self.pos_emb(positions)

        combined = torch.cat(
            [mel_feat, ioi_feat, kind_feat, big_feat, pos_feat], dim=-1,
        )
        tokens = self.input_proj(combined)

        if self.config.temporal_bias:
            bias = self._build_temporal_bias(onset_bins)
            out = self.transformer(tokens, temporal_bias=bias, padding_mask=mask)
        else:
            out = self.transformer(tokens, src_key_padding_mask=mask)

        center = self.config.past_context
        target_feat = out[:, center, :]

        type_logit = self.type_head(target_feat).squeeze(-1)
        strength_logit = self.strength_head(target_feat).squeeze(-1)

        return TypingOutput(
            type_logit=type_logit,
            strength_logit=strength_logit,
        )
