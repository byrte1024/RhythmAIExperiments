"""Conformer block — convolution-augmented transformer encoder layer.

Implements the canonical Conformer block from
[Gulati et al., INTERSPEECH 2020](https://arxiv.org/abs/2005.08100):

    x = x + 0.5 * FFN_macaron_1(x)         # half-residual
    x = x + MHSA(x)
    x = x + ConvModule(x)
    x = x + 0.5 * FFN_macaron_2(x)         # half-residual
    x = LayerNorm(x)

The convolution module operates along the time axis with a depthwise
convolution and is the core novelty over a vanilla transformer
encoder layer:

    LN -> PointwiseConv(d -> 2d) -> GLU -> DepthwiseConv1d(k) ->
    {GroupNorm, BatchNorm} -> Swish -> PointwiseConv(d -> d) -> Dropout

This file mirrors `torchaudio.models.Conformer`'s internal layer with
two changes:
1. Exposed as a single-block module with a `(B, T, d) -> (B, T, d)`
   forward, callable like `nn.TransformerEncoderLayer`. Lets the
   parent detector's existing `for layer, film in zip(self.layers,
   self.film_layers)` loop work without touching the forward path.
2. ``GroupNorm`` is selectable as an alternative to ``BatchNorm1d``
   inside the conv module — robust on small batches and the modern
   convention (used by BeatThis, ISMIR 2024).

Hyperparameter conventions from the audio literature:
- ``d_model``: 256-768 typical; 384 from #007 baseline.
- ``ffn_dim``: ``4 * d_model`` is canonical (macaron expansion).
- ``depthwise_kernel_size``: 31 standard; 17-31 productive band per
  Multi-Convformer 2024. Must be odd for centered conv.
- ``dropout``: 0.1 typical.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MacaronFFN(nn.Module):
    """Conformer's macaron-style feed-forward module.

    Shape: ``(B, T, d) -> (B, T, d)``. Applied with a 0.5 residual
    factor on each side of the attention + conv pair, per the paper.

    Layout (pre-norm):
        LN -> Linear(d -> ffn_dim) -> Swish -> Dropout ->
        Linear(ffn_dim -> d) -> Dropout
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.act = nn.SiLU()                   # Swish
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class _ConformerConvModule(nn.Module):
    """Conformer convolution module.

    Shape: ``(B, T, d) -> (B, T, d)``.

    Layout (pre-norm, attention-first ordering, paper default):
        LN ->                                # over channels of (B, T, d)
        transpose to (B, d, T) ->
        PointwiseConv1d(d -> 2d, k=1) ->
        GLU(dim=1) ->                        # halves channels back to d
        DepthwiseConv1d(d -> d, k, groups=d, padding=k//2) ->
        {GroupNorm | BatchNorm1d} ->
        Swish ->
        PointwiseConv1d(d -> d, k=1) ->
        Dropout ->
        transpose back to (B, T, d).

    GroupNorm with ``num_groups=1`` (effectively LayerNorm-over-channels
    per time step) is robust to small batches and matches the modern
    audio-paper convention. BatchNorm matches the original Conformer
    paper but is sensitive to batch size, especially during eval where
    running stats can drift on val distribution mismatches.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dropout: float,
        use_group_norm: bool,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for centered depthwise conv; "
                f"got {kernel_size}"
            )
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,
            kernel_size=1,
            bias=True,
        )
        self.depthwise = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            padding=kernel_size // 2,
            bias=False,
        )
        if use_group_norm:
            # num_groups=1 is LayerNorm-equivalent over channels per step,
            # but applied as a 1d operation that shares the same scaling.
            self.dw_norm = nn.GroupNorm(num_groups=1, num_channels=d_model)
        else:
            self.dw_norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()                   # Swish
        self.pointwise2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)                  # (B, d, T)
        x = self.pointwise1(x)                 # (B, 2d, T)
        x = F.glu(x, dim=1)                    # (B, d, T) — halves channels
        x = self.depthwise(x)                  # (B, d, T)
        x = self.dw_norm(x)
        x = self.act(x)
        x = self.pointwise2(x)                 # (B, d, T)
        x = self.dropout(x)
        x = x.transpose(1, 2)                  # (B, T, d)
        return x


class ConformerBlock(nn.Module):
    """One Conformer encoder block (Gulati et al., 2020).

    Shape: ``(B, T, d) -> (B, T, d)``. Drop-in replaceable for
    ``nn.TransformerEncoderLayer(d, nhead, dim_feedforward, dropout,
    activation, batch_first=True, norm_first=True)`` — the
    forward signature matches (`x` only, no mask), so existing trunks
    that loop over layers and apply per-layer FiLM still work without
    forward-path changes.

    Macaron-style residuals (factor 0.5) on the FFNs are critical —
    Gulati 2020's ablation showed ~0.4 pp WER regression when removed.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        depthwise_kernel_size: int,
        dropout: float = 0.1,
        use_group_norm: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.ffn1 = _MacaronFFN(d_model, ffn_dim, dropout)

        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mhsa_dropout = nn.Dropout(dropout)

        self.conv_module = _ConformerConvModule(
            d_model=d_model,
            kernel_size=depthwise_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _MacaronFFN(d_model, ffn_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Macaron FFN-1 with half residual.
        x = x + 0.5 * self.ffn1(x)

        # Multi-head self-attention (pre-norm + post-residual dropout).
        attn_in = self.mhsa_norm(x)
        attn_out, _ = self.mhsa(
            attn_in, attn_in, attn_in,
            need_weights=False,
        )
        x = x + self.mhsa_dropout(attn_out)

        # Convolution module (pre-norm internally, has its own dropout).
        x = x + self.conv_module(x)

        # Macaron FFN-2 with half residual.
        x = x + 0.5 * self.ffn2(x)

        # Final layer norm (canonical).
        return self.final_norm(x)
