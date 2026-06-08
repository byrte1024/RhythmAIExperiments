"""Onset typing transformer — predicts D/K type and normal/big strength.

Small bidirectional transformer over a window of 33 onset tokens
(16 past + 1 target + 16 future). Past tokens carry known D/K and
big/normal labels; target and future tokens carry UNK. The model
attends to the full window and predicts the target onset's type and
strength from the center token.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..domain.model import Model
from ..domain.typing import (
    TYPING_CONTEXT,
    TYPING_MEL_PATCH,
    TYPING_WINDOW,
    TypingInput,
    TypingModelConfig,
    TypingOutput,
)


class TypingTransformer(Model[TypingModelConfig, TypingInput, TypingOutput]):

    def __init__(self, config: TypingModelConfig):
        super().__init__(config)
        c = config

        # Per-onset feature encoders
        self.mel_proj = nn.Sequential(
            nn.Linear(c.n_mels * c.mel_patch, c.d_mel),
            nn.ReLU(),
        )
        self.ioi_proj = nn.Sequential(
            nn.Linear(3, c.d_ioi),
            nn.ReLU(),
        )
        self.kind_emb = nn.Embedding(3, c.d_kind)   # D=0, K=1, UNK=2
        self.big_emb = nn.Embedding(3, c.d_kind)    # normal=0, big=1, UNK=2
        self.pos_emb = nn.Embedding(TYPING_WINDOW, c.d_pos)

        feat_dim = c.d_mel + c.d_ioi + c.d_kind + c.d_kind + c.d_pos
        self.input_proj = nn.Linear(feat_dim, c.d_model)

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

    def predict(self, x: TypingInput) -> TypingOutput:
        return self.forward(
            x.mel_patches, x.ioi_features, x.kind_labels,
            x.big_labels, x.positions, x.mask,
        )

    def forward(
        self,
        mel_patches: torch.Tensor,    # (B, W, n_mels * mel_patch)
        ioi_features: torch.Tensor,   # (B, W, 3)
        kind_labels: torch.Tensor,    # (B, W) long
        big_labels: torch.Tensor,     # (B, W) long
        positions: torch.Tensor,      # (B, W) long
        mask: torch.Tensor,           # (B, W) bool, True=padded
    ) -> TypingOutput:
        mel_feat = self.mel_proj(mel_patches)           # (B, W, d_mel)
        ioi_feat = self.ioi_proj(ioi_features)          # (B, W, d_ioi)
        kind_feat = self.kind_emb(kind_labels)          # (B, W, d_kind)
        big_feat = self.big_emb(big_labels)             # (B, W, d_kind)
        pos_feat = self.pos_emb(positions)              # (B, W, d_pos)

        combined = torch.cat(
            [mel_feat, ioi_feat, kind_feat, big_feat, pos_feat], dim=-1,
        )
        tokens = self.input_proj(combined)              # (B, W, d_model)

        # Transformer with padding mask (True = ignore)
        out = self.transformer(tokens, src_key_padding_mask=mask)

        # Extract center token (target onset)
        center = self.config.context  # index 16 in a 33-token window
        target_feat = out[:, center, :]                 # (B, d_model)

        type_logit = self.type_head(target_feat).squeeze(-1)        # (B,)
        strength_logit = self.strength_head(target_feat).squeeze(-1) # (B,)

        return TypingOutput(
            type_logit=type_logit,
            strength_logit=strength_logit,
        )
