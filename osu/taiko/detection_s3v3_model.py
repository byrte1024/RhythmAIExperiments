"""S3v3 Pure Proposal Fusion: onset prediction from S1+S2v2 confidences only.

No audio features, no event embeddings, no density conditioning.
Just S1 (250,) + S2v2 (250,) → transformer → 251-class logits.

Experiment 65-S3v3.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        half = self.d_model // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        shape = x.shape
        x_flat = x.float().reshape(-1, 1)
        emb = x_flat * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.reshape(*shape, self.d_model)


class PureProposalFusion(nn.Module):
    """Predict onsets from S1+S2v2 confidence maps only.

    Each of 250 bins becomes a token with [s1_conf, s2_conf] features.
    Transformer self-attention over bins → cursor token → 251-class logits.
    """

    def __init__(self, d_model=128, n_layers=4, n_heads=4, n_classes=251,
                 b_pred=250, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.b_pred = b_pred

        # Per-bin feature: [s1_conf, s2_conf] → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_emb = SinusoidalPosEmb(d_model)

        # Transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])

        # Output from first token (cursor position = bin 0)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, s1_conf, s2_conf):
        """
        Args:
            s1_conf: (B, b_pred) S1 per-bin confidence
            s2_conf: (B, b_pred) S2v2 per-bin confidence

        Returns:
            logits: (B, n_classes)
        """
        B = s1_conf.size(0)
        device = s1_conf.device

        # Stack per-bin features
        x = torch.stack([s1_conf, s2_conf], dim=-1)  # (B, 250, 2)
        x = self.input_proj(x)  # (B, 250, d_model)

        # Positional encoding
        positions = torch.arange(self.b_pred, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        # Transformer
        for layer in self.layers:
            x = layer(x)

        # Predict from first token (bin 0 = cursor)
        logits = self.head(x[:, 0, :])  # (B, 251)

        return logits
