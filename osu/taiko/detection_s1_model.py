"""S1 Conformer Proposer: per-bin onset detection from audio alone.

No context, no events, no density conditioning.
Conformer blocks (conv inside transformer) with upsample head for per-bin output.

Experiment 65-S1.
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


class ConformerConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → batchnorm → swish → pointwise."""

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.pointwise1 = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, d_model)
        x = self.pointwise1(x)  # (B, T, 2*d)
        x = x[..., :x.size(-1) // 2] * torch.sigmoid(x[..., x.size(-1) // 2:])  # GLU
        x = x.transpose(1, 2)  # (B, d, T) for conv
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)  # (B, T, d)
        x = torch.nn.functional.silu(x)  # swish
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN(half) → MHSA → Conv → FFN(half) → LayerNorm."""

    def __init__(self, d_model, n_heads, ff_expansion=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        ff_dim = d_model * ff_expansion

        # First half-step FFN
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # MHSA
        self.mhsa_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mhsa_dropout = nn.Dropout(dropout)

        # Conv module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel, dropout=dropout)

        # Second half-step FFN
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # Final norm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Half-step FFN 1
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        # MHSA
        residual = x
        x_norm = self.mhsa_norm(x)
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm)
        x = residual + self.mhsa_dropout(attn_out)
        # Conv module
        x = x + self.conv_module(self.conv_norm(x))
        # Half-step FFN 2
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        # Final norm
        x = self.final_norm(x)
        return x


class ConformerProposer(nn.Module):
    """Audio-only onset proposer with Conformer blocks and per-bin output.

    mel → conv stem (4x downsample) → 8 conformer blocks → upsample → per-bin sigmoid
    """

    def __init__(self, n_mels=80, d_model=384, n_layers=8, n_heads=8,
                 conv_kernel=31, dropout=0.1, a_bins=500, b_bins=500, b_pred=250):
        super().__init__()
        self.d_model = d_model
        self.a_bins = a_bins
        self.b_bins = b_bins
        self.b_pred = b_pred
        self.n_audio_tokens = (a_bins + b_bins) // 4
        self.cursor_token = a_bins // 4

        # Conv stem (4x downsample)
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)

        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, conv_kernel=conv_kernel, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Upsample head: tokens → per-bin
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=4),
            nn.GELU(),
        )
        # Refine after upsample (smooth out transpose conv artifacts)
        self.refine = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, A_BINS + B_BINS)

        Returns:
            bin_logits: (B, b_pred) — per-bin onset logits (before sigmoid)
        """
        B = mel.size(0)

        # Conv stem
        x = self.conv(mel).transpose(1, 2)  # (B, n_tokens, d_model)
        x = self.conv_norm(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        # Upsample: tokens → mel frame resolution
        x = x.transpose(1, 2)  # (B, d_model, n_tokens)
        x = self.upsample(x)    # (B, d_model, n_tokens*4)
        x = self.refine(x)      # (B, d_model, ~n_frames)
        x = x.transpose(1, 2)   # (B, ~n_frames, d_model)

        # Slice to B_PRED range: bins 0..b_pred-1 start at cursor position
        cursor_frame = self.a_bins
        pred_end = cursor_frame + self.b_pred
        # Clamp to actual size
        actual_frames = x.size(1)
        pred_end = min(pred_end, actual_frames)
        cursor_frame = min(cursor_frame, actual_frames - 1)

        x_pred = x[:, cursor_frame:pred_end, :]  # (B, b_pred, d_model)

        # Pad if needed (transpose conv may produce slightly different size)
        if x_pred.size(1) < self.b_pred:
            pad = self.b_pred - x_pred.size(1)
            x_pred = torch.nn.functional.pad(x_pred, (0, 0, 0, pad))

        x_pred = x_pred[:, :self.b_pred, :]  # ensure exact size

        # Output projection
        bin_logits = self.output_proj(self.output_norm(x_pred)).squeeze(-1)  # (B, b_pred)

        return bin_logits
