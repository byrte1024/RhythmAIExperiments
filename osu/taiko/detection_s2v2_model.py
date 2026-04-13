"""S2-v2 Context Proposer: per-bin onset detection from rhythm patterns alone.

Same input as S2 (gap sequences + ratios + density), but output is per-bin
sigmoid (250 bins) instead of 251-class softmax. This matches S1's output
space for direct comparison and fusion.

Experiment 65-S2v2.
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


class FiLM(nn.Module):
    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x, cond):
        gamma_beta = self.proj(cond)
        if gamma_beta.dim() == 2:
            gamma_beta = gamma_beta.unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class ContextProposer(nn.Module):
    """Per-bin onset detection from context alone. Matches S1's output format.

    Input: gap sequences + ratios + density conditioning
    Output: (B, b_pred) per-bin logits (before sigmoid)
    """

    def __init__(self, d_model=256, n_gru_layers=4, b_pred=250,
                 max_events=128, cond_dim=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.b_pred = b_pred
        self.max_events = max_events

        # Event token encoding
        self.log_gap_emb = SinusoidalPosEmb(d_model)
        self.log_ratio_emb = SinusoidalPosEmb(d_model)
        self.raw_gap_emb = SinusoidalPosEmb(d_model)
        self.event_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
        )

        # Density conditioning
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.film_input = FiLM(cond_dim, d_model)

        # History encoder: bidirectional GRU
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        self.gru_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )

        # Per-bin output head
        # Context → expand to b_pred bins → per-bin logit
        self.bin_expand = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, b_pred * d_model // 4),
            nn.GELU(),
        )
        self.bin_head = nn.Sequential(
            nn.LayerNorm(d_model // 4),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, gap_sequence, ratio_sequence, event_mask, conditioning):
        """
        Args:
            gap_sequence: (B, 128) inter-onset gaps in bins
            ratio_sequence: (B, 128) gap ratios
            event_mask: (B, 128) True = padding
            conditioning: (B, 3) density

        Returns:
            bin_logits: (B, b_pred) per-bin onset logits
        """
        B = gap_sequence.size(0)

        # Encode events
        log_gaps = torch.log(gap_sequence.float().clamp(min=1))
        log_ratios = torch.log(ratio_sequence.float().clamp(min=0.1, max=10.0)) * 50.0
        raw_gaps = gap_sequence.float()

        feat_lg = self.log_gap_emb(log_gaps)
        feat_lr = self.log_ratio_emb(log_ratios)
        feat_rg = self.raw_gap_emb(raw_gaps)

        tokens = self.event_proj(torch.cat([feat_lg, feat_lr, feat_rg], dim=-1))
        tokens = tokens * (~event_mask).float().unsqueeze(-1)

        # Conditioning
        cond = self.cond_mlp(conditioning)
        tokens = self.film_input(tokens, cond)

        # GRU
        gru_out, _ = self.gru(tokens)

        # Extract context from last valid position
        valid = ~event_mask
        lengths = valid.long().sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, gru_out.size(2))
        context = gru_out.gather(1, idx).squeeze(1)
        context = self.gru_proj(context)  # (B, d_model)

        # Expand to per-bin features
        bin_features = self.bin_expand(context)  # (B, b_pred * d_model//4)
        bin_features = bin_features.view(B, self.b_pred, self.d_model // 4)  # (B, b_pred, d//4)

        # Per-bin logits
        bin_logits = self.bin_head(bin_features).squeeze(-1)  # (B, b_pred)

        return bin_logits
