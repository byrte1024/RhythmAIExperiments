"""S2 Context Predictor: predict next onset from rhythm patterns alone.

No audio input. Takes gap sequences + ratios + density conditioning,
outputs 251-class logits (250 bins + STOP).

Experiment 65-S2.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal positional encoding for scalar inputs."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        half = self.d_model // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        # x can be any shape — flatten, embed, reshape
        shape = x.shape
        x_flat = x.float().reshape(-1, 1)
        emb = x_flat * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.reshape(*shape, self.d_model)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation."""

    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x, cond):
        gamma_beta = self.proj(cond)
        if gamma_beta.dim() == 2:
            gamma_beta = gamma_beta.unsqueeze(1)  # (B, 1, 2*d)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class ContextPredictor(nn.Module):
    """Predict next onset from gap sequence + ratios + density. No audio.

    Architecture: event token encoding → bidirectional GRU → MLP head.
    """

    def __init__(self, d_model=256, n_gru_layers=4, n_classes=251,
                 max_events=128, cond_dim=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.max_events = max_events

        # Event token encoding: 3 sinusoidal features → d_model
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
        self.film_output = FiLM(cond_dim, d_model)

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

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, gap_sequence, ratio_sequence, event_mask, conditioning):
        """
        Args:
            gap_sequence: (B, 128) inter-onset gaps in bins
            ratio_sequence: (B, 128) gap ratios (gap[i] / gap[i-1])
            event_mask: (B, 128) True = padding (no event)
            conditioning: (B, 3) [density_mean, density_peak, density_std]

        Returns:
            logits: (B, n_classes)
        """
        B = gap_sequence.size(0)

        # Encode each event as a token
        log_gaps = torch.log(gap_sequence.float().clamp(min=1))
        log_ratios = torch.log(ratio_sequence.float().clamp(min=0.1, max=10.0)) * 50.0
        raw_gaps = gap_sequence.float()

        feat_lg = self.log_gap_emb(log_gaps)      # (B, 128, d_model)
        feat_lr = self.log_ratio_emb(log_ratios)   # (B, 128, d_model)
        feat_rg = self.raw_gap_emb(raw_gaps)       # (B, 128, d_model)

        tokens = self.event_proj(torch.cat([feat_lg, feat_lr, feat_rg], dim=-1))  # (B, 128, d_model)

        # Zero out padded events
        tokens = tokens * (~event_mask).float().unsqueeze(-1)

        # Density conditioning on input
        cond = self.cond_mlp(conditioning)
        tokens = self.film_input(tokens, cond)

        # GRU encoder
        gru_out, _ = self.gru(tokens)  # (B, 128, 2*d_model)

        # Extract context: last non-padded position's hidden state
        # Find last valid position per batch
        valid = ~event_mask  # (B, 128) True = real event
        # Use the last valid token's GRU output
        lengths = valid.long().sum(dim=1).clamp(min=1)  # (B,)
        # Gather last valid hidden state
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(B, 1, gru_out.size(2))
        context = gru_out.gather(1, idx).squeeze(1)  # (B, 2*d_model)

        context = self.gru_proj(context)  # (B, d_model)

        # Density conditioning on output
        context = self.film_output(context.unsqueeze(1), cond).squeeze(1)

        # Output head
        logits = self.head(context)  # (B, n_classes)

        return logits
