"""Onset detection model: predicts the next event's bin offset given audio + event context."""
import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional encoding for arbitrary integer positions."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """x: (batch, seq_len) integer positions → (batch, seq_len, dim)"""
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: γ * x + β, predicted from conditioning."""

    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feat_dim * 2)
        # init all zeros so γ=0, β=0 → FiLM starts as identity: (1+0)*x + 0 = x
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        """
        x: (B, T, D) feature tensor
        cond: (B, cond_dim) conditioning vector
        Returns: (B, T, D) modulated features
        """
        gamma_beta = self.fc(cond)  # (B, 2*D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, D)
        return (1 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)


class AudioEncoder(nn.Module):
    """Conv stack + FiLM-conditioned transformer encoder over mel spectrogram."""

    def __init__(self, n_mels=80, d_model=512, n_layers=6, n_heads=8, cond_dim=64, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers

        # 4x downsample: 1000 frames → 250 tokens
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)

        # FiLM after conv + after each transformer layer
        self.film_conv = FiLM(cond_dim, d_model)
        self.film_layers = nn.ModuleList([FiLM(cond_dim, d_model) for _ in range(n_layers)])

        # individual encoder layers (not wrapped in TransformerEncoder, so we can
        # apply FiLM between them)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(self, mel, cond):
        """
        mel: (B, n_mels, 1000)
        cond: (B, cond_dim) conditioning vector
        Returns: (B, 250, d_model)
        """
        x = self.conv(mel)  # (B, d_model, 250)
        x = x.transpose(1, 2)  # (B, 250, d_model)
        x = self.conv_norm(x)

        # positional encoding
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_emb(positions)

        # FiLM after conv
        x = self.film_conv(x, cond)

        # transformer layers with FiLM after each
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x)
            x = film(x, cond)

        return x


class EventDecoder(nn.Module):
    """Transformer decoder over past events with cross-attention to audio."""

    def __init__(self, d_model=512, n_layers=8, n_heads=8, n_classes=501, max_events=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.max_events = max_events

        # event position encoding: sinusoidal for bin offsets (can be large negatives)
        self.pos_enc = SinusoidalPosEmb(d_model)
        self.event_proj = nn.Linear(d_model, d_model)

        # learnable query token for "predict next"
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # sequence position embedding (order within the C events + query)
        self.seq_pos_emb = nn.Embedding(max_events + 1, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, audio_tokens, event_offsets, event_mask):
        """
        audio_tokens: (B, 250, d_model) from AudioEncoder
        event_offsets: (B, C) int32 — bin positions relative to cursor (negative/zero)
        event_mask: (B, C) bool — True = padded (ignore)
        Returns: (B, n_classes) logits
        """
        B, C = event_offsets.shape

        # encode past events via sinusoidal position of their bin offset
        evt_emb = self.pos_enc(event_offsets)  # (B, C, d_model)
        evt_emb = self.event_proj(evt_emb)

        # append query token
        query = self.query_token.expand(B, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([evt_emb, query], dim=1)  # (B, C+1, d_model)

        # add sequence-order position embeddings
        seq_positions = torch.arange(C + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.seq_pos_emb(seq_positions)

        # causal mask for self-attention (float type)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(C + 1, device=seq.device)

        # pad mask: extend event_mask with False for query token (bool type is fine here)
        query_pad = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        tgt_key_padding_mask = torch.cat([event_mask, query_pad], dim=1)  # (B, C+1)

        out = self.transformer(
            seq, audio_tokens,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # take only the query token output
        query_out = out[:, -1, :]  # (B, d_model)
        logits = self.head(query_out)  # (B, n_classes)
        return logits


class OnsetDetector(nn.Module):
    """Full model: audio encoder + FiLM conditioning + event decoder."""

    def __init__(
        self,
        n_mels=80,
        d_model=512,
        enc_layers=6,
        dec_layers=8,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # conditioning MLP: (mean_density, peak_density, density_std) → cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_decoder = EventDecoder(
            d_model=d_model, n_layers=dec_layers, n_heads=n_heads,
            n_classes=n_classes, max_events=max_events, dropout=dropout,
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        """
        mel: (B, n_mels, 1000)
        event_offsets: (B, C) past event bin positions relative to cursor
        event_mask: (B, C) bool, True = padding
        conditioning: (B, 3) [mean_density, peak_density, density_std]
        Returns: (B, 501) logits
        """
        cond = self.cond_mlp(conditioning)  # (B, cond_dim)
        audio_tokens = self.audio_encoder(mel, cond)
        logits = self.event_decoder(audio_tokens, event_offsets, event_mask)
        return logits
