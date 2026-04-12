"""Chart quality evaluator model.

Exp 66-1: Pairwise quality scoring via Bradley-Terry loss.
Architecture: 16-bin mel + scatter-add event embeddings + transformer + attention pool → scalar.
"""
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


class ChartQualityEvaluator(nn.Module):
    """Scores a (chart, audio) window as a scalar quality value.

    Input: 10s mel window + event positions within that window + star rating.
    Output: scalar score (higher = better quality).
    """

    def __init__(self, n_mels=80, mel_bins=16, d_model=256, n_layers=6, n_heads=8,
                 n_star_buckets=20, max_events=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_events = max_events

        # ── mel compression: 80 → 16 bins ──
        self.mel_compress = nn.Linear(n_mels, mel_bins)

        # ── conv stem: 4x downsample ──
        self.conv = nn.Sequential(
            nn.Conv1d(mel_bins, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)

        # ── star rating conditioning ──
        self.star_emb = nn.Embedding(n_star_buckets, d_model)

        # ── event embeddings: ratio_before + ratio_after + gap_ms ──
        self.ratio_before_emb = SinusoidalPosEmb(d_model)
        self.ratio_after_emb = SinusoidalPosEmb(d_model)
        self.gap_ms_emb = SinusoidalPosEmb(d_model)
        self.event_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── transformer ──
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # ── attention pooling ──
        self.pool_query = nn.Parameter(torch.randn(d_model))

        # ── output head ──
        self.out_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, 1)

    def _build_event_embeddings(self, events, event_mask):
        """Compute ratio-based event embeddings.

        events: (B, N_max) int64 — event positions (bin indices within window)
        event_mask: (B, N_max) bool — True = padding
        Returns: (B, N_max, d_model) embeddings, (B, N_max) token positions
        """
        B, N = events.shape
        device = events.device

        gaps = torch.zeros(B, N, dtype=torch.long, device=device)
        ratio_before = torch.ones(B, N, dtype=torch.long, device=device) * 50  # neutral = 1.0 * 50
        ratio_after = torch.ones(B, N, dtype=torch.long, device=device) * 50

        for b in range(B):
            valid = ~event_mask[b]
            n = valid.sum().item()
            if n < 1:
                continue
            pos = events[b][valid].long()

            # gaps
            g = torch.zeros(n, dtype=torch.long, device=device)
            if n >= 2:
                g[1:] = (pos[1:] - pos[:-1]).clamp(min=1)
            g[0] = g[1] if n >= 2 else torch.tensor(50, device=device)  # ~250ms default

            # gap in ms
            gap_ms = g * 5  # 5ms per bin
            gaps[b][valid] = gap_ms

            # ratios: gap[i] / gap[i-1], clamped [0.1, 10.0], scaled *50
            rb = torch.ones(n, dtype=torch.float32, device=device)
            ra = torch.ones(n, dtype=torch.float32, device=device)
            if n >= 2:
                g_f = g.float().clamp(min=1.0)
                rb[1:] = (g_f[1:] / g_f[:-1]).clamp(0.1, 10.0)
                ra[:-1] = (g_f[1:] / g_f[:-1]).clamp(0.1, 10.0)
            ratio_before[b][valid] = (rb * 50).long()
            ratio_after[b][valid] = (ra * 50).long()

        emb_parts = [
            self.ratio_before_emb(ratio_before),    # (B, N, d_model)
            self.ratio_after_emb(ratio_after),       # (B, N, d_model)
            self.gap_ms_emb(gaps),                   # (B, N, d_model)
        ]
        combined = torch.cat(emb_parts, dim=-1)      # (B, N, d_model*3)
        event_embs = self.event_proj(combined)        # (B, N, d_model)

        # token positions: event bin // 4 (conv stem does 4x downsample)
        token_pos = (events // 4).clamp(0, 499)      # max 500 tokens (2000 frames / 4)

        return event_embs, token_pos

    def forward(self, mel, events, event_mask, star_rating):
        """
        mel: (B, 80, 2000) — raw mel spectrogram window
        events: (B, N_max) int64 — event bin positions within window
        event_mask: (B, N_max) bool — True = padding
        star_rating: (B,) float — osu! star rating

        Returns: (B,) scalar quality scores
        """
        B = mel.size(0)

        # ── mel compression ──
        # (B, 80, 2000) → transpose → (B, 2000, 80) → linear → (B, 2000, 16) → transpose → (B, 16, 2000)
        x_mel = self.mel_compress(mel.transpose(1, 2)).transpose(1, 2)

        # ── conv stem ──
        x = self.conv(x_mel).transpose(1, 2)           # (B, 500, d_model)
        x = self.conv_norm(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        # ── star rating conditioning ──
        star_bucket = (star_rating / 0.5).long().clamp(0, self.star_emb.num_embeddings - 1)
        x = x + self.star_emb(star_bucket).unsqueeze(1)  # broadcast to all tokens

        # ── event embeddings: scatter-add into audio tokens ──
        event_embs, token_pos = self._build_event_embeddings(events, event_mask)
        for b in range(B):
            valid_idx = (~event_mask[b]).nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue
            tpos = token_pos[b, valid_idx]
            embs = event_embs[b, valid_idx].to(x.dtype)
            x[b].scatter_add_(0, tpos.unsqueeze(-1).expand(-1, self.d_model), embs)

        # ── transformer ──
        for layer in self.layers:
            x = layer(x)

        # ── attention pooling ──
        # query: (d_model,) → attn over (B, 500, d_model)
        attn_logits = (x @ self.pool_query) / math.sqrt(self.d_model)  # (B, 500)
        attn_weights = torch.softmax(attn_logits, dim=-1)               # (B, 500)
        pooled = (attn_weights.unsqueeze(-1) * x).sum(dim=1)            # (B, d_model)

        # ── output ──
        score = self.out_head(self.out_norm(pooled)).squeeze(-1)         # (B,)
        return score
