"""S3v2 Fusion Selector: single-onset classification with S1+S2v2 signals.

Like exp58's selector but with S2v2 context confidence added.
Audio tokens enriched with S1 proposals + S2v2 proposals + event embeddings + FiLM.
Output: 251-class logits (250 bins + STOP), single onset via argmax.

Experiment 65-S3v2.
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


class FusionClassifier(nn.Module):
    """Single-onset classifier with S1+S2v2 fusion.

    Takes audio tokens + S1 confidence + S2v2 confidence + events + density.
    Outputs 251-class logits from cursor token (like exp58 selector).
    """

    def __init__(self, d_model=384, d_audio=384, n_layers=8, n_heads=8,
                 n_classes=251, dropout=0.1, a_bins=500, b_bins=500,
                 max_events=128, cond_dim=64, gap_ratios=True):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.a_bins = a_bins
        self.b_bins = b_bins
        self.n_audio_tokens = (a_bins + b_bins) // 4
        self.cursor_token = a_bins // 4
        self.max_events = max_events
        self.gap_ratios = gap_ratios

        # Project S1 audio features to d_model
        self.audio_proj = nn.Linear(d_audio, d_model)

        # S1 confidence embedding (same as exp58 proposal_embed)
        self.s1_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # S2v2 confidence embedding (NEW)
        self.s2_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Positional encoding
        self.pos_emb = SinusoidalPosEmb(d_model)

        # Density conditioning
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )
        self.film_conv = FiLM(cond_dim, d_model)

        # Event embeddings
        self.event_presence_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.gap_before_emb = SinusoidalPosEmb(d_model)
        self.gap_after_emb = SinusoidalPosEmb(d_model)
        n_emb_inputs = 3
        if gap_ratios:
            self.gap_ratio_before_emb = SinusoidalPosEmb(d_model)
            self.gap_ratio_after_emb = SinusoidalPosEmb(d_model)
            n_emb_inputs = 5
        self.event_proj = nn.Sequential(
            nn.Linear(d_model * n_emb_inputs, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Transformer layers with FiLM
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        self.film_layers = nn.ModuleList([
            FiLM(cond_dim, d_model) for _ in range(n_layers)
        ])

        # Output head (from cursor token)
        self.head_norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def _build_event_embeddings(self, event_offsets, event_mask):
        B, C = event_offsets.shape
        device = event_offsets.device

        presence = self.event_presence_emb.expand(B, C, -1)
        gaps_before = torch.zeros(B, C, dtype=torch.long, device=device)
        gaps_after = torch.zeros(B, C, dtype=torch.long, device=device)

        for b in range(B):
            valid = ~event_mask[b]
            if valid.sum() < 2:
                continue
            offsets = event_offsets[b][valid]
            diffs = offsets[1:] - offsets[:-1]
            valid_idx = valid.nonzero(as_tuple=True)[0]
            gaps_before[b, valid_idx[1:]] = diffs
            gaps_before[b, valid_idx[0]] = diffs[0] if len(diffs) > 0 else 0
            gaps_after[b, valid_idx[:-1]] = diffs
            gaps_after[b, valid_idx[-1]] = gaps_before[b, valid_idx[-1]]

        gb_emb = self.gap_before_emb(gaps_before.abs())
        ga_emb = self.gap_after_emb(gaps_after.abs())

        if self.gap_ratios:
            ratios_before = torch.zeros(B, C, dtype=torch.float32, device=device)
            ratios_after = torch.zeros(B, C, dtype=torch.float32, device=device)
            for b in range(B):
                valid = ~event_mask[b]
                if valid.sum() < 3:
                    continue
                gb_vals = gaps_before[b][valid].float()
                ga_vals = gaps_after[b][valid].float()
                valid_idx = valid.nonzero(as_tuple=True)[0]
                for i in range(1, len(valid_idx)):
                    if gb_vals[i] > 0 and gb_vals[i - 1] > 0:
                        ratios_before[b, valid_idx[i]] = (gb_vals[i - 1] / gb_vals[i]).clamp(0.1, 10.0) * 50
                    if ga_vals[i] > 0 and ga_vals[i - 1] > 0:
                        ratios_after[b, valid_idx[i]] = (ga_vals[i - 1] / ga_vals[i]).clamp(0.1, 10.0) * 50

            grb_emb = self.gap_ratio_before_emb(ratios_before.long())
            gra_emb = self.gap_ratio_after_emb(ratios_after.long())
            combined = torch.cat([presence, gb_emb, ga_emb, grb_emb, gra_emb], dim=-1)
        else:
            combined = torch.cat([presence, gb_emb, ga_emb], dim=-1)

        event_embs = self.event_proj(combined)

        token_pos = ((self.a_bins + event_offsets) // 4).clamp(0, self.n_audio_tokens - 1)
        in_window = (token_pos >= 0) & (token_pos < self.n_audio_tokens) & (~event_mask)

        return event_embs, token_pos, in_window

    def forward(self, audio_features, s1_conf, s2_conf,
                event_offsets, event_mask, conditioning):
        """
        Args:
            audio_features: (B, n_tokens, d_audio) from S1 conv stem
            s1_conf: (B, B_PRED) S1 per-bin confidence
            s2_conf: (B, B_PRED) S2v2 per-bin confidence
            event_offsets: (B, 128)
            event_mask: (B, 128)
            conditioning: (B, 3)

        Returns:
            logits: (B, n_classes) — 251-class single-onset prediction
        """
        B = audio_features.size(0)
        device = audio_features.device

        # Project audio
        x = self.audio_proj(audio_features)  # (B, n_tokens, d_model)

        # Embed S1+S2v2 and add to prediction range tokens
        s1_emb = self.s1_embed(s1_conf.unsqueeze(-1))  # (B, B_PRED, d_model)
        s2_emb = self.s2_embed(s2_conf.unsqueeze(-1))  # (B, B_PRED, d_model)

        ct = self.cursor_token
        b_pred = s1_conf.size(1)
        end = min(ct + b_pred, self.n_audio_tokens)
        n = end - ct
        x[:, ct:end, :] = x[:, ct:end, :] + s1_emb[:, :n, :] + s2_emb[:, :n, :]

        # Position
        positions = torch.arange(self.n_audio_tokens, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(positions)

        # Density conditioning
        cond = self.cond_mlp(conditioning)
        x = self.film_conv(x, cond)

        # Event embeddings
        event_embs, token_pos, in_window = self._build_event_embeddings(event_offsets, event_mask)
        for b in range(B):
            valid_idx = in_window[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue
            tpos = token_pos[b, valid_idx]
            embs = event_embs[b, valid_idx]
            x[b].scatter_add_(0, tpos.unsqueeze(-1).expand(-1, self.d_model), embs)

        # Transformer with FiLM
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x)
            x = film(x, cond)

        # Output from cursor token
        cursor = x[:, ct, :]
        logits = self.head_proj(self.head_norm(cursor))  # (B, 251)
        logits = logits + self.head_smooth(logits.unsqueeze(1)).squeeze(1)

        return logits
