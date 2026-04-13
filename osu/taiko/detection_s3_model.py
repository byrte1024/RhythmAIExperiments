"""S3 Fusion Selector: DETR-style encoder-decoder for per-bin onset detection.

Fuses S1 (audio) and S2v2 (context) confidence maps with cross-bin coordination.
Encoder processes enriched audio tokens. Decoder uses per-bin queries with
self-attention (cross-bin) and cross-attention (to encoder).

Experiment 65-S3.

References:
  - Carion et al., DETR (ECCV 2020): auxiliary losses per decoder layer
  - Zhao et al., RT-DETR (CVPR 2024): content-based query init
  - Ye et al., SEDT (2021): 1D-DETR for audio detection
  - Li et al., T-UAED (2024): per-class queries, D=192
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


class FusionSelector(nn.Module):
    """DETR-style encoder-decoder for fusing S1+S2v2 into per-bin onset detection.

    Encoder: self-attention over enriched audio tokens (250 tokens).
    Decoder: per-bin queries with self-attention + cross-attention to encoder.
    Auxiliary detection head at every decoder layer.
    """

    def __init__(self, d_model=192, d_audio=384, n_enc_layers=4, n_dec_layers=3,
                 n_heads=8, ff_expansion=4, dropout=0.1,
                 a_bins=500, b_bins=500, b_pred=250,
                 max_events=128, cond_dim=64):
        super().__init__()
        self.d_model = d_model
        self.b_pred = b_pred
        self.a_bins = a_bins
        self.n_tokens = (a_bins + b_bins) // 4
        self.cursor_token = a_bins // 4

        # Project S1 audio features to S3's dimension
        self.audio_proj = nn.Linear(d_audio, d_model)

        # Confidence embeddings (S1 and S2v2)
        self.s1_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.s2_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Positional encoding
        self.pos_emb = SinusoidalPosEmb(d_model)

        # Event embeddings (simplified — presence + gap_before + gap_after)
        self.event_presence_emb = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.gap_before_emb = SinusoidalPosEmb(d_model)
        self.gap_after_emb = SinusoidalPosEmb(d_model)
        self.event_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Density conditioning
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )

        # ── Encoder ──
        self.enc_layers = nn.ModuleList()
        self.enc_films = nn.ModuleList()
        for _ in range(n_enc_layers):
            self.enc_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * ff_expansion,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ))
            self.enc_films.append(FiLM(cond_dim, d_model))

        # ── Decoder ──
        self.learned_bin_pos = nn.Parameter(torch.randn(b_pred, d_model) * 0.02)

        self.dec_self_attn = nn.ModuleList()
        self.dec_cross_attn = nn.ModuleList()
        self.dec_ffn = nn.ModuleList()
        self.dec_norm1 = nn.ModuleList()
        self.dec_norm2 = nn.ModuleList()
        self.dec_norm3 = nn.ModuleList()

        for _ in range(n_dec_layers):
            self.dec_self_attn.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.dec_cross_attn.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True))
            self.dec_ffn.append(nn.Sequential(
                nn.Linear(d_model, d_model * ff_expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * ff_expansion, d_model),
                nn.Dropout(dropout),
            ))
            self.dec_norm1.append(nn.LayerNorm(d_model))
            self.dec_norm2.append(nn.LayerNorm(d_model))
            self.dec_norm3.append(nn.LayerNorm(d_model))

        # Shared detection head (used at every decoder layer)
        self.detection_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.n_dec_layers = n_dec_layers

    def _build_event_embeddings(self, event_offsets, event_mask):
        """Build event embeddings and compute scatter positions."""
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
        combined = torch.cat([presence, gb_emb, ga_emb], dim=-1)
        event_embs = self.event_proj(combined)

        # Map to token positions
        token_pos = ((self.a_bins + event_offsets) // 4).clamp(0, self.n_tokens - 1)
        in_window = (token_pos >= 0) & (token_pos < self.n_tokens) & (~event_mask)

        return event_embs, token_pos, in_window

    def forward(self, audio_features, s1_conf, s2_conf,
                event_offsets, event_mask, conditioning):
        """
        Args:
            audio_features: (B, n_tokens, d_audio) from S1 conv stem
            s1_conf: (B, b_pred) S1 per-bin sigmoid confidence
            s2_conf: (B, b_pred) S2v2 per-bin sigmoid confidence
            event_offsets: (B, 128)
            event_mask: (B, 128)
            conditioning: (B, 3)

        Returns:
            final_logits: (B, b_pred)
            aux_logits: list of (B, b_pred) from each decoder layer
        """
        B = audio_features.size(0)
        device = audio_features.device

        # ── Build encoder input ──
        # Project audio features
        tokens = self.audio_proj(audio_features)  # (B, n_tokens, d_model)

        # Embed S1/S2v2 confidences and add to prediction-range tokens
        # S1/S2 are (B, b_pred) — need to map to token positions
        s1_emb = self.s1_embed(s1_conf.unsqueeze(-1))  # (B, b_pred, d_model)
        s2_emb = self.s2_embed(s2_conf.unsqueeze(-1))  # (B, b_pred, d_model)

        # Add to the prediction range of tokens
        ct = self.cursor_token
        end = min(ct + self.b_pred, self.n_tokens)
        n_to_add = end - ct
        tokens[:, ct:end, :] = tokens[:, ct:end, :] + s1_emb[:, :n_to_add, :] + s2_emb[:, :n_to_add, :]

        # Positional encoding
        positions = torch.arange(self.n_tokens, device=device).unsqueeze(0).expand(B, -1)
        pos_enc = self.pos_emb(positions)
        tokens = tokens + pos_enc

        # Event embeddings (scatter-add to past tokens)
        event_embs, token_pos, in_window = self._build_event_embeddings(event_offsets, event_mask)
        for b in range(B):
            valid_idx = in_window[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue
            tpos = token_pos[b, valid_idx]
            embs = event_embs[b, valid_idx]
            tokens[b].scatter_add_(0, tpos.unsqueeze(-1).expand(-1, self.d_model), embs)

        # Density conditioning
        cond = self.cond_mlp(conditioning)

        # ── Encoder ──
        x = tokens
        for enc_layer, film in zip(self.enc_layers, self.enc_films):
            x = enc_layer(x)
            x = film(x, cond)
            x = x + pos_enc  # re-add position per layer (DETR finding)

        fenc = x  # (B, n_tokens, d_model)

        # ── Decoder ──
        # Content-based query init from encoder (per RT-DETR)
        queries = fenc[:, ct:ct + self.b_pred, :].clone()  # (B, b_pred, d_model)
        if queries.size(1) < self.b_pred:
            pad = self.b_pred - queries.size(1)
            queries = torch.nn.functional.pad(queries, (0, 0, 0, pad))
        queries = queries[:, :self.b_pred, :]

        # Add learned per-bin positional embeddings
        queries = queries + self.learned_bin_pos.unsqueeze(0)

        # Key position bias for cross-attention
        key_pos = pos_enc  # (B, n_tokens, d_model)

        aux_logits = []
        for l in range(self.n_dec_layers):
            # Self-attention among queries (cross-bin coordination)
            q_norm = self.dec_norm1[l](queries)
            sa_out, _ = self.dec_self_attn[l](q_norm, q_norm, q_norm)
            queries = queries + sa_out

            # Cross-attention: queries attend to encoder
            q_norm = self.dec_norm2[l](queries)
            ca_out, _ = self.dec_cross_attn[l](
                query=q_norm,
                key=fenc + key_pos,
                value=fenc,
            )
            queries = queries + ca_out

            # FFN
            q_norm = self.dec_norm3[l](queries)
            queries = queries + self.dec_ffn[l](q_norm)

            # Auxiliary detection (shared head, per DETR)
            aux = self.detection_head(queries).squeeze(-1)  # (B, b_pred)
            aux_logits.append(aux)

        final_logits = aux_logits[-1]  # last layer is the primary output

        return final_logits, aux_logits
