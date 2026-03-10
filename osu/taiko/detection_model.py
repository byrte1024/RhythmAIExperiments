"""Onset detection model: two-path architecture with top-K reranking.

Audio Path (proposer): audio-primary with light event cross-attention → 501 logits
Context Path (selector): selects from audio's top-K candidates using event context
Final: context picks from audio's top-K proposals
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Feature-wise Linear Modulation: (1+γ)*x + β, predicted from conditioning."""

    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feat_dim * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        """x: (B, T, D), cond: (B, cond_dim) → (B, T, D)"""
        gamma_beta = self.fc(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return (1 + gamma).unsqueeze(1) * x + beta.unsqueeze(1)


class AudioEncoder(nn.Module):
    """Conv stack + FiLM-conditioned transformer encoder over mel spectrogram."""

    def __init__(self, n_mels=80, d_model=384, n_layers=4, n_heads=8, cond_dim=64, dropout=0.1):
        super().__init__()

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

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(self, mel, cond):
        """mel: (B, n_mels, 1000), cond: (B, cond_dim) → (B, 250, d_model)"""
        x = self.conv(mel).transpose(1, 2)
        x = self.conv_norm(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_emb(positions)
        x = self.film_conv(x, cond)
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x)
            x = film(x, cond)
        return x


class EventEncoder(nn.Module):
    """Bottlenecked transformer encoder over past events.

    Operates at d_event (small), projects up to d_model for cross-attention.
    """

    def __init__(self, d_model=384, d_event=128, n_layers=2, n_heads=4,
                 max_events=128, cond_dim=64, dropout=0.1):
        super().__init__()
        self.d_event = d_event

        self.pos_enc = SinusoidalPosEmb(d_event)
        self.event_proj = nn.Linear(d_event, d_event)
        self.seq_pos_emb = nn.Embedding(max_events, d_event)

        # FiLM after each layer
        self.film_layers = nn.ModuleList([FiLM(cond_dim, d_event) for _ in range(n_layers)])

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_event, nhead=n_heads, dim_feedforward=d_event * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # project up to d_model for cross-attention compatibility
        self.up_proj = nn.Linear(d_event, d_model)

    def forward(self, event_offsets, event_mask, cond):
        """
        event_offsets: (B, C) int - bin positions relative to cursor
        event_mask: (B, C) bool - True = padded
        cond: (B, cond_dim)
        Returns: (B, C, d_model)
        """
        B, C = event_offsets.shape

        # sinusoidal encoding of bin offset positions
        x = self.pos_enc(event_offsets)  # (B, C, d_event)
        x = self.event_proj(x)

        # add sequence-order position embeddings
        seq_pos = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.seq_pos_emb(seq_pos)

        # guard: if ALL events are masked for a sample, transformer self-attention
        # produces NaN (softmax over all -inf). Unmask the last position as a
        # no-op dummy - its content is just positional encoding of offset 0.
        all_masked = event_mask.all(dim=1)  # (B,)
        if all_masked.any():
            safe_mask = event_mask.clone()
            safe_mask[all_masked, -1] = False  # unmask last position
        else:
            safe_mask = event_mask

        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, src_key_padding_mask=safe_mask)
            x = film(x, cond)

        return self.up_proj(x)  # (B, C, d_model)


class AudioPath(nn.Module):
    """Audio-primary proposer: audio self-attention + cross-attention to events.

    Extracts the cursor token (center of window) and produces broad logits.
    """

    def __init__(self, d_model=384, n_layers=2, n_heads=8, n_classes=501,
                 cond_dim=64, dropout=0.1):
        super().__init__()

        # FiLM after each decoder layer
        self.film_layers = nn.ModuleList([FiLM(cond_dim, d_model) for _ in range(n_layers)])

        # individual decoder layers so we can apply FiLM between them
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # output head with smoothing
        self.head_norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def forward(self, audio_tokens, event_tokens, event_mask, cond):
        """
        audio_tokens: (B, 250, d_model)
        event_tokens: (B, C, d_model) from EventEncoder
        event_mask: (B, C) bool - True = padded
        cond: (B, cond_dim)
        Returns: (B, n_classes) audio logits
        """
        # guard: unmask at least one event position to prevent NaN in cross-attention
        all_masked = event_mask.all(dim=1)
        if all_masked.any():
            safe_mask = event_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = event_mask

        # audio self-attention + cross-attention to events
        x = audio_tokens
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, event_tokens, memory_key_padding_mask=safe_mask)
            x = film(x, cond)

        # extract cursor token (center of 5s window, token 125 after 4x downsample)
        cursor = x[:, 125, :]  # (B, d_model)

        logits = self.head_proj(self.head_norm(cursor))
        logits = logits + self.head_smooth(logits.unsqueeze(1)).squeeze(1)
        return logits


class ContextPath(nn.Module):
    """Top-K reranking selector: picks from audio's top-K candidates using event context.

    Architecture:
    1. Takes audio's top-K candidate bins + their scores + audio features at those positions
    2. Builds candidate embeddings from bin position + audio score + audio feature
    3. Processes event history through transformer decoder (event self-attn + audio cross-attn)
    4. Scores each candidate via dot product with context query vector
    5. Returns K-way selection logits

    This makes rubber-stamping architecturally impossible: context MUST pick one of K candidates.
    """

    def __init__(self, d_model=384, n_layers=3, n_heads=8, K=20,
                 max_events=128, cond_dim=64, dropout=0.1):
        super().__init__()
        self.K = K
        self.d_model = d_model

        # Candidate feature building
        self.bin_pos_emb = SinusoidalPosEmb(d_model)
        self.score_proj = nn.Sequential(
            nn.Linear(2, d_model),  # [audio_score, normalized_rank]
            nn.GELU(),
        )
        self.candidate_combine = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # bin_emb + score_proj + audio_feature
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Event context: transformer decoder (event self-attn + audio cross-attn)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seq_pos_emb = nn.Embedding(max_events + 1, d_model)

        self.film_layers = nn.ModuleList([FiLM(cond_dim, d_model) for _ in range(n_layers)])

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

        # Selection scoring: project query and candidates to scoring space
        d_score = 64
        self.q_proj = nn.Linear(d_model, d_score)
        self.k_proj = nn.Linear(d_model, d_score)
        self.score_scale = d_score ** -0.5

    def forward(self, event_tokens, audio_tokens, event_mask, cond, audio_logits):
        """
        event_tokens: (B, C, d_model) from EventEncoder
        audio_tokens: (B, 250, d_model)
        event_mask: (B, C) bool - True = padded
        cond: (B, cond_dim)
        audio_logits: (B, n_classes) from AudioPath (detached)
        Returns: (selection_logits: (B, K), top_k_indices: (B, K))
        """
        B, C, _ = event_tokens.shape

        # ── 1. Get top-K candidates from audio ──
        audio_logits_det = audio_logits.detach()
        top_k_scores, top_k_indices = audio_logits_det.topk(self.K, dim=-1)  # (B, K)

        # Force-include STOP (class 500) if not already in top-K
        stop_idx = audio_logits_det.size(1) - 1
        has_stop = (top_k_indices == stop_idx).any(dim=1)  # (B,)
        if not has_stop.all():
            no_stop = ~has_stop
            top_k_indices[no_stop, -1] = stop_idx
            top_k_scores[no_stop, -1] = audio_logits_det[no_stop, stop_idx]

        # ── 2. Build candidate feature embeddings ──
        # Bin position encoding
        bin_emb = self.bin_pos_emb(top_k_indices)  # (B, K, d_model)

        # Audio score + rank features
        ranks = torch.arange(1, self.K + 1, device=audio_logits.device, dtype=torch.float32)
        ranks = ranks.unsqueeze(0).expand(B, -1) / self.K  # (B, K) normalized 0-1
        score_feat = self.score_proj(
            torch.stack([top_k_scores, ranks], dim=-1)  # (B, K, 2)
        )  # (B, K, d_model)

        # Audio features at candidate temporal positions
        # Bin offset j → audio token index: (500 + j) / 4, clamped
        token_idx = ((500 + top_k_indices.clamp(max=499)).float() / 4.0).long()
        token_idx = token_idx.clamp(0, audio_tokens.size(1) - 1)  # (B, K)
        audio_feat = audio_tokens.gather(
            1, token_idx.unsqueeze(-1).expand(-1, -1, audio_tokens.size(-1))
        )  # (B, K, d_model)

        # Combine into candidate embeddings
        candidate_feat = self.candidate_combine(
            torch.cat([bin_emb, score_feat, audio_feat], dim=-1)
        )  # (B, K, d_model)

        # ── 3. Process event context ──
        query = self.query_token.expand(B, -1, -1)
        seq = torch.cat([event_tokens, query], dim=1)  # (B, C+1, d_model)

        seq_pos = torch.arange(C + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.seq_pos_emb(seq_pos)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(C + 1, device=seq.device)
        query_pad = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        tgt_pad_mask = torch.cat([event_mask, query_pad], dim=1)

        x = seq
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, audio_tokens,
                      tgt_mask=causal_mask,
                      tgt_key_padding_mask=tgt_pad_mask)
            x = film(x, cond)

        query_out = x[:, -1, :]  # (B, d_model) - context query vector

        # ── 4. Score candidates via scaled dot product ──
        q = self.q_proj(query_out)  # (B, d_score)
        k = self.k_proj(candidate_feat)  # (B, K, d_score)
        selection_logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.score_scale  # (B, K)

        return selection_logits, top_k_indices


class OnsetDetector(nn.Module):
    """Two-path onset detector: audio proposes, context selects from top-K.

    Audio path: audio self-attn + event cross-attn → 501 candidate logits
    Context path: top-K reranking — selects from audio's top-K proposals
    Final: context's selection mapped back to 501-way logits
    """

    def __init__(
        self,
        n_mels=80,
        d_model=384,
        d_event=128,
        enc_layers=4,
        enc_event_layers=2,
        audio_path_layers=2,
        context_path_layers=3,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
        top_k=20,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.top_k = top_k

        # conditioning MLP: (mean_density, peak_density, density_std) → cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # shared encoders
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            d_model=d_model, d_event=d_event, n_layers=enc_event_layers,
            n_heads=4, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )

        # two prediction paths
        self.audio_path = AudioPath(
            d_model=d_model, n_layers=audio_path_layers, n_heads=n_heads,
            n_classes=n_classes, cond_dim=cond_dim, dropout=dropout,
        )
        self.context_path = ContextPath(
            d_model=d_model, n_layers=context_path_layers, n_heads=n_heads,
            K=top_k, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        """
        mel: (B, n_mels, 1000)
        event_offsets: (B, C) past event bin positions relative to cursor
        event_mask: (B, C) bool, True = padding
        conditioning: (B, 3) [mean_density, peak_density, density_std]
        Returns: (logits, audio_logits, selection_logits, top_k_indices)
            logits: (B, 501) combined prediction (selection scattered to 501)
            audio_logits: (B, 501) audio-only prediction (for aux loss)
            selection_logits: (B, K) context's scores over K candidates
            top_k_indices: (B, K) which bins the K candidates correspond to
        """
        cond = self.cond_mlp(conditioning)

        # shared encoding
        audio_tokens = self.audio_encoder(mel, cond)
        event_tokens = self.event_encoder(event_offsets, event_mask, cond)

        # audio proposes
        audio_logits = self.audio_path(audio_tokens, event_tokens, event_mask, cond)

        # context selects from audio's top-K
        selection_logits, top_k_indices = self.context_path(
            event_tokens, audio_tokens, event_mask, cond, audio_logits
        )

        # scatter selection back to 501-way for compatibility with metrics
        logits = torch.full_like(audio_logits, -100.0)
        logits.scatter_(1, top_k_indices, selection_logits)

        return logits, audio_logits, selection_logits, top_k_indices
