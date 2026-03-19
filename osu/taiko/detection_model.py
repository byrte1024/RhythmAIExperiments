"""Onset detection model.

Exp 25+: Unified architecture - audio + gap tokens fused via self-attention.
Exp 24: Additive context logits (AdditiveOnsetDetector).
Exp 19-23: Gap-based reranker (RerankerOnsetDetector).
Exp 17-18: Shared encoder rerankers (legacy).
Exp 11-16: Legacy two-path additive (LegacyOnsetDetector).
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


class GapEncoder(nn.Module):
    """Gap-based event encoder (exp 25+).

    Replaces EventEncoder with the proven gap representation from exp 19-24.
    Computes inter-onset intervals, extracts mel snippets at event positions,
    and processes through self-attention to produce gap tokens at d_model.

    Output tokens participate in fusion self-attention alongside audio tokens.
    """

    def __init__(self, n_mels=80, d_model=384, n_layers=2, n_heads=6,
                 max_events=128, cond_dim=64, dropout=0.1, snippet_frames=10):
        super().__init__()
        self.d_model = d_model
        self.snippet_frames = snippet_frames
        self.max_events = max_events

        # Snippet encoder: ~50ms mel window → d_model feature vector
        self.snippet_encoder = nn.Sequential(
            nn.Linear(n_mels * snippet_frames, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Gap encoding: sinusoidal encoding of inter-onset intervals
        self.gap_emb = SinusoidalPosEmb(d_model)
        self.seq_pos_emb = nn.Embedding(max_events + 1, d_model)

        # Self-attention over gap sequence
        self.film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_model) for _ in range(n_layers)]
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])

    def _extract_snippets(self, mel, frame_positions, valid_mask):
        """Extract mel snippets at given positions and encode them.

        mel: (B, n_mels, T)
        frame_positions: (B, N) mel frame indices
        valid_mask: (B, N) bool - True = valid position
        Returns: (B, N, d_model)
        """
        B, n_mels, T = mel.shape
        N = frame_positions.size(1)
        half = self.snippet_frames // 2

        centers = frame_positions.clamp(half, T - half - 1)
        offsets = torch.arange(-half, half, device=mel.device)
        frame_idx = centers.unsqueeze(-1) + offsets
        frame_idx = frame_idx.clamp(0, T - 1)

        flat_idx = frame_idx.reshape(B, -1)
        flat_idx = flat_idx.unsqueeze(1).expand(-1, n_mels, -1)
        snippets = mel.gather(2, flat_idx)
        snippets = snippets.reshape(B, n_mels, N, self.snippet_frames)
        snippets = snippets.permute(0, 2, 1, 3).reshape(B, N, -1)
        snippets = snippets * valid_mask.unsqueeze(-1).float()

        return self.snippet_encoder(snippets)

    def forward(self, event_offsets, event_mask, mel, cond):
        """
        event_offsets: (B, C) int - bin positions relative to cursor (negative/zero)
        event_mask: (B, C) bool - True = padded
        mel: (B, n_mels, 1000)
        cond: (B, cond_dim)
        Returns: (gap_tokens: (B, C, d_model), gap_mask: (B, C) bool)
        """
        B, C = event_offsets.shape
        event_valid = ~event_mask

        # ── 1. Compute gap sequence ──
        gap_before = event_offsets[:, 1:] - event_offsets[:, :-1]
        gap_valid = event_valid[:, 1:] & event_valid[:, :-1]

        has_events = event_valid[:, -1]
        time_since_last = (-event_offsets[:, -1]).unsqueeze(1)

        all_gaps = torch.cat([gap_before, time_since_last], dim=1)  # (B, C)
        all_gap_valid = torch.cat([gap_valid, has_events.unsqueeze(1)], dim=1)
        all_gap_mask = ~all_gap_valid

        # ── 2. Build representations: gap encoding + audio snippets ──
        gap_features = self.gap_emb(all_gaps.abs())  # (B, C, d_model)

        event_mel_frames = 500 + event_offsets
        snippet_valid_events = event_valid & (event_mel_frames >= 0) & (event_mel_frames < mel.size(2))

        snippet_frames_for_gaps = torch.cat([
            event_mel_frames[:, 1:],
            torch.full((B, 1), 500, device=mel.device, dtype=event_mel_frames.dtype),
        ], dim=1)
        snippet_valid_for_gaps = torch.cat([
            snippet_valid_events[:, 1:],
            has_events.unsqueeze(1),
        ], dim=1)

        event_snippet_feat = self._extract_snippets(
            mel, snippet_frames_for_gaps, snippet_valid_for_gaps
        )  # (B, C, d_model)

        x = gap_features + event_snippet_feat

        seq_pos = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.seq_pos_emb(seq_pos)

        # NaN guard
        all_masked = all_gap_mask.all(dim=1)
        if all_masked.any():
            safe_mask = all_gap_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = all_gap_mask

        # ── 3. Self-attention over gap sequence ──
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, src_key_padding_mask=safe_mask)
            x = film(x, cond)

        return x, safe_mask  # (B, C, d_model), (B, C) mask


class AudioPath(nn.Module):
    """Audio-primary proposer: audio self-attention + cross-attention to events.

    Extracts the cursor token (center of window) and produces broad logits.
    Used by legacy detectors (exp 11-24).
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


class LegacyContextPath(nn.Module):
    """Legacy 501-way context path (exp 11-16). Used for loading old checkpoints."""

    def __init__(self, d_model=384, n_layers=3, n_heads=8, n_classes=501,
                 max_events=128, cond_dim=64, dropout=0.1):
        super().__init__()
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
        self.head_norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def forward(self, event_tokens, audio_tokens, event_mask, cond):
        B, C, _ = event_tokens.shape
        query = self.query_token.expand(B, -1, -1)
        seq = torch.cat([event_tokens, query], dim=1)
        seq_pos = torch.arange(C + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.seq_pos_emb(seq_pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(C + 1, device=seq.device)
        query_pad = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        tgt_pad_mask = torch.cat([event_mask, query_pad], dim=1)
        x = seq
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, audio_tokens, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad_mask)
            x = film(x, cond)
        query_out = x[:, -1, :]
        logits = self.head_proj(self.head_norm(query_out))
        logits = logits + self.head_smooth(logits.unsqueeze(1)).squeeze(1)
        return logits


class Exp17ContextPath(nn.Module):
    """Exp 17 ContextPath: top-K with dot-product scoring, audio cross-attention.

    Used for loading exp 17 checkpoints.
    """

    def __init__(self, d_model=384, n_layers=3, n_heads=8, K=20,
                 max_events=128, cond_dim=64, dropout=0.1):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.bin_pos_emb = SinusoidalPosEmb(d_model)
        self.score_proj = nn.Sequential(nn.Linear(2, d_model), nn.GELU())
        self.candidate_combine = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.seq_pos_emb = nn.Embedding(max_events + 1, d_model)
        self.film_layers = nn.ModuleList([FiLM(cond_dim, d_model) for _ in range(n_layers)])
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(n_layers)
        ])
        d_score = 64
        self.q_proj = nn.Linear(d_model, d_score)
        self.k_proj = nn.Linear(d_model, d_score)
        self.score_scale = d_score ** -0.5

    def forward(self, event_tokens, audio_tokens, event_mask, cond, audio_logits):
        B, C, _ = event_tokens.shape
        audio_logits_det = audio_logits.detach()
        top_k_scores, top_k_indices = audio_logits_det.topk(self.K, dim=-1)
        stop_idx = audio_logits_det.size(1) - 1
        has_stop = (top_k_indices == stop_idx).any(dim=1)
        if not has_stop.all():
            no_stop = ~has_stop
            top_k_indices[no_stop, -1] = stop_idx
            top_k_scores[no_stop, -1] = audio_logits_det[no_stop, stop_idx]

        bin_emb = self.bin_pos_emb(top_k_indices)
        ranks = torch.arange(1, self.K + 1, device=audio_logits.device, dtype=torch.float32)
        ranks = ranks.unsqueeze(0).expand(B, -1) / self.K
        score_feat = self.score_proj(torch.stack([top_k_scores, ranks], dim=-1))
        token_idx = ((500 + top_k_indices.clamp(max=499)).float() / 4.0).long()
        token_idx = token_idx.clamp(0, audio_tokens.size(1) - 1)
        audio_feat = audio_tokens.gather(
            1, token_idx.unsqueeze(-1).expand(-1, -1, audio_tokens.size(-1))
        )
        candidate_feat = self.candidate_combine(
            torch.cat([bin_emb, score_feat, audio_feat], dim=-1)
        )

        query = self.query_token.expand(B, -1, -1)
        seq = torch.cat([event_tokens, query], dim=1)
        seq_pos = torch.arange(C + 1, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.seq_pos_emb(seq_pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(C + 1, device=seq.device)
        query_pad = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        tgt_pad_mask = torch.cat([event_mask, query_pad], dim=1)
        x = seq
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, audio_tokens, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad_mask)
            x = film(x, cond)
        query_out = x[:, -1, :]

        q = self.q_proj(query_out)
        k = self.k_proj(candidate_feat)
        selection_logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.score_scale
        return selection_logits, top_k_indices


class Exp18ContextPath(nn.Module):
    """Exp 18 context path: two-stage with shared encoder features (detached).

    Used for loading exp 18 checkpoints.
    """

    def __init__(self, d_model=384, n_event_layers=2, n_select_layers=2,
                 n_heads=8, K=20, max_events=128, cond_dim=64, dropout=0.1):
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

        # Stage 1: Event understanding (pure self-attention)
        self.seq_pos_emb = nn.Embedding(max_events + 1, d_model)

        self.event_film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_model) for _ in range(n_event_layers)]
        )
        self.event_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_event_layers)
        ])

        # Stage 2: Candidate selection (self-attn + cross-attn to candidates)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.select_film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_model) for _ in range(n_select_layers)]
        )
        self.select_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_select_layers)
        ])

        # Selection scoring: dot product between query and each candidate
        d_score = 64
        self.q_proj = nn.Linear(d_model, d_score)
        self.k_proj = nn.Linear(d_model, d_score)
        self.score_scale = d_score ** -0.5

    def forward(self, event_tokens, audio_tokens, event_mask, cond, audio_logits):
        """
        All tensor inputs should be detached (stop-gradient).
        event_tokens: (B, C, d_model) from EventEncoder
        audio_tokens: (B, 250, d_model)
        event_mask: (B, C) bool - True = padded
        cond: (B, cond_dim)
        audio_logits: (B, n_classes) from AudioPath
        Returns: (selection_logits: (B, K), top_k_indices: (B, K))
        """
        B, C, _ = event_tokens.shape

        # ── 1. Get top-K candidates from audio (sorted by confidence) ──
        audio_logits_det = audio_logits.detach()
        top_k_scores, top_k_indices = audio_logits_det.topk(self.K, dim=-1)  # (B, K)

        # Force-include STOP (class 500) if not already in top-K
        stop_idx = audio_logits_det.size(1) - 1
        has_stop = (top_k_indices == stop_idx).any(dim=1)  # (B,)
        if not has_stop.all():
            no_stop = ~has_stop
            top_k_indices[no_stop, -1] = stop_idx
            top_k_scores[no_stop, -1] = audio_logits_det[no_stop, stop_idx]

        # ── 2. Build candidate embeddings (position + confidence + audio feature) ──
        bin_emb = self.bin_pos_emb(top_k_indices)  # (B, K, d_model)

        ranks = torch.arange(1, self.K + 1, device=audio_logits.device, dtype=torch.float32)
        ranks = ranks.unsqueeze(0).expand(B, -1) / self.K  # (B, K) normalized 0-1
        score_feat = self.score_proj(
            torch.stack([top_k_scores, ranks], dim=-1)  # (B, K, 2)
        )  # (B, K, d_model)

        token_idx = ((500 + top_k_indices.clamp(max=499)).float() / 4.0).long()
        token_idx = token_idx.clamp(0, audio_tokens.size(1) - 1)  # (B, K)
        audio_feat = audio_tokens.gather(
            1, token_idx.unsqueeze(-1).expand(-1, -1, audio_tokens.size(-1))
        )  # (B, K, d_model)

        candidate_feat = self.candidate_combine(
            torch.cat([bin_emb, score_feat, audio_feat], dim=-1)
        )  # (B, K, d_model)

        # ── 3. Stage 1: Event understanding (pure self-attention) ──
        # Events reason about temporal patterns before seeing any candidates
        x = event_tokens  # (B, C, d_model)
        seq_pos = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.seq_pos_emb(seq_pos)

        # guard: NaN from all-masked attention
        all_masked = event_mask.all(dim=1)
        if all_masked.any():
            safe_mask = event_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = event_mask

        for layer, film in zip(self.event_layers, self.event_film_layers):
            x = layer(x, src_key_padding_mask=safe_mask)
            x = film(x, cond)

        # ── 4. Stage 2: Candidate selection (cross-attend to candidates) ──
        # Append query token with its own position embedding
        query = self.query_token.expand(B, -1, -1)
        query_pos = self.seq_pos_emb(
            torch.full((B, 1), C, dtype=torch.long, device=x.device)
        )  # (B, 1, d_model)
        seq = torch.cat([x, query + query_pos], dim=1)  # (B, C+1, d_model)

        # No causal mask: stage 1 already built global event representations,
        # causal mask would undo that for early events. All events evaluate
        # candidates on equal footing, query collects consensus.
        query_pad = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
        tgt_pad_mask = torch.cat([safe_mask, query_pad], dim=1)

        for layer, film in zip(self.select_layers, self.select_film_layers):
            seq = layer(seq, candidate_feat,
                        tgt_key_padding_mask=tgt_pad_mask)
            seq = film(seq, cond)

        query_out = seq[:, -1, :]  # (B, d_model) - consensus from all events

        # ── 5. Score candidates via scaled dot product → K-way logits ──
        q = self.q_proj(query_out)  # (B, d_score)
        k = self.k_proj(candidate_feat)  # (B, K, d_score)
        selection_logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.score_scale  # (B, K)

        return selection_logits, top_k_indices


class RerankerContextPath(nn.Module):
    """Gap-based K-way reranker with own encoders (exp 19-23, legacy).

    Context operates entirely in "gap space" - inter-onset intervals rather than
    absolute positions. Each event is characterized by (gap_before, local_audio_snippet).
    Each candidate is characterized by (proposed_next_gap, local_audio_snippet, audio_score).

    Own encoders: gap encoder + snippet encoder, both trained by selection loss.
    No shared encoder dependencies - fully gradient-isolated by design.

    Architecture:
      1. Compute gaps from raw event offsets (diff of consecutive positions)
      2. Extract ~50ms mel snippets at each event and candidate position
      3. Gap encoder: gap_emb + snippet_feat → self-attention → rhythm_repr
      4. Candidate embeddings: proposed_gap_emb + snippet_feat + score/rank
      5. Selection: rhythm_repr cross-attends to candidates → query → dot-product → K-way
    """

    def __init__(self, n_mels=80, d_ctx=192, n_gap_layers=2, n_select_layers=2,
                 n_heads=6, K=20, max_events=128, cond_dim=64, dropout=0.1,
                 snippet_frames=10):
        super().__init__()
        self.K = K
        self.d_ctx = d_ctx
        self.snippet_frames = snippet_frames
        self.max_events = max_events

        # ── Snippet encoder (shared for events and candidates) ──
        # ~50ms mel window → d_ctx feature vector
        self.snippet_encoder = nn.Sequential(
            nn.Linear(n_mels * snippet_frames, d_ctx),
            nn.GELU(),
            nn.Linear(d_ctx, d_ctx),
        )
        # STOP has no meaningful audio - learned embedding
        self.stop_snippet_emb = nn.Parameter(torch.randn(d_ctx) * 0.02)

        # ── Gap encoder (processes rhythm pattern) ──
        self.gap_emb = SinusoidalPosEmb(d_ctx)
        self.seq_pos_emb = nn.Embedding(max_events + 1, d_ctx)

        self.gap_film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_ctx) for _ in range(n_gap_layers)]
        )
        self.gap_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_ctx, nhead=n_heads, dim_feedforward=d_ctx * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_gap_layers)
        ])

        # ── Candidate building ──
        self.score_proj = nn.Sequential(nn.Linear(1, d_ctx), nn.GELU())
        self.candidate_combine = nn.Sequential(
            nn.Linear(d_ctx * 3, d_ctx),  # gap_emb + snippet + score
            nn.GELU(),
            nn.LayerNorm(d_ctx),
        )

        # ── Selection head (cross-attend to candidates) ──
        self.query_token = nn.Parameter(torch.randn(1, 1, d_ctx) * 0.02)

        self.select_film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_ctx) for _ in range(n_select_layers)]
        )
        self.select_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_ctx, nhead=n_heads, dim_feedforward=d_ctx * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_select_layers)
        ])

        # ── Scoring ──
        d_score = 64
        self.q_proj = nn.Linear(d_ctx, d_score)
        self.k_proj = nn.Linear(d_ctx, d_score)
        self.score_scale = d_score ** -0.5

    def _extract_snippets(self, mel, frame_positions, valid_mask):
        """Extract mel snippets and encode them.

        mel: (B, n_mels, T)
        frame_positions: (B, N) mel frame indices
        valid_mask: (B, N) bool - True = valid position, False = skip (gets zeros)
        Returns: (B, N, d_ctx)
        """
        B, n_mels, T = mel.shape
        N = frame_positions.size(1)
        half = self.snippet_frames // 2

        # Clamp to valid range
        centers = frame_positions.clamp(half, T - half - 1)  # (B, N)

        # Gather snippet frames: for each position, grab [center-half : center+half]
        offsets = torch.arange(-half, half, device=mel.device)  # (snippet_frames,)
        frame_idx = centers.unsqueeze(-1) + offsets  # (B, N, snippet_frames)
        frame_idx = frame_idx.clamp(0, T - 1)

        # Gather from mel: (B, n_mels, T) → (B, N, n_mels * snippet_frames)
        flat_idx = frame_idx.reshape(B, -1)  # (B, N * snippet_frames)
        flat_idx = flat_idx.unsqueeze(1).expand(-1, n_mels, -1)  # (B, n_mels, N*sf)
        snippets = mel.gather(2, flat_idx)  # (B, n_mels, N * snippet_frames)
        snippets = snippets.reshape(B, n_mels, N, self.snippet_frames)
        snippets = snippets.permute(0, 2, 1, 3).reshape(B, N, -1)  # (B, N, n_mels*sf)

        # Zero out invalid positions (out-of-window events, padding)
        snippets = snippets * valid_mask.unsqueeze(-1).float()

        return self.snippet_encoder(snippets)  # (B, N, d_ctx)

    def forward(self, event_offsets, event_mask, mel, cond, audio_logits):
        """
        event_offsets: (B, C) int - bin positions relative to cursor (negative/zero)
        event_mask: (B, C) bool - True = padded
        mel: (B, n_mels, 1000) - raw mel spectrogram
        cond: (B, cond_dim) - conditioning (detached by caller)
        audio_logits: (B, n_classes) - from AudioPath
        Returns: (selection_logits: (B, K), top_k_indices: (B, K))
        """
        B, C = event_offsets.shape

        # ── 1. Get top-K candidates from audio ──
        audio_logits_det = audio_logits.detach()
        top_k_scores, top_k_indices = audio_logits_det.topk(self.K, dim=-1)  # (B, K)

        stop_idx = audio_logits_det.size(1) - 1
        has_stop = (top_k_indices == stop_idx).any(dim=1)
        if not has_stop.all():
            no_stop = ~has_stop
            top_k_indices[no_stop, -1] = stop_idx
            top_k_scores[no_stop, -1] = audio_logits_det[no_stop, stop_idx]

        # ── 2. Compute gap sequence from event offsets ──
        # event_offsets: right-aligned, e.g. [0,0,0,...,-500,-350,-200,-100,-50]
        #                                     masked          valid (sorted)
        # Each event[i] gets gap_before = offset[i] - offset[i-1]
        # First valid event: gap is meaningless (masked)
        # Append cursor element: gap = time_since_last = -offset[-1]

        event_valid = ~event_mask  # (B, C) True = real event

        # Gap before each event: offset[i] - offset[i-1]
        gap_before = event_offsets[:, 1:] - event_offsets[:, :-1]  # (B, C-1)
        # Valid only if both this event and previous are valid
        gap_valid = event_valid[:, 1:] & event_valid[:, :-1]  # (B, C-1)

        # Time since last event (cursor gap)
        has_events = event_valid[:, -1]  # (B,) - last position is valid if any events
        time_since_last = (-event_offsets[:, -1]).unsqueeze(1)  # (B, 1) always >= 0

        # Build full gap sequence: [gap_before_event1, ..., gap_before_eventN, cursor_gap]
        # Length: (C-1) + 1 = C
        all_gaps = torch.cat([gap_before, time_since_last], dim=1)  # (B, C)
        all_gap_valid = torch.cat([gap_valid, has_events.unsqueeze(1)], dim=1)  # (B, C)
        all_gap_mask = ~all_gap_valid  # True = masked (for transformer)

        # ── 3. Build event representations: gap encoding + audio snippets ──
        # Gap features (sinusoidal encoding of gap magnitude)
        gap_features = self.gap_emb(all_gaps.abs())  # (B, C, d_ctx)

        # Mel snippets at each event position + cursor
        # Event mel frames: 500 + offset (cursor at frame 500 in the 1000-frame window)
        event_mel_frames = 500 + event_offsets  # (B, C)
        # Valid snippet: event is real AND within mel window [0, 999]
        snippet_valid_events = event_valid & (event_mel_frames >= 0) & (event_mel_frames < mel.size(2))

        # For gap sequence positions [0..C-2]: snippet is at the TARGET event (index 1..C-1)
        # For gap position [C-1] (cursor): snippet at frame 500
        snippet_frames_for_gaps = torch.cat([
            event_mel_frames[:, 1:],  # (B, C-1) target event of each gap
            torch.full((B, 1), 500, device=mel.device, dtype=event_mel_frames.dtype),  # cursor
        ], dim=1)  # (B, C)
        snippet_valid_for_gaps = torch.cat([
            snippet_valid_events[:, 1:],  # (B, C-1)
            has_events.unsqueeze(1),  # (B, 1) cursor snippet valid if any events
        ], dim=1)  # (B, C)

        event_snippet_feat = self._extract_snippets(
            mel, snippet_frames_for_gaps, snippet_valid_for_gaps
        )  # (B, C, d_ctx)

        # Combine gap + snippet (additive)
        x = gap_features + event_snippet_feat  # (B, C, d_ctx)

        # Add sequence position embeddings
        seq_pos = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.seq_pos_emb(seq_pos)

        # NaN guard: unmask at least one position
        all_masked = all_gap_mask.all(dim=1)
        if all_masked.any():
            safe_mask = all_gap_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = all_gap_mask

        # Gap encoder: self-attention over rhythm pattern
        for layer, film in zip(self.gap_layers, self.gap_film_layers):
            x = layer(x, src_key_padding_mask=safe_mask)
            x = film(x, cond)

        # ── 4. Build candidate embeddings ──
        # Proposed gap: distance from last event to each candidate
        last_event_offset = event_offsets[:, -1]  # (B,) most recent event offset
        # For samples with no events, use 0 (candidates measure from cursor)
        last_event_offset = last_event_offset * has_events.long()
        proposed_gaps = top_k_indices.long() - last_event_offset.unsqueeze(1)  # (B, K)

        # Gap embedding for candidates
        cand_gap_emb = self.gap_emb(proposed_gaps.abs())  # (B, K, d_ctx)

        # Audio snippets at candidate positions
        cand_mel_frames = 500 + top_k_indices  # (B, K) - candidates are future bins
        is_stop = (top_k_indices == stop_idx)  # (B, K)
        cand_snippet_valid = ~is_stop & (cand_mel_frames < mel.size(2))

        cand_snippet_feat = self._extract_snippets(
            mel, cand_mel_frames.clamp(max=mel.size(2) - 1), cand_snippet_valid
        )  # (B, K, d_ctx)
        # STOP candidates get learned embedding instead
        if is_stop.any():
            cand_snippet_feat = cand_snippet_feat.clone()
            cand_snippet_feat[is_stop] = self.stop_snippet_emb

        # Audio confidence per candidate (softmax probability, no rank info)
        audio_probs = F.softmax(audio_logits_det, dim=-1)
        cand_probs = audio_probs.gather(1, top_k_indices.long())  # (B, K)
        score_feat = self.score_proj(cand_probs.unsqueeze(-1))  # (B, K, d_ctx)

        candidate_feat = self.candidate_combine(
            torch.cat([cand_gap_emb, cand_snippet_feat, score_feat], dim=-1)
        )  # (B, K, d_ctx)

        # Shuffle candidates so context can't learn positional bias (k=0 = audio's #1)
        if self.training:
            shuffle_idx = torch.stack([torch.randperm(self.K, device=mel.device) for _ in range(B)])
            candidate_feat = candidate_feat.gather(1, shuffle_idx.unsqueeze(-1).expand(-1, -1, candidate_feat.size(-1)))
            top_k_indices = top_k_indices.gather(1, shuffle_idx)

        # ── 5. Selection: cross-attend to candidates ──
        query = self.query_token.expand(B, -1, -1)
        query_pos = self.seq_pos_emb(
            torch.full((B, 1), C, dtype=torch.long, device=x.device)
        )
        seq = torch.cat([x, query + query_pos], dim=1)  # (B, C+1, d_ctx)

        query_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        tgt_pad_mask = torch.cat([safe_mask, query_pad], dim=1)

        for layer, film in zip(self.select_layers, self.select_film_layers):
            seq = layer(seq, candidate_feat, tgt_key_padding_mask=tgt_pad_mask)
            seq = film(seq, cond)

        query_out = seq[:, -1, :]  # (B, d_ctx)

        # ── 6. Score candidates via scaled dot product ──
        q = self.q_proj(query_out)  # (B, d_score)
        k = self.k_proj(candidate_feat)  # (B, K, d_score)
        selection_logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.score_scale

        return selection_logits, top_k_indices


class ContextPath(nn.Module):
    """Additive context path with own encoders (exp 24+).

    Produces 501-way logits from rhythm patterns (gaps + mel snippets).
    Added to audio logits before softmax - soft influence, not hard override.

    Keeps the proven gap encoder from exp 19-23 but replaces the K-way
    selection head with a simple output projection to n_classes logits.

    Architecture:
      1. Compute gaps from raw event offsets (diff of consecutive positions)
      2. Extract ~50ms mel snippets at each event position
      3. Gap encoder: gap_emb + snippet_feat → self-attention + FiLM → rhythm_repr
      4. Output: cursor token → project to n_classes logits
    """

    def __init__(self, n_mels=80, d_ctx=192, n_gap_layers=2,
                 n_heads=6, n_classes=501, max_events=128, cond_dim=64,
                 dropout=0.1, snippet_frames=10):
        super().__init__()
        self.d_ctx = d_ctx
        self.n_classes = n_classes
        self.snippet_frames = snippet_frames
        self.max_events = max_events

        # ── Snippet encoder (shared for all event positions) ──
        self.snippet_encoder = nn.Sequential(
            nn.Linear(n_mels * snippet_frames, d_ctx),
            nn.GELU(),
            nn.Linear(d_ctx, d_ctx),
        )

        # ── Gap encoder (processes rhythm pattern) ──
        self.gap_emb = SinusoidalPosEmb(d_ctx)
        self.seq_pos_emb = nn.Embedding(max_events + 1, d_ctx)

        self.gap_film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_ctx) for _ in range(n_gap_layers)]
        )
        self.gap_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_ctx, nhead=n_heads, dim_feedforward=d_ctx * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_gap_layers)
        ])

        # ── Output head: cursor token → n_classes logits ──
        self.cursor_token = nn.Parameter(torch.randn(1, 1, d_ctx) * 0.02)
        self.output_head = nn.Sequential(
            nn.Linear(d_ctx, d_ctx),
            nn.GELU(),
            nn.Linear(d_ctx, n_classes),
        )

    def _extract_snippets(self, mel, frame_positions, valid_mask):
        """Extract mel snippets and encode them.

        mel: (B, n_mels, T)
        frame_positions: (B, N) mel frame indices
        valid_mask: (B, N) bool - True = valid position, False = skip (gets zeros)
        Returns: (B, N, d_ctx)
        """
        B, n_mels, T = mel.shape
        N = frame_positions.size(1)
        half = self.snippet_frames // 2

        centers = frame_positions.clamp(half, T - half - 1)
        offsets = torch.arange(-half, half, device=mel.device)
        frame_idx = centers.unsqueeze(-1) + offsets
        frame_idx = frame_idx.clamp(0, T - 1)

        flat_idx = frame_idx.reshape(B, -1)
        flat_idx = flat_idx.unsqueeze(1).expand(-1, n_mels, -1)
        snippets = mel.gather(2, flat_idx)
        snippets = snippets.reshape(B, n_mels, N, self.snippet_frames)
        snippets = snippets.permute(0, 2, 1, 3).reshape(B, N, -1)

        snippets = snippets * valid_mask.unsqueeze(-1).float()

        return self.snippet_encoder(snippets)

    def forward(self, event_offsets, event_mask, mel, cond):
        """
        event_offsets: (B, C) int - bin positions relative to cursor (negative/zero)
        event_mask: (B, C) bool - True = padded
        mel: (B, n_mels, 1000) - raw mel spectrogram
        cond: (B, cond_dim) - conditioning (detached by caller)
        Returns: context_logits (B, n_classes)
        """
        B, C = event_offsets.shape

        event_valid = ~event_mask

        # ── 1. Compute gap sequence ──
        gap_before = event_offsets[:, 1:] - event_offsets[:, :-1]
        gap_valid = event_valid[:, 1:] & event_valid[:, :-1]

        has_events = event_valid[:, -1]
        time_since_last = (-event_offsets[:, -1]).unsqueeze(1)

        all_gaps = torch.cat([gap_before, time_since_last], dim=1)
        all_gap_valid = torch.cat([gap_valid, has_events.unsqueeze(1)], dim=1)
        all_gap_mask = ~all_gap_valid

        # ── 2. Build event representations: gap encoding + audio snippets ──
        gap_features = self.gap_emb(all_gaps.abs())

        event_mel_frames = 500 + event_offsets
        snippet_valid_events = event_valid & (event_mel_frames >= 0) & (event_mel_frames < mel.size(2))

        snippet_frames_for_gaps = torch.cat([
            event_mel_frames[:, 1:],
            torch.full((B, 1), 500, device=mel.device, dtype=event_mel_frames.dtype),
        ], dim=1)
        snippet_valid_for_gaps = torch.cat([
            snippet_valid_events[:, 1:],
            has_events.unsqueeze(1),
        ], dim=1)

        event_snippet_feat = self._extract_snippets(
            mel, snippet_frames_for_gaps, snippet_valid_for_gaps
        )

        x = gap_features + event_snippet_feat

        seq_pos = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.seq_pos_emb(seq_pos)

        # NaN guard
        all_masked = all_gap_mask.all(dim=1)
        if all_masked.any():
            safe_mask = all_gap_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = all_gap_mask

        # ── 3. Gap encoder: self-attention over rhythm ──
        for layer, film in zip(self.gap_layers, self.gap_film_layers):
            x = layer(x, src_key_padding_mask=safe_mask)
            x = film(x, cond)

        # ── 4. Append cursor token and read it out ──
        cursor = self.cursor_token.expand(B, -1, -1)
        cursor_pos = self.seq_pos_emb(
            torch.full((B, 1), C, dtype=torch.long, device=x.device)
        )
        x = torch.cat([x, cursor + cursor_pos], dim=1)

        # One more self-attention pass so cursor attends to rhythm
        cursor_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([safe_mask, cursor_mask], dim=1)
        # Re-use last gap layer for cursor attention (saves params)
        x = self.gap_layers[-1](x, src_key_padding_mask=full_mask)
        x = self.gap_film_layers[-1](x, cond)

        cursor_out = x[:, -1, :]  # (B, d_ctx)

        # ── 5. Project to logits ──
        return self.output_head(cursor_out)  # (B, n_classes)


class OnsetDetector(nn.Module):
    """Unified onset detector (exp 25+).

    Single path: AudioEncoder → GapEncoder → Fusion self-attention → 501 logits.
    Audio and gap tokens concatenated and jointly attended - no separate paths.

    Architecture:
      1. AudioEncoder: mel → 250 audio tokens (d_model)
      2. GapEncoder: event gaps + snippets → C gap tokens (d_model)
      3. Concatenate [audio_tokens; gap_tokens] → 250+C tokens
      4. FusionTransformer: N self-attention layers over combined tokens
      5. Extract cursor at position 125 → output head → 501 logits
    """

    def __init__(
        self,
        n_mels=80,
        d_model=384,
        enc_layers=4,
        gap_enc_layers=2,
        fusion_layers=4,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
        snippet_frames=10,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # audio encoder (warm-start from exp 14)
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )

        # gap encoder (replaces EventEncoder - proven gap representation)
        self.gap_encoder = GapEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=gap_enc_layers,
            n_heads=n_heads, max_events=max_events, cond_dim=cond_dim,
            dropout=dropout, snippet_frames=snippet_frames,
        )

        # fusion: self-attention over concatenated [audio; gap] tokens
        self.fusion_film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_model) for _ in range(fusion_layers)]
        )
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(fusion_layers)
        ])

        # output head (same as AudioPath)
        self.head_norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def _embed_events_in_mel(self, mel, event_offsets, event_mask):
        """Embed event positions as gradient ramps in reserved mel bands.

        Fully vectorized — no Python loops over batch or events.

        For each frame t, computes "time since last event" as a normalized ramp:
        ramp(t) = clamp(1 - (t - last_event_before_t) / gap_to_next_event, 0, 1)

        Bands 0-2 (bottom) and 77-79 (top), with fading intensity inward.
        """
        B, n_mels, T = mel.shape
        cursor_frame = T // 2  # 500

        mel = mel.clone()

        # frame positions of events: (B, C)
        frame_pos = (cursor_frame + event_offsets).float()  # (B, C)
        valid = ~event_mask  # (B, C)

        # replace invalid positions with large negative so they sort to the front
        # and don't affect the ramp computation
        frame_pos = torch.where(valid, frame_pos, torch.full_like(frame_pos, -1e6))

        # sort events by position for each sample
        frame_pos_sorted, _ = frame_pos.sort(dim=1)  # (B, C)

        # append cursor frame as final position: (B, C+1)
        cursor_col = torch.full((B, 1), cursor_frame, device=mel.device, dtype=frame_pos.dtype)
        positions = torch.cat([frame_pos_sorted, cursor_col], dim=1)  # (B, C+1)

        # for each frame t in [0, T), find which event interval it falls in
        # t_grid: (1, 1, T)
        t_grid = torch.arange(T, device=mel.device, dtype=torch.float32).view(1, 1, T)
        # positions: (B, C+1, 1)
        pos_exp = positions.unsqueeze(2)  # (B, C+1, 1)

        # for each frame, find the last event at or before it
        # mask: (B, C+1, T) — True if position <= t
        before_mask = pos_exp <= t_grid  # (B, C+1, T)
        # also exclude the dummy -1e6 positions
        real_mask = (pos_exp > -1e5) & before_mask

        # last event position before each frame: take max of valid positions
        neg_inf = torch.full_like(pos_exp, -1e6)
        masked_pos = torch.where(real_mask, pos_exp, neg_inf)  # (B, C+1, T)
        last_event, last_idx = masked_pos.max(dim=1)  # (B, T)

        # next event position after last_event: positions[last_idx + 1]
        # clamp index to valid range
        next_idx = (last_idx + 1).clamp(max=positions.size(1) - 1)  # (B, T)
        next_event = positions.gather(1, next_idx)  # (B, T)

        # gap between consecutive events
        gap = next_event - last_event  # (B, T)
        gap = gap.clamp(min=1.0)  # avoid div by zero

        # exponential decay ramp: sharp spike at event, fast falloff
        # at 3% of the gap, signal decays to 50% → half_life = 0.03 * gap
        # ramp(t) = exp(-ln(2) * (t - last_event) / half_life)
        t_flat = torch.arange(T, device=mel.device, dtype=torch.float32).unsqueeze(0)  # (1, T)
        elapsed = (t_flat - last_event).clamp(min=0.0)  # time since last event
        half_life = (gap * 0.03).clamp(min=0.5)  # 3% of gap, min 0.5 frames
        ramp = torch.exp(-0.693147 * elapsed / half_life)  # ln(2) ≈ 0.693147

        # zero out ramp where no valid events exist (last_event == -1e6)
        ramp = torch.where(last_event > -1e5, ramp, torch.zeros_like(ramp))

        # amplitude jitter: random scaling of audio (0.25-0.75) per sample during training
        if self.training:
            audio_scale = 0.25 + 0.5 * torch.rand(B, 1, 1, device=mel.device)
        else:
            audio_scale = 0.5
        mel = mel * audio_scale
        ramp = ramp * 10.0  # (B, T)
        mel = mel + ramp.unsqueeze(1)  # broadcast (B, 1, T) across all mel bands

        return mel

    def forward(self, mel, event_offsets, event_mask, conditioning):
        """
        mel: (B, n_mels, 1000)
        event_offsets: (B, C) past event bin positions relative to cursor
        event_mask: (B, C) bool, True = padding
        conditioning: (B, 3) [mean_density, peak_density, density_std]
        Returns: logits (B, 501)
        """
        cond = self.cond_mlp(conditioning)

        # embed event ramps into mel before audio encoding
        mel = self._embed_events_in_mel(mel, event_offsets, event_mask)

        # encode audio → 250 tokens
        audio_tokens = self.audio_encoder(mel, cond)  # (B, 250, d_model)

        # encode gaps → C tokens
        gap_tokens, gap_mask = self.gap_encoder(
            event_offsets, event_mask, mel, cond
        )  # (B, C, d_model), (B, C)

        # concatenate [audio; gap] - audio first so cursor stays at position 125
        x = torch.cat([audio_tokens, gap_tokens], dim=1)  # (B, 250+C, d_model)

        # padding mask: audio tokens are never masked, gap tokens use gap_mask
        B = mel.size(0)
        audio_pad = torch.zeros(B, audio_tokens.size(1), dtype=torch.bool, device=mel.device)
        fused_mask = torch.cat([audio_pad, gap_mask], dim=1)  # (B, 250+C)

        # fusion: self-attention over all tokens
        for layer, film in zip(self.fusion_layers, self.fusion_film_layers):
            x = layer(x, src_key_padding_mask=fused_mask)
            x = film(x, cond)

        # extract cursor (center of audio window, position 125)
        cursor = x[:, 125, :]  # (B, d_model)

        logits = self.head_proj(self.head_norm(cursor))
        logits = logits + self.head_smooth(logits.unsqueeze(1)).squeeze(1)
        return logits


class ContextFiLMDetector(nn.Module):
    """Audio-only self-attention with context FiLM conditioning (exp 34+).

    Gap tokens are summarized into a conditioning vector via attention pooling,
    then applied as FiLM modulation on the audio fusion layers. Context does not
    compete with audio for attention — it modulates HOW audio features are
    interpreted, like density conditioning but learned from gap patterns.

    Architecture:
      1. AudioEncoder: mel → 250 audio tokens (d_model)
      2. GapEncoder: event gaps + snippets → C gap tokens (d_model)
      3. Context pooling: learned query attends to gap tokens → context vector
      4. Audio fusion: N self-attention layers over 250 audio tokens only,
         with FiLM from density + context at each layer
      5. Extract cursor at position 125 → output head → 501 logits
    """

    def __init__(
        self,
        n_mels=80,
        d_model=384,
        enc_layers=4,
        gap_enc_layers=2,
        fusion_layers=4,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
        snippet_frames=10,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # conditioning MLP (density → cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # audio encoder
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )

        # gap encoder
        self.gap_encoder = GapEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=gap_enc_layers,
            n_heads=n_heads, max_events=max_events, cond_dim=cond_dim,
            dropout=dropout, snippet_frames=snippet_frames,
        )

        # context pooling: learned query → attention over gap tokens → context vector
        self.ctx_pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.ctx_pool_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ctx_pool_norm = nn.LayerNorm(d_model)
        # project context to cond_dim for FiLM
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # audio-only fusion: self-attention over 250 audio tokens
        # FiLM from BOTH density cond AND context cond at each layer
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(fusion_layers)
        ])
        self.fusion_density_film = nn.ModuleList(
            [FiLM(cond_dim, d_model) for _ in range(fusion_layers)]
        )
        self.fusion_context_film = nn.ModuleList(
            [FiLM(cond_dim, d_model) for _ in range(fusion_layers)]
        )

        # output head
        self.head_norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        """
        mel: (B, n_mels, 1000)
        event_offsets: (B, C) past event bin positions relative to cursor
        event_mask: (B, C) bool, True = padding
        conditioning: (B, 3) [mean_density, peak_density, density_std]
        Returns: logits (B, 501)
        """
        cond = self.cond_mlp(conditioning)

        # encode audio → 250 tokens
        audio = self.audio_encoder(mel, cond)  # (B, 250, d_model)

        # encode gaps → C tokens
        gap_tokens, gap_mask = self.gap_encoder(
            event_offsets, event_mask, mel, cond
        )  # (B, C, d_model), (B, C)

        # context pooling: summarize gap tokens → single context vector
        B = mel.size(0)
        query = self.ctx_pool_query.expand(B, -1, -1)  # (B, 1, d_model)
        # NaN guard
        all_masked = gap_mask.all(dim=1)
        if all_masked.any():
            safe_mask = gap_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = gap_mask
        ctx_out, _ = self.ctx_pool_attn(
            query, gap_tokens, gap_tokens, key_padding_mask=safe_mask,
        )  # (B, 1, d_model)
        ctx_vec = self.ctx_proj(self.ctx_pool_norm(ctx_out.squeeze(1)))  # (B, cond_dim)

        # audio-only fusion with dual FiLM conditioning
        for layer, d_film, c_film in zip(
            self.fusion_layers, self.fusion_density_film, self.fusion_context_film,
        ):
            audio = layer(audio)           # audio self-attention (250 tokens only)
            audio = d_film(audio, cond)    # density FiLM
            audio = c_film(audio, ctx_vec) # context FiLM

        # extract cursor
        cursor = audio[:, 125, :]

        logits = self.head_proj(self.head_norm(cursor))
        logits = logits + self.head_smooth(logits.unsqueeze(1)).squeeze(1)
        return logits


class FramewiseOnsetDetector(nn.Module):
    """Framewise onset detector with causal future prediction (exp 38+).

    Predicts onset probability at every position in the future audio window.
    Past tokens have full bidirectional attention (with mel-embedded ramps as context).
    Future tokens use causal masking — each position conditions on all past
    context + all previous future predictions.

    Architecture:
      1. Conv stem: mel (80, 1000) → 250 tokens (d_model), 4x downsample
         Past frames (0-499) have exponential ramps embedded in the mel.
         Future frames (500-999) are clean audio.
      2. Self-attention with causal mask on future tokens:
         - Tokens 0-124 (past): full bidirectional attention
         - Tokens 125-249 (future): causal (each sees past + previous future)
      3. Per-token onset head: each future token → sigmoid P(onset)

    Input: mel (B, 80, 1000) with ramps + event_offsets/mask for ramp computation
    Output: (B, 125) onset probabilities for the future window
    """

    def __init__(
        self,
        n_mels=80,
        d_model=384,
        n_layers=6,
        n_heads=8,
        dropout=0.1,
        cond_dim=64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_past_tokens = 125
        self.n_future_tokens = 125

        # conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # conv stem: mel → 250 tokens
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.film_conv = FiLM(cond_dim, d_model)

        # self-attention layers with FiLM
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.film_layers = nn.ModuleList(
            [FiLM(cond_dim, d_model) for _ in range(n_layers)]
        )

        # per-token onset prediction head (future tokens only)
        self.onset_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # onset feedback: learned embedding added to future tokens
        # where the PREVIOUS token had an onset (shifted by 1)
        self.onset_feedback_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # register causal mask as buffer
        self._register_causal_mask()

    def _register_causal_mask(self):
        """Build attention mask: past tokens bidirectional, future tokens causal."""
        n = self.n_past_tokens + self.n_future_tokens  # 250
        mask = torch.zeros(n, n, dtype=torch.bool)

        # future tokens (125-249) can only attend to:
        # - all past tokens (0-124): always visible
        # - previous and current future tokens: causal
        for i in range(self.n_past_tokens, n):
            # block attention to future tokens beyond current position
            mask[i, i + 1:] = True

        self.register_buffer("causal_mask", mask)

    def _embed_events_in_mel(self, mel, event_offsets, event_mask):
        """Embed exponential decay ramps at past event positions in the mel.

        Same vectorized implementation as OnsetDetector._embed_events_in_mel.
        Only embeds in the past half (frames 0-499). Future half stays clean.
        """
        B, n_mels, T = mel.shape
        cursor_frame = T // 2  # 500

        mel = mel.clone()

        frame_pos = (cursor_frame + event_offsets).float()
        valid = ~event_mask

        frame_pos = torch.where(valid, frame_pos, torch.full_like(frame_pos, -1e6))
        frame_pos_sorted, _ = frame_pos.sort(dim=1)

        cursor_col = torch.full((B, 1), cursor_frame, device=mel.device, dtype=frame_pos.dtype)
        positions = torch.cat([frame_pos_sorted, cursor_col], dim=1)

        t_grid = torch.arange(T, device=mel.device, dtype=torch.float32).view(1, 1, T)
        pos_exp = positions.unsqueeze(2)

        before_mask = pos_exp <= t_grid
        real_mask = (pos_exp > -1e5) & before_mask

        neg_inf = torch.full_like(pos_exp, -1e6)
        masked_pos = torch.where(real_mask, pos_exp, neg_inf)
        last_event, last_idx = masked_pos.max(dim=1)

        next_idx = (last_idx + 1).clamp(max=positions.size(1) - 1)
        next_event = positions.gather(1, next_idx)

        gap = (next_event - last_event).clamp(min=1.0)

        t_flat = torch.arange(T, device=mel.device, dtype=torch.float32).unsqueeze(0)
        elapsed = (t_flat - last_event).clamp(min=0.0)
        half_life = (gap * 0.03).clamp(min=0.5)
        ramp = torch.exp(-0.693147 * elapsed / half_life)

        ramp = torch.where(last_event > -1e5, ramp, torch.zeros_like(ramp))

        # only embed in past half (frames 0-499)
        ramp[:, cursor_frame:] = 0.0

        # amplitude jitter during training
        if self.training:
            audio_scale = 0.25 + 0.5 * torch.rand(B, 1, 1, device=mel.device)
        else:
            audio_scale = 0.5
        mel = mel * audio_scale
        ramp = ramp * 10.0
        mel = mel + ramp.unsqueeze(1)

        return mel

    def forward(self, mel, event_offsets, event_mask, conditioning,
                future_onsets=None):
        """
        mel: (B, 80, 1000)
        event_offsets: (B, C) past event bin positions relative to cursor
        event_mask: (B, C) bool, True = padding
        conditioning: (B, 3) [mean_density, peak_density, density_std]
        future_onsets: (B, 125) binary ground truth for teacher forcing (training only)
                       If None during training, no onset feedback is applied.
        Returns: onset_probs (B, 125) sigmoid probabilities for future tokens
        """
        cond = self.cond_mlp(conditioning)

        # embed past event ramps into mel
        mel = self._embed_events_in_mel(mel, event_offsets, event_mask)

        # conv stem → 250 tokens
        x = self.conv(mel).transpose(1, 2)
        x = self.conv_norm(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_emb(positions)
        x = self.film_conv(x, cond)

        # onset feedback: add onset embedding to future tokens where
        # the PREVIOUS position had an onset (teacher forcing during training)
        if future_onsets is not None:
            # shift right by 1: token i gets feedback from onset at token i-1
            shifted = torch.zeros_like(future_onsets)
            shifted[:, 1:] = future_onsets[:, :-1]
            # add onset embedding scaled by onset presence
            feedback = shifted.unsqueeze(-1) * self.onset_feedback_emb  # (B, 125, d_model)
            x[:, self.n_past_tokens:, :] = x[:, self.n_past_tokens:, :] + feedback

        # self-attention with causal mask on future tokens
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, src_mask=self.causal_mask)
            x = film(x, cond)

        # onset prediction from future tokens only (125-249)
        future_tokens = x[:, self.n_past_tokens:, :]  # (B, 125, d_model)
        onset_logits = self.onset_head(future_tokens).squeeze(-1)  # (B, 125)

        return torch.sigmoid(onset_logits)


class CrossAttentionFusionLayer(nn.Module):
    """Bidirectional cross-attention: audio attends to gap, gap attends to audio."""

    def __init__(self, d_model, n_heads, dropout=0.1, cond_dim=64):
        super().__init__()
        # audio → gap cross-attention
        self.audio_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.audio_cross_norm = nn.LayerNorm(d_model)
        self.audio_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.audio_film = FiLM(cond_dim, d_model)

        # gap → audio cross-attention
        self.gap_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.gap_cross_norm = nn.LayerNorm(d_model)
        self.gap_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.gap_film = FiLM(cond_dim, d_model)

    def forward(self, audio, gap, gap_mask, cond):
        """
        audio: (B, Na, d_model)
        gap: (B, Ng, d_model)
        gap_mask: (B, Ng) bool - True = padded
        cond: (B, cond_dim)
        Returns: (audio_out, gap_out)
        """
        # NaN guard: if all gap tokens are masked for a sample, unmask the last one
        all_masked = gap_mask.all(dim=1)
        if all_masked.any():
            safe_mask = gap_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = gap_mask

        # zero out padded gap tokens so they don't contribute garbage
        gap_valid_mask = (~safe_mask).unsqueeze(-1).float()  # (B, Ng, 1)
        gap = gap * gap_valid_mask

        # audio cross-attends to gap (pre-norm)
        a_norm = self.audio_cross_norm(audio)
        a_cross, _ = self.audio_cross_attn(
            a_norm, gap, gap, key_padding_mask=safe_mask,
        )
        audio = audio + a_cross
        audio = audio + self.audio_ffn(audio)
        audio = self.audio_film(audio, cond)

        # gap cross-attends to audio (pre-norm, no mask needed for audio)
        g_norm = self.gap_cross_norm(gap)
        g_cross, _ = self.gap_cross_attn(g_norm, audio, audio)
        gap = gap + g_cross * gap_valid_mask
        gap = gap + self.gap_ffn(gap) * gap_valid_mask
        gap = self.gap_film(gap, cond)

        # clamp to prevent rare activation explosion after many training steps
        audio = audio.clamp(-1e4, 1e4)
        gap = gap.clamp(-1e4, 1e4)

        return audio, gap


class DualStreamOnsetDetector(nn.Module):
    """Dual-stream onset detector with late cross-attention fusion (exp 31+).

    Two independent streams process audio and context in parallel, each
    developing strong representations before exchanging information via
    cross-attention in the final layers. This prevents audio tokens from
    drowning out gap tokens in early self-attention.

    Architecture:
      1. AudioEncoder: mel → 250 audio tokens (d_model), 4 self-attn layers
      2. GapEncoder: event gaps + snippets → C gap tokens (d_model), 4 self-attn layers
      3. Cross-attention fusion: N layers of bidirectional cross-attention
      4. Extract cursor at position 125 from audio stream → output head → 501 logits
    """

    def __init__(
        self,
        n_mels=80,
        d_model=384,
        enc_layers=4,
        gap_enc_layers=4,
        cross_attn_layers=2,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
        snippet_frames=10,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # audio stream: AudioEncoder with enc_layers self-attention layers
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )

        # context stream: GapEncoder with gap_enc_layers self-attention layers
        self.gap_encoder = GapEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=gap_enc_layers,
            n_heads=n_heads, max_events=max_events, cond_dim=cond_dim,
            dropout=dropout, snippet_frames=snippet_frames,
        )

        # late fusion: bidirectional cross-attention
        self.cross_attn_fusion = nn.ModuleList([
            CrossAttentionFusionLayer(d_model, n_heads, dropout, cond_dim)
            for _ in range(cross_attn_layers)
        ])

        # output head (same as OnsetDetector)
        self.head_norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        """
        mel: (B, n_mels, 1000)
        event_offsets: (B, C) past event bin positions relative to cursor
        event_mask: (B, C) bool, True = padding
        conditioning: (B, 3) [mean_density, peak_density, density_std]
        Returns: logits (B, 501)
        """
        cond = self.cond_mlp(conditioning)

        # stream 1: encode audio → 250 tokens
        audio = self.audio_encoder(mel, cond)  # (B, 250, d_model)

        # stream 2: encode gaps → C tokens
        gap, gap_mask = self.gap_encoder(
            event_offsets, event_mask, mel, cond
        )  # (B, C, d_model), (B, C)

        # save pre-fusion audio cursor (fine-grained temporal features)
        audio_pre_cursor = audio[:, 125, :]  # (B, d_model)

        # late fusion: bidirectional cross-attention
        for layer in self.cross_attn_fusion:
            audio, gap = layer(audio, gap, gap_mask, cond)

        # extract cursor (center of audio window, position 125)
        # skip connection: add pre-fusion audio to preserve fine-grained features
        cursor = audio[:, 125, :] + audio_pre_cursor  # (B, d_model)

        logits = self.head_proj(self.head_norm(cursor))
        logits = logits + self.head_smooth(logits.unsqueeze(1)).squeeze(1)
        return logits


class InterleavedOnsetDetector(nn.Module):
    """Interleaved dual-stream onset detector (exp 33+).

    Audio and context tokens are processed with alternating self-attention
    and cross-attention layers. Each cycle:
      1. Audio self-attention (audio attends to audio)
      2. Gap self-attention (gap attends to gap)
      3. Bidirectional cross-attention (audio↔gap)

    This weaves context into audio processing at every stage — audio never
    goes more than 1 layer without seeing context, AND it consolidates its
    own fine-grained features between each cross-attention injection.

    Architecture:
      1. Audio conv stem: mel → 250 tokens (d_model)
      2. GapEncoder feature extraction: gaps + snippets → C tokens (d_model)
      3. N interleaved blocks of [audio-self, gap-self, cross-attn]
      4. Extract cursor at position 125 from audio stream → output head
    """

    def __init__(
        self,
        n_mels=80,
        d_model=384,
        n_blocks=4,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
        snippet_frames=10,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        # conditioning MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # audio conv stem (from AudioEncoder, without transformer layers)
        self.audio_conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.GroupNorm(1, d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        self.audio_conv_norm = nn.LayerNorm(d_model)
        self.audio_pos_emb = SinusoidalPosEmb(d_model)
        self.audio_conv_film = FiLM(cond_dim, d_model)

        # gap feature extraction (from GapEncoder, without transformer layers)
        self.gap_snippet_encoder = nn.Sequential(
            nn.Linear(n_mels * snippet_frames, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gap_emb = SinusoidalPosEmb(d_model)
        self.gap_seq_pos_emb = nn.Embedding(max_events + 1, d_model)
        self.snippet_frames = snippet_frames
        self.max_events = max_events

        # interleaved blocks: [audio-self, gap-self, cross-attn] × N
        self.audio_self_layers = nn.ModuleList()
        self.audio_self_film = nn.ModuleList()
        self.gap_self_layers = nn.ModuleList()
        self.gap_self_film = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.audio_self_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                )
            )
            self.audio_self_film.append(FiLM(cond_dim, d_model))

            self.gap_self_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                )
            )
            self.gap_self_film.append(FiLM(cond_dim, d_model))

            self.cross_attn_layers.append(
                CrossAttentionFusionLayer(d_model, n_heads, dropout, cond_dim)
            )

        # output head
        self.head_norm = nn.LayerNorm(d_model)
        self.head_proj = nn.Linear(d_model, n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    def _extract_snippets(self, mel, frame_positions, valid_mask):
        """Extract mel snippets at given positions (same as GapEncoder)."""
        B, n_mels, T = mel.shape
        N = frame_positions.size(1)
        half = self.snippet_frames // 2

        centers = frame_positions.clamp(half, T - half - 1)
        offsets = torch.arange(-half, half, device=mel.device)
        frame_idx = centers.unsqueeze(-1) + offsets
        frame_idx = frame_idx.clamp(0, T - 1)

        flat_idx = frame_idx.reshape(B, -1)
        flat_idx = flat_idx.unsqueeze(1).expand(-1, n_mels, -1)
        snippets = mel.gather(2, flat_idx)
        snippets = snippets.reshape(B, n_mels, N, self.snippet_frames)
        snippets = snippets.permute(0, 2, 1, 3).reshape(B, N, -1)
        snippets = snippets * valid_mask.unsqueeze(-1).float()

        return self.gap_snippet_encoder(snippets)

    def _prepare_gap_tokens(self, event_offsets, event_mask, mel, cond):
        """Build gap token representations (same logic as GapEncoder.forward)."""
        B, C = event_offsets.shape
        event_valid = ~event_mask

        # compute gap sequence
        gap_before = event_offsets[:, 1:] - event_offsets[:, :-1]
        gap_valid = event_valid[:, 1:] & event_valid[:, :-1]
        has_events = event_valid[:, -1]
        time_since_last = (-event_offsets[:, -1]).unsqueeze(1)

        all_gaps = torch.cat([gap_before, time_since_last], dim=1)
        all_gap_valid = torch.cat([gap_valid, has_events.unsqueeze(1)], dim=1)
        all_gap_mask = ~all_gap_valid

        # gap features + snippets
        gap_features = self.gap_emb(all_gaps.abs())

        event_mel_frames = 500 + event_offsets
        snippet_valid_events = event_valid & (event_mel_frames >= 0) & (event_mel_frames < mel.size(2))

        snippet_frames_for_gaps = torch.cat([
            event_mel_frames[:, 1:],
            torch.full((B, 1), 500, device=mel.device, dtype=event_mel_frames.dtype),
        ], dim=1)
        snippet_valid_for_gaps = torch.cat([
            snippet_valid_events[:, 1:],
            has_events.unsqueeze(1),
        ], dim=1)

        snippet_feat = self._extract_snippets(mel, snippet_frames_for_gaps, snippet_valid_for_gaps)
        x = gap_features + snippet_feat

        seq_pos = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.gap_seq_pos_emb(seq_pos)

        # NaN guard
        all_masked = all_gap_mask.all(dim=1)
        if all_masked.any():
            safe_mask = all_gap_mask.clone()
            safe_mask[all_masked, -1] = False
        else:
            safe_mask = all_gap_mask

        return x, safe_mask

    def forward(self, mel, event_offsets, event_mask, conditioning):
        """
        mel: (B, n_mels, 1000)
        event_offsets: (B, C) past event bin positions relative to cursor
        event_mask: (B, C) bool, True = padding
        conditioning: (B, 3) [mean_density, peak_density, density_std]
        Returns: logits (B, 501)
        """
        cond = self.cond_mlp(conditioning)

        # audio stem: conv → 250 tokens
        audio = self.audio_conv(mel).transpose(1, 2)
        audio = self.audio_conv_norm(audio)
        positions = torch.arange(audio.size(1), device=audio.device).unsqueeze(0).expand(audio.size(0), -1)
        audio = audio + self.audio_pos_emb(positions)
        audio = self.audio_conv_film(audio, cond)

        # gap stem: feature extraction → C tokens
        gap, gap_mask = self._prepare_gap_tokens(event_offsets, event_mask, mel, cond)

        # interleaved blocks: [audio-self, gap-self, cross-attn] × N
        for a_self, a_film, g_self, g_film, cross in zip(
            self.audio_self_layers, self.audio_self_film,
            self.gap_self_layers, self.gap_self_film,
            self.cross_attn_layers,
        ):
            # audio self-attention
            audio = a_self(audio)
            audio = a_film(audio, cond)

            # gap self-attention
            gap = g_self(gap, src_key_padding_mask=gap_mask)
            gap = g_film(gap, cond)

            # bidirectional cross-attention
            audio, gap = cross(audio, gap, gap_mask, cond)

        # extract cursor (center of audio window, position 125)
        cursor = audio[:, 125, :]

        logits = self.head_proj(self.head_norm(cursor))
        logits = logits + self.head_smooth(logits.unsqueeze(1)).squeeze(1)
        return logits


class AdditiveOnsetDetector(nn.Module):
    """Two-path additive onset detector (exp 24).

    Audio path: audio self-attn + event cross-attn → 501 logits
    Context path: gap-based with own encoders → 501 logits
    Final: audio_logits + context_logits (soft influence)
    """

    def __init__(
        self,
        n_mels=80,
        d_model=384,
        d_event=128,
        d_ctx=192,
        enc_layers=4,
        enc_event_layers=2,
        audio_path_layers=2,
        context_gap_layers=2,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
        snippet_frames=10,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            d_model=d_model, d_event=d_event, n_layers=enc_event_layers,
            n_heads=4, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )
        self.audio_path = AudioPath(
            d_model=d_model, n_layers=audio_path_layers, n_heads=n_heads,
            n_classes=n_classes, cond_dim=cond_dim, dropout=dropout,
        )
        self.context_path = ContextPath(
            n_mels=n_mels, d_ctx=d_ctx, n_gap_layers=context_gap_layers,
            n_heads=6, n_classes=n_classes, max_events=max_events,
            cond_dim=cond_dim, dropout=dropout, snippet_frames=snippet_frames,
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        cond = self.cond_mlp(conditioning)
        audio_tokens = self.audio_encoder(mel, cond)
        event_tokens = self.event_encoder(event_offsets, event_mask, cond)
        audio_logits = self.audio_path(audio_tokens, event_tokens, event_mask, cond)
        context_logits = self.context_path(
            event_offsets, event_mask, mel, cond.detach()
        )
        logits = audio_logits + context_logits
        return logits, audio_logits, context_logits


class RerankerOnsetDetector(nn.Module):
    """Reranker two-path detector (exp 19-23) with top-K selection.

    For loading legacy reranker checkpoints. Uses RerankerContextPath.
    """

    def __init__(self, n_mels=80, d_model=384, d_event=128, d_ctx=192,
                 enc_layers=4, enc_event_layers=2, audio_path_layers=2,
                 context_gap_layers=2, context_select_layers=2, n_heads=8,
                 n_classes=501, max_events=128, dropout=0.1, cond_dim=64,
                 top_k=20, snippet_frames=10):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.top_k = top_k

        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            d_model=d_model, d_event=d_event, n_layers=enc_event_layers,
            n_heads=4, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )
        self.audio_path = AudioPath(
            d_model=d_model, n_layers=audio_path_layers, n_heads=n_heads,
            n_classes=n_classes, cond_dim=cond_dim, dropout=dropout,
        )
        self.context_path = RerankerContextPath(
            n_mels=n_mels, d_ctx=d_ctx, n_gap_layers=context_gap_layers,
            n_select_layers=context_select_layers,
            n_heads=6, K=top_k, max_events=max_events,
            cond_dim=cond_dim, dropout=dropout, snippet_frames=snippet_frames,
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        cond = self.cond_mlp(conditioning)
        audio_tokens = self.audio_encoder(mel, cond)
        event_tokens = self.event_encoder(event_offsets, event_mask, cond)
        audio_logits = self.audio_path(audio_tokens, event_tokens, event_mask, cond)
        selection_logits, top_k_indices = self.context_path(
            event_offsets, event_mask, mel, cond.detach(), audio_logits
        )
        logits = torch.full_like(audio_logits, -100.0)
        logits.scatter_(1, top_k_indices, selection_logits)
        return logits, audio_logits, selection_logits, top_k_indices


class LegacyOnsetDetector(nn.Module):
    """Legacy two-path detector (exp 11-16) with additive logits.

    Returns the same 4-tuple as OnsetDetector for interface compatibility,
    but selection_logits and top_k_indices are synthesized from the combined output.
    """

    def __init__(self, n_mels=80, d_model=384, d_event=128, enc_layers=4,
                 enc_event_layers=2, audio_path_layers=2, context_path_layers=3,
                 n_heads=8, n_classes=501, max_events=128, dropout=0.1, cond_dim=64):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            d_model=d_model, d_event=d_event, n_layers=enc_event_layers,
            n_heads=4, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )
        self.audio_path = AudioPath(
            d_model=d_model, n_layers=audio_path_layers, n_heads=n_heads,
            n_classes=n_classes, cond_dim=cond_dim, dropout=dropout,
        )
        self.context_path = LegacyContextPath(
            d_model=d_model, n_layers=context_path_layers, n_heads=n_heads,
            n_classes=n_classes, max_events=max_events, cond_dim=cond_dim,
            dropout=dropout,
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        cond = self.cond_mlp(conditioning)
        audio_tokens = self.audio_encoder(mel, cond)
        event_tokens = self.event_encoder(event_offsets, event_mask, cond)
        audio_logits = self.audio_path(audio_tokens, event_tokens, event_mask, cond)
        context_logits = self.context_path(event_tokens, audio_tokens, event_mask, cond)
        logits = audio_logits + context_logits

        # Synthesize 4-tuple interface: use combined logits top-20 as fake selection
        top_k_scores, top_k_indices = logits.topk(20, dim=-1)
        return logits, audio_logits, top_k_scores, top_k_indices


class Exp17OnsetDetector(nn.Module):
    """Exp 17 detector: top-K reranking with shared encoder gradients (no stop-gradient).

    Used for loading exp 17 checkpoints.
    """

    def __init__(self, n_mels=80, d_model=384, d_event=128, enc_layers=4,
                 enc_event_layers=2, audio_path_layers=2, context_path_layers=3,
                 n_heads=8, n_classes=501, max_events=128, dropout=0.1,
                 cond_dim=64, top_k=20):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.top_k = top_k

        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            d_model=d_model, d_event=d_event, n_layers=enc_event_layers,
            n_heads=4, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )
        self.audio_path = AudioPath(
            d_model=d_model, n_layers=audio_path_layers, n_heads=n_heads,
            n_classes=n_classes, cond_dim=cond_dim, dropout=dropout,
        )
        self.context_path = Exp17ContextPath(
            d_model=d_model, n_layers=context_path_layers, n_heads=n_heads,
            K=top_k, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        cond = self.cond_mlp(conditioning)
        audio_tokens = self.audio_encoder(mel, cond)
        event_tokens = self.event_encoder(event_offsets, event_mask, cond)
        audio_logits = self.audio_path(audio_tokens, event_tokens, event_mask, cond)
        selection_logits, top_k_indices = self.context_path(
            event_tokens, audio_tokens, event_mask, cond, audio_logits
        )
        logits = torch.full_like(audio_logits, -100.0)
        logits.scatter_(1, top_k_indices, selection_logits)
        return logits, audio_logits, selection_logits, top_k_indices


class Exp18OnsetDetector(nn.Module):
    """Exp 18 detector: two-stage context with shared encoder features (stop-gradient).

    Used for loading exp 18 checkpoints.
    """

    def __init__(self, n_mels=80, d_model=384, d_event=128, enc_layers=4,
                 enc_event_layers=2, audio_path_layers=2,
                 context_event_layers=2, context_select_layers=2,
                 n_heads=8, n_classes=501, max_events=128, dropout=0.1,
                 cond_dim=64, top_k=20):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model
        self.top_k = top_k

        self.cond_mlp = nn.Sequential(
            nn.Linear(3, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim),
        )
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            d_model=d_model, d_event=d_event, n_layers=enc_event_layers,
            n_heads=4, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )
        self.audio_path = AudioPath(
            d_model=d_model, n_layers=audio_path_layers, n_heads=n_heads,
            n_classes=n_classes, cond_dim=cond_dim, dropout=dropout,
        )
        self.context_path = Exp18ContextPath(
            d_model=d_model, n_event_layers=context_event_layers,
            n_select_layers=context_select_layers, n_heads=n_heads,
            K=top_k, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )

    def forward(self, mel, event_offsets, event_mask, conditioning):
        cond = self.cond_mlp(conditioning)
        audio_tokens = self.audio_encoder(mel, cond)
        event_tokens = self.event_encoder(event_offsets, event_mask, cond)
        audio_logits = self.audio_path(audio_tokens, event_tokens, event_mask, cond)
        selection_logits, top_k_indices = self.context_path(
            event_tokens.detach(), audio_tokens.detach(), event_mask,
            cond.detach(), audio_logits
        )
        logits = torch.full_like(audio_logits, -100.0)
        logits.scatter_(1, top_k_indices, selection_logits)
        return logits, audio_logits, selection_logits, top_k_indices
