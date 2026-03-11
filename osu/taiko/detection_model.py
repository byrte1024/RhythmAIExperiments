"""Onset detection model: two-path architecture with top-K reranking.

Audio Path (proposer): audio-primary with light event cross-attention → 501 logits
Context Path (selector): K-way classifier over audio's top-K candidates
Final: context picks class 0-K from audio's ranked proposals

Gradient isolation (exp 18+): stop-gradient between shared encoders and context path.
Audio loss trains: audio_path + audio_encoder + event_encoder + cond_mlp
Selection loss trains: context_path ONLY (encoder outputs detached)
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

        query_out = seq[:, -1, :]  # (B, d_model) — consensus from all events

        # ── 5. Score candidates via scaled dot product → K-way logits ──
        q = self.q_proj(query_out)  # (B, d_score)
        k = self.k_proj(candidate_feat)  # (B, K, d_score)
        selection_logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) * self.score_scale  # (B, K)

        return selection_logits, top_k_indices


class ContextPath(nn.Module):
    """Gap-based K-way classifier with own encoders (exp 19+).

    Context operates entirely in "gap space" — inter-onset intervals rather than
    absolute positions. Each event is characterized by (gap_before, local_audio_snippet).
    Each candidate is characterized by (proposed_next_gap, local_audio_snippet, audio_score).

    Own encoders: gap encoder + snippet encoder, both trained by selection loss.
    No shared encoder dependencies — fully gradient-isolated by design.

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
        # STOP has no meaningful audio — learned embedding
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
        self.score_proj = nn.Sequential(nn.Linear(2, d_ctx), nn.GELU())
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
        valid_mask: (B, N) bool — True = valid position, False = skip (gets zeros)
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
        event_offsets: (B, C) int — bin positions relative to cursor (negative/zero)
        event_mask: (B, C) bool — True = padded
        mel: (B, n_mels, 1000) — raw mel spectrogram
        cond: (B, cond_dim) — conditioning (detached by caller)
        audio_logits: (B, n_classes) — from AudioPath
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
        has_events = event_valid[:, -1]  # (B,) — last position is valid if any events
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
        cand_mel_frames = 500 + top_k_indices  # (B, K) — candidates are future bins
        is_stop = (top_k_indices == stop_idx)  # (B, K)
        cand_snippet_valid = ~is_stop & (cand_mel_frames < mel.size(2))

        cand_snippet_feat = self._extract_snippets(
            mel, cand_mel_frames.clamp(max=mel.size(2) - 1), cand_snippet_valid
        )  # (B, K, d_ctx)
        # STOP candidates get learned embedding instead
        if is_stop.any():
            cand_snippet_feat = cand_snippet_feat.clone()
            cand_snippet_feat[is_stop] = self.stop_snippet_emb

        # Audio confidence + rank
        ranks = torch.arange(1, self.K + 1, device=mel.device, dtype=torch.float32)
        ranks = ranks.unsqueeze(0).expand(B, -1) / self.K
        score_feat = self.score_proj(
            torch.stack([top_k_scores, ranks], dim=-1)
        )  # (B, K, d_ctx)

        candidate_feat = self.candidate_combine(
            torch.cat([cand_gap_emb, cand_snippet_feat, score_feat], dim=-1)
        )  # (B, K, d_ctx)

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


class OnsetDetector(nn.Module):
    """Two-path onset detector: audio proposes, context selects from top-K.

    Audio path: audio self-attn + event cross-attn → 501 candidate logits
    Context path: gap-based reranking with own encoders (exp 19+)
    Final: context's selection mapped back to 501-way logits
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
        context_select_layers=2,
        n_heads=8,
        n_classes=501,
        max_events=128,
        dropout=0.1,
        cond_dim=64,
        top_k=20,
        snippet_frames=10,
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

        # shared encoders (used by audio path only)
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels, d_model=d_model, n_layers=enc_layers,
            n_heads=n_heads, cond_dim=cond_dim, dropout=dropout,
        )
        self.event_encoder = EventEncoder(
            d_model=d_model, d_event=d_event, n_layers=enc_event_layers,
            n_heads=4, max_events=max_events, cond_dim=cond_dim, dropout=dropout,
        )

        # audio path (proposer)
        self.audio_path = AudioPath(
            d_model=d_model, n_layers=audio_path_layers, n_heads=n_heads,
            n_classes=n_classes, cond_dim=cond_dim, dropout=dropout,
        )
        # context path (gap-based selector with own encoders)
        self.context_path = ContextPath(
            n_mels=n_mels, d_ctx=d_ctx, n_gap_layers=context_gap_layers,
            n_select_layers=context_select_layers,
            n_heads=6, K=top_k, max_events=max_events,
            cond_dim=cond_dim, dropout=dropout, snippet_frames=snippet_frames,
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

        # shared encoding (for audio path)
        audio_tokens = self.audio_encoder(mel, cond)
        event_tokens = self.event_encoder(event_offsets, event_mask, cond)

        # audio proposes (gradients flow through shared encoders)
        audio_logits = self.audio_path(audio_tokens, event_tokens, event_mask, cond)

        # context selects from audio's top-K using own encoders
        # Takes raw inputs (event_offsets, mel) — no shared encoder outputs needed
        # cond detached to protect cond_mlp from selection loss
        selection_logits, top_k_indices = self.context_path(
            event_offsets, event_mask, mel,
            cond.detach(), audio_logits
        )

        # scatter selection back to 501-way for compatibility with metrics
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
