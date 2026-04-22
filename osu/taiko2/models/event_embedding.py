"""Event-embedding onset detector — taiko1's exp 45 architecture ported
onto the taiko2 `Model` ABC.

The model injects learned embeddings of past onsets directly into the
audio-token stream at those events' mel-frame positions, then runs a
pre-norm transformer with per-layer FiLM conditioning. The cursor
token (center of the audio window) is the classification head's input;
output is a softmax over `b_pred + 1` classes — bin-offset 0..b_pred-1
or STOP at index `b_pred`.

Window geometry — what each bin range means:
  - ``a_bins``  = mel frames of audio BEFORE the cursor.
  - ``b_bins``  = mel frames of audio AFTER the cursor.
  - ``b_pred``  = prediction range forward of the cursor; must be
                  ≤ b_bins. The STOP class covers anything at or past
                  offset ``b_pred`` (including the "no onset within
                  prediction range" case).

Audio token index ↔ mel frame:
  mel frames 0..A+B-1 pass through a stride-4 conv stem → tokens
  0..(A+B)/4 - 1. Event offset `e` (negative/zero = past) maps to
  token `(a_bins + e) // 4`. The cursor sits at token `a_bins // 4`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..domain.model import Model, ModelConfig, ModelInput, ModelOutput, ModelTarget
from .common import AudioConvStem, FiLM, SinusoidalPosEmb


# ─────────────────────────── config ───────────────────────────────────

@dataclass(frozen=True, slots=True)
class EventEmbeddingConfig(ModelConfig):
    """Hyperparameters for `EventEmbeddingDetector`.

    `b_pred` may be smaller than `b_bins` — the model sees a larger
    audio window than it predicts into. This lets the future-audio
    context inform near-term predictions. The STOP class is always
    the last (index `b_pred`).
    """
    # Mel / audio encoder
    n_mels: int = 80
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1

    # Event embeddings
    c_events: int = 128
    gap_ratios: bool = True         # exp 45 default

    # Conditioning
    cond_dim: int = 64

    # Window geometry
    a_bins: int = 500
    b_bins: int = 500
    b_pred: int = 500

    def __post_init__(self):
        if self.b_pred > self.b_bins:
            raise ValueError(
                f"b_pred ({self.b_pred}) must be <= b_bins ({self.b_bins})"
            )
        if (self.a_bins + self.b_bins) % 4 != 0:
            raise ValueError(
                f"a_bins + b_bins must be divisible by 4 (conv stride); "
                f"got a_bins={self.a_bins}, b_bins={self.b_bins}"
            )
        if self.a_bins % 4 != 0:
            raise ValueError(
                f"a_bins must be divisible by 4 (cursor token alignment); "
                f"got a_bins={self.a_bins}"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )

    @property
    def n_audio_tokens(self) -> int:
        return (self.a_bins + self.b_bins) // 4

    @property
    def cursor_token(self) -> int:
        return self.a_bins // 4

    @property
    def n_classes(self) -> int:
        """`b_pred` bin-offset classes + 1 STOP."""
        return self.b_pred + 1


# ─────────────────────────── IO types ─────────────────────────────────

@dataclass(frozen=True, slots=True)
class EventEmbeddingInput(ModelInput):
    mel: torch.Tensor               # (B, n_mels, a_bins + b_bins)    float32
    event_offsets: torch.Tensor     # (B, c_events)                   int64
    event_mask: torch.Tensor        # (B, c_events) True = padding    bool
    conditioning: torch.Tensor      # (B, 3)        [mean, peak, std] float32


@dataclass(frozen=True, slots=True)
class EventEmbeddingOutput(ModelOutput):
    logits: torch.Tensor            # (B, n_classes)  float32


@dataclass(frozen=True, slots=True)
class EventEmbeddingTarget(ModelTarget):
    """Training target: a single bin index per sample.

    Class `n_classes - 1` is STOP (no onset within `b_pred`). The loss
    derives `is_stop = (target_bin == n_classes - 1)` rather than
    carrying a redundant bool.

    `all_future_bins` / `all_future_mask` are OPTIONAL metric-side
    state: the full list of cursor-relative offsets for any future
    onset falling inside [0, b_pred) (not just the next one). Required
    for I-variant metrics (IHIT / IGOOD / IBAD) that count a
    prediction correct if it matches ANY upcoming onset. The loss
    ignores these fields entirely.
    """
    target_bin: torch.Tensor                # (B,) int64 in [0, n_classes-1]
    all_future_bins: torch.Tensor | None = None  # (B, K) int64 — cursor offsets
    all_future_mask: torch.Tensor | None = None  # (B, K) bool, True = padded


# ─────────────────────────── model ────────────────────────────────────

class EventEmbeddingDetector(
    Model[EventEmbeddingConfig, EventEmbeddingInput, EventEmbeddingOutput]
):
    """Exp 45's event-embedding architecture.

    Architecture summary (see `experiments/001-exp45-port/ARCHITECTURE.md`
    for the full spec):

      1. Conditioning MLP:        (B, 3)  → (B, cond_dim).
      2. Conv stem:               (B, n_mels, T)  → (B, T/4, d_model).
      3. Audio pos-emb + FiLM.
      4. Build per-event feature vectors (presence + gap_before +
         gap_after [+ gap_ratio_before + gap_ratio_after]); project
         into (B, c_events, d_model).
      5. Scatter-add event embeddings onto audio tokens at their
         corresponding token positions (past audio only, tokens
         0..cursor_token-1).
      6. N transformer encoder layers with per-layer FiLM.
      7. Head: cursor-token layer-norm → linear → conv-smooth → logits.
    """

    def __init__(self, config: EventEmbeddingConfig):
        super().__init__(config)
        c = config
        d = c.d_model

        # 1. Conditioning MLP — (B, 3) → (B, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(3, c.cond_dim),
            nn.GELU(),
            nn.Linear(c.cond_dim, c.cond_dim),
        )

        # 2. Conv stem — mel → audio tokens
        self.conv_stem = AudioConvStem(c.n_mels, d)
        self.audio_pos_emb = SinusoidalPosEmb(d)
        self.film_conv = FiLM(c.cond_dim, d)

        # 3. Event embedding parts
        # Presence: a single learned vector broadcast to every slot.
        self.event_presence_emb = nn.Parameter(torch.randn(1, d) * 0.02)
        self.gap_before_emb = SinusoidalPosEmb(d)
        self.gap_after_emb = SinusoidalPosEmb(d)
        if c.gap_ratios:
            self.gap_ratio_before_emb = SinusoidalPosEmb(d)
            self.gap_ratio_after_emb = SinusoidalPosEmb(d)
            n_emb_inputs = 5
        else:
            n_emb_inputs = 3
        self.event_proj = nn.Sequential(
            nn.Linear(d * n_emb_inputs, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # 4. Transformer trunk — pre-norm, FiLM after each layer.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=c.n_heads,
                dim_feedforward=d * 4,
                dropout=c.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(c.n_layers)
        ])
        self.film_layers = nn.ModuleList(
            [FiLM(c.cond_dim, d) for _ in range(c.n_layers)]
        )

        # 5. Output head
        self.head_norm = nn.LayerNorm(d)
        self.head_proj = nn.Linear(d, c.n_classes)
        self.head_smooth = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2),
        )

    # ── forward / predict ────────────────────────────────────────────

    def forward(
        self,
        mel: torch.Tensor,
        event_offsets: torch.Tensor,
        event_mask: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Raw-tensor forward.

        Shapes:
            mel:            (B, n_mels, a_bins + b_bins)
            event_offsets:  (B, c_events)          int64
            event_mask:     (B, c_events) True = padding
            conditioning:   (B, 3)

        Returns:
            logits:         (B, n_classes)
        """
        c = self.config
        B = mel.size(0)
        d = c.d_model

        # 1. Conditioning vector.
        cond = self.cond_mlp(conditioning)

        # 2. Conv stem + 3. audio pos-emb + FiLM.
        x = self.conv_stem(mel)                              # (B, T/4, d)
        audio_positions = torch.arange(
            x.size(1), device=x.device
        ).unsqueeze(0).expand(B, -1)
        x = x + self.audio_pos_emb(audio_positions)
        x = self.film_conv(x, cond)

        # 4. Build event embeddings.
        event_embs, token_pos, in_window = self._build_event_embeddings(
            event_offsets, event_mask
        )

        # 5. Scatter-add event embeddings to their audio tokens.
        # Per-batch loop kept for faithfulness to taiko1; vectorizing
        # is a follow-up if it shows up in profiles.
        for b in range(B):
            valid_idx = in_window[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            tpos = token_pos[b, valid_idx]
            embs = event_embs[b, valid_idx]
            x[b].scatter_add_(
                0, tpos.unsqueeze(-1).expand(-1, d), embs,
            )

        # 6. Transformer trunk with per-layer FiLM.
        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x)
            x = film(x, cond)

        # 7. Output head on the cursor token.
        cursor = x[:, c.cursor_token, :]                      # (B, d)
        logits = self.head_proj(self.head_norm(cursor))       # (B, n_classes)
        logits = logits + self.head_smooth(
            logits.unsqueeze(1)
        ).squeeze(1)
        return logits

    def predict(self, x: EventEmbeddingInput) -> EventEmbeddingOutput:
        logits = self.forward(
            x.mel, x.event_offsets, x.event_mask, x.conditioning,
        )
        return EventEmbeddingOutput(logits=logits)

    # ── event-embedding construction ─────────────────────────────────

    def _build_event_embeddings(
        self,
        event_offsets: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build per-event feature vectors and their mel-token positions.

        Feature vectors per event (concatenated on `d_model` then
        linearly projected back to `d_model`):

          - **presence** — learned parameter (1, d) broadcast to every slot.
          - **gap_before[i]** — `|offsets[i] - offsets[i-1]|` clamped
            to ≥1; for i=0 a placeholder of 50 is used.
          - **gap_after[i]**  — `|offsets[i+1] - offsets[i]|` clamped
            to ≥1. For the LAST valid event in each row, `gap_after`
            would leak the target offset; we overwrite it with that
            event's `gap_before` as a structure-preserving proxy.
          - (gap_ratios only) **gap_ratio_before[i]** — `gap_before[i-1]
            / gap_before[i]`, clamped to [0.1, 10.0] then scaled by 50.
          - (gap_ratios only) **gap_ratio_after[i]**  — `gap_after[i+1]
            / gap_after[i]`, same clamp + scale.

        Mel-frame mapping:
            mel_frame[i] = a_bins + event_offsets[i]    # offsets are ≤ 0
            token_pos[i] = mel_frame[i] // 4            # conv stride 4
            in_window[i] = valid[i] AND 0 ≤ token_pos[i] < cursor_token

        `in_window` excludes events that map at or beyond the cursor
        token — the model is only supposed to inject *past* events.

        Returns:
          event_embs:  (B, c_events, d_model)  — pre-scatter features.
          token_pos:   (B, c_events)           — clamped to valid range.
          in_window:   (B, c_events) bool      — True = inject this event.
        """
        c = self.config
        B, C = event_offsets.shape
        valid = ~event_mask                                    # (B, C)
        offsets = event_offsets.float()

        # gap_before: pairwise diff; placeholder 50 at index 0.
        gap_before = torch.zeros(B, C, device=offsets.device)
        gap_before[:, 1:] = offsets[:, 1:] - offsets[:, :-1]
        gap_before[:, 0] = 50.0
        gap_before = gap_before.abs().clamp(min=1.0)

        # gap_after: same deltas shifted one slot left.
        gap_after = torch.zeros(B, C, device=offsets.device)
        gap_after[:, :-1] = offsets[:, 1:] - offsets[:, :-1]
        gap_after = gap_after.abs().clamp(min=1.0)

        # Mask the last valid event's `gap_after` — it would span the
        # target offset otherwise. Use `gap_before` at that position
        # as a same-structure proxy.
        for b in range(B):
            valid_indices = valid[b].nonzero(as_tuple=True)[0]
            if valid_indices.numel() > 0:
                last_idx = int(valid_indices[-1].item())
                gap_after[b, last_idx] = gap_before[b, last_idx]

        parts = [
            self.event_presence_emb.expand(B, C, -1),          # (B, C, d)
            self.gap_before_emb(gap_before),                   # (B, C, d)
            self.gap_after_emb(gap_after),                     # (B, C, d)
        ]
        if c.gap_ratios:
            ratio_before = torch.ones(B, C, device=offsets.device)
            ratio_before[:, 1:] = gap_before[:, :-1] / gap_before[:, 1:]
            ratio_before = (ratio_before.clamp(0.1, 10.0) * 50.0)

            ratio_after = torch.ones(B, C, device=offsets.device)
            ratio_after[:, :-1] = gap_after[:, 1:] / gap_after[:, :-1]
            ratio_after = (ratio_after.clamp(0.1, 10.0) * 50.0)

            parts.append(self.gap_ratio_before_emb(ratio_before))
            parts.append(self.gap_ratio_after_emb(ratio_after))

        combined = torch.cat(parts, dim=-1)                    # (B, C, n*d)
        event_embs = self.event_proj(combined)                 # (B, C, d)

        # Token positions (see docstring for math).
        mel_frames = c.a_bins + event_offsets                  # (B, C) int64
        token_pos = mel_frames // 4
        in_window = valid & (token_pos >= 0) & (token_pos < c.cursor_token)
        token_pos_clamped = token_pos.clamp(0, c.n_audio_tokens - 1)
        return event_embs, token_pos_clamped, in_window
