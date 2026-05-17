"""``FramewiseDetector`` -- single-shot framewise onset detector (#017).

Same trunk as ``EventEmbeddingDetector``. Output head is a Conv1D-on-
bin-axis stack conditioned on future-half audio features + cursor token
+ sinusoidal positional embedding. Produces a ``(B, n_bins)`` logit map
(one logit per future-time bin); ``sigmoid(logits)`` is the confidence
map consumed by the decoder and diagnostics.

No diffusion machinery -- the confidence map is produced in a single
forward pass. Designed so that a future diffusion variant can produce
the same ``confidence_map`` shape via iterative refinement and share
all downstream consumers (decoder, metrics, diagnostics).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion.denoisers import sinusoidal_pos_embed
from ..domain.model import ModelOutput
from .event_embedding import (
    EventEmbeddingConfig,
    EventEmbeddingDetector,
    EventEmbeddingInput,
)


# ─────────────────────────── Output type ──────────────────────────────


@dataclass(frozen=True, slots=True)
class FramewiseDetectorOutput(ModelOutput):
    """Single-shot framewise detector output.

    ``logits`` is the raw pre-sigmoid ``(B, n_bins)`` map -- used by
    the BCE loss for numerically stable backprop.

    ``confidence_map`` is ``sigmoid(logits).detach()`` in ``[0, 1]``
    -- consumed by the decoder, metrics, and diagnostics. Detached so
    downstream code never accidentally back-props through it.
    """
    logits: torch.Tensor                              # (B, n_bins) raw
    confidence_map: torch.Tensor                      # (B, n_bins) [0, 1]
    cursor_token: torch.Tensor                        # (B, d_model)
    audio_features: torch.Tensor | None = None        # (B, T_audio, d_model)


# ─────────────────────────── Config ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class FramewiseDetectorConfig(EventEmbeddingConfig):
    """``EventEmbeddingConfig`` + framewise Conv1D head sub-config.

    No STOP class -- ``b_pred == n_bins`` directly. The head produces
    one logit per bin; the decoder thresholds + NMS to emit onsets.
    """
    head_channels: int = 256
    head_kernels: tuple[int, ...] = (31, 15, 15)
    head_pos_embed_dim: int = 32
    head_cursor_proj_dim: int = 32
    head_dropout: float = 0.1

    def __post_init__(self) -> None:
        EventEmbeddingConfig.__post_init__(self)
        if self.b_pred != self.b_bins:
            raise ValueError(
                f"b_pred ({self.b_pred}) must == b_bins ({self.b_bins}) "
                f"for framewise mode (no STOP class)"
            )
        if self.head_channels < 1:
            raise ValueError(
                f"head_channels must be >= 1 (got {self.head_channels})"
            )
        if len(self.head_kernels) < 1:
            raise ValueError("head_kernels must be non-empty")
        if any(k % 2 == 0 for k in self.head_kernels):
            raise ValueError(
                f"head_kernels must all be odd (got {self.head_kernels})"
            )
        if self.head_channels % 8 != 0:
            raise ValueError(
                f"head_channels must be a multiple of 8 for GroupNorm "
                f"(got {self.head_channels})"
            )
        if self.head_pos_embed_dim < 2 or self.head_pos_embed_dim % 2 != 0:
            raise ValueError(
                f"head_pos_embed_dim must be even and >= 2 "
                f"(got {self.head_pos_embed_dim})"
            )
        if self.head_cursor_proj_dim < 1:
            raise ValueError(
                f"head_cursor_proj_dim must be >= 1 "
                f"(got {self.head_cursor_proj_dim})"
            )


# ─────────────────────────── Detector ─────────────────────────────────


class FramewiseDetector(EventEmbeddingDetector):
    """Single-shot framewise onset detector.

    Trunk: identical to ``EventEmbeddingDetector`` (inherited).
    Head: Conv1D-on-bin-axis stack with per-bin audio features,
    sinusoidal positional embedding, and broadcast cursor projection.

    Parent's standard head (``head_norm``, ``head_proj``,
    ``head_smooth``) is unused; left in place for parameter-dict
    compatibility.
    """

    config: FramewiseDetectorConfig

    def __init__(self, config: FramewiseDetectorConfig):
        super().__init__(config)
        c = config
        n_bins = c.b_pred

        self.cursor_head_proj = nn.Linear(c.d_model, c.head_cursor_proj_dim)

        pos = sinusoidal_pos_embed(n_bins, c.head_pos_embed_dim)
        self.register_buffer("_head_pos_embed", pos)

        in_ch = (
            c.head_pos_embed_dim
            + c.d_model
            + c.head_cursor_proj_dim
        )

        convs: list[nn.Module] = []
        cur_in = in_ch
        for k in c.head_kernels:
            convs.append(
                nn.Conv1d(cur_in, c.head_channels, kernel_size=k, padding=k // 2)
            )
            cur_in = c.head_channels
        self.head_convs = nn.ModuleList(convs)
        self.head_norms = nn.ModuleList([
            nn.GroupNorm(8, c.head_channels) for _ in c.head_kernels
        ])
        self.head_act = nn.SiLU()
        self.head_drop = nn.Dropout(c.head_dropout)
        self.head_out = nn.Conv1d(c.head_channels, 1, kernel_size=1)

    # ── trunk forward returning both cursor + full audio tokens ──────

    def _trunk_forward_full(
        self, x: EventEmbeddingInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk and return ``(cursor_token, all_audio_tokens)``.

        ``all_audio_tokens`` is ``(B, n_tokens, d_model)`` -- the full
        sequence after the transformer layers. The future-half audio
        features are ``all_audio_tokens[:, cursor_token_idx:, :]``.
        """
        c = self.config
        B = x.mel.size(0)
        d = c.d_model

        cond = self.cond_mlp(x.conditioning)
        h = self.conv_stem(x.mel)
        audio_positions = torch.arange(
            h.size(1), device=h.device,
        ).unsqueeze(0).expand(B, -1)
        h = h + self.audio_pos_emb(audio_positions)
        h = self.film_conv(h, cond)

        event_embs, token_pos, in_window = self._build_event_embeddings(
            x.event_offsets, x.event_mask,
        )
        for b in range(B):
            valid_idx = in_window[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            tpos = token_pos[b, valid_idx]
            embs = event_embs[b, valid_idx]
            h[b].scatter_add_(
                0, tpos.unsqueeze(-1).expand(-1, d), embs,
            )

        for layer, film in zip(self.layers, self.film_layers):
            h = layer(h)
            h = film(h, cond)

        cursor_tok = h[:, c.cursor_token, :]
        return cursor_tok, h

    # ── framewise head ───────────────────────────────────────────────

    def _framewise_head(
        self,
        cursor_token: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Conv1D head: ``(B, d_model)`` cursor + ``(B, T_audio, d_model)``
        audio features -> ``(B, n_bins)`` logits."""
        c = self.config
        B = cursor_token.size(0)
        n_bins = c.b_pred

        per_bin: list[torch.Tensor] = []

        # Positional embed: (n_bins, pos_dim) -> (B, pos_dim, n_bins).
        pos = self._head_pos_embed.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        per_bin.append(pos)

        # Audio features: (B, T_audio, d_model) -> (B, d_model, n_bins).
        audio = audio_features.transpose(1, 2)
        audio = F.interpolate(audio, size=n_bins, mode="linear", align_corners=False)
        per_bin.append(audio)

        # Cursor broadcast: (B, cursor_proj_dim, n_bins).
        cursor_emb = self.cursor_head_proj(cursor_token)
        cur_brd = cursor_emb.unsqueeze(-1).expand(-1, -1, n_bins)
        per_bin.append(cur_brd)

        h = torch.cat(per_bin, dim=1)

        for conv, norm in zip(self.head_convs, self.head_norms):
            h = conv(h)
            h = norm(h)
            h = self.head_act(h)
            h = self.head_drop(h)

        return self.head_out(h).squeeze(1)

    # ── predict ──────────────────────────────────────────────────────

    def predict(self, x: EventEmbeddingInput) -> FramewiseDetectorOutput:
        cursor_token, all_tokens = self._trunk_forward_full(x)
        c = self.config
        audio_features = all_tokens[:, c.cursor_token:, :]
        logits = self._framewise_head(cursor_token, audio_features)
        return FramewiseDetectorOutput(
            logits=logits,
            confidence_map=torch.sigmoid(logits).detach(),
            cursor_token=cursor_token,
            audio_features=audio_features,
        )
