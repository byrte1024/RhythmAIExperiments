"""Ratio-decomposed onset detector (taiko1 exp 67 design on taiko2).

Inherits the full EventEmbeddingDetector backbone; replaces the output
head with 3 decomposed heads:

  1. **Divisor** — the dominant rhythmic gap ("what's the beat?").
  2. **Offset** — cursor distance from last event (0 normally, >0 after
     STOP hops).
  3. **Ratio** — what multiple of the divisor is the next onset? Log-
     spaced 0.125×–8.0× in R bins + STOP.

The ratio head receives soft expectations from divisor + offset as
additive embeddings, so it can condition its prediction on "what tempo
am I in?" and "where am I relative to the last event?". Conv1d
smoothing on the ratio logits prevents the collapse-to-few-values
pathology taiko1 exp 67 observed.

Final position at inference: ``round(divisor × ratio_value - offset)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .event_embedding import (
    EventEmbeddingConfig,
    EventEmbeddingDetector,
    EventEmbeddingInput,
    EventEmbeddingOutput,
)


@dataclass(frozen=True, slots=True)
class RatioDetectorConfig(EventEmbeddingConfig):
    """Config for the ratio-decomposed detector.

    Inherits all EventEmbeddingConfig fields (backbone geometry).
    The parent's ``n_classes`` / ``n_output_dims`` are NOT used — the
    ratio detector's output width is ``divisor_bins + offset_bins +
    ratio_bins + 1``.
    """
    ratio_bins: int = 255         # log-spaced 0.125×–8.0×
    divisor_bins: int = 500       # dominant gap size
    offset_bins: int = 100        # cursor-to-last-event distance
    # Ratio head Conv1d smoothing. Prevents collapse to few values
    # but can over-spread if too wide. k=5/ch=8 was #010's default.
    ratio_smooth_kernel: int = 5
    ratio_smooth_channels: int = 8
    # Stop-gradient on the divisor/offset soft expectations before
    # they enter the ratio head. True (default, taiko1 exp 67 design)
    # means div/off heads only train via their own aux losses. False
    # lets the ratio loss gradient flow back through the soft
    # expectations and shape div/off predictions too.
    aux_stop_gradient: bool = True


def build_ratio_bin_centers(ratio_bins: int = 255) -> torch.Tensor:
    """Log-spaced ratio centers from 0.125× to 8.0× (6 octaves).

    Returns a (ratio_bins,) tensor. The last class in the ratio head
    is STOP (index ratio_bins), not covered here.
    """
    log_min = math.log(0.125)
    log_max = math.log(8.0)
    return torch.exp(torch.linspace(log_min, log_max, ratio_bins))


class RatioDetector(EventEmbeddingDetector):
    """EventEmbeddingDetector with ratio decomposition heads.

    The backbone (conv stem → event embeddings → transformer trunk) is
    inherited unchanged. Only the output head is replaced.
    """

    config: RatioDetectorConfig

    def __init__(self, config: RatioDetectorConfig):
        super().__init__(config)
        d = config.d_model
        R = config.ratio_bins
        D = config.divisor_bins
        O = config.offset_bins

        # Head 1 — Divisor (auxiliary).
        self.divisor_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, D),
        )

        # Head 2 — Offset (auxiliary).
        self.offset_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d // 2), nn.GELU(),
            nn.Linear(d // 2, O),
        )

        # Embeddings for soft expectations → ratio head input.
        self.divisor_val_emb = nn.Linear(1, d)
        self.offset_val_emb = nn.Linear(1, d)

        # Head 3 — Ratio (primary). R bins + 1 STOP.
        self.ratio_head_mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, R + 1),
        )
        k = config.ratio_smooth_kernel
        ch = config.ratio_smooth_channels
        self.ratio_smooth = nn.Sequential(
            nn.Conv1d(1, ch, kernel_size=k, padding=k // 2),
            nn.GELU(),
            nn.Conv1d(ch, 1, kernel_size=k, padding=k // 2),
        )

        # Precompute ratio bin centers (not a parameter — just a buffer).
        self.register_buffer(
            "ratio_centers",
            build_ratio_bin_centers(R),
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _apply_head(self, cursor_tok: torch.Tensor) -> torch.Tensor:
        """Override parent's head with ratio decomposition.

        Returns (B, D + O + R + 1) packed tensor:
          [:D]        divisor logits
          [D:D+O]     offset logits
          [D+O:]      ratio logits (R bins + 1 STOP)
        """
        cfg = self.config
        D = cfg.divisor_bins
        O = cfg.offset_bins

        div_logits = self.divisor_head(cursor_tok)              # (B, D)
        off_logits = self.offset_head(cursor_tok)               # (B, O)

        # Soft expectations → scalar → (optional detach) → embed → cursor.
        # When ``aux_stop_gradient`` is True (default, taiko1 exp 67
        # design), detach so ratio_loss gradient doesn't backprop
        # into div/off heads. When False (#010d), gradients flow —
        # ratio loss can shape div/off predictions too.
        div_probs = torch.softmax(div_logits, dim=-1)
        off_probs = torch.softmax(off_logits, dim=-1)
        div_bins = torch.arange(D, device=div_logits.device, dtype=torch.float32)
        off_bins = torch.arange(O, device=off_logits.device, dtype=torch.float32)
        div_val = (div_probs * div_bins).sum(-1, keepdim=True)          # (B, 1)
        off_val = (off_probs * off_bins).sum(-1, keepdim=True)          # (B, 1)
        if cfg.aux_stop_gradient:
            div_val = div_val.detach()
            off_val = off_val.detach()

        ratio_in = (
            cursor_tok
            + self.divisor_val_emb(div_val)
            + self.offset_val_emb(off_val)
        )
        ratio_logits = self.ratio_head_mlp(ratio_in)            # (B, R+1)
        ratio_logits = ratio_logits + self.ratio_smooth(
            ratio_logits.unsqueeze(1),
        ).squeeze(1)

        return torch.cat([div_logits, off_logits, ratio_logits], dim=-1)
