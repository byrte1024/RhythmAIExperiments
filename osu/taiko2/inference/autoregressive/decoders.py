"""Abstract + concrete AR decoders.

- ABC: `ARDecoder[Config, ModelOutput] → ARDecision`.
- Concrete: `ArgmaxDecoder` — deterministic argmax over the bin-offset
  logits, STOP class at index `b_pred`. Attaches top-5 + entropy in
  `extras` so AR traces are diagnostic.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import torch.nn.functional as F

from ...domain.model import ModelOutput
from ...models.event_embedding import EventEmbeddingOutput
from .types import ARContext, ARDecision, ARDecoderConfig

DCfg = TypeVar("DCfg", bound=ARDecoderConfig)
Out = TypeVar("Out", bound=ModelOutput)


class ARDecoder(ABC, Generic[DCfg, Out]):
    """Decides what the model's output means for one AR step."""
    config: DCfg

    def __init__(self, config: DCfg):
        self.config = config

    @abstractmethod
    def decode(self, output: Out, context: ARContext) -> ARDecision:
        """Return the AR step's decision.

        May emit 0..N onsets via `ARDecision.bin_offsets` (cursor-
        relative). Empty tuple ⇒ STOP for this step. Single-onset
        models emit length-1; multi-onset models (exp 62, 64) emit
        up to `n_onsets`, truncated at the first internal STOP.

        Must attach diagnostic values (top-k candidates, entropy,
        sampled temperature, etc.) via `ARDecision.extras` so they
        can be logged to disk.
        """
        ...


# ─────────────────────────── ArgmaxDecoder ────────────────────────────

@dataclass(frozen=True, slots=True)
class ArgmaxDecoderConfig(ARDecoderConfig):
    """Argmax + STOP-at-b_pred. Must match the trained model's
    `EventEmbeddingConfig.b_pred`."""
    b_pred: int = 500
    top_k_log: int = 5      # how many top-k entries to report in extras


class ArgmaxDecoder(ARDecoder[ArgmaxDecoderConfig, EventEmbeddingOutput]):
    """Deterministic argmax over `(1, n_classes)` logits.

    Decision rule:
      - argmax index == `b_pred` → STOP  (bin_offsets = ()).
      - otherwise              → onset, bin_offset = index, confidence
                                   = softmax probability.

    Always attaches `top{1..K}_bin`, `top{1..K}_prob`, and `entropy`
    in `extras` for per-step logging.
    """

    def decode(self, output: EventEmbeddingOutput, context: ARContext) -> ARDecision:
        logits = output.logits
        if logits.dim() != 2 or logits.size(0) != 1:
            raise ValueError(
                f"ArgmaxDecoder expects (1, n_classes); got shape {tuple(logits.shape)}"
            )
        probs = F.softmax(logits, dim=-1)[0]                          # (n_classes,)
        cls = int(logits[0].argmax().item())
        conf = float(probs[cls].item())

        # Entropy + top-k for trace logging.
        extras: dict[str, float] = {}
        log_probs = probs.clamp(min=1e-10).log()
        extras["entropy"] = float(-(probs * log_probs).sum().item())

        k = min(self.config.top_k_log, probs.size(0))
        top_vals, top_idx = torch.topk(probs, k=k)
        for i, (v, idx) in enumerate(
            zip(top_vals.tolist(), top_idx.tolist()), start=1,
        ):
            extras[f"top{i}_bin"] = float(idx)
            extras[f"top{i}_prob"] = float(v)

        stop_idx = self.config.b_pred
        if cls == stop_idx:
            return ARDecision(
                bin_offsets=(),
                confidences=(),
                extras=extras,
            )
        return ARDecision(
            bin_offsets=(cls,),
            confidences=(conf,),
            extras=extras,
        )
