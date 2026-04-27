"""Abstract + concrete AR decoders.

- ABC: `ARDecoder[Config, ModelOutput] → ARDecision`.
- Concrete: `ArgmaxDecoder` — deterministic argmax over the bin-offset
  logits, STOP class at index `b_pred`. Attaches top-5 + entropy in
  `extras` so AR traces are diagnostic.
- Concrete: `MdnDecoder` — pick highest-weight MDN component, round μ
  to nearest bin. STOP via sigmoid gate.
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
from ...training.losses import parse_mdn_params
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


# ─────────────────────────── MdnDecoder ─────────────────────────────

@dataclass(frozen=True, slots=True)
class MdnDecoderConfig(ARDecoderConfig):
    """MDN decoder: pick highest-weight component, round μ to bin.
    STOP if sigmoid(stop_logit) > stop_threshold."""
    b_pred: int = 500
    n_components: int = 3
    stop_threshold: float = 0.5


class MdnDecoder(ARDecoder[MdnDecoderConfig, EventEmbeddingOutput]):
    """Pick highest-weight MDN component.

    Decision rule:
      - sigmoid(stop_logit) > threshold → STOP.
      - else → round(μ of highest-π component) = bin offset.
    """

    def decode(self, output: EventEmbeddingOutput, context: ARContext) -> ARDecision:
        raw = output.logits
        if raw.dim() != 2 or raw.size(0) != 1:
            raise ValueError(
                f"MdnDecoder expects (1, K*3+1); got shape {tuple(raw.shape)}"
            )

        stop_logit, mu, sigma, pi = parse_mdn_params(
            raw, self.config.n_components, self.config.b_pred,
        )

        p_stop = float(torch.sigmoid(stop_logit[0]).item())

        extras: dict[str, float] = {"p_stop": p_stop}
        # Per-component diagnostics.
        for k in range(self.config.n_components):
            extras[f"comp{k}_mu"] = float(mu[0, k].item())
            extras[f"comp{k}_sigma"] = float(sigma[0, k].item())
            extras[f"comp{k}_pi"] = float(pi[0, k].item())

        if p_stop > self.config.stop_threshold:
            return ARDecision(
                bin_offsets=(),
                confidences=(),
                extras=extras,
            )

        # Pick highest-weight component.
        best_k = int(pi[0].argmax().item())
        bin_pred = int(round(float(mu[0, best_k].item())))
        bin_pred = max(0, min(bin_pred, self.config.b_pred - 1))
        conf = float(pi[0, best_k].item())

        return ARDecision(
            bin_offsets=(bin_pred,),
            confidences=(conf,),
            extras=extras,
        )
