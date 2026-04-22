"""Abstract AR decoder — translates a model's output into a single
AR step's decision.

Concrete decoders (Argmax, TopK, TopUnique, metronome-aware, …) live
in the same package once the first model ports in. The decoder is
independent of the model's architecture as long as the output type
matches — two models that both emit `(logits, stop_logit_optional)`
share decoders.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ...domain.model import ModelOutput
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
