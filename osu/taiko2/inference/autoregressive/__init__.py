"""AR-style inference machinery.

Public surface: `AutoregressivePredictor` + its config. The ABCs
underneath (`ARDecoder`, `ARInputBuilder`) are exported so concrete
models can provide their own decoder/builder, but are never needed by
callers that just want `predictor.predict(chart)`.
"""
from .builders import ARInputBuilder
from .decoders import ARDecoder
from .predictor import AutoregressivePredictor, AutoregressivePredictorConfig
from .types import (
    ARContext,
    ARDecision,
    ARDecoderConfig,
    ARInputBuilderConfig,
)

__all__ = [
    "ARContext",
    "ARDecision",
    "ARDecoder",
    "ARDecoderConfig",
    "ARInputBuilder",
    "ARInputBuilderConfig",
    "AutoregressivePredictor",
    "AutoregressivePredictorConfig",
]
