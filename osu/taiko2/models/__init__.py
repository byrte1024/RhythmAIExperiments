"""Concrete models for taiko2."""
from .common import AudioConvStem, FiLM, SinusoidalPosEmb
from .event_embedding import (
    EventEmbeddingConfig,
    EventEmbeddingDetector,
    EventEmbeddingInput,
    EventEmbeddingOutput,
    EventEmbeddingTarget,
)
from .conformer_block import ConformerBlock
from .conformer_detector import ConformerDetector, ConformerDetectorConfig
from .diffusion_detector import (
    DiffusionDetector,
    DiffusionDetectorConfig,
    DiffusionModelOutput,
)
from .onset_augmented import OnsetAugmentedConfig, OnsetAugmentedDetector

__all__ = [
    "AudioConvStem",
    "ConformerBlock",
    "ConformerDetector",
    "ConformerDetectorConfig",
    "DiffusionDetector",
    "DiffusionDetectorConfig",
    "DiffusionModelOutput",
    "EventEmbeddingConfig",
    "EventEmbeddingDetector",
    "EventEmbeddingInput",
    "EventEmbeddingOutput",
    "EventEmbeddingTarget",
    "FiLM",
    "OnsetAugmentedConfig",
    "OnsetAugmentedDetector",
    "SinusoidalPosEmb",
]
