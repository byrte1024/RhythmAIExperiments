"""Training-time concretes: losses, adapters, hooks, loops."""
from .adapters import DetectionSampleAdapter, DetectionSampleAdapterConfig
from .losses import OnsetLoss, OnsetLossConfig

__all__ = [
    "DetectionSampleAdapter",
    "DetectionSampleAdapterConfig",
    "OnsetLoss",
    "OnsetLossConfig",
]
