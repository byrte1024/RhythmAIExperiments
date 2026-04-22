"""Training-time concretes: losses, adapters, metrics, hooks, loop."""
from .adapters import DetectionSampleAdapter, DetectionSampleAdapterConfig
from .artifacts import (
    DistributionArtifact,
    ErrorHistogramArtifact,
    PredictionScatterArtifact,
    RatioErrorScatterArtifact,
)
from .hooks import CheckpointHook, MetricLoggerHook
from .loop import train
from .losses import OnsetLoss, OnsetLossConfig
from .metrics_onset import OnsetMetric, OnsetMetricConfig

__all__ = [
    "CheckpointHook",
    "DetectionSampleAdapter",
    "DetectionSampleAdapterConfig",
    "DistributionArtifact",
    "ErrorHistogramArtifact",
    "MetricLoggerHook",
    "OnsetLoss",
    "OnsetLossConfig",
    "OnsetMetric",
    "OnsetMetricConfig",
    "PredictionScatterArtifact",
    "RatioErrorScatterArtifact",
    "train",
]
