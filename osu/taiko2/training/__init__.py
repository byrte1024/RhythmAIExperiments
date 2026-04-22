"""Training-time concretes: losses, adapters, metrics, hooks, loop."""
from .adapters import DetectionSampleAdapter, DetectionSampleAdapterConfig
from .artifacts import (
    DistributionArtifact,
    ErrorHistogramArtifact,
    PredictionScatterArtifact,
    RatioErrorScatterArtifact,
)
from .augmentations import (
    ConditioningJitter,
    ContextTruncation,
    EventDropout,
    EventInsertion,
    EventJitter,
    LargeTimeShift,
    MelFreqJitter,
    MelGainJitter,
    MelGaussianNoise,
    PartialAdvMetronome,
    PartialMetronome,
    SpecAugFreq,
    SpecAugTime,
    build_exp45_post_augs,
)
from .hooks import (
    CheckpointHook,
    ConsoleLoggerHook,
    CurveSpec,
    MetricCurvesHook,
    MetricLoggerHook,
    PerEvalJsonHook,
)
from .loop import train
from .losses import OnsetLoss, OnsetLossConfig
from .metrics_onset import OnsetMetric, OnsetMetricConfig

__all__ = [
    "CheckpointHook",
    "ConditioningJitter",
    "ConsoleLoggerHook",
    "ContextTruncation",
    "CurveSpec",
    "MetricCurvesHook",
    "DetectionSampleAdapter",
    "DetectionSampleAdapterConfig",
    "DistributionArtifact",
    "ErrorHistogramArtifact",
    "EventDropout",
    "EventInsertion",
    "EventJitter",
    "LargeTimeShift",
    "MelFreqJitter",
    "MelGainJitter",
    "MelGaussianNoise",
    "MetricLoggerHook",
    "OnsetLoss",
    "OnsetLossConfig",
    "OnsetMetric",
    "OnsetMetricConfig",
    "PartialAdvMetronome",
    "PartialMetronome",
    "PredictionScatterArtifact",
    "RatioErrorScatterArtifact",
    "SpecAugFreq",
    "SpecAugTime",
    "build_exp45_post_augs",
    "train",
]
