"""Domain types for taiko2."""
from .adapter import SampleToModelAdapter
from .augmentation import (
    AugmentationPipeline,
    PostSampleAugmentation,
    PreSampleAugmentation,
)
from .beatmap import (
    AudioRef,
    Density,
    Difficulty,
    Onset,
    OnsetBinned,
    OnsetKind,
    Pack,
    RelativeOnset,
    Track,
)
from .chart import Chart, ChartComparison, ChartMetrics
from .dataset import (
    AudioSampler,
    AudioSamplerConfig,
    ChartEntry,
    DatasetManifest,
    EventSampler,
    EventSamplerConfig,
    MelSamplerConfig,
)
from .loss import Loss, LossConfig, LossResult
from .metrics import (
    Metric,
    MetricConfig,
    MetricInput,
    MetricSet,
    MetricWindow,
    MetricsReport,
)
from .model import Model, ModelConfig, ModelInput, ModelOutput, ModelTarget
from .sampling import DataSample, DataSampler, DataSamplerConfig
from .training import (
    CheckpointMeta,
    RunSpec,
    TrainerConfig,
    TrainerHook,
    TrainingState,
)

__all__ = [
    "AudioRef",
    "AudioSampler",
    "AudioSamplerConfig",
    "AugmentationPipeline",
    "Chart",
    "ChartComparison",
    "ChartEntry",
    "ChartMetrics",
    "CheckpointMeta",
    "DataSample",
    "DataSampler",
    "DataSamplerConfig",
    "DatasetManifest",
    "Density",
    "Difficulty",
    "EventSampler",
    "EventSamplerConfig",
    "Loss",
    "LossConfig",
    "LossResult",
    "MelSamplerConfig",
    "Metric",
    "MetricConfig",
    "MetricInput",
    "MetricSet",
    "MetricWindow",
    "MetricsReport",
    "Model",
    "ModelConfig",
    "ModelInput",
    "ModelOutput",
    "ModelTarget",
    "Onset",
    "OnsetBinned",
    "OnsetKind",
    "Pack",
    "PostSampleAugmentation",
    "PreSampleAugmentation",
    "RelativeOnset",
    "RunSpec",
    "SampleToModelAdapter",
    "Track",
    "TrainerConfig",
    "TrainerHook",
    "TrainingState",
]
