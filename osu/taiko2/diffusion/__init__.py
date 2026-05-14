"""Concrete diffusion components (schedules, processes, denoisers,
samplers) implementing the ABCs in ``osu.taiko2.domain.diffusion``.

Each module is independently composable: concrete schedules pair with
concrete processes that pair with concrete denoisers and samplers.
The ``ConfigSelector`` machinery in the training CLI picks the
combination from JSON config at runtime.
"""
from .schedules import (
    CosineSchedule,
    CosineScheduleConfig,
    LinearSchedule,
    LinearScheduleConfig,
)
from .processes import (
    FramewiseActivationProcess,
    FramewiseActivationProcessConfig,
    GaussianContinuousProcess,
    GaussianContinuousProcessConfig,
)
from .denoisers import (
    Conv1DDenoiser,
    Conv1DDenoiserConfig,
    MLPDenoiser,
    MLPDenoiserConfig,
)
from .samplers import DDIMSampler, DDIMSamplerConfig, DDPMSampler

__all__ = [
    # schedules
    "LinearSchedule",
    "LinearScheduleConfig",
    "CosineSchedule",
    "CosineScheduleConfig",
    # processes
    "GaussianContinuousProcess",
    "GaussianContinuousProcessConfig",
    "FramewiseActivationProcess",
    "FramewiseActivationProcessConfig",
    # denoisers
    "MLPDenoiser",
    "MLPDenoiserConfig",
    "Conv1DDenoiser",
    "Conv1DDenoiserConfig",
    # samplers
    "DDPMSampler",
    "DDIMSampler",
    "DDIMSamplerConfig",
]
