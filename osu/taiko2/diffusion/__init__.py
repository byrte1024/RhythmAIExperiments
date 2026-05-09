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
    GaussianContinuousProcess,
    GaussianContinuousProcessConfig,
)
from .denoisers import MLPDenoiser, MLPDenoiserConfig
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
    # denoisers
    "MLPDenoiser",
    "MLPDenoiserConfig",
    # samplers
    "DDPMSampler",
    "DDIMSampler",
    "DDIMSamplerConfig",
]
