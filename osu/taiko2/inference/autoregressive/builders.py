"""Abstract AR input builder — constructs one-step `ModelInput` from
live AR state.

Separate from `SampleToModelAdapter` because AR has no `DataSample`
to collate — the input is assembled from (cursor, past_onsets, full
audio features, conditioning) on the fly each step.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import torch

from ...domain.beatmap import OnsetBinned
from ...domain.inference import Conditioning
from ...domain.model import ModelInput
from .types import ARInputBuilderConfig

BCfg = TypeVar("BCfg", bound=ARInputBuilderConfig)
Inp = TypeVar("Inp", bound=ModelInput)


class ARInputBuilder(ABC, Generic[BCfg, Inp]):
    """Assembles a single-batch `ModelInput` tensor bundle for one AR
    step. One concrete per (model family × data layout) pair; reusable
    across any predictor that runs the same model family.
    """
    config: BCfg

    def __init__(self, config: BCfg):
        self.config = config

    @abstractmethod
    def build(
        self,
        *,
        cursor_bin: int,
        past_onsets: tuple[OnsetBinned, ...],
        audio_features: np.ndarray,       # (F, T) precomputed once per chart
        conditioning: Conditioning | None,
        device: torch.device,
    ) -> Inp:
        """Return one batch-of-1 `ModelInput` ready for `model.predict`."""
        ...
