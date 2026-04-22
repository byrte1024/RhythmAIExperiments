"""Abstract loss contract.

Every loss produces a `LossResult` — a graph-connected scalar plus a
dict of detached floats. The detached dict is what the metrics logger
writes to disk, so sub-components (hard CE, soft CE, STOP weight, …)
are never lost to "printed but not saved" oblivion.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
import torch.nn as nn

from .model import ModelOutput, ModelTarget


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Base for loss hyperparameters. Subclass to add fields."""


@dataclass(frozen=True, slots=True)
class LossResult:
    """Loss output. `loss` is graph-connected (backprop); `metrics` is a
    dict of detached floats (logging-only)."""
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


LCfg = TypeVar("LCfg", bound=LossConfig)
Out = TypeVar("Out", bound=ModelOutput)
Tgt = TypeVar("Tgt", bound=ModelTarget)


class Loss(nn.Module, ABC, Generic[LCfg, Out, Tgt]):
    """Typed loss. Subclasses implement `forward(output, target)`."""

    config: LCfg

    def __init__(self, config: LCfg):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, output: Out, target: Tgt) -> LossResult:
        """Compute the loss. Must return `LossResult(loss, metrics)`.

        `metrics` keys should be stable strings — they become column
        names in the run's `metrics.jsonl`.
        """
        ...
