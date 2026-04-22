"""Abstract model contract.

`Model` is a typed nn.Module: `predict(input) -> output` is the
framework-wide entry point, while the subclass's own `forward()` stays
free-form for torch.compile / TorchScript / ONNX tracing.

The three IO base types — `ModelInput`, `ModelOutput`, `ModelTarget` —
are empty frozen dataclasses. They exist only to give TypeVar bounds,
so generic code (adapter, loss, metrics, trainer) can be written once
against "any model" and stay type-checkable. Concrete models subclass
each with the tensor fields they actually carry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch.nn as nn


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Base for model hyperparameters. Subclass to add fields."""


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Base for batched tensors fed to a model's `predict`. Subclass."""


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Base for tensors a model's `predict` returns. Subclass."""


@dataclass(frozen=True, slots=True)
class ModelTarget:
    """Base for a training target. Subclass."""


Cfg = TypeVar("Cfg", bound=ModelConfig)
Inp = TypeVar("Inp", bound=ModelInput)
Out = TypeVar("Out", bound=ModelOutput)


class Model(nn.Module, ABC, Generic[Cfg, Inp, Out]):
    """Typed nn.Module base. Framework code only touches `predict`."""

    config: Cfg

    def __init__(self, config: Cfg):
        super().__init__()
        self.config = config

    @abstractmethod
    def predict(self, x: Inp) -> Out:
        """Typed forward. Subclasses normally unpack `x` into tensors,
        invoke their own `forward(...)`, and wrap the result in an
        `Out` dataclass."""
        ...

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
