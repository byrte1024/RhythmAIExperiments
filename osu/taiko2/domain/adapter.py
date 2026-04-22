"""Adapter between a data sampler and a model.

`SampleToModelAdapter` is the single place where task-specific
interpretations happen (STOP-class derivation, density conditioning
lookup, mel concatenation, tensor collation, device placement). Both
the sampler (pure geometry) and the model (pure math on tensors) stay
decoupled.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

from .model import ModelInput, ModelTarget
from .sampling import DataSample


S = TypeVar("S", bound=DataSample)
Inp = TypeVar("Inp", bound=ModelInput)
Tgt = TypeVar("Tgt", bound=ModelTarget)


class SampleToModelAdapter(ABC, Generic[S, Inp, Tgt]):
    """Collates a list of DataSamples into a batched `(Inp, Tgt)` pair.

    One adapter per (sampler type × model type) pair. Training / eval
    code calls `make_batch(samples, device=...)` and never touches the
    sample internals itself.
    """

    @abstractmethod
    def make_input(self, samples: list[S], *, device: torch.device) -> Inp:
        """Collate + encode a batch into typed model input tensors."""
        ...

    @abstractmethod
    def make_target(self, samples: list[S], *, device: torch.device) -> Tgt:
        """Build the typed training target for the same batch."""
        ...

    def make_batch(
        self, samples: list[S], *, device: torch.device,
    ) -> tuple[Inp, Tgt]:
        return (
            self.make_input(samples, device=device),
            self.make_target(samples, device=device),
        )
