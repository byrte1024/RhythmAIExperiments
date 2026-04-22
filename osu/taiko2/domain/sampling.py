"""Data sampler abstraction: the contract between a built dataset on disk
and whatever consumes samples (training loop, eval script, exporter).

A `DataSampler` knows how to load a dataset and hand out typed samples and
batches by index. It's the third sampler layer, sibling to:
  - `AudioSampler` (audio-file → feature array)
  - `EventSampler` (onset timestamps → bin indices)
  - `DataSampler` (dataset → typed training/eval samples)

Concrete subclasses extend:
  - `DataSample`  to define what a single sample carries (mel window,
     event context, target, …).
  - `DataSamplerConfig` to declare their parameters (window size, split,
     shuffle seed, …).
  - `DataSampler` to implement `load_data` / `count_samples` / `get_sample`.

`count_batches` and `get_batch` have default implementations in terms of
the other three; override only if the sampler wants non-contiguous or
non-uniform batching.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar


@dataclass(frozen=True, slots=True)
class DataSample:
    """Base sample type. Concrete samplers extend this with payload fields.

    Carries only the sample's index within the dataset — enough to trace
    a prediction back to the exact source and to reconstruct determinism
    in shuffled pipelines.
    """
    sample_id: int


@dataclass(frozen=True, slots=True)
class DataSamplerConfig:
    """Base config. `batch_size` is the only field required by the default
    `count_batches` / `get_batch` implementations; subclasses add their own.
    """
    batch_size: int


S = TypeVar("S", bound=DataSample)
C = TypeVar("C", bound=DataSamplerConfig)


class DataSampler(ABC, Generic[S, C]):
    """Typed sampler interface over a built dataset.

    Lifecycle::

        sampler = ConcreteSampler(config)
        sampler.load_data()
        n_samples  = sampler.count_samples()
        n_batches  = sampler.count_batches()
        sample     = sampler.get_sample(i)   # -> S (subclass of DataSample)
        batch      = sampler.get_batch(i)    # -> list[S]

    `load_data` is separate from `__init__` so tests / tooling can construct
    a sampler with config alone (no disk reads) and only pay the I/O when
    they actually need samples.
    """
    config: C

    def __init__(self, config: C):
        self.config = config

    @abstractmethod
    def load_data(self) -> None:
        """Read source data into memory and build any indices required to
        serve `get_sample` / `get_batch` in O(1)–ish time.

        Idempotent: calling twice must be safe (either re-load or no-op).
        """
        ...

    @abstractmethod
    def count_samples(self) -> int:
        """Total sample count after `load_data`. Must not change between
        calls unless `load_data` is invoked again with altered config.
        """
        ...

    def count_batches(self) -> int:
        """Number of batches. Default: ``ceil(count_samples / batch_size)``.

        Override for samplers that use drop-last, padded-last, or
        bucketed batching schemes.
        """
        n = self.count_samples()
        bs = self.config.batch_size
        if bs <= 0:
            raise ValueError(f"batch_size must be >= 1, got {bs}")
        return (n + bs - 1) // bs

    @abstractmethod
    def get_sample(self, n: int) -> S:
        """Return the n-th sample. `n` is a 0-indexed integer in
        ``[0, count_samples())``. Out-of-range access should raise
        `IndexError`.
        """
        ...

    def get_batch(self, n: int) -> list[S]:
        """Return the n-th batch as a contiguous slice of samples.

        Default batching is deterministic, contiguous, and uniform in size
        except for the last batch which may be shorter. Override for
        shuffled, bucketed, or padded schemes.
        """
        total = self.count_samples()
        bs = self.config.batch_size
        start = n * bs
        if start < 0 or start >= total:
            raise IndexError(
                f"batch index {n} out of range for {self.count_batches()} batches"
            )
        end = min(start + bs, total)
        return [self.get_sample(i) for i in range(start, end)]
