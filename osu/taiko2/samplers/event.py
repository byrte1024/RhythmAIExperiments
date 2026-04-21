"""Concrete EventSampler implementations."""
from __future__ import annotations

from ..types.dataset import EventSampler, EventSamplerConfig


class FixedRateEventSampler(EventSampler):
    """Uniform grid at `config.bins_per_second`. Bin = floor(time_ms / bin_ms)."""

    def __init__(self, config: EventSamplerConfig):
        self.config = config

    @property
    def bin_ms(self) -> float:
        return 1000.0 / self.config.bins_per_second

    def bin_of(self, time_ms: float) -> int:
        return int(time_ms / self.bin_ms)
