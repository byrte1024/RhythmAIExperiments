"""Universal inference contract — audio in, Chart out.

The only framework-level abstraction on the inference side is
`ChartPredictor`. Everything that distinguishes AR from framewise from
dense end-to-end prediction lives inside concrete subclasses, never in
this file. Consumers only ever see `predictor.predict(chart)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .chart import Chart


@dataclass(frozen=True, slots=True)
class Conditioning:
    """User intent for generation: target density shape.

    Some predictors require it (typically AR/generative models that use
    density as a soft hint); others ignore it entirely. Concrete
    `ChartPredictor` subclasses are responsible for asserting whether
    they need it — the base type accepts `None` so predictors that
    don't condition on anything aren't forced to receive placeholders.
    """
    density_mean: float
    density_peak: int
    density_std: float


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    """Base for any ChartPredictor's hyperparameters."""


PCfg = TypeVar("PCfg", bound=PredictorConfig)


class ChartPredictor(ABC, Generic[PCfg]):
    """Audio in, Chart out.

    Contract: take a `Chart` (audio bytes + track metadata; onsets are
    ignored) and an optional `Conditioning`, return a new Chart with
    the same track metadata and filled-in onsets.

    The returned Chart preserves `audio` so the output can be rendered
    back to `.osz` or played in the viewer without reattaching. A
    predictor that wants to drop audio (e.g. to reduce memory) can
    override — but by default, audio round-trips.

    Subclasses are fully responsible for their own mechanics: AR loop,
    framewise windowing, dense single-pass, ensemble. The framework
    treats them uniformly.
    """
    config: PCfg

    def __init__(self, config: PCfg):
        self.config = config

    @abstractmethod
    def predict(
        self,
        chart: Chart,
        *,
        conditioning: Conditioning | None = None,
    ) -> Chart:
        """Run prediction on one chart. Subclass responsibility: decide
        whether `conditioning` is required and raise clearly if missing.
        """
        ...

    def predict_many(
        self,
        charts: Iterable[Chart],
        *,
        conditioning: Conditioning | None = None,
    ) -> list[Chart]:
        """Default: loop `predict`. Override when an implementation can
        batch charts together in one model call for throughput.
        """
        return [self.predict(c, conditioning=conditioning) for c in charts]
