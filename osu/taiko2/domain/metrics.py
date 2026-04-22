"""Abstract metric contract.

A `Metric` accumulates across batches and emits a `dict[str, float]`
when `compute()` is called. `MetricInput` is a struct that carries
everything a metric might need — model output, target, raw model
input, original untensorized samples, and any loss-reported sub-metrics
— so we don't need separate Metric ABCs for "output-only" vs. "needs
sample data" varieties.

Separation of concerns for reporting — every `MetricsReport` line
written to `metrics.jsonl` is tagged with:

  - `split`   — which data source ("train" / "val" / any user-defined).
  - `window`  — the reset cadence (`MetricWindow` enum):
                BATCH        — last batch only (reset every step).
                SINCE_EVAL   — running average since the last eval.
                SINCE_EPOCH  — running average since epoch start.
                OVERALL      — running average since training start.
                SINGLE       — one full pass, no running state (typical
                               for val/test evals).

This lets the same metric (e.g. `loss`) be reported under every window
for training, and under `SINGLE` for validation, without name clashes.

Invariant: every value in the final dict returned by `compute()` must
be JSON-safe (Python `int` / `float`). The metrics logger persists
these to `{run_dir}/metrics.jsonl` — nothing gets printed to console
alone.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .model import ModelInput, ModelOutput, ModelTarget


class MetricWindow(Enum):
    """Reset cadence for a metric accumulator.

    The trainer creates separate `Metric` instances for each (split,
    window) it cares about and issues `reset()` on its own schedule.
    """
    BATCH = "batch"              # resets every training step
    SINCE_EVAL = "since_eval"    # resets on each eval
    SINCE_EPOCH = "since_epoch"  # resets on each epoch
    OVERALL = "overall"          # never resets
    SINGLE = "single"            # one-shot full pass (no running state)


@dataclass(frozen=True, slots=True)
class MetricInput:
    """Everything a metric can see about one batch.

    Fields past `output` and `target` are optional; a metric that only
    needs output + target can ignore the rest. The typed fields are
    `ModelOutput` / `ModelTarget` — concrete metrics cast internally if
    they need fields their subclasses declare.
    """
    output: ModelOutput
    target: ModelTarget
    input: ModelInput | None = None
    samples: tuple[Any, ...] | None = None
    loss: float | None = None
    loss_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Base for metric hyperparameters (tolerance, top-k, etc.)."""


class Metric(ABC):
    """Accumulates batch statistics and emits a flat `dict[str, float]`.

    Implementations follow the `reset` → `update` → `compute` pattern:
    tests / eval loops call `reset()`, then `update(batch)` over every
    batch, then `compute()` once to read the final numbers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier — used as a prefix in the output
        dict keys so two metrics' values don't collide in the log."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulator state. Called at the start of each eval."""
        ...

    @abstractmethod
    def update(self, batch: MetricInput) -> None:
        """Accumulate one batch's contribution."""
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return the final metric values. Keys should be prefixed with
        `self.name` so multiple metrics can merge into one flat dict."""
        ...


class MetricSet:
    """Run many metrics at once. Collects their `compute()` dicts into
    one flat dict; key collisions between metrics are a bug (they
    should all prefix with their unique name).
    """

    def __init__(self, *metrics: Metric):
        self._metrics: tuple[Metric, ...] = tuple(metrics)

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return self._metrics

    def reset(self) -> None:
        for m in self._metrics:
            m.reset()

    def update(self, batch: MetricInput) -> None:
        for m in self._metrics:
            m.update(batch)

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in self._metrics:
            for k, v in m.compute().items():
                if k in out:
                    raise RuntimeError(
                        f"metric key collision: {k!r} already produced by a prior metric"
                    )
                out[k] = v
        return out


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """One line of `metrics.jsonl` — a labelled snapshot of numbers.

    `event` is the kind of moment this report describes (e.g.
    ``"step"`` fired every batch, ``"eval"`` at eval boundaries). The
    trainer picks stable strings here; the convention lets downstream
    tools filter logs without parsing column names.

    `values` is keyed by ``f"{split}/{window.value}/{metric_key}"`` so
    a single report line can mix train batch / train running / val
    single-pass values without collision.
    """
    event: str                      # "step" | "eval" | "epoch" | "resume" | ...
    step: int
    epoch: int
    wall_time: float                # seconds since epoch (time.time())
    values: dict[str, float] = field(default_factory=dict)

    def with_values(
        self, split: str, window: MetricWindow, extra: dict[str, float],
    ) -> "MetricsReport":
        """Return a new report with `extra` merged under `split/window/`."""
        merged = dict(self.values)
        prefix = f"{split}/{window.value}/"
        for k, v in extra.items():
            merged[prefix + k] = float(v)
        return MetricsReport(
            event=self.event,
            step=self.step,
            epoch=self.epoch,
            wall_time=self.wall_time,
            values=merged,
        )
