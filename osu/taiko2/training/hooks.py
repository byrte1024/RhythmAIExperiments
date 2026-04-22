"""Standard trainer hooks: metric logging (save-before-print) and
checkpointing (atomic, at eval boundaries only).

Both enforce the "save don't print" invariant: metrics hit disk before
anything is ever logged to stdout; `best.pt` lands on disk the moment
the watched metric improves.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..domain.loss import Loss, LossResult
from ..domain.metrics import MetricsReport, MetricWindow
from ..domain.model import Model
from ..domain.training import (
    RunSpec,
    TrainerConfig,
    TrainerHook,
    TrainingState,
)
from ..persistence.checkpoint import Checkpoint


class MetricLoggerHook(TrainerHook):
    """Append-only JSONL metrics log.

    Every step writes one line under ``event="step"`` with the train
    loss + sub-metrics. Every eval writes one line under ``event="eval"``
    with all val metrics. The file is opened once at train_start and
    closed at train_end; line-buffered flush on every write keeps the
    log intact across crashes.
    """

    def __init__(self, spec: RunSpec, *, step_log_every: int = 1):
        self._path = spec.metrics_path
        self._step_every = max(1, int(step_log_every))
        self._fh: Any = None

    def on_train_start(self, state: TrainingState, spec: RunSpec) -> None:
        spec.ensure()
        self._fh = self._path.open("a", encoding="utf-8")

    def _write(self, payload: dict[str, Any]) -> None:
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def on_step_end(
        self, state: TrainingState, train_loss: LossResult,
    ) -> None:
        if state.step % self._step_every != 0:
            return
        # Promote loss sub-metrics into the `train/batch/*` namespace.
        report = MetricsReport(
            event="step",
            step=state.step,
            epoch=state.epoch,
            wall_time=time.time(),
        ).with_values("train", MetricWindow.BATCH, dict(train_loss.metrics))
        self._write({
            "event": report.event,
            "step": report.step,
            "epoch": report.epoch,
            "wall_time": report.wall_time,
            **report.values,
        })

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        report = MetricsReport(
            event="eval",
            step=state.step,
            epoch=state.epoch,
            wall_time=time.time(),
        ).with_values("val", MetricWindow.SINGLE, val_metrics)
        self._write({
            "event": report.event,
            "step": report.step,
            "epoch": report.epoch,
            "wall_time": report.wall_time,
            **report.values,
        })

    def on_train_end(
        self, state: TrainingState, exc: BaseException | None,
    ) -> None:
        if self._fh is not None:
            try:
                self._write({
                    "event": "train_end",
                    "step": state.step,
                    "epoch": state.epoch,
                    "wall_time": time.time(),
                    "exc": type(exc).__name__ if exc is not None else None,
                })
            finally:
                self._fh.close()
                self._fh = None


class CheckpointHook(TrainerHook):
    """Saves `latest.pt` after every eval; writes `best.pt` when the
    watched metric improves. Extra safety `latest.pt` save in
    `on_train_end` so a crash mid-epoch still leaves a resume point.

    Checkpoints fire **only at eval boundaries** — per the design,
    per-step checkpoints of tens-of-megabytes PyTorch state dicts are
    too expensive to be useful.
    """

    def __init__(
        self,
        spec: RunSpec,
        model: Model,
        loss: Loss,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        trainer_config: TrainerConfig,
    ):
        self._spec = spec
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._trainer_config = trainer_config

    def _snapshot(self, state: TrainingState) -> Checkpoint:
        return Checkpoint.from_runtime(
            model=self._model,
            loss=self._loss,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            trainer_config=self._trainer_config,
            training_state=state,
        )

    def on_train_start(self, state: TrainingState, spec: RunSpec) -> None:
        spec.ensure()

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        ckpt = self._snapshot(state)
        ckpt.save(self._spec.latest_checkpoint)

        watched = self._trainer_config.metric_to_watch
        if watched not in val_metrics:
            return
        value = float(val_metrics[watched])
        lower_better = self._trainer_config.metric_lower_is_better
        is_better = (
            state.best_metric is None
            or (lower_better and value < state.best_metric)
            or (not lower_better and value > state.best_metric)
        )
        if is_better:
            state.best_metric = value
            state.best_metric_step = state.step
            ckpt.save(self._spec.best_checkpoint)

    def on_train_end(
        self, state: TrainingState, exc: BaseException | None,
    ) -> None:
        try:
            self._snapshot(state).save(self._spec.latest_checkpoint)
        except Exception:
            pass  # never raise from a finally hook
