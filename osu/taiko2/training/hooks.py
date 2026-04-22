"""Standard trainer hooks: metric logging (save-before-print) and
checkpointing (atomic, at eval boundaries only).

Both enforce the "save don't print" invariant: metrics hit disk before
anything is ever logged to stdout; `best.pt` lands on disk the moment
the watched metric improves.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
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

        eval_dir = self._spec.run_dir / f"eval_{state.step}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        ckpt.save(eval_dir / "checkpoint.pt")

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


class ConsoleLoggerHook(TrainerHook):
    """Human-readable eval summaries in the terminal.

    The metric dict is already written to ``metrics.jsonl`` by
    ``MetricLoggerHook`` before this hook runs, so nothing printed here
    is "only to stdout" — it's a surfaced view of persisted data.

    Uses ``tqdm.write`` when available so printed lines don't collide
    with in-progress epoch/eval progress bars.
    """

    def __init__(
        self,
        *,
        keys: tuple[str, ...] = (
            "loss", "onset/hit", "onset/miss", "onset/exact",
        ),
    ):
        self._keys = tuple(keys)

    def _write(self, line: str) -> None:
        try:
            from tqdm.auto import tqdm
            tqdm.write(line)
        except ImportError:
            print(line)

    def on_train_start(self, state: TrainingState, spec: RunSpec) -> None:
        self._write(f"[train_start] run={spec.name!r}  dir={spec.run_dir}")

    def on_epoch_start(self, state: TrainingState) -> None:
        self._write(
            f"[epoch_start] epoch={state.epoch + 1}  step={state.step}"
        )

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        parts = [f"step={state.step}", f"epoch={state.epoch + 1}"]
        for k in self._keys:
            if k in val_metrics:
                parts.append(f"{k}={val_metrics[k]:.4f}")
        if state.best_metric is not None:
            parts.append(
                f"best={state.best_metric:.4f}@{state.best_metric_step}"
            )
        self._write("[eval] " + "  ".join(parts))

    def on_train_end(
        self, state: TrainingState, exc: BaseException | None,
    ) -> None:
        status = (
            f"CRASHED ({type(exc).__name__})" if exc is not None else "OK"
        )
        self._write(
            f"[train_end] {status}  step={state.step}  "
            f"epoch={state.epoch}  "
            f"best={state.best_metric}@{state.best_metric_step}"
        )


@dataclass(frozen=True, slots=True)
class CurveSpec:
    """One time-series plot that `MetricCurvesHook` will render.

    `key` is the **un-namespaced** metric name — the same string that
    appears in `LossResult.metrics` / the eval metrics dict. The hook
    translates to `train/batch/{key}` and `val/single/{key}` when
    reading `metrics.jsonl` on resume.
    """
    name: str
    key: str
    log_y: bool = False
    title: str | None = None


DEFAULT_CURVES: tuple[CurveSpec, ...] = (
    # Loss — log-y because it spans multiple orders of magnitude.
    CurveSpec(name="loss",           key="loss",               log_y=True),
    # Primary composites.
    CurveSpec(name="onset_hit",      key="onset/hit",          log_y=False),
    CurveSpec(name="onset_good",     key="onset/good",         log_y=False),
    CurveSpec(name="onset_miss",     key="onset/miss",         log_y=False),
    CurveSpec(name="onset_exact",    key="onset/exact",        log_y=False),
    # Frame-based (±2 / ±7 bin windows).
    CurveSpec(name="onset_fhit",     key="onset/fhit",         log_y=False),
    CurveSpec(name="onset_fgood",    key="onset/fgood",        log_y=False),
    CurveSpec(name="onset_fmiss",    key="onset/fmiss",        log_y=False),
    # Ratio-based (±3 % / ±10 % log-ratio).
    CurveSpec(name="onset_rhit",     key="onset/rhit",         log_y=False),
    CurveSpec(name="onset_rgood",    key="onset/rgood",        log_y=False),
    CurveSpec(name="onset_rmiss",    key="onset/rmiss",        log_y=False),
    # Any-future variants — populated when future_events[0..] carries
    # real events (d_events ≥ 1).
    CurveSpec(name="onset_ihit",     key="onset/ihit",         log_y=False),
    CurveSpec(name="onset_igood",    key="onset/igood",        log_y=False),
    CurveSpec(name="onset_imiss",    key="onset/imiss",        log_y=False),
    # Diagnostic: how often the model chooses STOP on non-STOP targets.
    CurveSpec(name="pred_stop_rate", key="onset/pred_stop_rate", log_y=False),
    # STOP-class precision / recall / F1 — catches over- or under-STOPping.
    CurveSpec(name="onset_stop_precision", key="onset/stop_precision", log_y=False),
    CurveSpec(name="onset_stop_recall",    key="onset/stop_recall",    log_y=False),
    CurveSpec(name="onset_stop_f1",        key="onset/stop_f1",        log_y=False),
    # Frame-error aggregates on non-STOP pairs.
    CurveSpec(name="onset_frame_err_mean",   key="onset/frame_err_mean",   log_y=False),
    CurveSpec(name="onset_frame_err_median", key="onset/frame_err_median", log_y=False),
    CurveSpec(name="onset_frame_err_p90",    key="onset/frame_err_p90",    log_y=False),
)


class MetricCurvesHook(TrainerHook):
    """Writes one time-series PNG per tracked metric under
    ``{run_dir}/curves/{name}.png``. Updated after every eval and at
    train end; atomic via tmp + ``os.replace``.

    Each PNG plots:
      - Per-step train values (raw thin line + smoothed rolling mean).
      - Per-eval val values (marker + line).

    Resume-safe: ``on_train_start`` seeds the in-memory arrays from
    the existing ``metrics.jsonl`` so the curves pick up where a
    killed run left off.

    This hook is a **derived artifact** of ``metrics.jsonl`` — every
    value plotted is already persisted as JSONL before it's ever
    rendered.
    """

    def __init__(
        self,
        *,
        curves: "tuple[CurveSpec, ...] | list[CurveSpec]" = DEFAULT_CURVES,
        smoothing_window: int = 50,
        output_dir_name: str = "curves",
    ):
        self._curves: tuple[CurveSpec, ...] = tuple(curves)
        self._smoothing = max(1, int(smoothing_window))
        self._dir_name = output_dir_name
        self._spec: RunSpec | None = None
        # Per-metric series keyed by `CurveSpec.key`.
        self._train: dict[str, list[tuple[int, float]]] = {}
        self._eval: dict[str, list[tuple[int, float]]] = {}

    # ── lifecycle ─────────────────────────────────────────────────────

    def on_train_start(self, state: TrainingState, spec: RunSpec) -> None:
        self._spec = spec
        self._seed_from_jsonl(spec.metrics_path)

    def on_step_end(self, state: TrainingState, train_loss: LossResult) -> None:
        step = int(state.step)
        metrics = train_loss.metrics
        for curve in self._curves:
            if curve.key in metrics:
                self._train.setdefault(curve.key, []).append(
                    (step, float(metrics[curve.key])),
                )

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        step = int(state.step)
        for curve in self._curves:
            if curve.key in val_metrics:
                self._eval.setdefault(curve.key, []).append(
                    (step, float(val_metrics[curve.key])),
                )
        self._render_atomic()

    def on_train_end(
        self, state: TrainingState, exc: BaseException | None,
    ) -> None:
        self._render_atomic()

    # ── internals ─────────────────────────────────────────────────────

    def _seed_from_jsonl(self, path: Path) -> None:
        """Populate arrays from a pre-existing ``metrics.jsonl``.
        JSON-decode failures are silent — partial lines at the end of a
        killed run shouldn't block resume."""
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event = d.get("event")
                    step = d.get("step")
                    if step is None:
                        continue
                    if event == "step":
                        for curve in self._curves:
                            k = f"train/batch/{curve.key}"
                            if k in d:
                                self._train.setdefault(curve.key, []).append(
                                    (int(step), float(d[k])),
                                )
                    elif event == "eval":
                        for curve in self._curves:
                            k = f"val/single/{curve.key}"
                            if k in d:
                                self._eval.setdefault(curve.key, []).append(
                                    (int(step), float(d[k])),
                                )
        except OSError:
            pass

    def _render_atomic(self) -> None:
        if self._spec is None:
            return
        out_dir = self._spec.run_dir / self._dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for curve in self._curves:
            self._render_one(curve, out_dir)

    def _render_one(self, curve: CurveSpec, out_dir: Path) -> None:
        train_pts = self._train.get(curve.key, [])
        eval_pts = self._eval.get(curve.key, [])
        if not train_pts and not eval_pts:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        path = out_dir / f"{curve.name}.png"
        tmp = path.with_suffix(path.suffix + ".tmp")

        fig, ax = plt.subplots(figsize=(11, 5))

        if train_pts:
            xs = [p[0] for p in train_pts]
            ys = [p[1] for p in train_pts]
            ax.plot(
                xs, ys,
                color="#4a90d9", alpha=0.25, linewidth=0.6,
                label="train/batch (raw)",
            )
            if len(ys) >= self._smoothing:
                arr = np.asarray(ys, dtype=np.float64)
                w = self._smoothing
                kernel = np.ones(w, dtype=np.float64) / w
                smoothed = np.convolve(arr, kernel, mode="valid")
                smoothed_x = xs[w - 1:]
                ax.plot(
                    smoothed_x, smoothed,
                    color="#1f4fa3", linewidth=1.6,
                    label=f"train/batch (avg {w})",
                )

        if eval_pts:
            xs = [p[0] for p in eval_pts]
            ys = [p[1] for p in eval_pts]
            ax.plot(
                xs, ys,
                color="#e86850", marker="o", linewidth=1.8,
                markersize=5, label="val/single",
            )

        if curve.log_y:
            # log scale requires strictly positive values; fall back
            # to linear if any non-positive crept in.
            vals = [v for _, v in train_pts] + [v for _, v in eval_pts]
            if vals and min(vals) > 0:
                ax.set_yscale("log")

        title = curve.title or curve.key
        ax.set_xlabel("Step")
        ax.set_ylabel(curve.key)
        ax.set_title(
            f"{title} — {len(train_pts):,} train pts, {len(eval_pts):,} eval pts"
        )
        ax.grid(True, which="both", alpha=0.15)
        ax.legend(loc="best")
        fig.tight_layout()
        try:
            fig.savefig(tmp, dpi=120, format="png")
        finally:
            plt.close(fig)
        os.replace(tmp, path)


class PerEvalJsonHook(TrainerHook):
    """Writes ``{run_dir}/eval_{step}/eval.json`` after every eval.

    One self-contained JSON per eval. Complements `metrics.jsonl`
    (append-only stream of every event) by giving each eval its own
    standalone record — easier to `diff` two evals, grep a specific
    step, or feed into a downstream analysis script without parsing
    the whole jsonl stream.

    The file includes the full val metric dict (un-namespaced keys,
    as the training loop produces them) plus the corresponding
    `val/single/…` namespaced mirror — that way consumers can
    reference either convention without re-mapping.
    """

    def __init__(self, *, filename: str = "eval.json"):
        self._filename = filename
        self._spec: RunSpec | None = None

    def on_train_start(self, state: TrainingState, spec: RunSpec) -> None:
        self._spec = spec

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        if self._spec is None:
            return
        eval_dir = self._spec.run_dir / f"eval_{state.step}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        path = eval_dir / self._filename
        tmp = path.with_suffix(path.suffix + ".tmp")

        # Mirror MetricLoggerHook's namespacing so downstream tools can
        # reference either form without custom code.
        namespaced = {f"val/single/{k}": v for k, v in val_metrics.items()}

        payload = {
            "event": "eval",
            "step": state.step,
            "epoch": state.epoch,
            "wall_time": time.time(),
            "best_metric": state.best_metric,
            "best_metric_step": state.best_metric_step,
            "metrics": dict(val_metrics),
            "namespaced": namespaced,
        }
        try:
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
