"""Training domain types: RunSpec, TrainerConfig, TrainingState,
TrainerHook ABC, CheckpointMeta.

Persistence invariants the framework enforces (documented here;
implementations live in `persistence/checkpoint.py` and
`training/hooks.py`):

  - **Checkpoints fire at eval boundaries only.** Model + optimizer +
    scheduler snapshots are too big to keep per-step; evals per epoch
    is the cadence knob. `latest.pt` is rewritten atomically after
    every eval; a new best metric additionally writes `best.pt`.
  - **On every eval** the full `MetricsReport` is appended to
    `{run_dir}/metrics.jsonl` before any console output.
  - **On every training step** the per-step report (split `train`,
    window `batch`) is appended to the same file. Running /
    epoch / overall windows are included in each eval report.
  - **Training exits** (normal, exception, signal) → final save in a
    `finally` block at the trainer level. Resume picks up from there.
  - No metric is ever printed that isn't also in `metrics.jsonl`.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

from .loss import LossConfig, LossResult
from .model import ModelConfig


# ─────────────────────────── run identity ─────────────────────────────

@dataclass(frozen=True, slots=True)
class RunSpec:
    """Identifies a run on disk. Separate from `TrainerConfig` so the
    same hyperparameters can be replayed under different run names
    (different seeds, different machines).
    """
    root: Path
    name: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoints_dir / "latest.pt"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints_dir / "best.pt"

    def step_checkpoint(self, step: int) -> Path:
        return self.checkpoints_dir / f"step_{step}.pt"

    def ensure(self) -> None:
        """Create the directory layout if missing."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── trainer config ───────────────────────────

@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Orchestration hyperparameters. Not about the model or the loss —
    those have their own config types. Subclass for extra fields only
    if you genuinely add new knobs; 90% of runs fit this shape."""
    epochs: int = 50
    batch_size: int = 48
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    evals_per_epoch: int = 4
    amp: bool = False
    num_workers: int = 0
    seed: int = 42

    # Persistence. Checkpoints are written at eval boundaries only (the
    # `evals_per_epoch` knob above sets the cadence). `latest.pt` is
    # overwritten after every eval; `best.pt` is rewritten when
    # `metric_to_watch` improves. No step-based checkpoint because the
    # model+optimizer payload is large — if you want finer resume
    # granularity, raise `evals_per_epoch`.
    # Key into the dict returned by the eval pass — NOT the namespaced
    # form that `metrics.jsonl` writes (the logger adds the
    # `val/single/…` prefix separately). See `_run_eval` for the raw
    # keys: `loss`, `hard_ce`, `onset/hit`, etc.
    metric_to_watch: str = "loss"
    metric_lower_is_better: bool = True


# ─────────────────────────── runtime state ────────────────────────────

@dataclass
class TrainingState:
    """Mutable state the trainer mutates as training advances. Snapshotted
    into every checkpoint so resume is lossless at the epoch/step level.
    """
    epoch: int = 0
    step: int = 0
    samples_seen: int = 0
    best_metric: float | None = None
    best_metric_step: int | None = None
    last_eval_metrics: dict[str, float] = field(default_factory=dict)
    started_at: str = ""


# ─────────────────────────── checkpoint metadata ──────────────────────

@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    """What every checkpoint records alongside the tensor state dicts.
    Everything needed to reconstruct the model class and resume training."""
    model_class: str           # fully-qualified "module:Class" for `_import_symbol`
    loss_class: str
    model_config: ModelConfig
    loss_config: LossConfig
    trainer_config: TrainerConfig
    training_state: TrainingState
    created_at: str


# ─────────────────────────── trainer hooks ────────────────────────────

class TrainerHook(ABC):
    """Optional observer plugged into the training loop.

    All methods are no-op defaults; hooks override only what they need.
    The trainer guarantees each is called exactly once per event, in
    the documented order, even under exceptions.

    Standard hooks live under `training/hooks.py` (CheckpointHook,
    MetricLoggerHook, ConsoleLoggerHook, EarlyStoppingHook). The
    trainer installs a default set that enforces the persistence
    invariants above unless the user disables them.
    """

    def on_train_start(self, state: TrainingState, spec: RunSpec) -> None:
        """Called once, before any epoch runs."""

    def on_epoch_start(self, state: TrainingState) -> None:
        """Called before each epoch's forward passes."""

    def on_step_end(
        self, state: TrainingState, train_loss: LossResult,
    ) -> None:
        """Called after every optimizer step. `state.step` is the
        newly-incremented value."""

    def on_eval_end(
        self, state: TrainingState, val_metrics: dict[str, float],
    ) -> None:
        """Called after each evaluation pass. `state.last_eval_metrics`
        is set to `val_metrics` by the trainer before this fires."""

    def on_epoch_end(self, state: TrainingState) -> None:
        """Called after the epoch's steps complete (but before the final
        eval of that epoch, if scheduled)."""

    def on_train_end(
        self, state: TrainingState, exc: BaseException | None,
    ) -> None:
        """Called exactly once, in a `finally` block. `exc` is the
        exception that terminated training if any — hooks MUST tolerate
        partial state (may have 0 steps, no eval data)."""
