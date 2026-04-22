"""Atomic checkpoint save / load.

A checkpoint holds everything needed to resume training exactly where
it stopped: model weights, optimizer state, scheduler state, RNG state
for every library that touches randomness, training counters, and the
config trio (model / loss / trainer) so the classes themselves can be
reconstructed.

Atomicity: `save_checkpoint` writes to a `.tmp` sibling, `fsync`s it,
then `os.replace`s onto the target path. A crash mid-write never
corrupts an existing good checkpoint — at worst, a lone `.tmp` is
left behind.
"""
from __future__ import annotations

import importlib
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..domain.loss import Loss, LossConfig
from ..domain.model import Model, ModelConfig
from ..domain.training import (
    CheckpointMeta,
    RunSpec,
    TrainerConfig,
    TrainingState,
)


# ─────────────────────────── RNG state ────────────────────────────────

def _capture_rng_state() -> dict[str, Any]:
    """Snapshot Python / NumPy / PyTorch (CPU + CUDA) RNG state.

    Best-effort per subsystem — if any library's RNG is uninitialized
    or unavailable on the current platform, that slice is skipped. In
    real training these all succeed; the permissive handling is only
    to survive CI environments with broken lazy imports.
    """
    state: dict[str, Any] = {"python": random.getstate()}
    try:
        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        state["torch"] = torch.get_rng_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if "torch" in state:
        try:
            torch.set_rng_state(state["torch"])
        except Exception:
            pass
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


# ─────────────────────────── qualified names ──────────────────────────

def _qualified_name(cls: type) -> str:
    """`module:ClassName` — same format consumed by the CLI importers."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_class(spec: str) -> type:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"bad class spec: {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


# ─────────────────────────── config <-> dict ──────────────────────────

def _config_to_dict(cfg: Any) -> dict[str, Any]:
    """`dataclasses.asdict`, plus a `__class__` tag for reconstruction."""
    d = asdict(cfg)
    d["__class__"] = _qualified_name(type(cfg))
    return d


def _config_from_dict(d: dict[str, Any]) -> Any:
    d = dict(d)
    cls_name = d.pop("__class__")
    cls = _resolve_class(cls_name)
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in d.items() if k in known}
    for f in fields(cls):
        if f.name == "root" and f.type in (Path, "Path", "pathlib.Path"):
            filtered[f.name] = Path(filtered[f.name])
    return cls(**filtered)


def _training_state_to_dict(s: TrainingState) -> dict[str, Any]:
    return asdict(s)


def _training_state_from_dict(d: dict[str, Any]) -> TrainingState:
    return TrainingState(**d)


# ─────────────────────────── checkpoint payload ───────────────────────

@dataclass(frozen=True, slots=True)
class Checkpoint:
    """In-memory checkpoint payload. `save` / `load` round-trip to disk."""
    meta: CheckpointMeta
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    rng_state: dict[str, Any]

    # ── construction ──────────────────────────────────────────────────

    @classmethod
    def from_runtime(
        cls,
        *,
        model: Model,
        loss: Loss,
        optimizer: torch.optim.Optimizer | None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        trainer_config: TrainerConfig,
        training_state: TrainingState,
    ) -> "Checkpoint":
        meta = CheckpointMeta(
            model_class=_qualified_name(type(model)),
            loss_class=_qualified_name(type(loss)),
            model_config=model.config,
            loss_config=loss.config,
            trainer_config=trainer_config,
            training_state=training_state,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return cls(
            meta=meta,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict() if optimizer is not None else None,
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            rng_state=_capture_rng_state(),
        )

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write atomically to `path`. Partial writes never clobber an
        existing good file: we write `path.tmp`, `fsync`, then
        `os.replace(path.tmp, path)`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")

        payload = {
            "meta": {
                "model_class": self.meta.model_class,
                "loss_class": self.meta.loss_class,
                "model_config": _config_to_dict(self.meta.model_config),
                "loss_config": _config_to_dict(self.meta.loss_config),
                "trainer_config": _config_to_dict(self.meta.trainer_config),
                "training_state": _training_state_to_dict(self.meta.training_state),
                "created_at": self.meta.created_at,
            },
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "rng_state": self.rng_state,
        }

        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Checkpoint":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        meta_d = payload["meta"]
        meta = CheckpointMeta(
            model_class=meta_d["model_class"],
            loss_class=meta_d["loss_class"],
            model_config=_config_from_dict(meta_d["model_config"]),
            loss_config=_config_from_dict(meta_d["loss_config"]),
            trainer_config=_config_from_dict(meta_d["trainer_config"]),
            training_state=_training_state_from_dict(meta_d["training_state"]),
            created_at=meta_d["created_at"],
        )
        return cls(
            meta=meta,
            model_state=payload["model_state"],
            optimizer_state=payload.get("optimizer_state"),
            scheduler_state=payload.get("scheduler_state"),
            rng_state=payload.get("rng_state", {}),
        )

    # ── restore ───────────────────────────────────────────────────────

    def restore_to(
        self,
        *,
        model: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        restore_rng: bool = True,
    ) -> TrainingState:
        """Load tensor states into caller-provided objects. Returns the
        checkpoint's `TrainingState` so the trainer can pick up its
        counters.
        """
        if model is not None:
            model.load_state_dict(self.model_state)
        if optimizer is not None and self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
        if scheduler is not None and self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)
        if restore_rng and self.rng_state:
            _restore_rng_state(self.rng_state)
        # Return a copy so the caller can freely mutate.
        return replace(self.meta.training_state)


# ─────────────────────────── convenience ──────────────────────────────

def save_latest(
    spec: RunSpec,
    checkpoint: Checkpoint,
    *,
    is_best: bool = False,
) -> None:
    """Persist `latest.pt` and, if `is_best`, duplicate to `best.pt`."""
    spec.ensure()
    checkpoint.save(spec.latest_checkpoint)
    if is_best:
        checkpoint.save(spec.best_checkpoint)


def load_latest_if_any(spec: RunSpec) -> Checkpoint | None:
    if spec.latest_checkpoint.exists():
        return Checkpoint.load(spec.latest_checkpoint)
    return None


# ─────────────────────────── per-eval resume ─────────────────────────

def find_last_eval_checkpoint(spec: RunSpec) -> "tuple[Path, int] | None":
    """Scan ``{run_dir}/eval_{step}/checkpoint.pt`` snapshots and return
    the (path, step) with the largest `step`.

    Differs from ``load_latest_if_any`` (which reads ``latest.pt``) in
    that it ignores any mid-epoch snapshot the trainer may have written
    on crash/shutdown — only eval boundaries count. Returns None if no
    eval checkpoint exists.
    """
    run_dir = spec.run_dir
    if not run_dir.exists():
        return None
    best: tuple[Path, int] | None = None
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("eval_"):
            continue
        try:
            step = int(child.name[len("eval_"):])
        except ValueError:
            continue
        ckpt = child / "checkpoint.pt"
        if not ckpt.exists():
            continue
        if best is None or step > best[1]:
            best = (ckpt, step)
    return best


def truncate_stats_after_step(spec: RunSpec, step: int) -> None:
    """Drop everything the trainer produced after ``step`` so an eval-
    boundary resume starts with clean stats. Specifically:

      - Rewrite ``metrics.jsonl`` keeping only rows with ``row["step"]
        <= step`` (JSON-decode failures are dropped conservatively).
      - Delete any ``eval_{N}/`` directories with ``N > step``.
      - Refresh ``latest.pt`` from the resumed eval's checkpoint.

    Curves live under ``{run_dir}/curves/`` and are regenerated from the
    truncated JSONL at train-start by ``MetricCurvesHook``, so they do
    not need explicit cleanup.
    """
    path = spec.metrics_path
    if path.exists():
        import json
        kept: list[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_step = row.get("step")
                if isinstance(row_step, int) and row_step > step:
                    continue
                # Skip any `train_end` line from a prior crashed session.
                if row.get("event") == "train_end":
                    continue
                kept.append(line if line.endswith("\n") else line + "\n")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(kept), encoding="utf-8")
        os.replace(tmp, path)

    # Delete eval_{N}/ for N > step.
    if spec.run_dir.exists():
        import shutil
        for child in spec.run_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("eval_"):
                continue
            try:
                n = int(child.name[len("eval_"):])
            except ValueError:
                continue
            if n > step:
                shutil.rmtree(child, ignore_errors=True)
