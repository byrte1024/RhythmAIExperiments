"""Reconstruct a Model (+ Loss) from a saved checkpoint.

Universal across any `ChartPredictor` — used by AR, framewise, or any
future predictor variant.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import torch

from ..domain.loss import Loss
from ..domain.model import Model
from ..domain.training import CheckpointMeta
from ..persistence.checkpoint import Checkpoint


def _resolve_class(spec: str) -> type:
    """`module:Class` → the class object."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"bad class spec: {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def load_model_from_checkpoint(
    path: Path,
    *,
    device: torch.device | str = torch.device("cpu"),
) -> tuple[Model, Loss, CheckpointMeta]:
    """Load a checkpoint, reconstruct the Model + Loss classes, restore
    weights, move to `device`, return everything.

    Model + Loss classes are stored in `CheckpointMeta.model_class` /
    `loss_class` as fully-qualified ``module:Class`` strings. Both
    classes must be importable in the current process — the typical
    failure mode is a missing import path, fixed by adding the concrete
    module to `sys.path` or installing the package.
    """
    checkpoint = Checkpoint.load(Path(path))
    device = torch.device(device) if isinstance(device, str) else device

    model_cls = _resolve_class(checkpoint.meta.model_class)
    loss_cls = _resolve_class(checkpoint.meta.loss_class)

    model = model_cls(checkpoint.meta.model_config)
    loss = loss_cls(checkpoint.meta.loss_config)

    model.load_state_dict(checkpoint.model_state)
    model.to(device)

    return model, loss, checkpoint.meta
