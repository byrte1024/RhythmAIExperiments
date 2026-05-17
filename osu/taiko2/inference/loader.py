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


def _model_cls_from_config(config: "ModelConfig") -> type:
    """Resolve the concrete Model class from its config type.

    Used when the checkpoint's ``model_class`` points at
    ``torch._dynamo.OptimizedModule`` (from ``torch.compile``) instead
    of the real model class.
    """
    from ..models.event_embedding import EventEmbeddingConfig, EventEmbeddingDetector
    from ..models.framewise_detector import FramewiseDetector, FramewiseDetectorConfig

    _CONFIG_TO_MODEL: dict[type, type] = {
        FramewiseDetectorConfig: FramewiseDetector,
        EventEmbeddingConfig: EventEmbeddingDetector,
    }
    # Lazy imports for optional model types.
    try:
        from ..models.diffusion_detector import DiffusionDetector, DiffusionDetectorConfig
        _CONFIG_TO_MODEL[DiffusionDetectorConfig] = DiffusionDetector
    except ImportError:
        pass
    try:
        from ..models.framewise_diffusion_detector import (
            FramewiseDiffusionDetector, FramewiseDiffusionDetectorConfig,
        )
        _CONFIG_TO_MODEL[FramewiseDiffusionDetectorConfig] = FramewiseDiffusionDetector
    except ImportError:
        pass
    try:
        from ..models.ratio_detector import RatioDetector, RatioDetectorConfig
        _CONFIG_TO_MODEL[RatioDetectorConfig] = RatioDetector
    except ImportError:
        pass

    cfg_type = type(config)
    # Check exact type first, then MRO for subclass matches.
    if cfg_type in _CONFIG_TO_MODEL:
        return _CONFIG_TO_MODEL[cfg_type]
    for cfg_cls, model_cls in _CONFIG_TO_MODEL.items():
        if isinstance(config, cfg_cls):
            return model_cls
    raise TypeError(
        f"cannot resolve Model class for config type "
        f"{cfg_type.__module__}:{cfg_type.__qualname__}; "
        f"add it to inference.loader._model_cls_from_config"
    )


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
    state_dict = checkpoint.model_state

    # torch.compile wraps models in OptimizedModule and prefixes state
    # dict keys with ``_orig_mod.``. Unwrap both at load time so
    # inference never needs torch.compile.
    if model_cls.__name__ == "OptimizedModule":
        model_cls = _model_cls_from_config(checkpoint.meta.model_config)
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {
            k.removeprefix("_orig_mod."): v for k, v in state_dict.items()
        }

    loss_cls = _resolve_class(checkpoint.meta.loss_class)

    model = model_cls(checkpoint.meta.model_config)
    loss = loss_cls(checkpoint.meta.loss_config)

    model.load_state_dict(state_dict)
    model.to(device)

    return model, loss, checkpoint.meta
