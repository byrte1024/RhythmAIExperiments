"""Shared infrastructure for CLIs that assemble a `ChartPredictor` from
a spec JSON. Extracted from `cli/infer.py` so both `infer.py` and
`infer_corpus.py` consume the same resolver / builder logic without
copy-paste drift.

Private (leading underscore) — imported only by sibling CLIs.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

from ..domain.inference import ChartPredictor
from ..domain.model import Model
from ..domain.training import CheckpointMeta
from ..inference.loader import load_model_from_checkpoint


# ─────────────────────────── class / config resolution ────────────────

def resolve_class(spec: str) -> type:
    """``module:Class`` → class object. Same convention the checkpoint
    loader uses."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"bad class spec: {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def build_config(node: dict[str, Any]) -> Any:
    """``{__class__, ...fields}`` → instance of that dataclass.

    Nested dataclass fields are recursed into when their value is a
    dict that itself carries ``__class__``. Enum-valued fields (e.g.
    `OnsetKind`) are decoded from string names.

    Field-type resolution uses ``typing.get_type_hints`` because every
    config dataclass in the repo uses ``from __future__ import
    annotations``, which stores field types as strings rather than
    class objects. Without this, the enum-coercion branch would silently
    leave string values in place and downstream code would attribute-
    error (e.g. ``chart.save`` accessing ``onset.kind.value`` on a
    string).
    """
    import typing
    cls_spec = node.get("__class__")
    if cls_spec is None:
        raise ValueError(
            f"config node missing __class__: keys={sorted(node)!r}"
        )
    cls = resolve_class(cls_spec)
    if not is_dataclass(cls):
        raise TypeError(f"{cls_spec!r} is not a dataclass")

    resolved = typing.get_type_hints(cls)
    fieldmap = {f.name: f for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in node.items():
        if key == "__class__" or key.startswith("_"):
            continue
        if key not in fieldmap:
            raise ValueError(
                f"{cls_spec}: unknown field {key!r} "
                f"(known: {sorted(fieldmap)!r})"
            )
        type_hint = resolved.get(key, fieldmap[key].type)
        kwargs[key] = _coerce_field_value(type_hint, value)
    return cls(**kwargs)


def _coerce_field_value(type_hint: Any, value: Any) -> Any:
    """Tuples and enums only — the two coercions config JSON routinely
    needs. Everything else is passed through."""
    if isinstance(value, dict) and "__class__" in value:
        return build_config(value)
    # Enum coercion — pull the enum member matching the string name.
    # Handles both the direct-class form (resolved via get_type_hints)
    # and Optional[Enum] / Enum | None via typing.get_args.
    if isinstance(value, str):
        import enum
        import typing
        candidates: list[type] = []
        if isinstance(type_hint, type):
            candidates.append(type_hint)
        for arg in typing.get_args(type_hint) or ():
            if isinstance(arg, type):
                candidates.append(arg)
        for cand in candidates:
            try:
                if issubclass(cand, enum.Enum):
                    return cand[value]
            except TypeError:
                continue
    if isinstance(value, list) and value and isinstance(value[0], list):
        return tuple(tuple(inner) for inner in value)
    return value


def build_component(node: dict[str, Any]) -> Any:
    """``{__class__: "m:Cls", config: {...}}`` → ``Cls(config=cfg)``.

    The class constructor is invoked with a single positional config
    argument — every concrete ABC in taiko2 follows this convention
    (AudioSampler, EventSampler, ARDecoder, ARInputBuilder).
    """
    cls = resolve_class(node["__class__"])
    cfg = build_config(node["config"])
    return cls(cfg)


# ─────────────────────────── spec IO ──────────────────────────────────

REQUIRED_SPEC_KEYS: frozenset[str] = frozenset({
    "checkpoint", "predictor", "decoder",
    "input_builder", "audio_sampler", "event_sampler",
})


def load_spec(
    *, config: Path | None, config_json: str | None,
) -> dict[str, Any]:
    if config is not None and config_json is not None:
        raise SystemExit("pass one of --config / --config-json, not both")
    if config is not None:
        text = Path(config).read_text(encoding="utf-8")
    elif config_json is not None:
        text = config_json
    else:
        raise SystemExit("one of --config / --config-json is required")
    spec = json.loads(text)
    missing = REQUIRED_SPEC_KEYS - set(spec)
    if missing:
        raise SystemExit(f"spec missing keys: {sorted(missing)!r}")
    return spec


# ─────────────────────────── predictor assembly ───────────────────────

def assemble_predictor(
    *,
    spec: dict[str, Any],
    device: torch.device,
    per_step_log_path: Path | None = None,
) -> tuple[ChartPredictor, CheckpointMeta]:
    """Build a ready-to-run predictor from a spec dict. Returns the
    predictor along with the checkpoint's `CheckpointMeta` so callers
    can read `training_state.step`, the model / loss class strings,
    etc. without reloading the checkpoint.

    If `per_step_log_path` is supplied and the predictor's config
    carries a `per_step_log_path` field, the path is attached.
    """
    ckpt_path = Path(spec["checkpoint"])
    model, _loss, meta = load_model_from_checkpoint(ckpt_path, device=device)
    model.eval()

    decoder = build_component(spec["decoder"])
    # Some decoders (e.g. DiffusionDecoder) need a reference to the
    # model to wire their internal sampler against the model's
    # process / denoiser. The bind_model hook is opt-in.
    if hasattr(decoder, "bind_model"):
        decoder.bind_model(model)
    input_builder = build_component(spec["input_builder"])
    if hasattr(decoder, "bind_input_builder"):
        decoder.bind_input_builder(input_builder)
    audio_sampler = build_component(spec["audio_sampler"])
    event_sampler = build_component(spec["event_sampler"])

    predictor_cls = resolve_class(spec["predictor"]["__class__"])
    pred_cfg = build_config(spec["predictor"]["config"])
    if per_step_log_path is not None and hasattr(pred_cfg, "per_step_log_path"):
        pred_cfg = dataclasses.replace(
            pred_cfg, per_step_log_path=per_step_log_path,
        )

    predictor = predictor_cls(
        pred_cfg,
        model=model,
        decoder=decoder,
        input_builder=input_builder,
        audio_sampler=audio_sampler,
        event_sampler=event_sampler,
        device=device,
    )
    return predictor, meta


def assemble_predictor_with_model(
    *,
    spec: dict[str, Any],
    model: Model,
    device: torch.device,
    per_step_log_path: Path | None = None,
) -> ChartPredictor:
    """Like `assemble_predictor`, but takes an already-live Model
    instance instead of loading from the spec's checkpoint. Used by
    the training-loop `InferCorpusHook` to reuse the model that's
    currently in memory rather than paying a disk round-trip every
    eval.

    The spec's `checkpoint` field is read but NOT used to load weights
    — only the other components (decoder / input_builder / samplers /
    predictor config) are built from the spec.
    """
    decoder = build_component(spec["decoder"])
    if hasattr(decoder, "bind_model"):
        decoder.bind_model(model)
    input_builder = build_component(spec["input_builder"])
    if hasattr(decoder, "bind_input_builder"):
        decoder.bind_input_builder(input_builder)
    audio_sampler = build_component(spec["audio_sampler"])
    event_sampler = build_component(spec["event_sampler"])

    predictor_cls = resolve_class(spec["predictor"]["__class__"])
    pred_cfg = build_config(spec["predictor"]["config"])
    if per_step_log_path is not None and hasattr(pred_cfg, "per_step_log_path"):
        pred_cfg = dataclasses.replace(
            pred_cfg, per_step_log_path=per_step_log_path,
        )

    return predictor_cls(
        pred_cfg,
        model=model,
        decoder=decoder,
        input_builder=input_builder,
        audio_sampler=audio_sampler,
        event_sampler=event_sampler,
        device=device,
    )
