"""Save/load DatasetManifest to/from JSON.

Sampler config is polymorphic: a `"type"` tag carries the concrete config
class name so the manifest can round-trip through JSON without the caller
knowing which sampler wrote it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ..types.dataset import (
    AudioSamplerConfig,
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)

_CONFIG_REGISTRY: dict[str, type[AudioSamplerConfig]] = {
    "AudioSamplerConfig": AudioSamplerConfig,
    "MelSamplerConfig": MelSamplerConfig,
}


def _config_to_dict(cfg: AudioSamplerConfig) -> dict[str, Any]:
    return {"type": type(cfg).__name__, **asdict(cfg)}


def _config_from_dict(data: dict[str, Any]) -> AudioSamplerConfig:
    data = dict(data)
    tag = data.pop("type", "AudioSamplerConfig")
    cls = _CONFIG_REGISTRY.get(tag)
    if cls is None:
        raise ValueError(f"Unknown sampler config type: {tag!r}")
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)


def _chart_to_dict(c: ChartEntry) -> dict[str, Any]:
    d = asdict(c)
    d["features_path"] = str(c.features_path)
    return d


def _chart_from_dict(d: dict[str, Any]) -> ChartEntry:
    known = {f.name for f in fields(ChartEntry)}
    filtered = {k: v for k, v in d.items() if k in known}
    filtered["features_path"] = Path(filtered["features_path"])
    return ChartEntry(**filtered)


def save_manifest(manifest: DatasetManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "name": manifest.name,
        "created_at": manifest.created_at,
        "sampler_config": _config_to_dict(manifest.sampler_config),
        "charts": [_chart_to_dict(c) for c in manifest.charts],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_manifest(path: Path) -> DatasetManifest:
    with open(Path(path), "r", encoding="utf-8") as f:
        obj = json.load(f)
    return DatasetManifest(
        name=obj["name"],
        created_at=obj["created_at"],
        sampler_config=_config_from_dict(obj["sampler_config"]),
        charts=tuple(_chart_from_dict(c) for c in obj["charts"]),
    )
