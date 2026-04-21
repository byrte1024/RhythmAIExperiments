"""Build a taiko2 dataset from a directory of .osz packs.

Accepts sampler implementations by short alias (``mel``, ``fixed_rate``) or
by fully-qualified ``module:ClassName`` path. Sampler configs are given as
JSON — either inline (``--audio-config '{...}'``) or as a file path
(``--audio-config path/to/config.json``). Any field not present in the JSON
falls back to the config dataclass default.

Output layout::

    osu/taiko2/datasets/{name}/
        manifest.json
        features/{stem}.npy      # float16 (F, T)
        events/{chart_id}.npz    # bins + times_ms + kind_ids

Usage::

    python -m osu.taiko2.prepare_dataset \\
        --name taiko2_v1 \\
        --charts-dir osu/taiko/charts \\
        --audio-config '{"sample_rate": 22050}' \\
        --event-config '{"bins_per_second": 200.454}'
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .dataset import build_dataset
from .parsing import load_audio_waveform, load_pack
from .domain.beatmap import AudioRef, Pack
from .domain.dataset import AudioSampler, EventSampler

# ─────────────────────────── sampler registry ──────────────────────────
# Short aliases map to (sampler_spec, config_spec) where each spec is
# "module:Class". Fully-qualified paths bypass the registry entirely.

AUDIO_SAMPLERS: dict[str, tuple[str, str]] = {
    "mel": (
        "osu.taiko2.samplers:MelSampler",
        "osu.taiko2.domain:MelSamplerConfig",
    ),
}

EVENT_SAMPLERS: dict[str, tuple[str, str]] = {
    "fixed_rate": (
        "osu.taiko2.samplers:FixedRateEventSampler",
        "osu.taiko2.domain:EventSamplerConfig",
    ),
}


def _import_symbol(spec: str) -> Any:
    """Resolve ``module:Name`` (preferred) or ``module.Name`` to an object."""
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
    else:
        module_name, _, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Bad symbol spec: {spec!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"{module_name!r} has no attribute {attr!r}") from e


def _resolve_sampler(
    selector: str, registry: dict[str, tuple[str, str]],
) -> tuple[type, type]:
    """Return (sampler_class, config_class) from alias or dotted path."""
    if selector in registry:
        sampler_spec, config_spec = registry[selector]
    else:
        # Fully-qualified: expect "sampler_spec|config_spec"
        if "|" not in selector:
            raise ValueError(
                f"Unknown sampler {selector!r}. Use a registered alias "
                f"({', '.join(sorted(registry))}) or "
                f"'sampler_module:Class|config_module:Class'."
            )
        sampler_spec, config_spec = selector.split("|", 1)
    return _import_symbol(sampler_spec), _import_symbol(config_spec)


def _load_config_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_config(config_class: type, values: dict[str, Any]) -> Any:
    """Instantiate a sampler config dataclass from a (possibly partial) dict."""
    known = {f.name for f in fields(config_class)}
    missing = [f.name for f in fields(config_class)
               if f.default is f.default_factory is None  # type: ignore[attr-defined]
               and f.name not in values]
    unknown = [k for k in values if k not in known]
    if unknown:
        raise ValueError(
            f"Unknown fields for {config_class.__name__}: {unknown}. "
            f"Valid: {sorted(known)}"
        )
    filtered = {k: v for k, v in values.items() if k in known}
    try:
        return config_class(**filtered)
    except TypeError as e:
        raise ValueError(
            f"Failed to construct {config_class.__name__}: {e}. "
            f"Required fields not provided via config JSON: {missing}"
        ) from e


def _instantiate_sampler(
    selector: str,
    config_json: str | None,
    registry: dict[str, tuple[str, str]],
):
    sampler_cls, config_cls = _resolve_sampler(selector, registry)
    values = _load_config_json(config_json)
    config = _build_config(config_cls, values)
    return sampler_cls(config)


# ─────────────────────────── pack loading ──────────────────────────────

def _scan_charts_dir(charts_dir: Path) -> list[Pack]:
    osz_paths = sorted(charts_dir.rglob("*.osz"))
    print(f"Scanning {len(osz_paths)} .osz files under {charts_dir}")

    packs: list[Pack] = []
    errors = 0
    for p in tqdm(osz_paths, desc="Parsing packs", unit="pack"):
        pack = load_pack(p)
        if pack is None:
            errors += 1
            continue
        packs.append(pack)

    n_tracks = sum(len(p.tracks) for p in packs)
    n_audio = sum(len(p.audio_files) for p in packs)
    print(f"Parsed {len(packs)} packs / {n_tracks} tracks / {n_audio} audio files "
          f"({errors} archives skipped)")
    return packs


def _make_waveform_loader(target_sr: int):
    """Closure matching build_dataset's ``load_waveform`` signature."""
    def _load(pack: Pack, audio: AudioRef):
        try:
            return load_audio_waveform(pack.source_path, audio.filename, target_sr)
        except Exception as e:
            tqdm.write(f"  audio decode failed: {pack.basename}/{audio.filename}: {e}")
            return None
    return _load


# ─────────────────────────── CLI ───────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a taiko2 dataset from a directory of .osz packs.",
    )
    p.add_argument("--name", required=True,
                   help="Dataset name; output written to {out_dir}/{name}/")
    p.add_argument("--charts-dir", type=Path, required=True,
                   help="Directory containing .osz archives (recursed).")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent / "datasets",
                   help="Root output directory (default: osu/taiko2/datasets)")

    p.add_argument("--audio-sampler", default="mel",
                   help=f"Audio sampler. Aliases: {', '.join(AUDIO_SAMPLERS)} "
                        f"— or 'sampler_mod:Class|config_mod:Class'.")
    p.add_argument("--audio-config", default=None,
                   help="JSON literal or path to a JSON file with config "
                        "overrides for the audio sampler config class.")

    p.add_argument("--event-sampler", default="fixed_rate",
                   help=f"Event sampler. Aliases: {', '.join(EVENT_SAMPLERS)} "
                        f"— or 'sampler_mod:Class|config_mod:Class'.")
    p.add_argument("--event-config", default=None,
                   help="JSON literal or path to a JSON file with config "
                        "overrides for the event sampler config class.")

    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm bars.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    charts_dir = args.charts_dir.resolve()
    if not charts_dir.is_dir():
        print(f"ERROR: --charts-dir {charts_dir} is not a directory", file=sys.stderr)
        return 2

    audio_sampler: AudioSampler = _instantiate_sampler(
        args.audio_sampler, args.audio_config, AUDIO_SAMPLERS,
    )
    event_sampler: EventSampler = _instantiate_sampler(
        args.event_sampler, args.event_config, EVENT_SAMPLERS,
    )

    print(f"Audio sampler: {type(audio_sampler).__name__} {audio_sampler.config}")
    print(f"Event sampler: {type(event_sampler).__name__} {event_sampler.config}")
    print(f"Frame rate:    {audio_sampler.frame_ms:.4f} ms/frame "
          f"({1000 / audio_sampler.frame_ms:.3f} fps)")
    print(f"Bin rate:      {event_sampler.bin_ms:.4f} ms/bin "
          f"({1000 / event_sampler.bin_ms:.3f} bins/s)")

    packs = _scan_charts_dir(charts_dir)
    if not packs:
        print("No valid taiko packs found; nothing to do.", file=sys.stderr)
        return 1

    load_waveform = _make_waveform_loader(audio_sampler.config.sample_rate)

    out_dir = (args.out_dir / args.name).resolve()
    manifest = build_dataset(
        packs=packs,
        audio_sampler=audio_sampler,
        event_sampler=event_sampler,
        load_waveform=load_waveform,
        out_dir=out_dir,
        name=args.name,
        progress=not args.no_progress,
    )

    print()
    print(f"Dataset '{manifest.name}' written to {out_dir}")
    print(f"  Charts:   {len(manifest.charts)}")
    print(f"  Manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
