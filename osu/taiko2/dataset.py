"""Build a DatasetManifest from Packs using an AudioSampler + EventSampler.

Orchestration only — file-format details live in `persistence/` and parsing
lives in `parsing/`. Audio source resolution is a caller-provided callable
so the builder stays decoupled from .osz specifics.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from .persistence.events import save_events
from .persistence.features import save_features
from .persistence.manifest import save_manifest
from .domain.beatmap import AudioRef, Pack, Track
from .domain.dataset import (
    AudioSampler,
    ChartEntry,
    DatasetManifest,
    EventSampler,
)

WaveformLoader = Callable[[Pack, AudioRef], "tuple[np.ndarray, int] | None"]


def _safe_filename(s: str, max_len: int = 120) -> str:
    for ch in '<>:"/\\|?*\n\r':
        s = s.replace(ch, "_")
    s = s.strip(". ")
    if len(s) > max_len:
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
        s = s[: max_len - 9] + "_" + h
    return s


def _features_stem(pack: Pack, audio: AudioRef) -> str:
    return _safe_filename(f"{pack.basename}__{audio.filename}")


def _chart_id(pack: Pack, track: Track) -> str:
    return f"{pack.basename} [{track.difficulty.version}]"


def _build_chart_entry(
    pack: Pack,
    track: Track,
    audio_frames: int,
    features_path: Path,
    features_root: Path,
) -> ChartEntry:
    return ChartEntry(
        chart_id=_chart_id(pack, track),
        beatmap_id=track.beatmap_id,
        beatmapset_id=track.beatmapset_id,
        artist=track.artist,
        title=track.title,
        difficulty_version=track.difficulty.version,
        overall_difficulty=track.difficulty.overall_difficulty,
        star_rating=track.difficulty.star_rating,
        density_mean=track.density.mean,
        density_peak=track.density.peak,
        density_std=track.density.std,
        duration_s=track.density.duration_s,
        total_events=track.density.total_events,
        audio_filename=track.audio.filename,
        features_path=features_path.relative_to(features_root),
        n_frames=audio_frames,
    )


def build_dataset(
    packs: Iterable[Pack],
    audio_sampler: AudioSampler,
    event_sampler: EventSampler,
    load_waveform: WaveformLoader,
    out_dir: Path,
    name: str,
    *,
    progress: bool = True,
) -> DatasetManifest:
    """Process a collection of Packs into an on-disk dataset.

    Steps per pack:
      1. For each unique `AudioRef`, `load_waveform(pack, audio)` returns
         `(waveform, sample_rate)` (or `None` to skip).
      2. Run `audio_sampler.sample_waveform(...)` → (F, T) features,
         persisted under `out_dir/features/`.
      3. For each `Track`, bin onsets with `event_sampler.sample(...)`,
         persist under `out_dir/events/`, and append a `ChartEntry`.

    A failure on a single audio or track logs and skips; builds are
    best-effort.
    """
    out_dir = Path(out_dir)
    features_dir = out_dir / "features"
    events_dir = out_dir / "events"
    features_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        try:
            from tqdm import tqdm
            pack_iter = tqdm(list(packs), desc="Building dataset", unit="pack")
        except ImportError:
            pack_iter = packs
    else:
        pack_iter = packs

    entries: list[ChartEntry] = []
    audio_frames_cache: dict[tuple[str, str], int] = {}
    audio_path_cache: dict[tuple[str, str], Path] = {}

    for pack in pack_iter:
        for audio in pack.audio_files:
            key = (pack.basename, audio.filename)
            if key in audio_frames_cache:
                continue

            stem = _features_stem(pack, audio)
            features_path = features_dir / f"{stem}.npy"

            if features_path.exists():
                try:
                    n_frames = int(np.load(features_path, mmap_mode="r").shape[1])
                except Exception:
                    n_frames = -1
                if n_frames > 0:
                    audio_frames_cache[key] = n_frames
                    audio_path_cache[key] = features_path
                    continue

            loaded = load_waveform(pack, audio)
            if loaded is None:
                continue
            waveform, sr = loaded
            try:
                features = audio_sampler.sample_waveform(waveform, sr)
            except Exception as e:
                print(f"  audio sample failed: {pack.basename} / {audio.filename}: {e}")
                continue
            save_features(features, features_path)
            audio_frames_cache[key] = int(features.shape[1])
            audio_path_cache[key] = features_path

        for track in pack.tracks:
            key = (pack.basename, track.audio.filename)
            if key not in audio_frames_cache:
                continue

            chart_id = _chart_id(pack, track)
            events_path = events_dir / f"{_safe_filename(chart_id)}.npz"
            try:
                binned = event_sampler.sample(track.onsets)
                save_events(binned, events_path)
            except Exception as e:
                print(f"  event sample failed: {chart_id}: {e}")
                continue

            entries.append(_build_chart_entry(
                pack=pack,
                track=track,
                audio_frames=audio_frames_cache[key],
                features_path=audio_path_cache[key],
                features_root=out_dir,
            ))

    manifest = DatasetManifest(
        name=name,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        sampler_config=audio_sampler.config,
        charts=tuple(entries),
    )
    save_manifest(manifest, out_dir / "manifest.json")
    return manifest
