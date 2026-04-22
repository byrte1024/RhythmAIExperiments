"""Layer 1: beatmap domain types.

Pure source-side data as it exists in osu! .osz packs. No ML preprocessing,
no bin quantization, no derived features beyond what osu! itself defines.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OnsetKind(Enum):
    DON = "don"
    KA = "ka"
    BIG_DON = "big_don"
    BIG_KA = "big_ka"
    DRUMROLL = "drumroll"
    SPINNER = "spinner"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Onset:
    time_ms: int
    kind: OnsetKind


@dataclass(frozen=True, slots=True)
class OnsetBinned(Onset):
    """An Onset with its time quantized to an event-sampler bin index.

    `bin` is always computed against a specific `EventSampler`'s framerate
    (bins per second). The original `time_ms` is preserved so the mapping
    stays invertible and re-binnable under a different framerate.
    """
    bin: int = 0


@dataclass(frozen=True, slots=True)
class RelativeOnset(OnsetBinned):
    """An OnsetBinned with its position relative to a sampler cursor.

    `cursor_offset = onset.bin - cursor_bin`. Negative values are past the
    cursor, positive are future. The absolute `bin` / `time_ms` are kept
    so a downstream consumer can still reconstruct absolute positions
    without carrying the cursor around.
    """
    cursor_offset: int = 0


@dataclass(frozen=True, slots=True)
class Density:
    mean: float
    peak: int
    std: float
    duration_s: float
    total_events: int


@dataclass(frozen=True, slots=True)
class Difficulty:
    version: str
    overall_difficulty: float
    star_rating: float | None = None


@dataclass(frozen=True, slots=True)
class AudioRef:
    filename: str
    format: str


@dataclass(frozen=True, slots=True)
class Track:
    beatmap_id: str
    beatmapset_id: str
    artist: str
    title: str
    difficulty: Difficulty
    audio: AudioRef
    onsets: tuple[Onset, ...]
    density: Density


@dataclass(frozen=True, slots=True)
class Pack:
    source_path: Path
    basename: str
    beatmapset_id: str
    tracks: tuple[Track, ...]
    audio_files: tuple[AudioRef, ...]
