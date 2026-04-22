"""Layer 2: dataset domain types.

Pure interfaces and configs for the audio- and event-sampling side of the
pipeline. Concrete sampler implementations live under `osu.taiko2.samplers`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .beatmap import Onset, OnsetBinned


# ─────────────────────────── audio sampler ────────────────────────────

@dataclass(frozen=True, slots=True)
class AudioSamplerConfig:
    """Base: every sampler operates at a known sample rate.

    Default 22000 Hz: divides evenly into common bin rates (200/400/500 bins
    per second → exact integer hop_lengths), avoiding the sub-ms drift that
    22050 Hz produced against a 5 ms event grid. See exp 14 for the bug
    this default is designed to prevent.
    """
    sample_rate: int = 22000


@dataclass(frozen=True, slots=True)
class MelSamplerConfig(AudioSamplerConfig):
    """Log-mel sampler config.

    Hop can be specified as one of three equivalent fields:
      - `hop_length`: samples (torchaudio's native parameter).
      - `hop_ms`: target bin duration in ms (converted via sample_rate).
      - `hop_divisor`: integer N for `hop_length = sample_rate // N`,
         giving an exact-integer bin rate of N bins/s (lesson of exp 14 —
         non-integer bin_ms like 4.9887ms accumulates drift).

    At most one may be set. Default → `hop_length=110`, which at the
    default `sample_rate=22000` yields exactly 5.000 ms / 200 fps.
    """
    n_fft: int = 2048
    hop_length: int | None = None
    hop_ms: float | None = None
    hop_divisor: int | None = None
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 8000.0
    power: float = 2.0
    top_db: float = 80.0

    def __post_init__(self):
        given = [x for x in (self.hop_length, self.hop_ms, self.hop_divisor)
                 if x is not None]
        if len(given) > 1:
            raise ValueError(
                "MelSamplerConfig: specify at most one of hop_length, "
                f"hop_ms, hop_divisor (got {len(given)})."
            )
        if not given:
            object.__setattr__(self, "hop_length", 110)

    @property
    def effective_hop_length(self) -> int:
        """Resolve hop_length regardless of which field the user set."""
        if self.hop_length is not None:
            return int(self.hop_length)
        if self.hop_ms is not None:
            return int(round(self.hop_ms * self.sample_rate / 1000.0))
        return int(self.sample_rate // self.hop_divisor)


class AudioSampler(ABC):
    """Turns audio into a (features, time) 2D float32 array.

    Contract: any concrete sampler — log-mel, CQT, MFCC, raw spectrogram,
    learned encoder — produces a fixed-frame-rate 2D representation with
    known frame duration and feature count.

    Subclasses implement `_transform(waveform)`; file and segment loading
    are shared in the base class.
    """
    config: AudioSamplerConfig

    @abstractmethod
    def _transform(self, waveform: np.ndarray) -> np.ndarray:
        """(N,) float32 mono waveform at `config.sample_rate` → (F, T) float32."""
        ...

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Size of the feature axis F."""
        ...

    @property
    @abstractmethod
    def frame_ms(self) -> float:
        """Duration of one time frame in milliseconds."""
        ...

    def sample(self, audio_path: Path) -> np.ndarray:
        """Load a full audio file and return its (F, T) features."""
        import librosa
        y, _ = librosa.load(str(audio_path), sr=self.config.sample_rate, mono=True)
        return self._transform(y.astype(np.float32))

    def sample_waveform(
        self, waveform: np.ndarray, sample_rate: int,
    ) -> np.ndarray:
        """Process an in-memory waveform. Resamples to `config.sample_rate`
        if `sample_rate` differs. Mono expected; (C, N) arrays are averaged.
        """
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        wav = np.ascontiguousarray(waveform, dtype=np.float32)

        if sample_rate != self.config.sample_rate:
            import librosa
            wav = librosa.resample(
                wav, orig_sr=sample_rate, target_sr=self.config.sample_rate,
            ).astype(np.float32)

        return self._transform(wav)

    def sample_segment(
        self, audio_path: Path, start_ms: float, end_ms: float,
    ) -> np.ndarray:
        """Load a time segment [start_ms, end_ms) from `audio_path`.

        Returns (F, T') features covering the requested range; T' is the
        number of frames produced by the transform for that segment length.
        """
        if end_ms <= start_ms:
            raise ValueError(f"end_ms ({end_ms}) must be > start_ms ({start_ms})")

        import librosa
        offset_s = max(0.0, start_ms / 1000.0)
        duration_s = (end_ms - start_ms) / 1000.0
        y, _ = librosa.load(
            str(audio_path), sr=self.config.sample_rate, mono=True,
            offset=offset_s, duration=duration_s,
        )
        return self._transform(y.astype(np.float32))


# ─────────────────────────── event sampler ────────────────────────────

@dataclass(frozen=True, slots=True)
class EventSamplerConfig:
    """Time-quantization grid for onsets.

    Specify at most one of:
      - `bins_per_second`: float, direct rate.
      - `bin_ms`: float, direct bin duration in ms.
      - `divisor`: integer N for `bin_ms = 1000 / N` (i.e. N bins/s with
         integer exactness — avoids the sub-ms drift that bit experiment
         14's data alignment).

    Default → `divisor=200` (5.000 ms per bin, matches the default audio
    frame rate of 200 fps at sample_rate=22000, hop_length=110).
    """
    bins_per_second: float | None = None
    bin_ms: float | None = None
    divisor: int | None = None

    def __post_init__(self):
        given = [x for x in (self.bins_per_second, self.bin_ms, self.divisor)
                 if x is not None]
        if len(given) > 1:
            raise ValueError(
                "EventSamplerConfig: specify at most one of bins_per_second, "
                f"bin_ms, divisor (got {len(given)})."
            )
        if not given:
            object.__setattr__(self, "divisor", 200)

    @property
    def effective_bin_ms(self) -> float:
        if self.bin_ms is not None:
            return float(self.bin_ms)
        if self.bins_per_second is not None:
            return 1000.0 / self.bins_per_second
        return 1000.0 / self.divisor


class EventSampler(ABC):
    """Quantizes onset timestamps to bin indices on a fixed grid."""
    config: EventSamplerConfig

    @property
    @abstractmethod
    def bin_ms(self) -> float:
        """Milliseconds per bin."""
        ...

    @abstractmethod
    def bin_of(self, time_ms: float) -> int:
        """Map a time in ms to a bin index."""
        ...

    def sample(self, onsets: Iterable[Onset]) -> tuple[OnsetBinned, ...]:
        """Bin a sequence of onsets against this sampler's grid."""
        return tuple(
            OnsetBinned(time_ms=o.time_ms, kind=o.kind, bin=self.bin_of(o.time_ms))
            for o in onsets
        )


# ─────────────────────────── dataset entries ───────────────────────────

@dataclass(frozen=True, slots=True)
class ChartEntry:
    """One chart in a processed dataset. References the sampled audio features
    on disk plus a flattened view of the source Track's metadata.
    """
    chart_id: str
    beatmap_id: str
    beatmapset_id: str
    artist: str
    title: str
    difficulty_version: str
    overall_difficulty: float
    star_rating: float | None
    density_mean: float
    density_peak: int
    density_std: float
    duration_s: float
    total_events: int
    audio_filename: str
    features_path: Path
    n_frames: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    name: str
    created_at: str
    sampler_config: AudioSamplerConfig
    charts: tuple[ChartEntry, ...]
