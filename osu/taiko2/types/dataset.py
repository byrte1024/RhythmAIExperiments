"""Layer 2: dataset domain types.

The audio-feature side of the pipeline. `AudioSampler` abstracts any method
that turns an audio file into a fixed-frame (F, T) 2D array; `MelSampler` is
the log-mel implementation. `ChartEntry` and `DatasetManifest` describe the
processed-on-disk form of a dataset.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .beatmap import Onset, OnsetBinned


# ─────────────────────────── sampler configs ───────────────────────────

@dataclass(frozen=True, slots=True)
class AudioSamplerConfig:
    """Base: every sampler operates at a known sample rate."""
    sample_rate: int


@dataclass(frozen=True, slots=True)
class MelSamplerConfig(AudioSamplerConfig):
    n_fft: int = 2048
    hop_length: int = 110
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 8000.0
    power: float = 2.0
    top_db: float = 80.0


# ─────────────────────────── sampler interface ─────────────────────────

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


class MelSampler(AudioSampler):
    """Log-mel spectrogram sampler.

    Produces (n_mels, T) float32 where T = ceil(audio_samples / hop_length).
    """

    def __init__(self, config: MelSamplerConfig):
        self.config = config
        self._mel_transform = None
        self._amp_to_db = None

    @property
    def n_features(self) -> int:
        return self.config.n_mels

    @property
    def frame_ms(self) -> float:
        return self.config.hop_length / self.config.sample_rate * 1000.0

    def _ensure_modules(self):
        if self._mel_transform is None:
            import torchaudio
            cfg = self.config
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
                f_min=cfg.f_min,
                f_max=cfg.f_max,
                power=cfg.power,
            )
            self._amp_to_db = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=cfg.top_db,
            )

    def _transform(self, waveform: np.ndarray) -> np.ndarray:
        import torch
        self._ensure_modules()
        wav = torch.from_numpy(waveform)
        with torch.no_grad():
            mel = self._amp_to_db(self._mel_transform(wav))
        return mel.numpy().astype(np.float32)


# ─────────────────────────── event sampler ────────────────────────────

@dataclass(frozen=True, slots=True)
class EventSamplerConfig:
    """Time-quantization grid for onsets. Framerate in bins per second.

    May be set to match an AudioSampler's frame rate (e.g. sample_rate /
    hop_length) or be independent — coarser for bar-level targets, finer
    for sub-frame precision.
    """
    bins_per_second: float


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


class FixedRateEventSampler(EventSampler):
    """Uniform grid at `config.bins_per_second`. Bin = floor(time_ms / bin_ms)."""

    def __init__(self, config: EventSamplerConfig):
        self.config = config

    @property
    def bin_ms(self) -> float:
        return 1000.0 / self.config.bins_per_second

    def bin_of(self, time_ms: float) -> int:
        return int(time_ms / self.bin_ms)


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
