"""Audio sampler that produces mel + coincidence summary features.

Outputs ``(n_mels + 13, T)`` float32 — the standard mel spectrogram
concatenated with the 13-row coincidence onset summary. Both share
the same hop_length so frames align exactly.

Drop-in replacement for ``MelSampler`` in dataset prep and inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.coincidence import compute_summary
from ..domain.dataset import AudioSampler, AudioSamplerConfig
from .mel import MelSampler, MelSamplerConfig


@dataclass(frozen=True, slots=True)
class CoincidenceMelSamplerConfig(AudioSamplerConfig):
    sample_rate: int = 22000
    n_fft: int = 2048
    hop_divisor: int = 200
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 8000.0
    power: float = 2.0
    top_db: float = 80.0
    coin_n_bands: int = 64
    coin_n_summary: int = 8


class CoincidenceMelSampler(AudioSampler):
    """Produces ``(n_mels + 13, T)`` features: mel + coincidence."""

    config: CoincidenceMelSamplerConfig

    def __init__(self, config: CoincidenceMelSamplerConfig):
        self.config = config
        self._mel_sampler = MelSampler(MelSamplerConfig(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_divisor=config.hop_divisor,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power,
            top_db=config.top_db,
        ))

    def _transform(self, waveform: np.ndarray) -> np.ndarray:
        mel = self._mel_sampler._transform(waveform)
        hop_length = self.config.sample_rate // self.config.hop_divisor
        coin = compute_summary(
            waveform, self.config.sample_rate,
            hop_length=hop_length,
            n_bands=self.config.coin_n_bands,
            n_summary=self.config.coin_n_summary,
        ).astype(np.float32)

        mel_T = mel.shape[1]
        coin_T = coin.shape[1]
        if coin_T >= mel_T:
            coin = coin[:, :mel_T]
        else:
            padded = np.zeros((coin.shape[0], mel_T), dtype=np.float32)
            padded[:, :coin_T] = coin
            coin = padded

        return np.concatenate([mel, coin], axis=0)

    @property
    def n_features(self) -> int:
        return self.config.n_mels + 5 + self.config.coin_n_summary  # 80 + 5 + 8 = 93

    @property
    def frame_ms(self) -> float:
        return self._mel_sampler.frame_ms
