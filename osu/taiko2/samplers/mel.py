"""Log-mel spectrogram AudioSampler."""
from __future__ import annotations

import numpy as np

from ..domain.dataset import AudioSampler, MelSamplerConfig


class MelSampler(AudioSampler):
    """Log-mel spectrogram sampler.

    Produces (n_mels, T) float32 where T = ceil(audio_samples / hop_length).
    Torchaudio modules are instantiated lazily and cached across calls so
    repeated `sample_segment` calls don't pay re-initialization cost.
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
        return self.config.effective_hop_length / self.config.sample_rate * 1000.0

    def _ensure_modules(self) -> None:
        if self._mel_transform is not None:
            return
        import torchaudio
        cfg = self.config
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.effective_hop_length,
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
