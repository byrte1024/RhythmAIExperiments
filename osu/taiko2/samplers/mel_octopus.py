"""Audio sampler producing mel + octopus gradient features.

Outputs ``(n_mels + n_octopus_cells, T)`` float32 — the standard mel
spectrogram concatenated with the octopus gradient onset map. Both
share the same hop rate (default 5ms / 200fps) so frames align.

Channel layout:
  rows 0..79        log-mel spectrogram (80 bands)
  rows 80..176      octopus gradient (97 cells, default config)

One dataset build supports multiple experiments:
  - mel only:     feature_rows [0, 80]
  - octopus only: feature_rows [80, 177]
  - dual channel: feature_rows null (all)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.dataset import AudioSampler, AudioSamplerConfig
from ..domain.octopus import compute_gradient
from .mel import MelSampler, MelSamplerConfig


@dataclass(frozen=True, slots=True)
class MelOctopusSamplerConfig(AudioSamplerConfig):
    sample_rate: int = 22000
    # Mel params.
    n_fft: int = 2048
    hop_divisor: int = 200
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 8000.0
    power: float = 2.0
    top_db: float = 80.0
    # Octopus params.
    oct_n_filters: int = 128
    oct_low_freq: float = 50.0
    oct_high_freq: float = 8000.0
    oct_gammatone_order: int = 4
    oct_onset_lookback: int = 3
    oct_sync_window_ms: float = 1.5
    oct_peak_percentile: float = 90.0
    oct_gradient_step: int = 1
    oct_gradient_cell_width_frac: float = 0.25
    oct_gradient_nonlinearity_exp: float = 1.5
    oct_compensate_group_delay: bool = True
    oct_max_workers: int = 8
    # Output row selection. None = all rows (177 for dataset build).
    # [0, 80] = mel only. [80, 177] = octopus only. Used at inference
    # time to match the model's expected input dimension.
    output_rows: tuple[int, int] | None = None


class MelOctopusSampler(AudioSampler):
    """Produces ``(n_mels + n_octopus_cells, T)`` features."""

    config: MelOctopusSamplerConfig

    def __init__(self, config: MelOctopusSamplerConfig):
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

        cfg = self.config
        gradient, _cfs, _n_cells = compute_gradient(
            waveform, cfg.sample_rate,
            n_filters=cfg.oct_n_filters,
            low_freq=cfg.oct_low_freq,
            high_freq=cfg.oct_high_freq,
            order=cfg.oct_gammatone_order,
            hop_ms=1.0,
            max_workers=cfg.oct_max_workers,
            onset_lookback=cfg.oct_onset_lookback,
            sync_window_ms=cfg.oct_sync_window_ms,
            peak_percentile=cfg.oct_peak_percentile,
            gradient_step=cfg.oct_gradient_step,
            gradient_cell_width_frac=cfg.oct_gradient_cell_width_frac,
            gradient_nonlinearity_exp=cfg.oct_gradient_nonlinearity_exp,
            compensate_group_delay=cfg.oct_compensate_group_delay,
        )

        # Max-pool octopus from 1ms to mel frame rate.
        pool = max(1, int(round(self.frame_ms / 1.0)))
        if pool > 1:
            n_cells, t_1ms = gradient.shape
            t_out = t_1ms // pool
            gradient = gradient[:, :t_out * pool].reshape(n_cells, t_out, pool).max(axis=2)

        # Align time axes.
        mel_T = mel.shape[1]
        oct_T = gradient.shape[1]
        if oct_T >= mel_T:
            gradient = gradient[:, :mel_T]
        else:
            padded = np.zeros((gradient.shape[0], mel_T), dtype=np.float32)
            padded[:, :oct_T] = gradient
            gradient = padded

        combined = np.concatenate([mel, gradient.astype(np.float32)], axis=0)
        if self.config.output_rows is not None:
            lo, hi = self.config.output_rows
            combined = combined[lo:hi]
        return combined

    @property
    def _total_features(self) -> int:
        w = max(4, int(self.config.oct_n_filters * self.config.oct_gradient_cell_width_frac))
        step = self.config.oct_gradient_step
        n = 0
        lo = 0
        while lo + w <= self.config.oct_n_filters:
            n += 1
            lo += step
        last_hi = (lo - step) + w if n > 0 else 0
        if n > 0 and last_hi < self.config.oct_n_filters:
            n += 1
        return self.config.n_mels + n

    @property
    def n_features(self) -> int:
        if self.config.output_rows is not None:
            lo, hi = self.config.output_rows
            return hi - lo
        return self._total_features

    @property
    def frame_ms(self) -> float:
        return self._mel_sampler.frame_ms
