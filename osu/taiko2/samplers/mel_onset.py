"""Log-mel sampler with sub-band spectral-flux onset rows appended.

Inherits everything from `MelSampler` (audio decode, STFT, mel filterbank,
AmpToDB) and overrides only the post-processing step: after computing
the log-mel spectrogram in dB, this sampler computes sub-band spectral
flux on the log-mel and appends it as additional rows.

The new rows are range-matched to the log-mel dB scale so the conv stem
in the downstream model can treat them as "more mel bands" — same
distribution shape, same init dynamics.

Output shape: ``(n_mels + n_onset_subbands, T)`` float32.

Design notes from #011 / #011b:
  - Sub-band split chosen because per-band rows give the conv stem
    multiple input planes per ODF (one row of activation is ~1 % of
    input among 80 mel bands, which the conv stem may ignore — see
    "ignorance" discussion in #012's pre-run README).
  - Spectral flux specifically because it tied for highest single-
    channel F1 in #011 (0.679 at ±10 frames) and is the cheapest
    candidate.
  - The activation is bucket-pool-friendly but we serve it at the
    raw 5 ms grid here so the model can choose to pool itself or
    use the raw values; bucket pooling can layer on later as a
    config flag if needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.onset_features import normalize_activation, subband_flux
from ..domain.dataset import MelSamplerConfig
from .mel import MelSampler


@dataclass(frozen=True, slots=True)
class MelOnsetSamplerConfig(MelSamplerConfig):
    """``MelSamplerConfig`` + sub-band onset detection function rows.

    Output rows = ``n_mels`` (existing log-mel) + ``n_onset_subbands``
    (sub-band spectral flux), in that order. The downstream model's
    ``n_mels`` config field should be set to the **total** —
    ``MelSamplerConfig.n_mels + n_onset_subbands`` — because the conv
    stem reads the concatenated tensor as one block.

    Range-matching: per-chart 99th-percentile-normalize each sub-band
    row, then linearly map to ``onset_target_db_range`` so the rows
    have the same distribution shape as log-mel. With the default
    ``(-80.0, 0.0)`` matching ``MelSampler``'s ``top_db = 80``, silent
    frames map to -80 dB (matching silent mel) and the 99th-percentile
    activation maps to 0 dB (matching loud mel).
    """
    n_onset_subbands: int = 4
    onset_norm_percentile: float = 99.0
    onset_target_db_range: tuple[float, float] = (-80.0, 0.0)


class MelOnsetSampler(MelSampler):
    """``MelSampler`` with sub-band spectral-flux rows appended.

    The first ``config.n_mels`` rows are the standard log-mel
    spectrogram (identical to ``MelSampler``); the next
    ``config.n_onset_subbands`` rows are sub-band spectral flux on the
    log-mel, range-matched to the same dB scale.

    The augmented features are mmap-able from disk just like plain
    log-mel — no special loader needed downstream. The training and
    inference paths see one feature tensor.
    """

    config: MelOnsetSamplerConfig

    def __init__(self, config: MelOnsetSamplerConfig):
        super().__init__(config)

    @property
    def n_features(self) -> int:
        return self.config.n_mels + self.config.n_onset_subbands

    def _transform(self, waveform: np.ndarray) -> np.ndarray:
        log_mel = super()._transform(waveform)              # (n_mels, T) float32

        # subband_flux + normalize_activation are torch ops; lazy import
        # so cold-load doesn't pay torch's startup cost when we only
        # want the audio decode path (e.g. tests, smoke).
        import torch
        cfg = self.config
        mel_t = torch.from_numpy(log_mel)
        sb = subband_flux(mel_t, n_bands=cfg.n_onset_subbands)  # (n_bands, T)
        normed = normalize_activation(
            sb, percentile=cfg.onset_norm_percentile,
        )                                                    # (n_bands, T) ~ [0, 1+]
        lo, hi = cfg.onset_target_db_range
        sb_dB = lo + normed.clamp(min=0.0, max=1.0) * (hi - lo)
        # Concatenate along the feature axis. Keep float32 to match
        # the existing log-mel dtype contract.
        out = torch.cat([mel_t, sb_dB.to(mel_t.dtype)], dim=0)
        return out.numpy().astype(np.float32, copy=False)
