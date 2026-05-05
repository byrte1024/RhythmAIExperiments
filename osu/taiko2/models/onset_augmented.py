"""EventEmbeddingDetector configured for an onset-augmented mel input.

Architecturally identical to ``EventEmbeddingDetector`` — same conv
stem, same event-embedding mixer, same transformer trunk, same head.
The only difference is the ``n_mels`` field of the config: the conv
stem reads ``n_mels = n_audio_mels + n_onset_channels`` rows from the
input tensor instead of the usual 80.

This file exists for two reasons:

1. **Explicit semantics in checkpoint metadata.** A saved
   ``EventEmbeddingConfig`` with ``n_mels=84`` is ambiguous — is it
   "we trained on a different audio sampler" or "we appended onset
   channels"? The ``OnsetAugmentedConfig`` carries
   ``n_onset_channels`` as a separate field so a future loader can
   tell at a glance.

2. **Future architectural divergence.** If we want to A/B different
   fusion strategies (e.g. a separate stem for the onset rows
   followed by late concat), this file is the natural home for
   those experiments — they all subclass the same explicit base.

The current implementation is a thin marker subclass: ``__init__``
validates the config and delegates to the parent. No new layers, no
new buffers, no new state.
"""
from __future__ import annotations

from dataclasses import dataclass

from .event_embedding import EventEmbeddingConfig, EventEmbeddingDetector


@dataclass(frozen=True, slots=True)
class OnsetAugmentedConfig(EventEmbeddingConfig):
    """``EventEmbeddingConfig`` + onset-channel structure.

    The parent's ``n_mels`` is the **total** input row count
    (= audio mels + onset channels). ``n_onset_channels`` records
    how many of those rows are pre-computed onset detection function
    features so the input shape contract is explicit in saved
    configs and checkpoints.

    Constraint: ``n_mels > n_onset_channels``. The first
    ``(n_mels - n_onset_channels)`` rows are log-mel; the last
    ``n_onset_channels`` rows are onset features.
    """
    n_onset_channels: int = 4

    def __post_init__(self):
        # Explicit parent call: Python's slots=True + frozen dataclass
        # inheritance breaks super() inside __post_init__ in some
        # CPython versions. Call the parent method directly.
        EventEmbeddingConfig.__post_init__(self)
        if self.n_onset_channels < 1:
            raise ValueError(
                f"n_onset_channels must be >= 1 (got {self.n_onset_channels})"
            )
        if self.n_onset_channels >= self.n_mels:
            raise ValueError(
                f"n_onset_channels ({self.n_onset_channels}) must be less "
                f"than n_mels ({self.n_mels}) — n_mels is the TOTAL input "
                f"rows including onset channels"
            )

    @property
    def n_audio_mels(self) -> int:
        """Number of true log-mel rows (i.e. produced by the AmpToDB
        stage of the audio sampler), excluding the onset channel rows.
        """
        return self.n_mels - self.n_onset_channels


class OnsetAugmentedDetector(EventEmbeddingDetector):
    """``EventEmbeddingDetector`` for an onset-augmented mel input.

    Identical architecture to the parent. The conv stem accepts
    ``config.n_mels`` total rows where the last
    ``config.n_onset_channels`` rows are sub-band spectral flux (or
    similar pre-computed onset features) appended by the upstream
    ``MelOnsetSampler``.
    """

    config: OnsetAugmentedConfig

    def __init__(self, config: OnsetAugmentedConfig):
        super().__init__(config)
