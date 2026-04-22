"""`SampleToModelAdapter` concretes for taiko2.

One adapter per (sampler × model) pair. `DetectionSampleAdapter`
turns a batch of `TaikoDetectionSample` into `EventEmbeddingInput` +
`EventEmbeddingTarget` — collation, mel concatenation, STOP class
derivation, device placement all live here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data_samplers.detection import TaikoDetectionSample
from ..domain.adapter import SampleToModelAdapter
from ..models.event_embedding import EventEmbeddingInput, EventEmbeddingTarget


@dataclass(frozen=True, slots=True)
class DetectionSampleAdapterConfig:
    """`b_pred` must match the model's `EventEmbeddingConfig.b_pred`.

    Any offset ≥ `b_pred` or negative is encoded as STOP (class index
    `b_pred`). A sample with no future event in its window (the
    sampler's ``future_events_mask[0] is True``) is always STOP.
    """
    b_pred: int = 500

    def __post_init__(self):
        if self.b_pred <= 0:
            raise ValueError(f"b_pred must be > 0, got {self.b_pred}")


class DetectionSampleAdapter(
    SampleToModelAdapter[
        TaikoDetectionSample,
        EventEmbeddingInput,
        EventEmbeddingTarget,
    ]
):
    """Collates `TaikoDetectionSample` batches for `EventEmbeddingDetector`."""

    def __init__(self, config: DetectionSampleAdapterConfig):
        self.config = config

    # ── model input ───────────────────────────────────────────────────

    def make_input(
        self, samples: list[TaikoDetectionSample], *, device: torch.device,
    ) -> EventEmbeddingInput:
        if not samples:
            raise ValueError("cannot build input from an empty batch")

        B = len(samples)
        # Mel: concatenate past + future along the time axis → (F, A+B).
        mel_np = np.stack(
            [np.concatenate([s.audio_past, s.audio_future], axis=1) for s in samples],
            axis=0,
        )                                                          # (B, F, A+B)
        mel = torch.from_numpy(mel_np).to(device=device, dtype=torch.float32)

        # Event offsets (cursor-relative, negative for past) + mask.
        c_events = len(samples[0].past_events)
        offsets_np = np.empty((B, c_events), dtype=np.int64)
        mask_np = np.empty((B, c_events), dtype=bool)
        for i, s in enumerate(samples):
            offsets_np[i] = np.fromiter(
                (o.cursor_offset for o in s.past_events),
                dtype=np.int64, count=c_events,
            )
            mask_np[i] = s.past_events_mask

        event_offsets = torch.from_numpy(offsets_np).to(device=device)
        event_mask = torch.from_numpy(mask_np).to(device=device)

        # Conditioning: density_mean / density_peak / density_std.
        cond_np = np.array(
            [[s.density_mean, s.density_peak, s.density_std] for s in samples],
            dtype=np.float32,
        )
        conditioning = torch.from_numpy(cond_np).to(device=device)

        return EventEmbeddingInput(
            mel=mel,
            event_offsets=event_offsets,
            event_mask=event_mask,
            conditioning=conditioning,
        )

    # ── target ────────────────────────────────────────────────────────

    def make_target(
        self, samples: list[TaikoDetectionSample], *, device: torch.device,
    ) -> EventEmbeddingTarget:
        stop_idx = self.config.b_pred
        targets = np.empty(len(samples), dtype=np.int64)
        for i, s in enumerate(samples):
            if bool(s.future_events_mask[0]):
                targets[i] = stop_idx
                continue
            offset = s.future_events[0].cursor_offset
            if offset < 0 or offset >= self.config.b_pred:
                targets[i] = stop_idx
            else:
                targets[i] = offset
        return EventEmbeddingTarget(
            target_bin=torch.from_numpy(targets).to(device=device),
        )
