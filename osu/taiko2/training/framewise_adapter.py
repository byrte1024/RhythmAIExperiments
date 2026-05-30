"""`FramewiseSampleAdapter` — DataSample → (EventEmbeddingInput, FramewiseTarget) (#016).

Mirrors `DetectionSampleAdapter` for the framewise-diffusion path. The
model input is identical (the trunk doesn't change), but the target is a
``FramewiseTarget`` produced by ``make_framewise_target`` over every
future event whose cursor offset lands in ``[0, b_pred)``.

Events outside the window are filtered (encoded as -1 padding); rows
with zero in-window future events emit all-zero activation maps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data_samplers.detection import TaikoDetectionSample
from ..domain.adapter import SampleToModelAdapter
from ..domain.framewise import FramewiseTarget, make_framewise_target
from ..models.event_embedding import EventEmbeddingInput
from .adapters import DetectionSampleAdapter, DetectionSampleAdapterConfig


@dataclass(frozen=True, slots=True)
class FramewiseSampleAdapterConfig:
    """Hyperparameters for ``FramewiseSampleAdapter``.

    - ``b_pred`` is the activation-map width. Onsets at offset ``e`` are
      included iff ``0 <= e < b_pred``.
    - ``sigma_frames`` controls Gaussian smoothing of the target map.
    - ``max_events_per_window`` upper-bounds the per-sample padded list
      of GT offsets. The sampler's ``d_events`` should be at least this
      large; the adapter silently clips any extra and warns at debug
      level only via the resulting ``n_gt`` (no console spam).
    """
    b_pred: int = 500
    sigma_frames: float = 2.0
    binary_only: bool = False
    max_events_per_window: int = 100
    feature_rows: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.b_pred <= 0:
            raise ValueError(f"b_pred must be > 0 (got {self.b_pred})")
        if not self.binary_only and self.sigma_frames <= 0.0:
            raise ValueError(
                f"sigma_frames must be > 0 (got {self.sigma_frames})"
            )
        if self.max_events_per_window < 1:
            raise ValueError(
                f"max_events_per_window must be >= 1 "
                f"(got {self.max_events_per_window})"
            )


class FramewiseSampleAdapter(
    SampleToModelAdapter[
        TaikoDetectionSample,
        EventEmbeddingInput,
        FramewiseTarget,
    ],
):
    """Collates ``TaikoDetectionSample`` batches for the framewise
    diffusion detector. Reuses ``DetectionSampleAdapter.make_input`` for
    the (B, n_mels, A+B) / events / conditioning tensors; overrides
    ``make_target`` to produce a ``FramewiseTarget`` instead of an
    ``EventEmbeddingTarget``.
    """

    def __init__(self, config: FramewiseSampleAdapterConfig):
        self.config = config
        # Reuse the detection adapter for the input side — identical
        # mel / event / conditioning collation.
        self._detection_adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=config.b_pred, ratio_mode=False),
        )

    # ── model input ───────────────────────────────────────────────────

    def make_input(
        self,
        samples: list[TaikoDetectionSample],
        *,
        device: torch.device,
    ) -> EventEmbeddingInput:
        fr = self.config.feature_rows
        if fr is not None:
            lo, hi = fr
            B = len(samples)
            mel_np = np.stack([
                np.concatenate([s.audio_past[lo:hi], s.audio_future[lo:hi]], axis=1)
                for s in samples
            ], axis=0)
            mel = torch.from_numpy(mel_np).to(device=device, dtype=torch.float32)

            c_events = len(samples[0].past_events)
            offsets_np = np.empty((B, c_events), dtype=np.int64)
            mask_np = np.empty((B, c_events), dtype=bool)
            cond_np = np.empty((B, 3), dtype=np.float32)
            for i, s in enumerate(samples):
                offsets_np[i] = np.fromiter(
                    (o.cursor_offset for o in s.past_events),
                    dtype=np.int64, count=c_events,
                )
                mask_np[i] = s.past_events_mask
                cond_np[i] = [s.density_mean, s.density_peak, s.density_std]

            return EventEmbeddingInput(
                mel=mel,
                event_offsets=torch.from_numpy(offsets_np).to(device=device),
                event_mask=torch.from_numpy(mask_np).to(device=device),
                conditioning=torch.from_numpy(cond_np).to(device=device),
            )
        return self._detection_adapter.make_input(samples, device=device)

    # ── target ────────────────────────────────────────────────────────

    def make_target(
        self,
        samples: list[TaikoDetectionSample],
        *,
        device: torch.device,
    ) -> FramewiseTarget:
        if not samples:
            raise ValueError("cannot build target from an empty batch")
        B = len(samples)
        b_pred = self.config.b_pred
        M = self.config.max_events_per_window

        offsets_np = np.full((B, M), -1, dtype=np.int64)
        for i, s in enumerate(samples):
            slot = 0
            for j, ev in enumerate(s.future_events):
                if slot >= M:
                    break
                if bool(s.future_events_mask[j]):
                    continue
                off = int(ev.cursor_offset)
                if 0 <= off < b_pred:
                    offsets_np[i, slot] = off
                    slot += 1

        future_offsets = torch.from_numpy(offsets_np).to(device=device)
        sigma = None if self.config.binary_only else self.config.sigma_frames
        return make_framewise_target(
            future_offsets,
            n_bins=b_pred,
            sigma=sigma,
        )
