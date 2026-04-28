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

    When ``ratio_mode`` is True, also computes ``divisor_target``
    (dominant past gap) and ``offset_target`` (cursor distance from
    last event) for the RatioDetector's loss.
    """
    b_pred: int = 500
    ratio_mode: bool = False

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
        B = len(samples)
        b_pred = self.config.b_pred
        stop_idx = b_pred

        # Primary target: next onset or STOP.
        targets = np.empty(B, dtype=np.int64)
        for i, s in enumerate(samples):
            if bool(s.future_events_mask[0]):
                targets[i] = stop_idx
                continue
            offset = s.future_events[0].cursor_offset
            if offset < 0 or offset >= b_pred:
                targets[i] = stop_idx
            else:
                targets[i] = offset

        # Optional all-future target (for I* metrics). Populate whenever
        # the sampler's d_events ≥ 1 — unused slots stay masked.
        K = len(samples[0].future_events)
        all_bins = np.zeros((B, K), dtype=np.int64)
        all_mask = np.ones((B, K), dtype=bool)
        for i, s in enumerate(samples):
            for j in range(K):
                if bool(s.future_events_mask[j]):
                    continue
                off = s.future_events[j].cursor_offset
                if 0 <= off < b_pred:
                    all_bins[i, j] = off
                    all_mask[i, j] = False

        # Ratio-mode: divisor + offset targets with validity masks.
        div_t = None
        off_t = None
        div_v = None
        off_v = None
        if self.config.ratio_mode:
            from collections import Counter
            div_arr = np.zeros(B, dtype=np.int64)
            off_arr = np.zeros(B, dtype=np.int64)
            div_valid = np.zeros(B, dtype=bool)
            off_valid = np.zeros(B, dtype=bool)
            for i, s in enumerate(samples):
                real_mask = ~s.past_events_mask
                n_real = int(real_mask.sum())

                # Offset: need ≥1 real past event.
                if n_real >= 1:
                    last_real_idx = int(np.where(real_mask)[0][-1])
                    last_offset = s.past_events[last_real_idx].cursor_offset
                    off_arr[i] = max(0, -last_offset)
                    off_valid[i] = True

                # Divisor: need ≥2 real past events to compute gaps,
                # AND the IOI mode must appear ≥2 times (clear peak).
                if n_real >= 2:
                    real_offsets = [
                        e.cursor_offset for e, m
                        in zip(s.past_events, s.past_events_mask) if not m
                    ]
                    gaps = [
                        abs(real_offsets[j] - real_offsets[j - 1])
                        for j in range(1, len(real_offsets))
                    ]
                    quantized = [round(g / 3) * 3 for g in gaps]
                    counts = Counter(quantized)
                    mode_gap, mode_count = counts.most_common(1)[0]
                    if mode_count >= 2:
                        div_arr[i] = max(1, mode_gap)
                        div_valid[i] = True
                    else:
                        # All gaps unique — no clear tempo. Loss masked.
                        div_arr[i] = max(1, mode_gap)  # best guess, but masked

            div_t = torch.from_numpy(div_arr).to(device=device)
            off_t = torch.from_numpy(off_arr).to(device=device)
            div_v = torch.from_numpy(div_valid).to(device=device)
            off_v = torch.from_numpy(off_valid).to(device=device)

        return EventEmbeddingTarget(
            target_bin=torch.from_numpy(targets).to(device=device),
            all_future_bins=torch.from_numpy(all_bins).to(device=device),
            all_future_mask=torch.from_numpy(all_mask).to(device=device),
            divisor_target=div_t,
            offset_target=off_t,
            divisor_valid=div_v,
            offset_valid=off_v,
        )
