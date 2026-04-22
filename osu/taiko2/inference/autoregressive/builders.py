"""Concrete `ARInputBuilder` for the event-embedding detector.

Mirrors the training-time `DetectionSampleAdapter` — same mel
window, same event encoding — but assembled from live AR state
(cursor, running past-onsets list, full-chart features,
conditioning) instead of a `TaikoDetectionSample`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch

from ...domain.beatmap import OnsetBinned
from ...domain.inference import Conditioning
from ...domain.model import ModelInput
from ...models.event_embedding import EventEmbeddingInput
from .types import ARInputBuilderConfig

BCfg = TypeVar("BCfg", bound=ARInputBuilderConfig)
Inp = TypeVar("Inp", bound=ModelInput)


class ARInputBuilder(ABC, Generic[BCfg, Inp]):
    """Assembles a single-batch `ModelInput` for one AR step."""
    config: BCfg

    def __init__(self, config: BCfg):
        self.config = config

    @abstractmethod
    def build(
        self,
        *,
        cursor_bin: int,
        past_onsets: tuple[OnsetBinned, ...],
        audio_features: np.ndarray,
        conditioning: Conditioning | None,
        device: torch.device,
    ) -> Inp:
        ...


# ─────────────────────────── concrete ──────────────────────────────────

@dataclass(frozen=True, slots=True)
class DetectionARBuilderConfig(ARInputBuilderConfig):
    """Must match the training-time `DetectionSampleAdapterConfig` +
    `EventEmbeddingConfig` window geometry used to train the model."""
    a_bins: int = 500
    b_bins: int = 500
    c_events: int = 128


class DetectionARInputBuilder(
    ARInputBuilder[DetectionARBuilderConfig, EventEmbeddingInput]
):
    """Build one batch-of-1 `EventEmbeddingInput` for live AR."""

    def build(
        self,
        *,
        cursor_bin: int,
        past_onsets: tuple[OnsetBinned, ...],
        audio_features: np.ndarray,
        conditioning: Conditioning | None,
        device: torch.device,
    ) -> EventEmbeddingInput:
        if conditioning is None:
            raise ValueError(
                "DetectionARInputBuilder requires conditioning "
                "(density_mean / density_peak / density_std)."
            )
        cfg = self.config

        # ── mel window, zero-padded at edges ──
        n_feat, total = audio_features.shape
        start = cursor_bin - cfg.a_bins
        end = cursor_bin + cfg.b_bins
        pad_left = max(0, -start)
        pad_right = max(0, end - total)
        s = max(0, start)
        e = min(total, end)
        core = audio_features[:, s:e].astype(np.float32, copy=False)
        if pad_left > 0 or pad_right > 0:
            core = np.pad(core, ((0, 0), (pad_left, pad_right)),
                          mode="constant")
        mel = (
            torch.from_numpy(np.ascontiguousarray(core))
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )  # (1, F, A+B)

        # ── event offsets (cursor-relative, ≤ 0 for past) + mask ──
        recent = past_onsets[-cfg.c_events:] if past_onsets else ()
        n_real = len(recent)
        offsets_np = np.zeros(cfg.c_events, dtype=np.int64)
        mask_np = np.ones(cfg.c_events, dtype=bool)  # True = padded
        if n_real > 0:
            for j, onset in enumerate(recent):
                idx = cfg.c_events - n_real + j
                offsets_np[idx] = int(onset.bin) - cursor_bin
                mask_np[idx] = False

        event_offsets = torch.from_numpy(offsets_np).unsqueeze(0).to(device=device)
        event_mask = torch.from_numpy(mask_np).unsqueeze(0).to(device=device)

        # ── conditioning ──
        cond_np = np.array(
            [[conditioning.density_mean, conditioning.density_peak,
              conditioning.density_std]],
            dtype=np.float32,
        )
        cond = torch.from_numpy(cond_np).to(device=device)

        return EventEmbeddingInput(
            mel=mel,
            event_offsets=event_offsets,
            event_mask=event_mask,
            conditioning=cond,
        )
