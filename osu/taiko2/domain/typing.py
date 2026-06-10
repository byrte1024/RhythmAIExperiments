"""Domain types for the onset-typing model.

The typing model is a secondary pass: given known onset positions, predict
each onset's type (DON/KA) and strength (normal/big). These dataclasses
define the sample, model IO, and target for that task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import ModelConfig, ModelInput, ModelOutput, ModelTarget
from .sampling import DataSample


TYPING_CONTEXT = 16  # default past and future onset count (legacy)
TYPING_MEL_PATCH = 5  # frames per onset mel patch
TYPING_WINDOW = 2 * TYPING_CONTEXT + 1  # legacy total tokens (33)


@dataclass(frozen=True, slots=True)
class TypingSample(DataSample):
    chart_id: str
    target_idx: int
    past_iois: np.ndarray       # (CONTEXT, 3) float32: [log_ioi_before, log_ioi_after, log_ratio]
    past_kinds: np.ndarray      # (CONTEXT,) uint8: 0=D, 1=K
    past_bigs: np.ndarray       # (CONTEXT,) uint8: 0=normal, 1=big
    past_mel: np.ndarray        # (CONTEXT, n_mels, MEL_PATCH) float32
    past_mask: np.ndarray       # (CONTEXT,) bool: True=padded
    target_iois: np.ndarray     # (3,) float32
    target_mel: np.ndarray      # (n_mels, MEL_PATCH) float32
    future_iois: np.ndarray     # (CONTEXT, 3) float32
    future_mel: np.ndarray      # (CONTEXT, n_mels, MEL_PATCH) float32
    future_mask: np.ndarray     # (CONTEXT,) bool: True=padded
    target_kind: int            # GT: 0=D, 1=K
    target_big: int             # GT: 0=normal, 1=big


@dataclass(frozen=True, slots=True)
class TypingModelConfig(ModelConfig):
    n_mels: int = 80
    mel_patch: int = TYPING_MEL_PATCH
    past_context: int = TYPING_CONTEXT
    future_context: int = TYPING_CONTEXT
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    d_mel: int = 32
    d_ioi: int = 16
    d_kind: int = 16
    d_pos: int = 32

    @property
    def window(self) -> int:
        return self.past_context + 1 + self.future_context


@dataclass(frozen=True, slots=True)
class TypingInput(ModelInput):
    mel_patches: torch.Tensor   # (B, W, n_mels * mel_patch) float32
    ioi_features: torch.Tensor  # (B, W, 3) float32
    kind_labels: torch.Tensor   # (B, W) long: 0=D, 1=K, 2=UNK
    big_labels: torch.Tensor    # (B, W) long: 0=normal, 1=big, 2=UNK
    positions: torch.Tensor     # (B, W) long: [-context .. 0 .. +context]
    mask: torch.Tensor          # (B, W) bool: True=padded


@dataclass(frozen=True, slots=True)
class TypingOutput(ModelOutput):
    type_logit: torch.Tensor      # (B,) raw, before sigmoid
    strength_logit: torch.Tensor  # (B,) raw, before sigmoid


@dataclass(frozen=True, slots=True)
class TypingTarget(ModelTarget):
    type_target: torch.Tensor      # (B,) float32: 0.0=K, 1.0=D
    strength_target: torch.Tensor  # (B,) float32: 0.0=normal, 1.0=big
