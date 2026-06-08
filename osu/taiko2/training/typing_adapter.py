"""Adapter: TypingSample batch -> TypingInput + TypingTarget tensors.

Handles the D<->K flip augmentation (50% probability during training)
and collation of variable-length mel patches / IOI features into
padded tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..domain.adapter import SampleToModelAdapter
from ..domain.typing import (
    TYPING_CONTEXT,
    TYPING_WINDOW,
    TypingInput,
    TypingSample,
    TypingTarget,
)

UNK_LABEL = 2


@dataclass(frozen=True, slots=True)
class TypingAdapterConfig:
    dk_flip_prob: float = 0.5
    future_dropout_prob: float = 0.1


class TypingSampleAdapter(
    SampleToModelAdapter[TypingSample, TypingInput, TypingTarget],
):
    def __init__(self, config: TypingAdapterConfig, *, training: bool = True):
        self.config = config
        self.training = training
        self._rng = np.random.default_rng(42)

    def make_input(
        self, samples: list[TypingSample], *, device: torch.device,
    ) -> TypingInput:
        B = len(samples)
        ctx = TYPING_CONTEXT
        W = TYPING_WINDOW
        n_mels = samples[0].target_mel.shape[0]
        mel_dim = samples[0].target_mel.size  # n_mels * mel_patch

        mel_all = np.zeros((B, W, mel_dim), dtype=np.float32)
        ioi_all = np.zeros((B, W, 3), dtype=np.float32)
        kind_all = np.full((B, W), UNK_LABEL, dtype=np.int64)
        big_all = np.full((B, W), UNK_LABEL, dtype=np.int64)
        pos_all = np.zeros((B, W), dtype=np.int64)
        mask_all = np.zeros((B, W), dtype=bool)

        # Position indices: 0..32 mapping to -16..+16
        positions = np.arange(W, dtype=np.int64)

        for i, s in enumerate(samples):
            # D<->K flip augmentation
            flip = (
                self.training
                and self._rng.random() < self.config.dk_flip_prob
            )

            # Past (indices 0..ctx-1)
            for j in range(ctx):
                mel_all[i, j] = s.past_mel[j].ravel()
                ioi_all[i, j] = s.past_iois[j]
                mask_all[i, j] = s.past_mask[j]
                if not s.past_mask[j]:
                    k = int(s.past_kinds[j])
                    kind_all[i, j] = (1 - k) if flip else k
                    big_all[i, j] = int(s.past_bigs[j])

            # Target (index ctx)
            mel_all[i, ctx] = s.target_mel.ravel()
            ioi_all[i, ctx] = s.target_iois
            # kind and big stay UNK for target

            # Future (indices ctx+1..W-1)
            drop_future = (
                self.training
                and self._rng.random() < self.config.future_dropout_prob
            )
            for j in range(ctx):
                fi = ctx + 1 + j
                if drop_future:
                    mask_all[i, fi] = True
                else:
                    mel_all[i, fi] = s.future_mel[j].ravel()
                    ioi_all[i, fi] = s.future_iois[j]
                    mask_all[i, fi] = s.future_mask[j]

            pos_all[i] = positions

        return TypingInput(
            mel_patches=torch.from_numpy(mel_all).to(device),
            ioi_features=torch.from_numpy(ioi_all).to(device),
            kind_labels=torch.from_numpy(kind_all).to(device),
            big_labels=torch.from_numpy(big_all).to(device),
            positions=torch.from_numpy(pos_all).to(device),
            mask=torch.from_numpy(mask_all).to(device),
        )

    def make_target(
        self, samples: list[TypingSample], *, device: torch.device,
    ) -> TypingTarget:
        # The flip state must match what make_input did for the same
        # batch. We achieve this by re-seeding with the same state —
        # but since make_input and make_target are called sequentially
        # from make_batch, we cache the flip decisions instead.
        #
        # Simpler approach: store flip decisions during make_input.
        # But the ABC calls make_input then make_target with no shared
        # state. So we re-derive: the RNG was advanced by make_input,
        # but we saved the flip decisions implicitly by convention:
        # make_batch calls make_input first, then make_target. We need
        # the flip state from make_input.
        #
        # Cleanest fix: override make_batch to share flip state.
        type_targets = np.zeros(len(samples), dtype=np.float32)
        str_targets = np.zeros(len(samples), dtype=np.float32)
        for i, s in enumerate(samples):
            type_targets[i] = float(s.target_kind)
            str_targets[i] = float(s.target_big)
        return TypingTarget(
            type_target=torch.from_numpy(type_targets).to(device),
            strength_target=torch.from_numpy(str_targets).to(device),
        )

    def make_batch(
        self, samples: list[TypingSample], *, device: torch.device,
    ) -> tuple[TypingInput, TypingTarget]:
        """Override to share D<->K flip state between input and target."""
        B = len(samples)
        ctx = TYPING_CONTEXT
        W = TYPING_WINDOW
        n_mels = samples[0].target_mel.shape[0]
        mel_dim = samples[0].target_mel.size

        mel_all = np.zeros((B, W, mel_dim), dtype=np.float32)
        ioi_all = np.zeros((B, W, 3), dtype=np.float32)
        kind_all = np.full((B, W), UNK_LABEL, dtype=np.int64)
        big_all = np.full((B, W), UNK_LABEL, dtype=np.int64)
        pos_all = np.zeros((B, W), dtype=np.int64)
        mask_all = np.zeros((B, W), dtype=bool)
        type_targets = np.zeros(B, dtype=np.float32)
        str_targets = np.zeros(B, dtype=np.float32)

        positions = np.arange(W, dtype=np.int64)

        for i, s in enumerate(samples):
            flip = (
                self.training
                and self._rng.random() < self.config.dk_flip_prob
            )

            for j in range(ctx):
                mel_all[i, j] = s.past_mel[j].ravel()
                ioi_all[i, j] = s.past_iois[j]
                mask_all[i, j] = s.past_mask[j]
                if not s.past_mask[j]:
                    k = int(s.past_kinds[j])
                    kind_all[i, j] = (1 - k) if flip else k
                    big_all[i, j] = int(s.past_bigs[j])

            mel_all[i, ctx] = s.target_mel.ravel()
            ioi_all[i, ctx] = s.target_iois

            drop_future = (
                self.training
                and self._rng.random() < self.config.future_dropout_prob
            )
            for j in range(ctx):
                fi = ctx + 1 + j
                if drop_future:
                    mask_all[i, fi] = True
                else:
                    mel_all[i, fi] = s.future_mel[j].ravel()
                    ioi_all[i, fi] = s.future_iois[j]
                    mask_all[i, fi] = s.future_mask[j]

            pos_all[i] = positions

            # Target with matching flip
            tk = s.target_kind
            type_targets[i] = float((1 - tk) if flip else tk)
            str_targets[i] = float(s.target_big)

        inp = TypingInput(
            mel_patches=torch.from_numpy(mel_all).to(device),
            ioi_features=torch.from_numpy(ioi_all).to(device),
            kind_labels=torch.from_numpy(kind_all).to(device),
            big_labels=torch.from_numpy(big_all).to(device),
            positions=torch.from_numpy(pos_all).to(device),
            mask=torch.from_numpy(mask_all).to(device),
        )
        tgt = TypingTarget(
            type_target=torch.from_numpy(type_targets).to(device),
            strength_target=torch.from_numpy(str_targets).to(device),
        )
        return inp, tgt
