"""Adapter: TypingSample batch -> TypingInput + TypingTarget tensors.

Augmentations (training only):
  - D<->K flip (50%): swap all D/K labels + target.
  - Future context dropout (10%): mask all future tokens.
  - Past label dropout (per-token 15%): set past kind/big to UNK.
  - IOI jitter (per-token): multiply IOI by exp(N(0, sigma)).
  - Mel noise: additive gaussian noise on mel patches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..domain.adapter import SampleToModelAdapter
from ..domain.typing import (
    TypingInput,
    TypingSample,
    TypingTarget,
)

UNK_LABEL = 2


@dataclass(frozen=True, slots=True)
class TypingAdapterConfig:
    dk_flip_prob: float = 0.5
    future_dropout_prob: float = 0.1
    past_label_dropout_prob: float = 0.0
    ioi_jitter_sigma: float = 0.0
    mel_noise_sigma: float = 0.0


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
        # Unused when make_batch is called (the normal path), but kept
        # for ABC compliance.
        raise NotImplementedError("Use make_batch directly")

    def make_target(
        self, samples: list[TypingSample], *, device: torch.device,
    ) -> TypingTarget:
        raise NotImplementedError("Use make_batch directly")

    def make_batch(
        self, samples: list[TypingSample], *, device: torch.device,
    ) -> tuple[TypingInput, TypingTarget]:
        B = len(samples)
        pc = samples[0].past_iois.shape[0]   # past context from sample
        fc = samples[0].future_iois.shape[0]  # future context from sample
        W = pc + 1 + fc
        n_mels = samples[0].target_mel.shape[0]
        mel_dim = samples[0].target_mel.size
        cfg = self.config
        aug = self.training

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
            flip = aug and self._rng.random() < cfg.dk_flip_prob
            drop_future = aug and self._rng.random() < cfg.future_dropout_prob

            # Past label dropout mask (per-token)
            past_label_drop = np.zeros(pc, dtype=bool)
            if aug and cfg.past_label_dropout_prob > 0:
                past_label_drop = self._rng.random(pc) < cfg.past_label_dropout_prob

            # Past tokens
            for j in range(pc):
                mel_all[i, j] = s.past_mel[j].ravel()
                ioi_all[i, j] = s.past_iois[j]
                mask_all[i, j] = s.past_mask[j]
                if not s.past_mask[j] and not past_label_drop[j]:
                    k = int(s.past_kinds[j])
                    kind_all[i, j] = (1 - k) if flip else k
                    big_all[i, j] = int(s.past_bigs[j])

            # Target token
            mel_all[i, pc] = s.target_mel.ravel()
            ioi_all[i, pc] = s.target_iois

            # Future tokens
            for j in range(fc):
                fi = pc + 1 + j
                if drop_future:
                    mask_all[i, fi] = True
                else:
                    mel_all[i, fi] = s.future_mel[j].ravel()
                    ioi_all[i, fi] = s.future_iois[j]
                    mask_all[i, fi] = s.future_mask[j]

            pos_all[i] = positions

            # Target labels with matching flip
            tk = s.target_kind
            type_targets[i] = float((1 - tk) if flip else tk)
            str_targets[i] = float(s.target_big)

        # IOI jitter (applied to all non-masked tokens)
        if aug and cfg.ioi_jitter_sigma > 0:
            noise = self._rng.normal(0, cfg.ioi_jitter_sigma, size=ioi_all.shape).astype(np.float32)
            noise[mask_all] = 0.0
            ioi_all += noise

        # Mel noise
        if aug and cfg.mel_noise_sigma > 0:
            noise = self._rng.normal(0, cfg.mel_noise_sigma, size=mel_all.shape).astype(np.float32)
            noise[mask_all] = 0.0
            mel_all += noise

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
