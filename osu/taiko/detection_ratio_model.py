"""Ratio-Based Onset Predictor.

Three-head architecture on top of ProposeSelectDetector's backbone:
  Head 1: Divisor (dominant rhythmic gap)
  Head 2: Offset (cursor distance from last event)
  Head 3: Ratio (multiple of divisor, sees Head 1+2 outputs)

Final onset = divisor * ratio - offset

Experiment 67.
"""

import math
import numpy as np
import torch
import torch.nn as nn

# Precompute ratio bins: 255 log-spaced from 0.125 to 8.0 + STOP
R_MIN = 0.125
R_MAX = 8.0
N_RATIO_BINS = 255
N_RATIO_CLASSES = 256  # 255 ratios + STOP at index 255
RATIO_STOP = 255

# Compute bin centers once
_ratio_bins = np.exp(np.linspace(np.log(R_MIN), np.log(R_MAX), N_RATIO_BINS)).astype(np.float32)


def get_ratio_bins():
    """Return (255,) array of ratio bin centers."""
    return _ratio_bins.copy()


def ratio_to_bin(ratio):
    """Snap a ratio value to the nearest bin index (0-254). Returns int."""
    return int(np.argmin(np.abs(_ratio_bins - ratio)))


def bin_to_ratio(bin_idx):
    """Convert bin index to ratio value."""
    if bin_idx >= N_RATIO_BINS:
        return 0.0  # STOP
    return float(_ratio_bins[bin_idx])


class RatioHeads(nn.Module):
    """Three-head ratio prediction on top of a cursor token.

    Head 1 (Divisor): predicts dominant rhythmic gap (D_BINS classes)
    Head 2 (Offset): predicts cursor-to-last-event distance (O_BINS classes)
    Head 3 (Ratio): predicts ratio of divisor (256 classes: 255 ratios + STOP)
         Sees Head 1+2 outputs embedded as vectors.
    """

    def __init__(self, d_model=384, d_bins=250, o_bins=100):
        super().__init__()
        self.d_model = d_model
        self.d_bins = d_bins
        self.o_bins = o_bins

        # Head 1: Divisor
        self.divisor_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_bins),
        )

        # Head 2: Offset
        self.offset_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, o_bins),
        )

        # Embeddings for Head 1+2 outputs -> Head 3 input
        self.divisor_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.offset_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Head 3: Ratio (sees cursor + divisor_emb + offset_emb)
        self.ratio_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, N_RATIO_CLASSES),
        )

    def forward(self, cursor_token):
        """
        Args:
            cursor_token: (B, d_model)

        Returns:
            divisor_logits: (B, d_bins)
            offset_logits: (B, o_bins)
            ratio_logits: (B, 256)
            derived_bin: (B,) integer bin offset = divisor * ratio - offset
        """
        B = cursor_token.size(0)
        device = cursor_token.device

        # Head 1: Divisor
        divisor_logits = self.divisor_head(cursor_token)  # (B, d_bins)
        # Soft expected value for embedding (differentiable)
        divisor_probs = torch.softmax(divisor_logits, dim=-1)
        divisor_bins = torch.arange(self.d_bins, device=device, dtype=torch.float32)
        divisor_value = (divisor_probs * (divisor_bins + 1)).sum(dim=-1, keepdim=True)  # (B, 1) +1 to avoid 0

        # Head 2: Offset
        offset_logits = self.offset_head(cursor_token)  # (B, o_bins)
        offset_probs = torch.softmax(offset_logits, dim=-1)
        offset_bins = torch.arange(self.o_bins, device=device, dtype=torch.float32)
        offset_value = (offset_probs * offset_bins).sum(dim=-1, keepdim=True)  # (B, 1)

        # Embed Head 1+2 values for Head 3
        div_emb = self.divisor_embed(divisor_value)  # (B, d_model)
        off_emb = self.offset_embed(offset_value)    # (B, d_model)

        # Head 3: Ratio (cursor + divisor_emb + offset_emb)
        ratio_input = cursor_token + div_emb + off_emb
        ratio_logits = self.ratio_head(ratio_input)  # (B, 256)

        # Derive final bin offset for metrics
        ratio_probs = torch.softmax(ratio_logits, dim=-1)
        ratio_bins_t = torch.tensor(_ratio_bins, device=device)
        # Expected ratio (exclude STOP)
        expected_ratio = (ratio_probs[:, :N_RATIO_BINS] * ratio_bins_t.unsqueeze(0)).sum(dim=-1)  # (B,)
        # Is STOP?
        is_stop = ratio_logits.argmax(dim=-1) == RATIO_STOP

        derived_bin = (divisor_value.squeeze(-1) * expected_ratio - offset_value.squeeze(-1)).round().long()
        derived_bin = derived_bin.clamp(0, 249)
        derived_bin[is_stop] = 250  # STOP

        return divisor_logits, offset_logits, ratio_logits, derived_bin


def compute_ratio_loss(ratio_logits, ratio_target, stop_weight=1.5):
    """Hill loss: distance in log-ratio space from correct answer.

    For non-STOP: loss = |log(pred_ratio) - log(target_ratio)| per bin, weighted by softmax
    For STOP: binary CE

    Args:
        ratio_logits: (B, 256) raw logits
        ratio_target: (B,) target bin indices (0-254 = ratio, 255 = STOP)
    """
    B = ratio_logits.size(0)
    device = ratio_logits.device

    bins_t = torch.tensor(_ratio_bins, device=device)  # (255,)
    log_bins = torch.log(bins_t)  # (255,)

    probs = torch.softmax(ratio_logits, dim=-1)  # (B, 256)

    is_stop = ratio_target == RATIO_STOP
    is_ratio = ~is_stop

    total_loss = torch.zeros(B, device=device)

    # Non-STOP: expected distance in log-ratio space
    if is_ratio.any():
        ratio_probs = probs[is_ratio, :N_RATIO_BINS]  # (M, 255)
        target_bins = ratio_target[is_ratio]  # (M,)
        target_log = log_bins[target_bins]  # (M,)

        # Distance of each bin from target in log space
        dist = torch.abs(log_bins.unsqueeze(0) - target_log.unsqueeze(1))  # (M, 255)

        # Expected distance = sum(prob * dist) over ratio bins
        expected_dist = (ratio_probs * dist).sum(dim=1)  # (M,)

        # Also penalize probability on STOP when target is ratio
        stop_penalty = probs[is_ratio, RATIO_STOP]  # (M,)

        total_loss[is_ratio] = expected_dist + stop_penalty

    # STOP: penalize probability NOT on STOP
    if is_stop.any():
        stop_prob = probs[is_stop, RATIO_STOP]  # (K,)
        total_loss[is_stop] = -torch.log(stop_prob.clamp(min=1e-8)) * stop_weight

    return total_loss.mean()
