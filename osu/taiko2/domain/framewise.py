"""Framewise diffusion target types (#016).

Defines the per-sample target for framewise diffusion: an activation map
over ``n_bins`` future-time bins, where each GT onset contributes a
Gaussian-smoothed bump at its bin offset. The smoothed map is what the
denoiser is trained against; the binary map is kept around for metric
computation and decoder thresholding.

This module is domain-level (no I/O, no torch.nn): just frozen
dataclasses and the factory ``make_framewise_target``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class FramewiseTarget:
    """Per-sample activation-map target for framewise diffusion.

    - target_map_binary: strict GT, 1.0 at bins with onsets, 0.0
      elsewhere.
    - target_map_smoothed: Gaussian sigma=2-frame smoothed version;
      what the model is actually trained against.
    - gt_bins_padded: per-sample list of GT bin offsets, padded with
      -1.
    - n_gt: scalar tensor per-sample; same as the number of valid
      entries in ``gt_bins_padded`` for that batch row.
    """
    target_map_binary: torch.Tensor       # (B, n_bins) float in {0, 1}
    target_map_smoothed: torch.Tensor     # (B, n_bins) float in [0, 1]
    gt_bins_padded: torch.Tensor          # (B, max_events) int, -1 for pad
    n_gt: torch.Tensor                    # (B,) int >= 0


def make_framewise_target(
    future_offsets: torch.Tensor,
    n_bins: int,
    sigma: float = 2.0,
) -> FramewiseTarget:
    """Build binary and smoothed activation maps from per-sample GT
    bin offsets.

    ``future_offsets`` is ``(B, max_events) int`` with entries in
    ``[0, n_bins)`` for valid GT and ``-1`` for padding / "no onset".
    The smoothed map at bin ``i`` is the elementwise max across all
    valid GT bins ``b`` of ``exp(-((i - b) ** 2) / (2 * sigma ** 2))``,
    clipped to ``[0, 1]``. Rows with zero valid GT produce all-zero
    maps.
    """
    if future_offsets.dim() != 2:
        raise ValueError(
            f"future_offsets must be (B, max_events) (got shape "
            f"{tuple(future_offsets.shape)})"
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1 (got {n_bins})")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0 (got {sigma})")

    device = future_offsets.device
    B, M = future_offsets.shape
    valid = future_offsets >= 0                                  # (B, M)
    n_gt = valid.sum(dim=1).to(torch.int64)                      # (B,)

    # Binary map: scatter 1.0 at valid bin offsets. Pad index 0 is
    # safe to scatter when masked-off since we multiply by the
    # validity mask.
    binary = torch.zeros(B, n_bins, dtype=torch.float32, device=device)
    if M > 0:
        idx_clamped = future_offsets.clamp(min=0, max=n_bins - 1).long()
        ones = valid.float()
        # scatter_add then clamp to {0, 1} in case duplicates land
        binary.scatter_add_(1, idx_clamped, ones)
        binary = binary.clamp(max=1.0)

    # Smoothed map: for each valid GT bin b, place a Gaussian over
    # the bin axis and take the elementwise max across all GT bins.
    smoothed = torch.zeros(B, n_bins, dtype=torch.float32, device=device)
    if M > 0:
        bins_axis = torch.arange(n_bins, dtype=torch.float32, device=device)
        # offsets: (B, M, n_bins) = bins_axis[None,None,:] - future[..., None]
        offsets = future_offsets.float().unsqueeze(-1)                 # (B, M, 1)
        deltas = bins_axis.view(1, 1, n_bins) - offsets                # (B, M, n_bins)
        gauss = torch.exp(-(deltas ** 2) / (2.0 * sigma * sigma))      # (B, M, n_bins)
        # Mask invalid entries to 0 so they don't contribute to max.
        gauss = gauss * valid.float().unsqueeze(-1)
        # Elementwise max across M.
        smoothed = gauss.max(dim=1).values
        smoothed = smoothed.clamp(0.0, 1.0)

    return FramewiseTarget(
        target_map_binary=binary,
        target_map_smoothed=smoothed,
        gt_bins_padded=future_offsets.to(torch.int64),
        n_gt=n_gt,
    )
