"""AR decoder for the #017 single-shot framewise detector.

Takes ``output.confidence_map`` directly (no sampler). Applies optional
NMS, thresholds, enforces minimum spacing, returns ``ARDecision`` with
multi-bin ``bin_offsets``. Empty positive-set -> STOP.

The core ``framewise_decision_from_map`` function is shared with
``FramewiseDiffusionDecoder`` (#016) so both decoders use identical
post-processing logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .decoders import ARDecoder, ARDecoderConfig
from .types import ARContext, ARDecision


# ─────────────────────────── shared logic ────────────────────────────


def framewise_decision_from_map(
    scores: torch.Tensor,
    *,
    decode_threshold: float,
    nms_kernel: int,
    min_emit_gap_bins: int,
    top_k_log: int,
) -> ARDecision:
    """Threshold-decode a ``(n_bins,)`` confidence map into an
    ``ARDecision``. Shared by both ``FramewiseDecoder`` and
    ``FramewiseDiffusionDecoder``."""
    extras: dict[str, float] = {
        "mean_act": float(scores.mean().item()),
        "max_act": float(scores.max().item()),
        "decode_threshold": float(decode_threshold),
        "nms_kernel": float(nms_kernel),
    }
    k = min(top_k_log, scores.numel())
    top_vals, top_idx = torch.topk(scores, k=k)
    for i, (v, idx) in enumerate(
        zip(top_vals.tolist(), top_idx.tolist()), start=1,
    ):
        extras[f"top{i}_bin"] = float(idx)
        extras[f"top{i}_score"] = float(v)

    if nms_kernel > 1:
        pooled = F.max_pool1d(
            scores.view(1, 1, -1), kernel_size=nms_kernel,
            stride=1, padding=nms_kernel // 2,
        ).view(-1)
        local_max = scores >= pooled - 1e-9
    else:
        local_max = torch.ones_like(scores, dtype=torch.bool)

    above = scores > decode_threshold
    keep_mask = above & local_max
    n_above_threshold = int(above.sum().item())
    extras["n_above_threshold"] = float(n_above_threshold)

    conf_map = tuple(scores.tolist())

    kept_bins = keep_mask.nonzero(as_tuple=True)[0].tolist()
    if not kept_bins:
        extras["n_emitted"] = 0.0
        return ARDecision(
            bin_offsets=(), confidences=(), extras=extras,
            confidence_map=conf_map,
        )

    kept_bins.sort()
    gap = int(min_emit_gap_bins)
    final: list[int] = []
    last = -10 ** 9
    for b in kept_bins:
        if b - last >= gap:
            final.append(b)
            last = b

    confidences = tuple(float(scores[b].item()) for b in final)
    extras["n_emitted"] = float(len(final))
    return ARDecision(
        bin_offsets=tuple(final),
        confidences=confidences,
        extras=extras,
        confidence_map=conf_map,
    )


# ─────────────────────────── FramewiseDecoder ────────────────────────


@dataclass(frozen=True, slots=True)
class FramewiseDecoderConfig(ARDecoderConfig):
    """Hyperparameters for ``FramewiseDecoder``."""
    b_pred: int = 500
    decode_threshold: float = 0.5
    nms_kernel: int = 3
    stop_hop_bins: int = 20
    min_emit_gap_bins: int = 1
    top_k_log: int = 5
    max_notes_per_step: int = 0

    def __post_init__(self) -> None:
        if self.b_pred < 1:
            raise ValueError(f"b_pred must be >= 1 (got {self.b_pred})")
        if not 0.0 <= self.decode_threshold <= 1.0:
            raise ValueError(
                f"decode_threshold must be in [0, 1] "
                f"(got {self.decode_threshold})"
            )
        if self.nms_kernel < 1:
            raise ValueError(
                f"nms_kernel must be >= 1 (got {self.nms_kernel})"
            )
        if self.nms_kernel > 1 and self.nms_kernel % 2 == 0:
            raise ValueError(
                f"nms_kernel must be odd when > 1 (got {self.nms_kernel})"
            )
        if self.stop_hop_bins < 1:
            raise ValueError(
                f"stop_hop_bins must be >= 1 (got {self.stop_hop_bins})"
            )
        if self.min_emit_gap_bins < 1:
            raise ValueError(
                f"min_emit_gap_bins must be >= 1 "
                f"(got {self.min_emit_gap_bins})"
            )
        if self.top_k_log < 1:
            raise ValueError(
                f"top_k_log must be >= 1 (got {self.top_k_log})"
            )


class FramewiseDecoder(ARDecoder[FramewiseDecoderConfig, "FramewiseDetectorOutput"]):
    """AR decoder for the single-shot framewise detector (#017)."""

    config: FramewiseDecoderConfig

    def __init__(self, config: FramewiseDecoderConfig):
        super().__init__(config)

    def decode(self, output: "FramewiseDetectorOutput", context: ARContext) -> ARDecision:
        conf = getattr(output, "confidence_map", None)
        if conf is None:
            logits = output.logits
            conf = torch.sigmoid(logits).clamp(0.0, 1.0)
        if conf.dim() != 2 or conf.size(0) != 1:
            raise ValueError(
                f"FramewiseDecoder expects confidence_map (1, n_bins); "
                f"got shape {tuple(conf.shape)}"
            )
        cfg = self.config
        decision = framewise_decision_from_map(
            conf[0],
            decode_threshold=cfg.decode_threshold,
            nms_kernel=cfg.nms_kernel,
            min_emit_gap_bins=cfg.min_emit_gap_bins,
            top_k_log=cfg.top_k_log,
        )
        if cfg.max_notes_per_step > 0 and len(decision.bin_offsets) > cfg.max_notes_per_step:
            n = cfg.max_notes_per_step
            decision = ARDecision(
                bin_offsets=decision.bin_offsets[:n],
                confidences=decision.confidences[:n],
                extras={**decision.extras, "n_emitted": float(n), "truncated": 1.0},
                confidence_map=decision.confidence_map,
            )
        return decision
