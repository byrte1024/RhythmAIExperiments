"""AR decoder for the RatioDetector.

Derives the predicted bin from divisor × ratio − offset. STOP when
the ratio head's argmax is the last class (STOP index).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ...models.event_embedding import EventEmbeddingOutput
from ...models.ratio_detector import build_ratio_bin_centers
from .decoders import ARDecoder, ARDecoderConfig
from .types import ARContext, ARDecision


@dataclass(frozen=True, slots=True)
class RatioDecoderConfig(ARDecoderConfig):
    b_pred: int = 500
    divisor_bins: int = 500
    offset_bins: int = 100
    ratio_bins: int = 255


class RatioDecoder(ARDecoder[RatioDecoderConfig, EventEmbeddingOutput]):
    """Decode ratio-mode output: divisor × ratio_value − offset.

    Per-step extras log all three head predictions for the AR trace.
    """

    def __init__(self, config: RatioDecoderConfig):
        super().__init__(config)
        self._centers = build_ratio_bin_centers(config.ratio_bins)

    def decode(self, output: EventEmbeddingOutput, context: ARContext) -> ARDecision:
        raw = output.logits
        if raw.dim() != 2 or raw.size(0) != 1:
            raise ValueError(
                f"RatioDecoder expects (1, D+O+R+1); got {tuple(raw.shape)}"
            )
        cfg = self.config
        D = cfg.divisor_bins
        O = cfg.offset_bins
        R = cfg.ratio_bins

        div_logits = raw[0, :D]
        off_logits = raw[0, D:D + O]
        ratio_logits = raw[0, D + O:]                                   # (R+1,)

        # Soft expectations for divisor + offset.
        div_probs = torch.softmax(div_logits, dim=-1)
        off_probs = torch.softmax(off_logits, dim=-1)
        div_bins = torch.arange(D, device=raw.device, dtype=torch.float32)
        off_bins = torch.arange(O, device=raw.device, dtype=torch.float32)
        div_val = float((div_probs * div_bins).sum())
        off_val = float((off_probs * off_bins).sum())

        # Ratio: argmax. Last class = STOP.
        ratio_idx = int(ratio_logits.argmax().item())

        extras: dict[str, float] = {
            "divisor": div_val,
            "offset": off_val,
            "ratio_idx": float(ratio_idx),
            "div_argmax": float(div_logits.argmax().item()),
            "off_argmax": float(off_logits.argmax().item()),
        }

        if ratio_idx == R:
            # STOP.
            extras["ratio_val"] = 0.0
            return ARDecision(
                bin_offsets=(),
                confidences=(),
                extras=extras,
            )

        ratio_val = float(self._centers[ratio_idx].item())
        extras["ratio_val"] = ratio_val

        predicted_bin = int(round(div_val * ratio_val - off_val))
        predicted_bin = max(0, min(predicted_bin, cfg.b_pred - 1))

        ratio_conf = float(
            torch.softmax(ratio_logits, dim=-1)[ratio_idx].item()
        )

        return ARDecision(
            bin_offsets=(predicted_bin,),
            confidences=(ratio_conf,),
            extras=extras,
        )
