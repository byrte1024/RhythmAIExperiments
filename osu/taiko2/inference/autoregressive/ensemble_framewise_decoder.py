"""Ensemble framewise decoder — runs K+1 predictions at offset cursors
and merges the confidence maps before decoding.

At each AR step with cursor C, the decoder runs the model at cursors
``[C, C+stride, C+2*stride, ..., C+K*stride]`` and obtains K+1
confidence maps. Each map is aligned back to C's coordinate space
(shifted by the offset) and merged using one of several strategies.
The merged map is then decoded with the standard threshold+NMS path.

Merge strategies:
  - ``add``: element-wise mean, then threshold.
  - ``multiply``: element-wise product, then threshold.
  - ``vote``: count how many of the K+1 maps have each bin above
    threshold; keep bins that pass in >= ``vote_quorum`` maps.
  - ``max``: element-wise max across all maps.
  - ``median``: element-wise median across all maps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ...domain.model import Model
from .decoders import ARDecoder
from .framewise_decoder import (
    ARDecision,
    ARDecoderConfig,
    FramewiseDecoderConfig,
    framewise_decision_from_map,
)
from .types import ARContext


@dataclass(frozen=True, slots=True)
class EnsembleFramewiseDecoderConfig(ARDecoderConfig):
    b_pred: int = 500
    decode_threshold: float = 0.5
    nms_kernel: int = 3
    stop_hop_bins: int = 20
    min_emit_gap_bins: int = 1
    top_k_log: int = 5
    max_notes_per_step: int = 0
    ensemble_k: int = 2
    ensemble_stride: int = 1
    merge_method: str = "add"
    vote_quorum: int = 2

    def __post_init__(self) -> None:
        if self.ensemble_k < 1:
            raise ValueError(
                f"ensemble_k must be >= 1 (got {self.ensemble_k})"
            )
        if self.ensemble_stride < 1:
            raise ValueError(
                f"ensemble_stride must be >= 1 (got {self.ensemble_stride})"
            )
        if self.merge_method not in {"add", "multiply", "vote", "max", "median"}:
            raise ValueError(
                f"merge_method must be one of add/multiply/vote/max/median "
                f"(got {self.merge_method!r})"
            )
        if self.vote_quorum < 1:
            raise ValueError(
                f"vote_quorum must be >= 1 (got {self.vote_quorum})"
            )


class EnsembleFramewiseDecoder(ARDecoder[EnsembleFramewiseDecoderConfig, "object"]):
    """Ensemble decoder that re-runs the model at offset cursors.

    Unlike the standard ``FramewiseDecoder``, this decoder needs
    access to the model and input builder to run additional forward
    passes. Bound via ``bind_model`` + ``bind_input_builder`` (called
    by ``assemble_predictor``). Per-chart runtime state (audio
    features, conditioning) bound via ``bind_runtime`` at the start
    of each AR loop.
    """

    config: EnsembleFramewiseDecoderConfig

    def __init__(self, config: EnsembleFramewiseDecoderConfig):
        super().__init__(config)
        self._model: Model | None = None
        self._input_builder = None
        self._audio_features = None
        self._conditioning = None
        self._device = torch.device("cpu")

    def bind_model(self, model: Model) -> None:
        self._model = model

    def bind_input_builder(self, input_builder: object) -> None:
        self._input_builder = input_builder

    def bind_runtime(
        self,
        *,
        audio_features: "object",
        conditioning: "object",
        device: torch.device,
    ) -> None:
        """Stash per-chart runtime state so ``decode`` can re-run the
        model without needing these passed through the ARDecoder ABC."""
        self._audio_features = audio_features
        self._conditioning = conditioning
        self._device = device

    def decode(
        self,
        output: object,
        context: ARContext,
    ) -> ARDecision:
        if self._model is None or self._input_builder is None:
            raise RuntimeError(
                "EnsembleFramewiseDecoder requires bind_model + "
                "bind_input_builder before decode"
            )
        if self._audio_features is None:
            raise RuntimeError(
                "EnsembleFramewiseDecoder requires bind_runtime before "
                "decode (called by the predictor at AR loop start)"
            )

        cfg = self.config
        n_bins = cfg.b_pred

        # Collect K+1 confidence maps at offset cursors.
        maps: list[torch.Tensor] = []

        # Map 0: the primary prediction (already computed by predictor).
        conf0 = getattr(output, "confidence_map", None)
        if conf0 is None:
            logits = output.logits
            conf0 = torch.sigmoid(logits).clamp(0.0, 1.0)
        maps.append(conf0[0].clone())  # (n_bins,)

        # Maps 1..K: re-run model with offset cursors.
        for k in range(1, cfg.ensemble_k + 1):
            offset = k * cfg.ensemble_stride
            offset_cursor = context.cursor_bin + offset
            inp = self._input_builder.build(
                cursor_bin=offset_cursor,
                past_onsets=context.past_onsets,
                audio_features=self._audio_features,
                conditioning=self._conditioning,
                device=self._device,
            )
            with torch.no_grad():
                out_k = self._model.predict(inp)
            conf_k = getattr(out_k, "confidence_map", None)
            if conf_k is None:
                conf_k = torch.sigmoid(out_k.logits).clamp(0.0, 1.0)
            # Align to cursor C's coordinate space.
            # The offset model's bin i = absolute position C+offset+i.
            # In C's coordinate space that's bin i+offset.
            # So: shifted[i + offset] = conf_k[i], for i in [0, n_bins - offset).
            shifted = torch.zeros(n_bins, device=conf_k.device)
            if offset < n_bins:
                shifted[offset:] = conf_k[0, :n_bins - offset]
            maps.append(shifted)

        # Merge.
        stacked = torch.stack(maps, dim=0)  # (K+1, n_bins)
        merged = _merge(stacked, cfg.merge_method, cfg.decode_threshold,
                        cfg.vote_quorum)

        # Decode the merged map.
        decision = framewise_decision_from_map(
            merged,
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
                extras={
                    **decision.extras,
                    "n_emitted": float(n),
                    "truncated": 1.0,
                },
                confidence_map=decision.confidence_map,
            )

        # Add ensemble diagnostics to extras.
        extras = dict(decision.extras)
        extras["ensemble_k"] = float(cfg.ensemble_k)
        extras["ensemble_stride"] = float(cfg.ensemble_stride)
        extras["merge_method"] = cfg.merge_method
        return ARDecision(
            bin_offsets=decision.bin_offsets,
            confidences=decision.confidences,
            extras=extras,
            confidence_map=decision.confidence_map,
        )


def _merge(
    stacked: torch.Tensor,
    method: str,
    threshold: float,
    vote_quorum: int,
) -> torch.Tensor:
    """Merge K+1 aligned confidence maps into one."""
    if method == "add":
        return stacked.mean(dim=0)
    if method == "multiply":
        return stacked.prod(dim=0)
    if method == "max":
        return stacked.max(dim=0).values
    if method == "median":
        return stacked.median(dim=0).values
    if method == "vote":
        votes = (stacked > threshold).sum(dim=0)
        passed = votes >= vote_quorum
        mean_conf = stacked.mean(dim=0)
        return torch.where(passed, mean_conf, torch.zeros_like(mean_conf))
    raise ValueError(f"unknown merge method {method!r}")
