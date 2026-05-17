"""AR decoder for the #016 framewise diffusion model.

Runs the diffusion sampler against ``cursor_token + audio_features`` to
produce a ``(1, n_bins)`` activation map. Optionally applies 1-D NMS via
max-pool, thresholds, and enforces a minimum spacing between emitted
bins. Returns an ``ARDecision`` that can carry MULTIPLE bin offsets per
step — the AR loop (predictor.py) already supports multi-bin emission
via the ``bin_offsets`` tuple, so no engine changes are needed.

Empty positive-set → STOP (predictor advances by ``hop_bins_on_stop``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ...diffusion.samplers import DDIMSamplerConfig
from ...domain.diffusion import DiffusionSampler, DiffusionSamplerConfig
from .decoders import ARDecoder, ARDecoderConfig
from .diffusion_decoder import _KNOWN_SAMPLERS
from .types import ARContext, ARDecision


def _default_framewise_sampler_config() -> DiffusionSamplerConfig:
    return DDIMSamplerConfig(
        n_inference_steps=16, eta=0.0,
        timestep_spacing="linspace", time_offset=0.0,
    )


@dataclass(frozen=True, slots=True)
class FramewiseDiffusionDecoderConfig(ARDecoderConfig):
    """Hyperparameters for ``FramewiseDiffusionDecoder``.

    - ``b_pred``: activation map width (== framewise model's ``b_pred``).
    - ``sampler_config``: DDPM / DDIM / etc; dispatched via the shared
      ``_KNOWN_SAMPLERS`` registry.
    - ``decode_threshold``: scalar τ for binarization.
    - ``nms_kernel``: 1 = pure threshold; >1 keeps a bin only if it
      equals the local max within a window of this size (odd integer
      recommended).
    - ``stop_hop_bins``: kept here for documentation; the AR loop reads
      its own ``hop_bins_on_stop`` from ``AutoregressivePredictorConfig``.
      We expose it so this decoder's config carries a self-contained
      "no positives → advance by N" semantic.
    - ``min_emit_gap_bins``: minimum spacing between consecutive
      emissions; greedily drops bins within this distance of an already
      kept bin (lower-bin wins).
    - ``top_k_log``: top-K activations attached in extras for tracing.
    """
    b_pred: int = 500
    sampler_config: DiffusionSamplerConfig = field(
        default_factory=_default_framewise_sampler_config,
    )
    decode_threshold: float = 0.5
    nms_kernel: int = 1
    stop_hop_bins: int = 20
    top_k_log: int = 5
    min_emit_gap_bins: int = 1

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
        if self.top_k_log < 1:
            raise ValueError(
                f"top_k_log must be >= 1 (got {self.top_k_log})"
            )
        if self.min_emit_gap_bins < 1:
            raise ValueError(
                f"min_emit_gap_bins must be >= 1 "
                f"(got {self.min_emit_gap_bins})"
            )


class FramewiseDiffusionDecoder(ARDecoder[FramewiseDiffusionDecoderConfig, "FramewiseModelOutput"]):
    """AR decoder for the framewise diffusion head."""

    config: FramewiseDiffusionDecoderConfig

    def __init__(self, config: FramewiseDiffusionDecoderConfig):
        super().__init__(config)
        self._sampler: DiffusionSampler | None = None

    def bind_model(self, model) -> None:  # type: ignore[no-untyped-def]
        sc = self.config.sampler_config
        sampler_cls = _KNOWN_SAMPLERS.get(type(sc))
        if sampler_cls is None:
            for cfg_cls, cls in _KNOWN_SAMPLERS.items():
                if isinstance(sc, cfg_cls):
                    sampler_cls = cls
                    break
        if sampler_cls is None:
            raise TypeError(
                f"unknown sampler config type {type(sc).__name__}; "
                "register it via inference.autoregressive.diffusion_decoder"
                ".register_sampler"
            )
        self._sampler = sampler_cls(sc, model.process, model.denoiser)

    def decode(self, output, context: ARContext) -> ARDecision:
        if self._sampler is None:
            raise RuntimeError(
                "FramewiseDiffusionDecoder.decode called before "
                "bind_model. assemble_predictor should call bind_model "
                "after the model is loaded."
            )
        cursor_token = output.cursor_token
        audio_features = output.audio_features
        if cursor_token.dim() != 2 or cursor_token.size(0) != 1:
            raise ValueError(
                "FramewiseDiffusionDecoder expects cursor_token (1, "
                f"d_model); got shape {tuple(cursor_token.shape)}"
            )

        # Sampler returns (1, n_bins) already passed through
        # decode_to_logits. The framewise activation process clamps
        # output to [0, 1].
        m_hat = self._sampler.sample(
            cursor_token,
            audio_features=audio_features,
        )                                              # (1, n_bins)
        m_hat = m_hat.clamp(0.0, 1.0)
        return self._decision_from_map(m_hat)

    def _decision_from_map(self, m_hat: torch.Tensor) -> ARDecision:
        from .framewise_decoder import framewise_decision_from_map
        cfg = self.config
        decision = framewise_decision_from_map(
            m_hat[0],
            decode_threshold=cfg.decode_threshold,
            nms_kernel=cfg.nms_kernel,
            min_emit_gap_bins=cfg.min_emit_gap_bins,
            top_k_log=cfg.top_k_log,
        )
        extras = dict(decision.extras)
        extras["n_inference_steps"] = float(cfg.sampler_config.n_inference_steps)
        return ARDecision(
            bin_offsets=decision.bin_offsets,
            confidences=decision.confidences,
            extras=extras,
        )
