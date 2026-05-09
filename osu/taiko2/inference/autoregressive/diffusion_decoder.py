"""AR decoder that runs a diffusion sampler against the trunk's
cursor token to produce per-step logits, then argmax-decodes.

Wraps a ``DiffusionSampler`` (DDPM / DDIM / future variants); the
sampler is built lazily after the model is loaded, so the decoder
can pull the model's ``process`` and ``denoiser`` out and wire them
into the sampler. This means ``infer.json`` only specifies the
*sampler* config — the diffusion stack proper lives on the model
checkpoint and travels with it.

Inference contract:

- Decoder is constructed from ``DiffusionDecoderConfig`` via the
  standard ``build_component(spec["decoder"])`` path. At this
  point ``self._sampler`` is None.
- ``assemble_predictor`` calls ``decoder.bind_model(model)`` after
  the model is loaded. The decoder picks the concrete sampler class
  by ``sampler_config``'s type and constructs it with refs to
  ``model.process``, ``model.denoiser``.
- During AR rollout, ``decode`` runs ``self._sampler.sample`` on
  ``output.cursor_token`` and argmaxes the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from ...diffusion.samplers import DDIMSampler, DDIMSamplerConfig, DDPMSampler
from ...domain.diffusion import (
    DiffusionSampler,
    DiffusionSamplerConfig,
)
from ...models.diffusion_detector import DiffusionDetector, DiffusionModelOutput
from .decoders import ARDecoder, ARDecoderConfig
from .types import ARContext, ARDecision


# Sampler-config to sampler-class registry (mirrors the schedule /
# process / denoiser registries on the model). Sampler classes are
# pure inference-side; importing this module pulls them in.
_KNOWN_SAMPLERS: dict[type, type] = {
    DiffusionSamplerConfig: DDPMSampler,
    DDIMSamplerConfig: DDIMSampler,
}


def register_sampler(cfg_cls: type, impl_cls: type) -> None:
    _KNOWN_SAMPLERS[cfg_cls] = impl_cls


# ─────────────────────────── Config ───────────────────────────────────


def _default_sampler_config() -> DiffusionSamplerConfig:
    return DDIMSamplerConfig(
        n_inference_steps=16, eta=0.0, timestep_spacing="linspace",
    )


@dataclass(frozen=True, slots=True)
class DiffusionDecoderConfig(ARDecoderConfig):
    """``DiffusionDecoder`` config.

    - ``b_pred``: must match the trained model's ``b_pred`` (= STOP
      class index). Standard 500.
    - ``sampler_config``: pluggable sampler config (DDPM / DDIM / …).
      The concrete class is dispatched at ``bind_model`` time.
    - ``decode_strategy``: how to convert the sampler's final logits
      to a single bin. ``"argmax"`` (default — apples-to-apples with
      ``ArgmaxDecoder``) or ``"sample"`` (categorical sample from
      softmax — useful for AR diversity, not for direct comparison).
    - ``n_samples``: at each AR step, run the sampler ``N`` times from
      different ``x_T`` initial noise; aggregate. ``1`` = no
      aggregation. ``> 1`` runs ``N`` independent samplings;
      aggregation is by mean-of-softmax (logits averaged after
      softmax, then argmax).
    - ``top_k_log``: ``top_k`` entries reported in ``extras`` per AR
      step for trace logging (parity with ``ArgmaxDecoder``).
    """
    b_pred: int = 500
    sampler_config: DiffusionSamplerConfig = field(
        default_factory=_default_sampler_config,
    )
    decode_strategy: str = "argmax"
    n_samples: int = 1
    top_k_log: int = 5

    def __post_init__(self) -> None:
        if self.b_pred < 1:
            raise ValueError(f"b_pred must be >= 1 (got {self.b_pred})")
        if self.decode_strategy not in {"argmax", "sample"}:
            raise ValueError(
                f"decode_strategy must be 'argmax' or 'sample' "
                f"(got {self.decode_strategy!r})"
            )
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1 (got {self.n_samples})")
        if self.top_k_log < 1:
            raise ValueError(f"top_k_log must be >= 1 (got {self.top_k_log})")


# ─────────────────────────── Decoder ──────────────────────────────────


class DiffusionDecoder(ARDecoder[DiffusionDecoderConfig, DiffusionModelOutput]):
    """AR decoder that runs a diffusion sampler against the model's
    cursor token, then argmax-decodes the resulting logits."""

    config: DiffusionDecoderConfig

    def __init__(self, config: DiffusionDecoderConfig):
        super().__init__(config)
        self._sampler: DiffusionSampler | None = None

    # ── Model binding ────────────────────────────────────────────────

    def bind_model(self, model: DiffusionDetector) -> None:
        """Construct the sampler against the loaded model's ``process``
        and ``denoiser``. Must be called once after the model is
        loaded and before the first ``decode`` call.

        ``assemble_predictor`` calls this automatically when the
        loaded model is a ``DiffusionDetector``.
        """
        sc = self.config.sampler_config
        sampler_cls = _KNOWN_SAMPLERS.get(type(sc))
        if sampler_cls is None:
            # Fallback: type(sc) might be a parent ABC config. Walk MRO.
            for cfg_cls, cls in _KNOWN_SAMPLERS.items():
                if isinstance(sc, cfg_cls):
                    sampler_cls = cls
                    break
        if sampler_cls is None:
            raise TypeError(
                f"unknown sampler config type {type(sc).__name__}; "
                f"register it via "
                f"inference.autoregressive.diffusion_decoder.register_sampler"
            )
        self._sampler = sampler_cls(sc, model.process, model.denoiser)

    # ── Decoding ─────────────────────────────────────────────────────

    def decode(
        self,
        output: DiffusionModelOutput,
        context: ARContext,
    ) -> ARDecision:
        if self._sampler is None:
            raise RuntimeError(
                "DiffusionDecoder.decode called before bind_model. "
                "assemble_predictor should call bind_model after the "
                "model is loaded."
            )
        cursor_token = output.cursor_token
        if cursor_token.dim() != 2 or cursor_token.size(0) != 1:
            raise ValueError(
                f"DiffusionDecoder expects cursor_token (1, d_model); "
                f"got shape {tuple(cursor_token.shape)}"
            )

        # Run the sampler n_samples times, aggregate via mean-of-softmax.
        logits = self._sample_aggregated(cursor_token)               # (1, n_bins)
        return self._decision_from_logits(logits)

    @torch.no_grad()
    def _sample_aggregated(self, cursor_token: torch.Tensor) -> torch.Tensor:
        n = self.config.n_samples
        if n == 1:
            return self._sampler.sample(cursor_token)
        # n > 1: average softmax distributions to marginalize over x_T.
        accum = torch.zeros(
            cursor_token.size(0), self.config.b_pred + 1,
            device=cursor_token.device, dtype=torch.float32,
        )
        for _ in range(n):
            logits_i = self._sampler.sample(cursor_token)
            accum = accum + F.softmax(logits_i.float(), dim=-1)
        mean_softmax = accum / float(n)
        # Convert back to logits-shape so downstream code stays the same.
        return mean_softmax.log()

    def _decision_from_logits(self, logits: torch.Tensor) -> ARDecision:
        probs = F.softmax(logits.float(), dim=-1)[0]                # (n_bins,)
        if self.config.decode_strategy == "sample":
            cls = int(torch.multinomial(probs, num_samples=1).item())
        else:
            cls = int(probs.argmax().item())
        conf = float(probs[cls].item())

        extras: dict[str, float] = {
            "decode_strategy": float(0.0 if self.config.decode_strategy == "argmax" else 1.0),
            "n_samples": float(self.config.n_samples),
            "n_inference_steps": float(self.config.sampler_config.n_inference_steps),
        }
        log_probs = probs.clamp(min=1e-10).log()
        extras["entropy"] = float(-(probs * log_probs).sum().item())

        k = min(self.config.top_k_log, probs.size(0))
        top_vals, top_idx = torch.topk(probs, k=k)
        for i, (v, idx) in enumerate(
            zip(top_vals.tolist(), top_idx.tolist()), start=1,
        ):
            extras[f"top{i}_bin"] = float(idx)
            extras[f"top{i}_prob"] = float(v)

        stop_idx = self.config.b_pred
        if cls == stop_idx:
            return ARDecision(bin_offsets=(), confidences=(), extras=extras)
        return ARDecision(
            bin_offsets=(cls,), confidences=(conf,), extras=extras,
        )
