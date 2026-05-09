"""``EventEmbeddingDetector`` with a diffusion-based prediction head.

The trunk (conv stem + audio + event-embedding mixer + transformer
trunk) is inherited unchanged. The output head is replaced by:

  - a ``NoiseSchedule``      (when noise is added per step),
  - a ``DiffusionProcess``   (forward / reverse process math),
  - a ``DenoiserHead``       (the neural net that maps
                              cursor_token + x_t + t → model_out).

Training-time forward returns a ``DiffusionModelOutput`` with the
denoiser output, the loss target, the sampled timesteps, and the
noised state — so the loss can compute a simple MSE without needing
to re-run any diffusion math.

Inference-time forward returns the cursor token only; the inference
``DiffusionDecoder`` runs a ``DiffusionSampler`` over the cursor token
during AR decoding.

Field-level polymorphism (schedule / process / denoiser pluggable via
``__class__`` strings in JSON config) leverages the standard
``inference.spec.build_config`` recursion: each sub-config field is
typed as the ABC, and the JSON provides the concrete ``__class__``
string. Python construction passes concrete config instances directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..diffusion import (
    CosineSchedule,
    CosineScheduleConfig,
    GaussianContinuousProcess,
    GaussianContinuousProcessConfig,
    MLPDenoiser,
    MLPDenoiserConfig,
)
from ..domain.diffusion import (
    DenoiserConfig,
    DenoiserHead,
    DiffusionProcess,
    DiffusionProcessConfig,
    NoiseSchedule,
    NoiseScheduleConfig,
)
from ..domain.model import ModelOutput
from .event_embedding import (
    EventEmbeddingConfig,
    EventEmbeddingDetector,
    EventEmbeddingInput,
)


# ─────────────────────────── Output type ──────────────────────────────


@dataclass(frozen=True, slots=True)
class DiffusionModelOutput(ModelOutput):
    """Diffusion-detector output.

    Two regimes:

    - **Training**: the model has access to the GT target_bin
      (passed via ``forward_diffusion``) and computes one denoising
      step. Fields ``model_out``, ``loss_target``, ``t``, ``x_t``
      are populated. ``logits`` carries a *placeholder* — the
      predicted-x_0 logits at the sampled t — used by metrics that
      want a per-bin prediction at training time. **It is not the
      same as the inference-time prediction**, which requires the
      full sampler.

    - **Inference**: only ``cursor_token`` is set; the decoder runs
      the sampler externally to produce final logits. ``logits``,
      ``model_out``, ``loss_target``, ``t``, ``x_t`` are ``None``.
    """
    logits: torch.Tensor                              # (B, n_bins) float32
    cursor_token: torch.Tensor                        # (B, d_model) float32
    model_out: torch.Tensor | None = None             # (B, n_bins) — denoiser raw output
    loss_target: torch.Tensor | None = None           # (B, n_bins) — target the denoiser should match
    t: torch.Tensor | None = None                     # (B,) int64 — sampled timesteps
    x_t: torch.Tensor | None = None                   # (B, n_bins) — noised state at t


# ─────────────────────────── Config ───────────────────────────────────


def _default_schedule_config() -> NoiseScheduleConfig:
    return CosineScheduleConfig(n_steps=64)


def _default_process_config() -> DiffusionProcessConfig:
    return GaussianContinuousProcessConfig(
        n_bins=501, parameterization="x0", x0_scale=2.0,
    )


def _default_denoiser_config() -> DenoiserConfig:
    return MLPDenoiserConfig(
        d_model=384, n_bins=501, hidden_dim=1536,
        time_embed_dim=128, time_embed_proj_dim=256, n_layers=3,
        dropout=0.1,
    )


@dataclass(frozen=True, slots=True)
class DiffusionDetectorConfig(EventEmbeddingConfig):
    """``EventEmbeddingConfig`` + diffusion-stack sub-configs.

    The three sub-configs are typed as the ABCs (``NoiseScheduleConfig``,
    ``DiffusionProcessConfig``, ``DenoiserConfig``) so JSON
    deserialization via ``build_config`` can dispatch on ``__class__``
    to construct the right concrete subclass.

    Defaults (cosine schedule + Gaussian-continuous process +
    3-layer MLP denoiser) match the canonical first-pass
    configuration for #014.
    """
    schedule_config: NoiseScheduleConfig = field(
        default_factory=_default_schedule_config,
    )
    process_config: DiffusionProcessConfig = field(
        default_factory=_default_process_config,
    )
    denoiser_config: DenoiserConfig = field(
        default_factory=_default_denoiser_config,
    )

    def __post_init__(self) -> None:
        EventEmbeddingConfig.__post_init__(self)
        # Cross-config sanity: the denoiser's output width must equal
        # the process's n_bins; the denoiser's d_model must equal the
        # trunk's d_model. These are easy to get wrong in JSON.
        if self.denoiser_config.n_bins != self.process_config.n_bins:
            raise ValueError(
                f"denoiser_config.n_bins ({self.denoiser_config.n_bins}) "
                f"!= process_config.n_bins ({self.process_config.n_bins})"
            )
        if self.denoiser_config.d_model != self.d_model:
            raise ValueError(
                f"denoiser_config.d_model ({self.denoiser_config.d_model}) "
                f"!= EventEmbeddingConfig.d_model ({self.d_model})"
            )
        # The bin-count of the diffusion output should match the
        # detector's output classes (b_pred + 1 for STOP).
        expected_n_bins = self.b_pred + 1
        if self.process_config.n_bins != expected_n_bins:
            raise ValueError(
                f"process_config.n_bins ({self.process_config.n_bins}) "
                f"!= b_pred + 1 ({expected_n_bins}); the diffusion head "
                f"produces n_bins outputs, which must equal the "
                f"classification space (b_pred bin offsets + 1 STOP)."
            )


# ─────────────────────────── Concrete-config dispatch ─────────────────


# When ``DiffusionDetectorConfig`` is built directly in Python (e.g.
# from tests), the sub-config types may already be concrete subclass
# instances. When loaded from JSON via ``build_config``, the recursive
# decoder dispatches on each sub-config's ``__class__`` field. The
# concrete classes that the model knows how to construct are listed
# here so the constructor can validate the type.

_KNOWN_SCHEDULES: dict[type, type] = {
    CosineScheduleConfig: CosineSchedule,
}
_KNOWN_PROCESSES: dict[type, type] = {
    GaussianContinuousProcessConfig: GaussianContinuousProcess,
}
_KNOWN_DENOISERS: dict[type, type] = {
    MLPDenoiserConfig: MLPDenoiser,
}


def _register_schedule(cfg_cls: type, impl_cls: type) -> None:
    """Future schedules register themselves here so DiffusionDetector
    can dispatch on the config class without an import cycle. Importing
    a new schedule module should call this in the module body."""
    _KNOWN_SCHEDULES[cfg_cls] = impl_cls


def _register_process(cfg_cls: type, impl_cls: type) -> None:
    _KNOWN_PROCESSES[cfg_cls] = impl_cls


def _register_denoiser(cfg_cls: type, impl_cls: type) -> None:
    _KNOWN_DENOISERS[cfg_cls] = impl_cls


# Make the registries importable for downstream registration.
register_schedule = _register_schedule
register_process = _register_process
register_denoiser = _register_denoiser


# Lazy-import non-default schedules/processes/denoisers as they're
# added.  All current concrete classes are registered above.
try:
    from ..diffusion.schedules import LinearSchedule, LinearScheduleConfig
    _KNOWN_SCHEDULES[LinearScheduleConfig] = LinearSchedule
except ImportError:
    pass


# ─────────────────────────── Detector ─────────────────────────────────


class DiffusionDetector(EventEmbeddingDetector):
    """``EventEmbeddingDetector`` with a diffusion head.

    Inherits the trunk: conv stem, conditioning MLP, audio + event
    mixer, transformer encoder, FiLM. **Replaces** the parent's
    classification head with a denoiser + diffusion process.

    Two forward modes:

    - ``predict(input)`` — inference. Runs the trunk and returns
      ``DiffusionModelOutput`` with ``cursor_token`` only. The
      decoder runs the sampler externally.
    - ``forward_diffusion(cursor_token, target_bin)`` — training. Samples
      ``t``, ``noise``; computes ``x_t = q_sample(x_0, t, noise)``;
      runs the denoiser; returns the structured fields the loss
      consumes. Idempotent in the sense that all randomness is
      sampled fresh per call.

    The trunk's parent owns ``head_norm``, ``head_proj``,
    ``head_smooth`` from ``EventEmbeddingDetector``. We do not use
    them. They remain in the parameter dict (unused-but-not-deleted)
    so any inherited code paths that reference them don't break;
    they receive zero gradient since the loss never touches them.
    """

    config: DiffusionDetectorConfig

    def __init__(self, config: DiffusionDetectorConfig):
        super().__init__(config)
        # Construct the diffusion stack from the sub-configs.
        sched_cfg = config.schedule_config
        proc_cfg = config.process_config
        den_cfg = config.denoiser_config

        sched_cls = _KNOWN_SCHEDULES.get(type(sched_cfg))
        if sched_cls is None:
            raise TypeError(
                f"unknown schedule config type {type(sched_cfg).__name__}; "
                f"register it via models.diffusion_detector.register_schedule"
            )
        proc_cls = _KNOWN_PROCESSES.get(type(proc_cfg))
        if proc_cls is None:
            raise TypeError(
                f"unknown process config type {type(proc_cfg).__name__}; "
                f"register it via models.diffusion_detector.register_process"
            )
        den_cls = _KNOWN_DENOISERS.get(type(den_cfg))
        if den_cls is None:
            raise TypeError(
                f"unknown denoiser config type {type(den_cfg).__name__}; "
                f"register it via models.diffusion_detector.register_denoiser"
            )

        self.schedule: NoiseSchedule = sched_cls(sched_cfg)
        self.process: DiffusionProcess = proc_cls(proc_cfg, self.schedule)
        # The denoiser is an nn.Module — register it so its parameters
        # are picked up by the optimizer + checkpoint.
        self.denoiser: DenoiserHead = den_cls(den_cfg)

    # ── inference-mode forward ───────────────────────────────────────

    def predict(self, x: EventEmbeddingInput) -> DiffusionModelOutput:
        """Trunk-only forward. The decoder is responsible for running
        the diffusion sampler against the returned ``cursor_token``."""
        cursor_token = self.get_cursor_token(
            x.mel, x.event_offsets, x.event_mask, x.conditioning,
        )
        # `logits` placeholder = zero — we don't run the sampler here.
        # This is fine: nothing in the inference pipeline reads
        # ``logits`` directly; the decoder pulls ``cursor_token`` out
        # of the output and produces logits via the sampler.
        return DiffusionModelOutput(
            logits=torch.zeros(
                cursor_token.size(0), self.process.config.n_bins,
                device=cursor_token.device, dtype=cursor_token.dtype,
            ),
            cursor_token=cursor_token,
        )

    # ── training-mode forward ────────────────────────────────────────

    def forward_diffusion(
        self,
        cursor_token: torch.Tensor,
        target_bin: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> DiffusionModelOutput:
        """Run one diffusion training step starting from a precomputed
        cursor token.

        Sampling of ``t`` and ``noise`` happens here; pass them in
        explicitly for reproducible eval / gradient checks. ``t`` is
        sampled uniformly over the schedule by default.

        Returns a ``DiffusionModelOutput`` with all training fields
        populated, plus a placeholder ``logits`` field that contains
        the predicted x_0 in logit space at the sampled timestep.
        Downstream metrics that want a "per-step prediction" can read
        ``logits``; the inference path ignores it.
        """
        B = cursor_token.size(0)
        device = cursor_token.device

        x_0 = self.process.encode_x0(target_bin)            # (B, n_bins)
        if t is None:
            t = torch.randint(
                0, self.schedule.n_steps, (B,),
                device=device, dtype=torch.long,
            )
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.process.q_sample(x_0, t, noise=noise)
        model_out = self.denoiser(cursor_token, x_t, t)
        loss_target = self.process.loss_target(x_0, x_t, t, noise=noise)

        # Predict x_0 logits at this sampled t; downstream metrics may
        # read this as "the model's best guess at this t."
        x0_hat = self.process.predict_x0(model_out, x_t, t)
        logits = self.process.decode_to_logits(x0_hat)

        return DiffusionModelOutput(
            logits=logits,
            cursor_token=cursor_token,
            model_out=model_out,
            loss_target=loss_target,
            t=t,
            x_t=x_t,
        )
