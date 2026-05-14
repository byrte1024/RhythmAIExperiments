"""``FramewiseDiffusionDetector`` — diffusion onset detector with a
framewise activation-map output (#016).

Same trunk as ``EventEmbeddingDetector`` / ``DiffusionDetector``. The
diffusion head operates on a ``(B, n_bins)`` activation map (no STOP
class) and conditions the denoiser on both the cursor token and the
future-half audio token features.

Training-time forward returns a ``FramewiseModelOutput`` with the
denoiser output, the loss target, the sampled timesteps, the noised
state, and the audio features used to condition the denoiser — so the
loss can re-run the diffusion forward inline when given an inference-
shape output.

Inference-time forward returns the cursor token + audio features only;
the decoder runs the sampler externally.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..diffusion import (
    Conv1DDenoiser,
    Conv1DDenoiserConfig,
    CosineSchedule,
    CosineScheduleConfig,
    FramewiseActivationProcess,
    FramewiseActivationProcessConfig,
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
from .diffusion_detector import (
    _KNOWN_DENOISERS,
    _KNOWN_PROCESSES,
    _KNOWN_SCHEDULES,
    register_denoiser,
    register_process,
)
from .event_embedding import (
    EventEmbeddingConfig,
    EventEmbeddingDetector,
    EventEmbeddingInput,
)


# Extend the shared registries used by ``DiffusionDetector`` so the CLI
# / build_config dispatch keeps working without changes.
register_process(FramewiseActivationProcessConfig, FramewiseActivationProcess)
register_denoiser(Conv1DDenoiserConfig, Conv1DDenoiser)


# ─────────────────────────── Output type ──────────────────────────────


@dataclass(frozen=True, slots=True)
class FramewiseModelOutput(ModelOutput):
    """Framewise-diffusion-detector output.

    Two regimes:

    - **Training**: ``forward_diffusion`` populates ``model_out``,
      ``loss_target``, ``t``, ``x_t``. ``logits`` is the predicted
      activation map (B, n_bins) at the sampled t — used by metrics
      that want a per-bin prediction at training time. It is NOT the
      same as the inference-time prediction (which requires the full
      sampler).
    - **Inference**: only ``cursor_token`` + ``audio_features`` are
      set; the decoder runs the sampler. Training fields are ``None``.
    """
    logits: torch.Tensor                              # (B, n_bins) float32
    cursor_token: torch.Tensor                        # (B, d_model) float32
    audio_features: torch.Tensor | None = None        # (B, T_audio, d_model)
    model_out: torch.Tensor | None = None             # (B, n_bins)
    loss_target: torch.Tensor | None = None           # (B, n_bins)
    t: torch.Tensor | None = None                     # (B,) int64
    x_t: torch.Tensor | None = None                   # (B, n_bins)


# ─────────────────────────── Config ───────────────────────────────────


def _default_schedule_config() -> NoiseScheduleConfig:
    return CosineScheduleConfig(n_steps=64)


def _default_process_config() -> DiffusionProcessConfig:
    return FramewiseActivationProcessConfig(
        n_bins=500, parameterization="x0",
    )


def _default_denoiser_config() -> DenoiserConfig:
    return Conv1DDenoiserConfig(
        d_model=384, n_bins=500,
        audio_feature_dim=384, audio_token_count=125,
        self_cond=True,
        conv_channels=256, conv_kernels=(31, 15, 15),
    )


@dataclass(frozen=True, slots=True)
class FramewiseDiffusionDetectorConfig(EventEmbeddingConfig):
    """``EventEmbeddingConfig`` + framewise diffusion-stack sub-configs.

    Unlike ``DiffusionDetectorConfig``, the activation map has no STOP
    class — ``b_pred == n_bins`` directly (not ``b_pred + 1``). The
    denoiser additionally exposes ``audio_feature_dim`` and
    ``audio_token_count`` so the Conv1d-on-bin-axis denoiser can mix
    in future-half audio tokens from the trunk.
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
        # Framewise: no STOP class. b_pred IS n_bins.
        if self.process_config.n_bins != self.b_pred:
            raise ValueError(
                f"process_config.n_bins ({self.process_config.n_bins}) "
                f"!= b_pred ({self.b_pred}); the framewise diffusion head "
                f"produces n_bins outputs which must equal b_pred (no "
                f"STOP class in framewise mode)."
            )
        # Denoiser needs audio_feature_dim + audio_token_count for the
        # Conv1d-on-bin-axis denoiser. If the concrete denoiser doesn't
        # need audio (e.g. MLPDenoiser), audio_feature_dim is irrelevant
        # — but we still check it exists when the field is declared.
        afd = getattr(self.denoiser_config, "audio_feature_dim", None)
        if afd is not None and afd != self.d_model:
            raise ValueError(
                f"denoiser_config.audio_feature_dim ({afd}) "
                f"!= EventEmbeddingConfig.d_model ({self.d_model})"
            )
        atc = getattr(self.denoiser_config, "audio_token_count", None)
        if atc is not None:
            # Future-half audio tokens = b_bins / 4 (conv stride 4).
            expected = self.b_bins // 4
            if atc != expected:
                raise ValueError(
                    f"denoiser_config.audio_token_count ({atc}) "
                    f"!= b_bins // 4 ({expected}); the audio features "
                    f"fed to the denoiser are the future-half audio "
                    f"tokens after stride-4 conv stem."
                )


# ─────────────────────────── Detector ─────────────────────────────────


class FramewiseDiffusionDetector(EventEmbeddingDetector):
    """``EventEmbeddingDetector`` with a framewise diffusion head.

    Same trunk as ``DiffusionDetector``. Two forwards:

    - ``predict(input)`` — trunk forward; returns ``cursor_token`` and
      ``audio_features`` (future-half audio tokens). The decoder runs
      the sampler externally against these.
    - ``forward_diffusion(cursor_token, audio_features, target_map)`` —
      training. Samples ``t`` and ``noise``; ``x_t = q_sample(x_0)``;
      runs the denoiser (with optional Analog-Bits two-pass self-cond
      when the denoiser opts in); returns the structured fields the
      loss consumes.

    Parent's standard head (``head_norm``, ``head_proj``,
    ``head_smooth``) is unused; left in place for parameter-dict
    compatibility (zero gradient since the loss never touches it).
    """

    config: FramewiseDiffusionDetectorConfig

    def __init__(self, config: FramewiseDiffusionDetectorConfig):
        super().__init__(config)
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
        self.denoiser: DenoiserHead = den_cls(den_cfg)

    # ── trunk forward returning both cursor + audio features ─────────

    def _trunk_forward(
        self, x: EventEmbeddingInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the trunk and return (cursor_token, audio_tokens).

        Mirrors ``EventEmbeddingDetector.get_cursor_token`` but also
        keeps the full per-token feature sequence for downstream
        audio-feature extraction. Returns:
          - cursor_token:  (B, d_model) — the token at the cursor.
          - audio_tokens:  (B, n_audio_tokens, d_model) — the full
                           audio token sequence after the trunk.
        """
        c = self.config
        B = x.mel.size(0)
        d = c.d_model

        cond = self.cond_mlp(x.conditioning)
        h = self.conv_stem(x.mel)
        audio_positions = torch.arange(
            h.size(1), device=h.device,
        ).unsqueeze(0).expand(B, -1)
        h = h + self.audio_pos_emb(audio_positions)
        h = self.film_conv(h, cond)

        event_embs, token_pos, in_window = self._build_event_embeddings(
            x.event_offsets, x.event_mask,
        )
        for b in range(B):
            valid_idx = in_window[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                continue
            tpos = token_pos[b, valid_idx]
            embs = event_embs[b, valid_idx]
            h[b].scatter_add_(
                0, tpos.unsqueeze(-1).expand(-1, d), embs,
            )

        for layer, film in zip(self.layers, self.film_layers):
            h = layer(h)
            h = film(h, cond)

        cursor_tok = h[:, c.cursor_token, :]
        return cursor_tok, h

    def get_audio_features(self, x: EventEmbeddingInput) -> torch.Tensor:
        """Return the future-half audio token sequence (B, b_bins//4, d_model).

        For ``a_bins=500, b_bins=500`` and stride-4 conv, the cursor
        token sits at index 125 and the future half is tokens [125:250],
        shape (B, 125, d_model).
        """
        _, audio_tokens = self._trunk_forward(x)
        c = self.config
        return audio_tokens[:, c.cursor_token:, :]

    # ── inference-mode forward ───────────────────────────────────────

    def predict(self, x: EventEmbeddingInput) -> FramewiseModelOutput:
        """Trunk-only forward. Returns cursor token + future audio
        features. The decoder runs the diffusion sampler against these."""
        cursor_token, audio_tokens = self._trunk_forward(x)
        c = self.config
        audio_features = audio_tokens[:, c.cursor_token:, :]
        return FramewiseModelOutput(
            logits=torch.zeros(
                cursor_token.size(0), self.process.config.n_bins,
                device=cursor_token.device, dtype=cursor_token.dtype,
            ),
            cursor_token=cursor_token,
            audio_features=audio_features,
        )

    # ── training-mode forward ────────────────────────────────────────

    def forward_diffusion(
        self,
        cursor_token: torch.Tensor,
        audio_features: torch.Tensor,
        target_map: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        self_cond_prob: float = 0.5,
    ) -> FramewiseModelOutput:
        """One diffusion training step from a precomputed cursor token +
        audio features.

        ``target_map`` is the (B, n_bins) smoothed activation map. The
        diffusion process treats it as the clean ``x_0`` directly (no
        one-hot encoding, no STOP).
        """
        B = cursor_token.size(0)
        device = cursor_token.device

        x_0 = self.process.encode_x0(target_map)                # (B, n_bins)
        if t is None:
            t = torch.randint(
                0, self.schedule.n_steps, (B,),
                device=device, dtype=torch.long,
            )
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.process.q_sample(x_0, t, noise=noise)

        self_cond = bool(self.denoiser.config.self_cond)
        prev_x0_hat: torch.Tensor | None = None
        if self_cond and self_cond_prob > 0.0 and self.training:
            mask = torch.rand(B, device=device) < self_cond_prob
            if bool(mask.any()):
                with torch.no_grad():
                    first_pass = self.denoiser(
                        cursor_token, x_t, t,
                        prev_x0_hat=None,
                        audio_features=audio_features,
                    )
                    sc_x0 = self.process.predict_x0(
                        first_pass, x_t, t,
                    ).detach()
                prev_x0_hat = torch.where(
                    mask.view(-1, 1), sc_x0, torch.zeros_like(sc_x0),
                )

        model_out = self.denoiser(
            cursor_token, x_t, t,
            prev_x0_hat=prev_x0_hat if self_cond else None,
            audio_features=audio_features,
        )
        loss_target = self.process.loss_target(x_0, x_t, t, noise=noise)
        x0_hat = self.process.predict_x0(model_out, x_t, t)
        logits = self.process.decode_to_logits(x0_hat)

        return FramewiseModelOutput(
            logits=logits,
            cursor_token=cursor_token,
            audio_features=audio_features,
            model_out=model_out,
            loss_target=loss_target,
            t=t,
            x_t=x_t,
        )
