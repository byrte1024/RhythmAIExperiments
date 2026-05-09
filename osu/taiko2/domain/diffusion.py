"""ABCs for the diffusion-based onset prediction head (#014).

The diffusion stack has four pluggable pieces, each behind an ABC, each
configured via a frozen dataclass:

  1. ``NoiseSchedule``      — schedule of noise levels across T steps.
  2. ``DiffusionProcess``   — forward (q) and reverse process; defines
                              the noise space (continuous Gaussian on
                              logits, discrete D3PM, GMM-parameter, …)
                              and the parameterization the denoiser
                              targets (predict-x0, predict-noise,
                              predict-v).
  3. ``DenoiserHead``       — the neural net that maps
                              ``(cursor_token, x_t, t) → model_out``.
                              Concrete impls: MLP, conv-on-bin-axis,
                              transformer-on-bin-axis, …
  4. ``DiffusionSampler``   — inference-time loop that drives the
                              denoiser from x_T to x_0. Concrete impls:
                              full DDPM, DDIM, ancestral, consistency.

Each ABC is intentionally minimal — enough for any concrete pair to
talk, but no more. Polymorphism on ``x_t`` shape lives at the
``(DiffusionProcess, DenoiserHead)`` pair level; the ABCs just specify
``torch.Tensor`` and let the pair agree on layout.

These ABCs are stable per CLAUDE.md §2: do not change signatures
without explicit approval. Concrete implementations live in
``osu/taiko2/diffusion/``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


# ─────────────────────────── NoiseSchedule ────────────────────────────


@dataclass(frozen=True, slots=True)
class NoiseScheduleConfig:
    """Schedule-level config common to all schedules.

    Concrete schedules add their own fields by subclassing.
    """
    n_steps: int = 64

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1 (got {self.n_steps})")


class NoiseSchedule(ABC):
    """Discrete sequence of noise levels indexed by ``t ∈ {0, …, n_steps-1}``.

    Conventions (chosen to match modern DDPM literature):

    - **t = 0** is the cleanest end (closest to ``x_0``).
    - **t = n_steps - 1** is the noisiest end (closest to the prior).
    - ``alpha_bar(t)`` is the cumulative signal preservation in ``[0, 1]``:
      ``alpha_bar(0) ≈ 1`` (signal almost intact), ``alpha_bar(T-1) ≈ 0``
      (signal nearly destroyed).
    - ``beta(t)`` is the per-step noise rate; concrete process classes
      interpret it (variance for Gaussian; absorbing-probability for
      D3PM; …).

    Schedules are stateless w.r.t. the noised data — they only carry
    the per-step parameters. They get instantiated once per training
    and once per inference and reused across batches.
    """

    config: NoiseScheduleConfig

    def __init__(self, config: NoiseScheduleConfig):
        self.config = config

    @property
    def n_steps(self) -> int:
        return self.config.n_steps

    @abstractmethod
    def betas(self) -> torch.Tensor:
        """Per-step beta values, shape ``(n_steps,)`` float32, on CPU.

        The interpretation of ``beta_t`` is process-specific:
        - Gaussian: variance of per-step noise; ``alpha_t = 1 - beta_t``.
        - D3PM absorbing: probability of being absorbed at step t.
        - D3PM uniform: probability of being uniformly randomized.

        Schedules construct this tensor lazily on first call and may
        cache; concrete implementations should not assume mutability.
        """
        ...

    def alphas(self) -> torch.Tensor:
        """``1 - betas``. Default implementation derives from ``betas()``."""
        return 1.0 - self.betas()

    def alphas_cumprod(self) -> torch.Tensor:
        """Cumulative product ``∏_{s ≤ t} alpha_s``, shape ``(n_steps,)``.

        ``alphas_cumprod[0] = alpha_0`` (clean end), and for the
        Gaussian process this is the squared scaling of ``x_0`` in
        ``x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise``.
        """
        return torch.cumprod(self.alphas(), dim=0)


# ─────────────────────────── DiffusionProcess ─────────────────────────


@dataclass(frozen=True, slots=True)
class DiffusionProcessConfig:
    """Process-level config common to all processes.

    Concrete processes add their own fields by subclassing.
    Subclasses choose how the schedule's ``betas()`` are interpreted.
    """
    n_bins: int = 501
    parameterization: str = "x0"      # "x0" | "noise" | "v"

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2 (got {self.n_bins})")
        if self.parameterization not in {"x0", "noise", "v"}:
            raise ValueError(
                f"parameterization must be one of x0/noise/v "
                f"(got {self.parameterization!r})"
            )


class DiffusionProcess(ABC):
    """Forward and reverse diffusion process.

    Defines:
    - The noise space (what shape ``x_t`` has and what it represents).
    - The forward noising distribution ``q(x_t | x_0)``.
    - The training target the denoiser predicts (``x_0``, noise, or v).
    - One step of the reverse process ``p(x_{t-1} | x_t)``.

    Each concrete subclass owns its ``NoiseSchedule`` (passed in at
    construction). The ABC does not own the schedule directly — that's
    a concrete-class concern, since the relationship between the
    schedule's ``betas`` and the process's noise dynamics is process-
    specific.
    """

    config: DiffusionProcessConfig
    schedule: NoiseSchedule

    def __init__(
        self,
        config: DiffusionProcessConfig,
        schedule: NoiseSchedule,
    ):
        self.config = config
        self.schedule = schedule

    # ── forward noising ──────────────────────────────────────────────

    @abstractmethod
    def encode_x0(self, target_bin: torch.Tensor) -> torch.Tensor:
        """Convert a GT bin index ``(B,) int64`` to the process's
        ``x_0`` representation.

        - Gaussian: a one-hot (or scaled one-hot) over ``n_bins``,
          shape ``(B, n_bins)`` float32.
        - D3PM absorbing: a one-hot vector over ``n_bins`` (the same
          tensor; D3PM operates in categorical space).
        - GMM: parameter tensor for a delta-mode mixture.

        Concrete implementations decide. The ABC commits only to
        "returns a torch.Tensor whose shape matches what subsequent
        process methods expect for x_0/x_t/etc."
        """
        ...

    @abstractmethod
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``x_t ~ q(x_t | x_0)``.

        ``t`` is shape ``(B,) int64`` with values in
        ``[0, n_steps - 1]``. ``noise`` is optional pre-sampled
        randomness (pass ``None`` to sample fresh; useful for
        reproducible eval).

        Returns ``x_t`` with the same shape as ``x_0``.
        """
        ...

    # ── parameterization conversions ─────────────────────────────────

    @abstractmethod
    def loss_target(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Given GT ``x_0``, sampled ``x_t``, timesteps, and (optional)
        the noise that produced ``x_t``, return the target the
        denoiser should predict.

        Interpretation depends on ``config.parameterization``:
        - ``x0``: target = ``x_0`` itself.
        - ``noise``: target = the noise.
        - ``v``: target = ``v_t = sqrt(alpha_bar_t) * noise -
                            sqrt(1 - alpha_bar_t) * x_0``.

        Concrete classes implement the parameterization they support;
        unsupported choices should raise ``ValueError`` at config-time.
        """
        ...

    @abstractmethod
    def predict_x0(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the denoiser's output back to a predicted ``x_0``.

        For ``parameterization="x0"`` this is identity; for ``"noise"``
        and ``"v"`` it requires the standard inverse formulas using
        ``alphas_cumprod``.
        """
        ...

    # ── reverse process ──────────────────────────────────────────────

    @abstractmethod
    def reverse_step(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse-process step ``x_t → x_{t_prev}``.

        ``t_prev`` is allowed to differ from ``t - 1`` for accelerated
        samplers (DDIM, consistency) which subsample timesteps.

        Returns ``x_{t_prev}`` with the same shape as ``x_t``.

        Stochasticity (Gaussian noise injection) is the concrete
        sampler's job; this method should be deterministic given
        ``model_out``, ``x_t``, ``t``, ``t_prev``. Samplers that want
        stochastic reverse steps add the noise themselves before/after.
        """
        ...

    # ── prior + decoding ─────────────────────────────────────────────

    @abstractmethod
    def sample_prior(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample ``x_T`` from the noisiest end of the process.

        - Gaussian: ``N(0, I)`` over ``n_bins``.
        - D3PM absorbing: all-absorbing-token state.
        - GMM: per-process prior on parameters.
        """
        ...

    @abstractmethod
    def decode_to_logits(self, x_0_hat: torch.Tensor) -> torch.Tensor:
        """Convert a (predicted) ``x_0`` into a ``(B, n_bins)`` float
        tensor that can be argmax'd or fed into downstream metrics.

        Argmax of the result is the predicted bin index. For
        processes whose ``x_0`` is already a logit-space tensor, this
        is identity. For D3PM (one-hot space) we return the one-hot
        directly so argmax recovers the bin.
        """
        ...


# ─────────────────────────── DenoiserHead ─────────────────────────────


@dataclass(frozen=True, slots=True)
class DenoiserConfig:
    """Denoiser-level config.

    ``d_model`` matches the trunk's d_model; ``n_bins`` the output
    bin count; ``time_embed_dim`` the sinusoidal-time-embedding size.
    Concrete denoisers add their own fields.
    """
    d_model: int = 384
    n_bins: int = 501
    time_embed_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1 (got {self.d_model})")
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2 (got {self.n_bins})")
        if self.time_embed_dim < 1:
            raise ValueError(
                f"time_embed_dim must be >= 1 (got {self.time_embed_dim})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1) (got {self.dropout})")


class DenoiserHead(nn.Module, ABC):
    """The neural net component of the diffusion head.

    Takes the trunk's per-cursor conditioning ``cursor_token``, the
    noised state ``x_t``, and the timestep index ``t``, and returns
    the denoiser output (interpretation per the paired
    ``DiffusionProcess.config.parameterization``).

    Concrete denoisers must agree with the paired process on the
    shape of ``x_t``. The ABC commits to:

    - cursor_token: ``(B, d_model)`` float32 conditioning.
    - x_t:          process-defined shape; usually ``(B, n_bins)``.
    - t:            ``(B,) int64`` timestep indices.
    - return:       same shape as ``x_t``.
    """

    config: DenoiserConfig

    def __init__(self, config: DenoiserConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        cursor_token: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        ...


# ─────────────────────────── DiffusionSampler ─────────────────────────


@dataclass(frozen=True, slots=True)
class DiffusionSamplerConfig:
    """Sampler-level config common to all samplers.

    ``n_inference_steps`` may differ from the training schedule's
    ``n_steps`` (accelerated samplers like DDIM subsample timesteps).
    """
    n_inference_steps: int = 16
    eta: float = 0.0                  # DDIM stochasticity (0 = deterministic)

    def __post_init__(self) -> None:
        if self.n_inference_steps < 1:
            raise ValueError(
                f"n_inference_steps must be >= 1 "
                f"(got {self.n_inference_steps})"
            )
        if self.eta < 0.0:
            raise ValueError(f"eta must be >= 0 (got {self.eta})")


class DiffusionSampler(ABC):
    """Inference-time loop driving the denoiser from ``x_T`` to ``x_0``.

    Owns references to the ``DiffusionProcess`` and ``DenoiserHead``;
    does not own the trunk (the trunk's ``cursor_token`` is passed in
    per call). This separation is intentional — the trunk runs once
    per AR step and produces conditioning; the sampler runs the
    denoiser ``n_inference_steps`` times against that conditioning.
    """

    config: DiffusionSamplerConfig
    process: DiffusionProcess
    denoiser: DenoiserHead

    def __init__(
        self,
        config: DiffusionSamplerConfig,
        process: DiffusionProcess,
        denoiser: DenoiserHead,
    ):
        if config.n_inference_steps > process.schedule.n_steps:
            raise ValueError(
                f"n_inference_steps ({config.n_inference_steps}) > "
                f"schedule.n_steps ({process.schedule.n_steps})"
            )
        self.config = config
        self.process = process
        self.denoiser = denoiser

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        cursor_token: torch.Tensor,
        x_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the reverse process from ``x_T`` to ``x_0``.

        ``cursor_token`` shape ``(B, d_model)`` from the trunk.
        ``x_T`` may be passed in for reproducible inference; if
        ``None``, sampled fresh from ``process.sample_prior``.

        Returns logits-shape ``(B, n_bins)`` ready for argmax (via
        ``process.decode_to_logits``). The sampler may internally
        track and return additional intermediate states, but the
        canonical return is the final logits.
        """
        ...

    @abstractmethod
    def timesteps(self) -> torch.Tensor:
        """The descending sequence of timesteps the sampler visits.

        Length ``= n_inference_steps``. For full DDPM this is
        ``[T-1, T-2, …, 0]``; for DDIM/accelerated samplers it
        subsamples.
        """
        ...
