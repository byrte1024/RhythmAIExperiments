"""Tests for the diffusion ABCs and reference concrete implementations.

Covers:
  - Abstract-method enforcement (subclassing without overrides fails).
  - Config validation (bad fields raise at construction time).
  - Schedule shape / monotonicity / endpoint conventions.
  - Process round-trips (encode, q_sample, predict_x0, parameterization
    consistency).
  - Denoiser forward shape + gradient flow.
  - Sampler timestep generation, sample shape, end-to-end training step.
  - Cross-product sanity: schedule × process × denoiser × sampler all
    compose without shape errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from osu.taiko2.diffusion import (
    CosineSchedule,
    CosineScheduleConfig,
    DDIMSampler,
    DDIMSamplerConfig,
    DDPMSampler,
    GaussianContinuousProcess,
    GaussianContinuousProcessConfig,
    LinearSchedule,
    LinearScheduleConfig,
    MLPDenoiser,
    MLPDenoiserConfig,
)
from osu.taiko2.domain.diffusion import (
    DenoiserConfig,
    DenoiserHead,
    DiffusionProcess,
    DiffusionProcessConfig,
    DiffusionSampler,
    DiffusionSamplerConfig,
    NoiseSchedule,
    NoiseScheduleConfig,
)


# ─────────────────────────── ABC enforcement ──────────────────────────


class TestAbcEnforcement:
    """Subclassing the ABCs without overriding their abstract methods
    must raise ``TypeError``."""

    def test_noise_schedule_abstract(self):
        class Empty(NoiseSchedule):
            pass
        with pytest.raises(TypeError, match="abstract"):
            Empty(NoiseScheduleConfig(n_steps=10))

    def test_diffusion_process_abstract(self):
        class Empty(DiffusionProcess):
            pass
        sched = LinearSchedule(LinearScheduleConfig(n_steps=10))
        with pytest.raises(TypeError, match="abstract"):
            Empty(DiffusionProcessConfig(n_bins=10), sched)

    def test_denoiser_head_abstract(self):
        class Empty(DenoiserHead):
            pass
        with pytest.raises(TypeError, match="abstract"):
            Empty(DenoiserConfig(d_model=8, n_bins=10))

    def test_diffusion_sampler_abstract(self):
        class Empty(DiffusionSampler):
            pass
        sched = LinearSchedule(LinearScheduleConfig(n_steps=10))
        proc = GaussianContinuousProcess(
            GaussianContinuousProcessConfig(n_bins=10), sched,
        )
        denoiser = MLPDenoiser(MLPDenoiserConfig(
            d_model=8, n_bins=10, hidden_dim=16, n_layers=1,
        ))
        with pytest.raises(TypeError, match="abstract"):
            Empty(DiffusionSamplerConfig(n_inference_steps=10), proc, denoiser)


# ─────────────────────────── Config validation ────────────────────────


class TestConfigValidation:
    """Bad config fields must raise at construction."""

    def test_noise_schedule_n_steps(self):
        with pytest.raises(ValueError, match="n_steps"):
            NoiseScheduleConfig(n_steps=0)
        with pytest.raises(ValueError, match="n_steps"):
            NoiseScheduleConfig(n_steps=-1)

    def test_linear_schedule_betas(self):
        with pytest.raises(ValueError, match="beta_start"):
            LinearScheduleConfig(beta_start=0.0)
        with pytest.raises(ValueError, match="beta_end"):
            LinearScheduleConfig(beta_end=1.0)
        with pytest.raises(ValueError, match="beta_start.*beta_end"):
            LinearScheduleConfig(beta_start=0.5, beta_end=0.1)

    def test_cosine_schedule(self):
        with pytest.raises(ValueError, match="^s "):
            CosineScheduleConfig(s=-0.1)
        with pytest.raises(ValueError, match="max_beta"):
            CosineScheduleConfig(max_beta=1.0)
        with pytest.raises(ValueError, match="max_beta"):
            CosineScheduleConfig(max_beta=0.0)

    def test_process_n_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            DiffusionProcessConfig(n_bins=1)

    def test_process_parameterization(self):
        with pytest.raises(ValueError, match="parameterization"):
            DiffusionProcessConfig(parameterization="bogus")

    def test_gaussian_x0_scale(self):
        with pytest.raises(ValueError, match="x0_scale"):
            GaussianContinuousProcessConfig(x0_scale=0.0)
        with pytest.raises(ValueError, match="x0_scale"):
            GaussianContinuousProcessConfig(x0_scale=-1.0)

    def test_denoiser_dim_validation(self):
        with pytest.raises(ValueError, match="d_model"):
            DenoiserConfig(d_model=0)
        with pytest.raises(ValueError, match="n_bins"):
            DenoiserConfig(n_bins=1)
        with pytest.raises(ValueError, match="time_embed_dim"):
            DenoiserConfig(time_embed_dim=0)
        with pytest.raises(ValueError, match="dropout"):
            DenoiserConfig(dropout=1.5)
        with pytest.raises(ValueError, match="dropout"):
            DenoiserConfig(dropout=-0.1)

    def test_mlp_denoiser_validation(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            MLPDenoiserConfig(hidden_dim=0)
        with pytest.raises(ValueError, match="time_embed_proj_dim"):
            MLPDenoiserConfig(time_embed_proj_dim=-1)
        with pytest.raises(ValueError, match="n_layers"):
            MLPDenoiserConfig(n_layers=0)

    def test_sampler_validation(self):
        with pytest.raises(ValueError, match="n_inference_steps"):
            DiffusionSamplerConfig(n_inference_steps=0)
        with pytest.raises(ValueError, match="eta"):
            DiffusionSamplerConfig(eta=-0.1)

    def test_ddim_spacing(self):
        with pytest.raises(ValueError, match="timestep_spacing"):
            DDIMSamplerConfig(timestep_spacing="bogus")


# ─────────────────────────── Schedules ────────────────────────────────


class TestSchedules:
    """Shape, monotonicity, and endpoint conventions."""

    @pytest.mark.parametrize("T", [4, 16, 64, 1000])
    def test_linear_shape_and_monotone(self, T):
        sched = LinearSchedule(LinearScheduleConfig(
            n_steps=T, beta_start=1e-4, beta_end=2e-2,
        ))
        b = sched.betas()
        assert b.shape == (T,)
        # monotonic increasing
        assert torch.all(b[1:] >= b[:-1])
        # in (0, 1)
        assert (b > 0).all() and (b < 1).all()

    @pytest.mark.parametrize("T", [4, 16, 64, 1000])
    def test_cosine_shape_and_alpha_bar(self, T):
        sched = CosineSchedule(CosineScheduleConfig(n_steps=T))
        b = sched.betas()
        assert b.shape == (T,)
        ab = sched.alphas_cumprod()
        assert ab.shape == (T,)
        # alpha_bar approaches 0 at the noisy end.
        assert ab[-1] < 0.05
        # Monotonically decreasing.
        assert torch.all(ab[1:] <= ab[:-1])
        # For non-trivial T, alpha_bar at the clean end (after one
        # step) should still be close to 1. At very small T (e.g. 4)
        # one step already takes a meaningful chunk out, so we relax
        # this assertion to T >= 32.
        if T >= 32:
            assert ab[0] > 0.95

    def test_alphas_match_1_minus_betas(self):
        sched = LinearSchedule(LinearScheduleConfig(n_steps=32))
        a = sched.alphas()
        b = sched.betas()
        torch.testing.assert_close(a, 1.0 - b)

    def test_cumprod_matches_manual(self):
        sched = CosineSchedule(CosineScheduleConfig(n_steps=16))
        ab = sched.alphas_cumprod()
        manual = torch.cumprod(sched.alphas(), dim=0)
        torch.testing.assert_close(ab, manual)


# ─────────────────────────── Process ──────────────────────────────────


@pytest.fixture
def gaussian_process():
    sched = CosineSchedule(CosineScheduleConfig(n_steps=64))
    proc = GaussianContinuousProcess(
        GaussianContinuousProcessConfig(
            n_bins=128, parameterization="x0", x0_scale=2.0,
        ),
        sched,
    )
    return proc


class TestGaussianProcess:
    def test_encode_x0_shape_and_scale(self, gaussian_process):
        target = torch.tensor([0, 50, 127, 1])
        x_0 = gaussian_process.encode_x0(target)
        assert x_0.shape == (4, 128)
        # argmax recovers the target
        assert torch.equal(x_0.argmax(-1), target)
        # scaled by x0_scale=2.0
        assert float(x_0.max()) == pytest.approx(2.0)

    def test_q_sample_shape(self, gaussian_process):
        target = torch.tensor([10, 20, 30])
        x_0 = gaussian_process.encode_x0(target)
        t = torch.tensor([0, 32, 63])
        x_t = gaussian_process.q_sample(x_0, t)
        assert x_t.shape == x_0.shape

    def test_q_sample_at_t0_is_close_to_x0(self, gaussian_process):
        # At t=0, alpha_bar ≈ 1, so x_t ≈ x_0.
        target = torch.tensor([10, 20])
        x_0 = gaussian_process.encode_x0(target)
        t = torch.zeros(2, dtype=torch.long)
        noise = torch.zeros_like(x_0)               # zero noise for cleanliness
        x_t = gaussian_process.q_sample(x_0, t, noise=noise)
        # With cosine schedule the cleanest end has alpha_bar very close to 1
        sqrt_ab0 = float(gaussian_process._sqrt_alphas_cumprod[0])
        torch.testing.assert_close(x_t, sqrt_ab0 * x_0)

    @pytest.mark.parametrize("param", ["x0", "noise", "v"])
    def test_loss_target_param_round_trip(self, param):
        sched = CosineSchedule(CosineScheduleConfig(n_steps=32))
        proc = GaussianContinuousProcess(
            GaussianContinuousProcessConfig(
                n_bins=64, parameterization=param, x0_scale=2.0,
            ),
            sched,
        )
        target_bin = torch.randint(0, 64, (3,))
        x_0 = proc.encode_x0(target_bin)
        t = torch.tensor([5, 15, 25])
        noise = torch.randn_like(x_0)
        x_t = proc.q_sample(x_0, t, noise=noise)
        target = proc.loss_target(x_0, x_t, t, noise=noise)

        # Round-trip: predict_x0 with target as model_out should give x_0
        x_0_recovered = proc.predict_x0(target, x_t, t)
        torch.testing.assert_close(x_0_recovered, x_0, rtol=1e-4, atol=1e-4)

    def test_reverse_step_shape(self, gaussian_process):
        x_0 = gaussian_process.encode_x0(torch.tensor([10, 20, 30]))
        t = torch.tensor([5, 15, 25])
        x_t = gaussian_process.q_sample(x_0, t)
        model_out = x_0                              # x0 parameterization, oracle output
        t_prev = t - 1
        x_prev = gaussian_process.reverse_step(model_out, x_t, t, t_prev)
        assert x_prev.shape == x_t.shape

    def test_reverse_step_oracle_recovers_x0(self, gaussian_process):
        # If denoiser perfectly outputs x_0, reverse_step at t=0 should
        # be ~ x_0 itself.
        x_0 = gaussian_process.encode_x0(torch.tensor([5]))
        t = torch.tensor([0])
        x_t = gaussian_process.q_sample(x_0, t, noise=torch.zeros_like(x_0))
        t_prev = torch.tensor([-1])
        out = gaussian_process.reverse_step(x_0, x_t, t, t_prev)
        torch.testing.assert_close(out, x_0, rtol=1e-3, atol=1e-3)

    def test_sample_prior_shape(self, gaussian_process):
        x_T = gaussian_process.sample_prior(7, torch.device("cpu"))
        assert x_T.shape == (7, 128)
        # Should be ~unit-variance Gaussian
        assert abs(float(x_T.std()) - 1.0) < 0.5

    def test_decode_to_logits_argmax_preserved(self, gaussian_process):
        x_0 = gaussian_process.encode_x0(torch.tensor([10, 50, 100]))
        logits = gaussian_process.decode_to_logits(x_0)
        assert torch.equal(logits.argmax(-1), torch.tensor([10, 50, 100]))


# ─────────────────────────── Denoiser ─────────────────────────────────


class TestMLPDenoiser:
    def test_forward_shape(self):
        d = MLPDenoiser(MLPDenoiserConfig(
            d_model=64, n_bins=32, hidden_dim=128, n_layers=2,
        ))
        cursor = torch.randn(4, 64)
        x_t = torch.randn(4, 32)
        t = torch.randint(0, 16, (4,))
        out = d(cursor, x_t, t)
        assert out.shape == x_t.shape

    def test_gradient_flow(self):
        d = MLPDenoiser(MLPDenoiserConfig(
            d_model=32, n_bins=16, hidden_dim=64, n_layers=1,
        ))
        cursor = torch.randn(2, 32, requires_grad=True)
        x_t = torch.randn(2, 16)
        t = torch.zeros(2, dtype=torch.long)
        out = d(cursor, x_t, t)
        loss = out.pow(2).mean()
        loss.backward()
        # All denoiser params should have non-None grads
        assert all(p.grad is not None for p in d.parameters())
        assert cursor.grad is not None

    def test_time_embedding_changes_output(self):
        d = MLPDenoiser(MLPDenoiserConfig(
            d_model=32, n_bins=16, hidden_dim=64, n_layers=1,
            dropout=0.0,
        ))
        d.eval()
        cursor = torch.randn(1, 32)
        x_t = torch.randn(1, 16)
        out_t0 = d(cursor, x_t, torch.tensor([0]))
        out_t10 = d(cursor, x_t, torch.tensor([10]))
        # Different timesteps must produce different outputs
        assert not torch.allclose(out_t0, out_t10)

    def test_no_proj_path(self):
        # time_embed_proj_dim=0 path uses raw sinusoidal embed
        d = MLPDenoiser(MLPDenoiserConfig(
            d_model=32, n_bins=16, hidden_dim=64,
            time_embed_dim=64, time_embed_proj_dim=0,
        ))
        cursor = torch.randn(2, 32)
        x_t = torch.randn(2, 16)
        t = torch.tensor([3, 7])
        out = d(cursor, x_t, t)
        assert out.shape == x_t.shape


# ─────────────────────────── Samplers ─────────────────────────────────


def _make_pipeline(n_steps=32, n_bins=64, d_model=32):
    """Helper: build a full schedule+process+denoiser stack."""
    sched = CosineSchedule(CosineScheduleConfig(n_steps=n_steps))
    proc = GaussianContinuousProcess(
        GaussianContinuousProcessConfig(
            n_bins=n_bins, parameterization="x0", x0_scale=2.0,
        ),
        sched,
    )
    denoiser = MLPDenoiser(MLPDenoiserConfig(
        d_model=d_model, n_bins=n_bins, hidden_dim=64, n_layers=1,
    )).eval()
    return sched, proc, denoiser


class TestDDPMSampler:
    def test_requires_full_schedule(self):
        sched, proc, denoiser = _make_pipeline(n_steps=32)
        with pytest.raises(ValueError, match="schedule.n_steps"):
            DDPMSampler(
                DiffusionSamplerConfig(n_inference_steps=8, eta=1.0),
                proc, denoiser,
            )

    def test_timesteps_descending_full(self):
        sched, proc, denoiser = _make_pipeline(n_steps=8)
        s = DDPMSampler(
            DiffusionSamplerConfig(n_inference_steps=8, eta=1.0),
            proc, denoiser,
        )
        ts = s.timesteps()
        assert ts.tolist() == list(range(7, -1, -1))

    def test_sample_shape(self):
        sched, proc, denoiser = _make_pipeline(n_steps=8, n_bins=32, d_model=16)
        s = DDPMSampler(
            DiffusionSamplerConfig(n_inference_steps=8, eta=0.0),
            proc, denoiser,
        )
        cursor = torch.randn(3, 16)
        out = s.sample(cursor)
        assert out.shape == (3, 32)


class TestDDIMSampler:
    def test_timesteps_descending(self):
        sched, proc, denoiser = _make_pipeline(n_steps=64)
        s = DDIMSampler(
            DDIMSamplerConfig(
                n_inference_steps=16, eta=0.0, timestep_spacing="linspace",
            ),
            proc, denoiser,
        )
        ts = s.timesteps()
        # descending
        assert torch.all(ts[1:] < ts[:-1])
        # all in [0, 63]
        assert ts.min() >= 0 and ts.max() <= 63

    @pytest.mark.parametrize("spacing", ["linspace", "trailing", "leading"])
    def test_spacing_variants(self, spacing):
        sched, proc, denoiser = _make_pipeline(n_steps=64)
        s = DDIMSampler(
            DDIMSamplerConfig(
                n_inference_steps=8, eta=0.0, timestep_spacing=spacing,
            ),
            proc, denoiser,
        )
        ts = s.timesteps()
        # All distinct, descending
        assert len(set(ts.tolist())) == len(ts)
        assert torch.all(ts[1:] < ts[:-1])

    def test_sample_shape(self):
        sched, proc, denoiser = _make_pipeline(n_steps=32, n_bins=64, d_model=16)
        s = DDIMSampler(
            DDIMSamplerConfig(
                n_inference_steps=8, eta=0.0, timestep_spacing="linspace",
            ),
            proc, denoiser,
        )
        cursor = torch.randn(2, 16)
        out = s.sample(cursor)
        assert out.shape == (2, 64)

    def test_reproducible_with_fixed_x_T(self):
        sched, proc, denoiser = _make_pipeline(n_steps=16, n_bins=32, d_model=8)
        s = DDIMSampler(
            DDIMSamplerConfig(
                n_inference_steps=4, eta=0.0, timestep_spacing="linspace",
            ),
            proc, denoiser,
        )
        cursor = torch.randn(2, 8)
        x_T = torch.randn(2, 32)
        out1 = s.sample(cursor, x_T=x_T)
        out2 = s.sample(cursor, x_T=x_T)
        torch.testing.assert_close(out1, out2)

    def test_validates_inference_steps(self):
        sched, proc, denoiser = _make_pipeline(n_steps=16)
        with pytest.raises(ValueError, match="n_inference_steps"):
            DDIMSampler(
                DDIMSamplerConfig(n_inference_steps=32, eta=0.0),
                proc, denoiser,
            )


# ─────────────────────────── End-to-end ────────────────────────────────


class TestEndToEndTrainingStep:
    """A full forward + loss + backward pass through a minimal pipeline."""

    @pytest.mark.parametrize("param", ["x0", "noise", "v"])
    def test_training_step_runs_and_grads_flow(self, param):
        n_bins, d_model, B = 32, 16, 4
        sched = CosineSchedule(CosineScheduleConfig(n_steps=16))
        proc = GaussianContinuousProcess(
            GaussianContinuousProcessConfig(
                n_bins=n_bins, parameterization=param, x0_scale=2.0,
            ),
            sched,
        )
        denoiser = MLPDenoiser(MLPDenoiserConfig(
            d_model=d_model, n_bins=n_bins, hidden_dim=64, n_layers=1,
        ))

        target = torch.randint(0, n_bins, (B,))
        x_0 = proc.encode_x0(target)
        t = torch.randint(0, 16, (B,))
        noise = torch.randn_like(x_0)
        x_t = proc.q_sample(x_0, t, noise=noise)

        cursor = torch.randn(B, d_model)
        out = denoiser(cursor, x_t, t)
        target_for_loss = proc.loss_target(x_0, x_t, t, noise=noise)
        loss = ((out - target_for_loss) ** 2).mean()
        loss.backward()

        assert torch.isfinite(loss).item()
        assert all(p.grad is not None for p in denoiser.parameters())

    def test_inference_step_runs(self):
        n_bins, d_model, B = 32, 16, 3
        sched = CosineSchedule(CosineScheduleConfig(n_steps=16))
        proc = GaussianContinuousProcess(
            GaussianContinuousProcessConfig(n_bins=n_bins), sched,
        )
        denoiser = MLPDenoiser(MLPDenoiserConfig(
            d_model=d_model, n_bins=n_bins, hidden_dim=64, n_layers=1,
        )).eval()
        s = DDIMSampler(
            DDIMSamplerConfig(
                n_inference_steps=4, eta=0.0, timestep_spacing="linspace",
            ),
            proc, denoiser,
        )
        cursor = torch.randn(B, d_model)
        with torch.no_grad():
            logits = s.sample(cursor)
        assert logits.shape == (B, n_bins)
        assert torch.isfinite(logits).all()

    def test_perfect_denoiser_recovers_target(self):
        """A denoiser that always outputs x_0 should let the sampler
        recover the target when started from a clean ``x_T = x_0``.

        This exercises the math of reverse_step end-to-end with no
        noise injection.
        """
        n_bins, d_model = 32, 8
        sched = CosineSchedule(CosineScheduleConfig(n_steps=16))
        proc = GaussianContinuousProcess(
            GaussianContinuousProcessConfig(
                n_bins=n_bins, parameterization="x0", x0_scale=1.0,
            ),
            sched,
        )

        target = torch.tensor([5, 15])
        x_0 = proc.encode_x0(target)

        class OracleDenoiser(DenoiserHead):
            def __init__(self, config, x_0):
                super().__init__(config)
                self._x_0 = x_0

            def forward(self, cursor_token, x_t, t):
                return self._x_0

        denoiser = OracleDenoiser(
            MLPDenoiserConfig(d_model=d_model, n_bins=n_bins, hidden_dim=8),
            x_0,
        ).eval()
        s = DDIMSampler(
            DDIMSamplerConfig(
                n_inference_steps=4, eta=0.0, timestep_spacing="linspace",
            ),
            proc, denoiser,
        )
        cursor = torch.randn(2, d_model)
        # Start from x_T = pure noise; oracle should pull it back to x_0
        with torch.no_grad():
            logits = s.sample(cursor)
        assert torch.equal(logits.argmax(-1), target)
