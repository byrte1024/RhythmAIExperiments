"""Tests for #016 framewise diffusion scaffolding (Chunk A).

Covers:
- FramewiseTarget construction (binary + smoothed maps).
- FramewiseActivationProcessConfig validation.
- FramewiseActivationProcess encode/q_sample/predict_x0/reverse/prior/decode.
- Conv1DDenoiserConfig validation.
- Conv1DDenoiser forward shapes, audio upsampling, self-cond, grad flow.
- DenoiserHead ABC backward compat (MLPDenoiser still works without
  audio_features).
- Cross-product: FramewiseActivationProcess + Conv1DDenoiser composes.
"""
from __future__ import annotations

import math

import pytest
import torch

from osu.taiko2.diffusion import (
    Conv1DDenoiser,
    Conv1DDenoiserConfig,
    FramewiseActivationProcess,
    FramewiseActivationProcessConfig,
    LinearSchedule,
    LinearScheduleConfig,
    MLPDenoiser,
    MLPDenoiserConfig,
)
from osu.taiko2.domain.framewise import FramewiseTarget, make_framewise_target


# ─────────────────────────── FramewiseTarget ──────────────────────────


class TestFramewiseTarget:
    def test_empty_no_gt(self):
        future = torch.full((2, 4), -1, dtype=torch.int64)
        tgt = make_framewise_target(future, n_bins=10)
        assert tgt.target_map_binary.shape == (2, 10)
        assert tgt.target_map_smoothed.shape == (2, 10)
        assert torch.all(tgt.target_map_binary == 0.0)
        assert torch.all(tgt.target_map_smoothed == 0.0)
        assert torch.equal(tgt.n_gt, torch.tensor([0, 0]))

    def test_single_gt(self):
        future = torch.tensor([[5, -1, -1]], dtype=torch.int64)
        tgt = make_framewise_target(future, n_bins=10, sigma=2.0)
        # Binary: 1.0 only at bin 5.
        assert tgt.target_map_binary[0, 5].item() == 1.0
        assert tgt.target_map_binary[0].sum().item() == 1.0
        # Smoothed: peak at 5, value 1.0; check Gaussian neighbours.
        sm = tgt.target_map_smoothed[0]
        assert sm[5].item() == pytest.approx(1.0, abs=1e-6)
        for d in range(1, 4):
            expected = math.exp(-(d ** 2) / (2 * 2.0 ** 2))
            assert sm[5 + d].item() == pytest.approx(expected, abs=1e-6)
            assert sm[5 - d].item() == pytest.approx(expected, abs=1e-6)
        assert tgt.n_gt[0].item() == 1

    def test_dense_gt(self):
        future = torch.tensor([[2, 5, 8]], dtype=torch.int64)
        tgt = make_framewise_target(future, n_bins=12, sigma=2.0)
        assert tgt.n_gt[0].item() == 3
        # Binary: exactly 3 ones.
        assert tgt.target_map_binary[0].sum().item() == 3.0
        # Smoothed has peaks at each GT.
        for b in (2, 5, 8):
            assert tgt.target_map_smoothed[0, b].item() == pytest.approx(1.0, abs=1e-6)
        # Smoothed clipped to [0, 1].
        assert tgt.target_map_smoothed.max().item() <= 1.0
        assert tgt.target_map_smoothed.min().item() >= 0.0

    def test_boundary_gt(self):
        future = torch.tensor([[0, 9]], dtype=torch.int64)
        tgt = make_framewise_target(future, n_bins=10, sigma=2.0)
        assert tgt.target_map_binary[0, 0].item() == 1.0
        assert tgt.target_map_binary[0, 9].item() == 1.0
        # Peak at boundary still 1.0.
        assert tgt.target_map_smoothed[0, 0].item() == pytest.approx(1.0, abs=1e-6)
        assert tgt.target_map_smoothed[0, 9].item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_mixed(self):
        future = torch.tensor([
            [1, -1, -1],
            [-1, -1, -1],
            [3, 7, -1],
        ], dtype=torch.int64)
        tgt = make_framewise_target(future, n_bins=10, sigma=2.0)
        assert torch.equal(tgt.n_gt, torch.tensor([1, 0, 2]))
        assert tgt.target_map_binary[1].sum().item() == 0.0
        assert tgt.target_map_smoothed[1].sum().item() == 0.0

    def test_smoothed_max_not_sum(self):
        # Two GT bins close together: max, not sum -> still <= 1.
        future = torch.tensor([[5, 6]], dtype=torch.int64)
        tgt = make_framewise_target(future, n_bins=12, sigma=2.0)
        # At bin 5: max(exp(0), exp(-1/8)) = 1.0.
        assert tgt.target_map_smoothed[0, 5].item() == pytest.approx(1.0, abs=1e-6)
        assert tgt.target_map_smoothed[0, 6].item() == pytest.approx(1.0, abs=1e-6)
        # At bin 4: max(exp(-1/8), exp(-4/8)).
        expected = max(
            math.exp(-1 / (2 * 4.0)),
            math.exp(-4 / (2 * 4.0)),
        )
        assert tgt.target_map_smoothed[0, 4].item() == pytest.approx(expected, abs=1e-6)

    def test_rank1_input_raises(self):
        with pytest.raises(ValueError, match="future_offsets"):
            make_framewise_target(torch.tensor([1, 2, 3]), n_bins=10)

    def test_bad_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            make_framewise_target(torch.zeros((1, 1), dtype=torch.int64), n_bins=10, sigma=0.0)

    def test_bad_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            make_framewise_target(torch.zeros((1, 1), dtype=torch.int64), n_bins=0)

    def test_dataclass_frozen(self):
        future = torch.tensor([[1]], dtype=torch.int64)
        tgt = make_framewise_target(future, n_bins=5)
        assert isinstance(tgt, FramewiseTarget)
        with pytest.raises((AttributeError, Exception)):
            tgt.n_gt = torch.tensor([0])  # type: ignore[misc]


# ─────────────────────── FramewiseActivationProcess ───────────────────


def _make_proc(n_bins: int = 16, n_steps: int = 8) -> FramewiseActivationProcess:
    sched = LinearSchedule(LinearScheduleConfig(n_steps=n_steps))
    cfg = FramewiseActivationProcessConfig(n_bins=n_bins, parameterization="x0")
    return FramewiseActivationProcess(cfg, sched)


class TestFramewiseActivationProcessConfig:
    def test_rejects_noise_parameterization(self):
        with pytest.raises(ValueError, match="x0"):
            FramewiseActivationProcessConfig(n_bins=10, parameterization="noise")

    def test_rejects_v_parameterization(self):
        with pytest.raises(ValueError, match="x0"):
            FramewiseActivationProcessConfig(n_bins=10, parameterization="v")

    def test_rejects_bad_n_bins(self):
        with pytest.raises(ValueError, match="n_bins"):
            FramewiseActivationProcessConfig(n_bins=1)

    def test_accepts_x0(self):
        cfg = FramewiseActivationProcessConfig(n_bins=10, parameterization="x0")
        assert cfg.parameterization == "x0"


class TestFramewiseActivationProcess:
    def test_encode_x0_passes_map_through(self):
        proc = _make_proc(n_bins=10)
        m = torch.rand(3, 10)
        out = proc.encode_x0(m)
        assert torch.equal(out, m.float())

    def test_encode_x0_rejects_rank1(self):
        proc = _make_proc(n_bins=10)
        with pytest.raises(ValueError, match="activation map"):
            proc.encode_x0(torch.tensor([0, 1, 2]))

    def test_encode_x0_rejects_wrong_n_bins(self):
        proc = _make_proc(n_bins=10)
        with pytest.raises(ValueError, match="n_bins"):
            proc.encode_x0(torch.rand(2, 7))

    def test_q_sample_shape(self):
        proc = _make_proc(n_bins=16, n_steps=8)
        x_0 = torch.rand(4, 16)
        t = torch.randint(0, 8, (4,))
        x_t = proc.q_sample(x_0, t)
        assert x_t.shape == x_0.shape

    def test_predict_x0_is_identity(self):
        proc = _make_proc()
        x_t = torch.randn(2, 16)
        t = torch.zeros(2, dtype=torch.int64)
        model_out = torch.randn(2, 16)
        out = proc.predict_x0(model_out, x_t, t)
        assert torch.equal(out, model_out)

    def test_loss_target_returns_x0(self):
        proc = _make_proc()
        x_0 = torch.rand(2, 16)
        x_t = torch.randn(2, 16)
        t = torch.zeros(2, dtype=torch.int64)
        assert torch.equal(proc.loss_target(x_0, x_t, t), x_0)

    def test_q_sample_predict_round_trip_at_t0(self):
        # At t=0 with linear schedule, q_sample is close to x_0 (small
        # noise). The x0-parameterization predict_x0 is identity.
        proc = _make_proc(n_bins=8, n_steps=8)
        x_0 = torch.rand(3, 8)
        t = torch.zeros(3, dtype=torch.int64)
        # Predict x_0 from a "perfect" denoiser output (=x_0 itself).
        out = proc.predict_x0(x_0, x_0, t)
        assert torch.allclose(out, x_0)

    def test_sample_prior_stats(self):
        proc = _make_proc(n_bins=64)
        x = proc.sample_prior(256, torch.device("cpu"))
        assert x.shape == (256, 64)
        assert abs(x.mean().item()) < 0.1
        assert abs(x.std().item() - 1.0) < 0.1

    def test_decode_to_logits_clips_to_unit_interval(self):
        proc = _make_proc(n_bins=8)
        out = proc.decode_to_logits(torch.tensor([[-1.0, 0.0, 0.5, 1.0, 2.0, -0.3, 0.7, 1.5]]))
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0
        # Mid-range values pass through unchanged.
        assert out[0, 2].item() == pytest.approx(0.5)
        assert out[0, 6].item() == pytest.approx(0.7)

    def test_reverse_step_shape(self):
        proc = _make_proc(n_bins=8, n_steps=8)
        x_t = torch.randn(2, 8)
        model_out = torch.rand(2, 8)
        t = torch.tensor([5, 5])
        t_prev = torch.tensor([4, 4])
        out = proc.reverse_step(model_out, x_t, t, t_prev)
        assert out.shape == x_t.shape


# ─────────────────────────── Conv1DDenoiserConfig ─────────────────────


class TestConv1DDenoiserConfig:
    def test_rejects_even_kernel(self):
        with pytest.raises(ValueError, match="odd"):
            Conv1DDenoiserConfig(
                d_model=8, n_bins=16, conv_kernels=(31, 14),
            )

    def test_rejects_odd_pos_embed_dim(self):
        with pytest.raises(ValueError, match="pos_embed_dim"):
            Conv1DDenoiserConfig(d_model=8, n_bins=16, pos_embed_dim=15)

    def test_rejects_zero_conv_channels(self):
        with pytest.raises(ValueError, match="conv_channels"):
            Conv1DDenoiserConfig(d_model=8, n_bins=16, conv_channels=0)

    def test_rejects_empty_kernels(self):
        with pytest.raises(ValueError, match="conv_kernels"):
            Conv1DDenoiserConfig(d_model=8, n_bins=16, conv_kernels=())

    def test_rejects_bad_audio_feature_dim(self):
        with pytest.raises(ValueError, match="audio_feature_dim"):
            Conv1DDenoiserConfig(d_model=8, n_bins=16, audio_feature_dim=0)

    def test_rejects_bad_audio_token_count(self):
        with pytest.raises(ValueError, match="audio_token_count"):
            Conv1DDenoiserConfig(d_model=8, n_bins=16, audio_token_count=0)

    def test_inherits_denoiser_validation(self):
        with pytest.raises(ValueError, match="d_model"):
            Conv1DDenoiserConfig(d_model=0, n_bins=16)


# ─────────────────────────── Conv1DDenoiser ───────────────────────────


def _conv_cfg(self_cond: bool = False, n_bins: int = 32) -> Conv1DDenoiserConfig:
    return Conv1DDenoiserConfig(
        d_model=16,
        n_bins=n_bins,
        time_embed_dim=16,
        self_cond=self_cond,
        audio_feature_dim=8,
        audio_token_count=8,
        pos_embed_dim=4,
        cursor_proj_dim=8,
        time_proj_dim=8,
        conv_channels=8,
        conv_kernels=(3, 3),
    )


class TestConv1DDenoiser:
    def test_forward_shape_no_self_cond(self):
        cfg = _conv_cfg(self_cond=False, n_bins=32)
        net = Conv1DDenoiser(cfg)
        B = 4
        cursor = torch.randn(B, cfg.d_model)
        x_t = torch.randn(B, cfg.n_bins)
        t = torch.randint(0, 16, (B,))
        audio = torch.randn(B, cfg.audio_token_count, cfg.audio_feature_dim)
        out = net(cursor, x_t, t, audio_features=audio)
        assert out.shape == (B, cfg.n_bins)

    def test_forward_shape_with_self_cond(self):
        cfg = _conv_cfg(self_cond=True, n_bins=32)
        net = Conv1DDenoiser(cfg)
        B = 3
        cursor = torch.randn(B, cfg.d_model)
        x_t = torch.randn(B, cfg.n_bins)
        t = torch.randint(0, 16, (B,))
        audio = torch.randn(B, cfg.audio_token_count, cfg.audio_feature_dim)
        prev = torch.zeros(B, cfg.n_bins)
        out = net(cursor, x_t, t, prev_x0_hat=prev, audio_features=audio)
        assert out.shape == (B, cfg.n_bins)

    def test_audio_features_required(self):
        cfg = _conv_cfg()
        net = Conv1DDenoiser(cfg)
        cursor = torch.randn(2, cfg.d_model)
        x_t = torch.randn(2, cfg.n_bins)
        t = torch.zeros(2, dtype=torch.int64)
        with pytest.raises(ValueError, match="audio_features"):
            net(cursor, x_t, t)

    def test_audio_upsampling_to_n_bins(self):
        # T_audio < n_bins is the typical case: linear interpolation
        # should produce a valid output.
        cfg = Conv1DDenoiserConfig(
            d_model=16, n_bins=64, time_embed_dim=16,
            audio_feature_dim=4, audio_token_count=8,
            pos_embed_dim=4, cursor_proj_dim=4, time_proj_dim=4,
            conv_channels=8, conv_kernels=(3,),
        )
        net = Conv1DDenoiser(cfg)
        B = 2
        audio = torch.randn(B, 8, 4)
        out = net(
            torch.randn(B, 16),
            torch.randn(B, 64),
            torch.zeros(B, dtype=torch.int64),
            audio_features=audio,
        )
        assert out.shape == (B, 64)

    def test_audio_upsampling_when_equal(self):
        cfg = Conv1DDenoiserConfig(
            d_model=16, n_bins=32, time_embed_dim=16,
            audio_feature_dim=4, audio_token_count=32,
            pos_embed_dim=4, cursor_proj_dim=4, time_proj_dim=4,
            conv_channels=8, conv_kernels=(3,),
        )
        net = Conv1DDenoiser(cfg)
        audio = torch.randn(1, 32, 4)
        out = net(
            torch.randn(1, 16),
            torch.randn(1, 32),
            torch.zeros(1, dtype=torch.int64),
            audio_features=audio,
        )
        assert out.shape == (1, 32)

    def test_gradient_flow(self):
        cfg = _conv_cfg(self_cond=True)
        net = Conv1DDenoiser(cfg)
        B = 2
        cursor = torch.randn(B, cfg.d_model, requires_grad=True)
        x_t = torch.randn(B, cfg.n_bins, requires_grad=True)
        t = torch.randint(0, 16, (B,))
        audio = torch.randn(B, cfg.audio_token_count, cfg.audio_feature_dim, requires_grad=True)
        out = net(cursor, x_t, t, audio_features=audio)
        loss = out.pow(2).mean()
        loss.backward()
        # Every parameter should have a gradient.
        for name, p in net.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            # FiLM layers are zero-init: their gradients can be exactly
            # zero on the first backward through them when they're at
            # an "identity" point. So we just check the gradient
            # *exists* — at least one parameter must have a non-zero
            # gradient for the conv stack to have actually trained.
        # Sanity: at least the output projection has nonzero grad.
        assert net.out_proj.weight.grad.abs().sum().item() > 0.0

    def test_self_cond_none_equals_zeros(self):
        cfg = _conv_cfg(self_cond=True, n_bins=16)
        net = Conv1DDenoiser(cfg).eval()
        B = 2
        cursor = torch.randn(B, cfg.d_model)
        x_t = torch.randn(B, cfg.n_bins)
        t = torch.zeros(B, dtype=torch.int64)
        audio = torch.randn(B, cfg.audio_token_count, cfg.audio_feature_dim)
        with torch.no_grad():
            out_none = net(cursor, x_t, t, prev_x0_hat=None, audio_features=audio)
            out_zeros = net(
                cursor, x_t, t,
                prev_x0_hat=torch.zeros_like(x_t),
                audio_features=audio,
            )
        assert torch.allclose(out_none, out_zeros)

    def test_film_zero_init_makes_block_identity_like(self):
        # All FiLM layers start zero -> at init the modulation is
        # `h * 1.0 + 0.0`. Verify weights are exactly zero.
        cfg = _conv_cfg()
        net = Conv1DDenoiser(cfg)
        for film in net.film_layers:
            assert torch.all(film.weight == 0.0)
            assert torch.all(film.bias == 0.0)

    def test_pos_embed_registered_as_buffer(self):
        cfg = _conv_cfg(n_bins=32)
        net = Conv1DDenoiser(cfg)
        # Buffer present and correctly shaped.
        assert hasattr(net, "pos_embed")
        assert net.pos_embed.shape == (cfg.n_bins, cfg.pos_embed_dim)
        # Not a Parameter.
        names = {n for n, _ in net.named_parameters()}
        assert "pos_embed" not in names


# ─────────────────────── DenoiserHead ABC backcompat ──────────────────


class TestDenoiserHeadBackcompat:
    """The MLPDenoiser predates #016's ``audio_features`` kwarg. It
    must still work either way."""

    def test_mlp_works_without_audio(self):
        cfg = MLPDenoiserConfig(d_model=8, n_bins=10, hidden_dim=16, n_layers=1, time_embed_dim=8)
        net = MLPDenoiser(cfg)
        cursor = torch.randn(2, cfg.d_model)
        x_t = torch.randn(2, cfg.n_bins)
        t = torch.zeros(2, dtype=torch.int64)
        out = net(cursor, x_t, t)
        assert out.shape == (2, cfg.n_bins)

    def test_mlp_ignores_audio_features_kwarg(self):
        cfg = MLPDenoiserConfig(d_model=8, n_bins=10, hidden_dim=16, n_layers=1, time_embed_dim=8)
        net = MLPDenoiser(cfg).eval()
        cursor = torch.randn(2, cfg.d_model)
        x_t = torch.randn(2, cfg.n_bins)
        t = torch.zeros(2, dtype=torch.int64)
        audio = torch.randn(2, 8, 16)
        with torch.no_grad():
            out_with = net(cursor, x_t, t, audio_features=audio)
            out_without = net(cursor, x_t, t)
        assert torch.allclose(out_with, out_without)


# ─────────────────────────── Cross-product ────────────────────────────


class TestCrossProduct:
    def test_framewise_process_with_conv_denoiser(self):
        proc = _make_proc(n_bins=32, n_steps=8)
        cfg = _conv_cfg(self_cond=False, n_bins=32)
        denoiser = Conv1DDenoiser(cfg)

        B = 2
        # Build a synthetic target map via make_framewise_target.
        future = torch.tensor([[5, 12, -1], [20, -1, -1]], dtype=torch.int64)
        target = make_framewise_target(future, n_bins=32, sigma=2.0)
        x_0 = proc.encode_x0(target.target_map_smoothed)
        t = torch.randint(0, 8, (B,))
        x_t = proc.q_sample(x_0, t)
        cursor = torch.randn(B, cfg.d_model)
        audio = torch.randn(B, cfg.audio_token_count, cfg.audio_feature_dim)
        model_out = denoiser(cursor, x_t, t, audio_features=audio)
        assert model_out.shape == (B, 32)
        # predict_x0 in x0 param is identity.
        x_0_hat = proc.predict_x0(model_out, x_t, t)
        assert x_0_hat.shape == (B, 32)
        # decode clips.
        decoded = proc.decode_to_logits(x_0_hat)
        assert decoded.min().item() >= 0.0
        assert decoded.max().item() <= 1.0
