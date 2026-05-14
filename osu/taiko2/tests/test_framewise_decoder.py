"""Tests for `FramewiseDiffusionDecoder` (#016, Chunk C)."""
from __future__ import annotations

import pytest
import torch

from osu.taiko2.diffusion import (
    Conv1DDenoiser,
    Conv1DDenoiserConfig,
    CosineSchedule,
    CosineScheduleConfig,
    FramewiseActivationProcess,
    FramewiseActivationProcessConfig,
)
from osu.taiko2.diffusion.samplers import (
    DDIMSampler,
    DDIMSamplerConfig,
    DDPMSampler,
)
from osu.taiko2.domain.diffusion import DiffusionSamplerConfig
from osu.taiko2.inference.autoregressive.framewise_diffusion_decoder import (
    FramewiseDiffusionDecoder,
    FramewiseDiffusionDecoderConfig,
)
from osu.taiko2.inference.autoregressive.types import ARContext


def _tiny_model_components(n_bins: int = 16) -> tuple[object, object, object]:
    sched = CosineSchedule(CosineScheduleConfig(n_steps=4))
    proc = FramewiseActivationProcess(
        FramewiseActivationProcessConfig(n_bins=n_bins), sched,
    )
    den = Conv1DDenoiser(Conv1DDenoiserConfig(
        d_model=8, n_bins=n_bins,
        audio_feature_dim=8, audio_token_count=4,
        self_cond=False,
        conv_channels=8, conv_kernels=(3,),
        time_embed_dim=4,
        cursor_proj_dim=2, time_proj_dim=2, pos_embed_dim=2,
        dropout=0.0,
    ))
    class M:
        pass
    m = M()
    m.process = proc
    m.denoiser = den
    m.schedule = sched
    return m, proc, den


class TestConfig:
    def test_defaults(self):
        c = FramewiseDiffusionDecoderConfig()
        assert c.b_pred == 500
        assert c.decode_threshold == 0.5
        assert c.nms_kernel == 1

    def test_b_pred_validation(self):
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(b_pred=0)

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(decode_threshold=1.5)
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(decode_threshold=-0.1)

    def test_nms_kernel_even(self):
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(nms_kernel=2)

    def test_nms_kernel_lt1(self):
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(nms_kernel=0)

    def test_min_emit_gap_validation(self):
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(min_emit_gap_bins=0)

    def test_stop_hop_validation(self):
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(stop_hop_bins=0)

    def test_top_k_validation(self):
        with pytest.raises(ValueError):
            FramewiseDiffusionDecoderConfig(top_k_log=0)


class TestBinding:
    def test_bind_constructs_ddim(self):
        m, _, _ = _tiny_model_components()
        d = FramewiseDiffusionDecoder(FramewiseDiffusionDecoderConfig(
            b_pred=16,
            sampler_config=DDIMSamplerConfig(n_inference_steps=4),
        ))
        assert d._sampler is None
        d.bind_model(m)
        assert isinstance(d._sampler, DDIMSampler)

    def test_decode_without_bind_raises(self):
        d = FramewiseDiffusionDecoder(FramewiseDiffusionDecoderConfig(b_pred=16))
        class O:
            pass
        out = O()
        out.cursor_token = torch.zeros(1, 8)
        out.audio_features = torch.zeros(1, 4, 8)
        ctx = ARContext(cursor_bin=0, step=0, max_bin=100, past_onsets=())
        with pytest.raises(RuntimeError):
            d.decode(out, ctx)


def _decode_with_scores(scores: torch.Tensor, **cfg_overrides):
    """Patch the sampler to return ``scores`` (shape (1, n_bins))."""
    m, _, _ = _tiny_model_components(n_bins=scores.shape[-1])
    decoder = FramewiseDiffusionDecoder(FramewiseDiffusionDecoderConfig(
        b_pred=scores.shape[-1],
        sampler_config=DDIMSamplerConfig(n_inference_steps=2),
        **cfg_overrides,
    ))
    decoder.bind_model(m)

    class _StubSampler:
        config = decoder._sampler.config

        def sample(self, cursor_token, audio_features=None, x_T=None):
            del cursor_token, audio_features, x_T
            return scores
    decoder._sampler = _StubSampler()

    class O:
        pass
    out = O()
    out.cursor_token = torch.zeros(1, 8)
    out.audio_features = torch.zeros(1, 4, 8)
    ctx = ARContext(cursor_bin=0, step=0, max_bin=100, past_onsets=())
    return decoder.decode(out, ctx)


class TestDecodeBehavior:
    def test_empty_below_threshold_is_stop(self):
        scores = torch.zeros(1, 10)
        dec = _decode_with_scores(scores, decode_threshold=0.5)
        assert dec.is_stop
        assert dec.bin_offsets == ()

    def test_low_threshold_many_bins(self):
        scores = torch.tensor([[0.1, 0.4, 0.6, 0.7, 0.2, 0.55, 0.0, 0.9, 0.05, 0.0]])
        dec_low = _decode_with_scores(scores, decode_threshold=0.05, min_emit_gap_bins=1)
        dec_high = _decode_with_scores(scores, decode_threshold=0.8, min_emit_gap_bins=1)
        assert len(dec_low.bin_offsets) > len(dec_high.bin_offsets)
        assert len(dec_high.bin_offsets) == 1
        assert dec_high.bin_offsets[0] == 7

    def test_nms_suppresses_adjacent(self):
        # Two adjacent bins both above threshold; with nms_kernel=3 only the higher wins.
        scores = torch.tensor([[0.0, 0.7, 0.8, 0.6, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0]])
        dec = _decode_with_scores(scores, decode_threshold=0.5, nms_kernel=3, min_emit_gap_bins=1)
        bins = set(dec.bin_offsets)
        # bin 2 is local max in [1,2,3] → kept. bins 1, 3 are NOT local max → dropped.
        # bin 6 is isolated, kept.
        assert 2 in bins
        assert 6 in bins
        assert 1 not in bins
        assert 3 not in bins

    def test_min_emit_gap(self):
        # All bins above threshold; with min_emit_gap_bins=3, only every-3rd kept.
        scores = torch.ones(1, 10) * 0.9
        dec = _decode_with_scores(
            scores, decode_threshold=0.5, nms_kernel=1, min_emit_gap_bins=3,
        )
        bins = list(dec.bin_offsets)
        # Greedy: 0, 3, 6, 9.
        assert bins == [0, 3, 6, 9]

    def test_multi_bin_decision(self):
        scores = torch.tensor([[0.0, 0.9, 0.0, 0.0, 0.95, 0.0, 0.0, 0.7]])
        dec = _decode_with_scores(scores, decode_threshold=0.5, min_emit_gap_bins=1)
        assert tuple(dec.bin_offsets) == (1, 4, 7)
        # Confidences ascending with bin order, but values are correct.
        assert dec.confidences[0] == pytest.approx(0.9, abs=1e-6)
        assert dec.confidences[1] == pytest.approx(0.95, abs=1e-6)

    def test_extras_present(self):
        scores = torch.tensor([[0.1, 0.6, 0.0, 0.8]])
        dec = _decode_with_scores(scores, decode_threshold=0.5, top_k_log=3)
        assert "mean_act" in dec.extras
        assert "max_act" in dec.extras
        assert "n_emitted" in dec.extras
        assert "n_above_threshold" in dec.extras
        assert dec.extras["n_emitted"] == 2
        assert dec.extras["n_above_threshold"] == 2
        assert "top1_bin" in dec.extras
        assert dec.extras["top1_bin"] == 3

    def test_extras_on_stop(self):
        scores = torch.zeros(1, 10)
        dec = _decode_with_scores(scores, decode_threshold=0.5)
        assert dec.is_stop
        assert dec.extras["n_emitted"] == 0
        assert "mean_act" in dec.extras

    def test_confidences_match_scores(self):
        scores = torch.tensor([[0.0, 0.7, 0.0, 0.85, 0.0]])
        dec = _decode_with_scores(scores, decode_threshold=0.5, min_emit_gap_bins=1)
        assert dec.bin_offsets == (1, 3)
        assert dec.confidences == (
            pytest.approx(0.7, abs=1e-6),
            pytest.approx(0.85, abs=1e-6),
        )

    def test_cursor_token_shape_check(self):
        d = FramewiseDiffusionDecoder(FramewiseDiffusionDecoderConfig(
            b_pred=8, sampler_config=DDIMSamplerConfig(n_inference_steps=2),
        ))
        m, _, _ = _tiny_model_components(n_bins=8)
        d.bind_model(m)

        class O:
            pass
        out = O()
        out.cursor_token = torch.zeros(2, 8)  # wrong batch size
        out.audio_features = None
        ctx = ARContext(cursor_bin=0, step=0, max_bin=100, past_onsets=())
        with pytest.raises(ValueError):
            d.decode(out, ctx)

    def test_bin_offsets_sorted(self):
        scores = torch.tensor([[0.0, 0.95, 0.0, 0.85, 0.7, 0.0]])
        dec = _decode_with_scores(scores, decode_threshold=0.5, min_emit_gap_bins=1)
        assert list(dec.bin_offsets) == sorted(dec.bin_offsets)


class TestFullSamplerIntegration:
    """Smoke test against a real model + sampler — no stub."""
    def test_real_decode_returns_decision(self):
        n_bins = 16
        m, proc, den = _tiny_model_components(n_bins=n_bins)
        decoder = FramewiseDiffusionDecoder(FramewiseDiffusionDecoderConfig(
            b_pred=n_bins,
            sampler_config=DDIMSamplerConfig(n_inference_steps=2),
        ))
        decoder.bind_model(m)

        class O:
            pass
        out = O()
        out.cursor_token = torch.randn(1, 8)
        out.audio_features = torch.randn(1, 4, 8)
        ctx = ARContext(cursor_bin=0, step=0, max_bin=200, past_onsets=())
        decision = decoder.decode(out, ctx)
        # Whether bins emit or not depends on random init — just verify the
        # type contract.
        assert isinstance(decision.bin_offsets, tuple)
        assert isinstance(decision.confidences, tuple)
        assert "mean_act" in decision.extras
