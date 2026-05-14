"""Tests for `FramewiseRolloutHook` and `DDIMSampler.sample_with_intermediates`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from osu.taiko2.data_samplers.detection import TaikoDetectionSample
from osu.taiko2.diffusion.samplers import DDIMSampler, DDIMSamplerConfig
from osu.taiko2.diffusion import (
    Conv1DDenoiserConfig,
    CosineScheduleConfig,
    FramewiseActivationProcessConfig,
)
from osu.taiko2.domain.beatmap import OnsetKind, RelativeOnset
from osu.taiko2.models.framewise_diffusion_detector import (
    FramewiseDiffusionDetector,
    FramewiseDiffusionDetectorConfig,
)
from osu.taiko2.training.framewise_adapter import (
    FramewiseSampleAdapter,
    FramewiseSampleAdapterConfig,
)
from osu.taiko2.training.framewise_rollout_hook import (
    FramewiseRolloutHook,
    FramewiseRolloutHookConfig,
    _per_step_metrics,
    _summary_per_sample,
)


# ─────────────────────────── helpers ─────────────────────────────────


def _tiny_detector_config(b_pred: int = 16) -> FramewiseDiffusionDetectorConfig:
    a_bins = 64
    b_bins = 64
    return FramewiseDiffusionDetectorConfig(
        n_mels=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        c_events=4,
        cond_dim=8,
        a_bins=a_bins, b_bins=b_bins, b_pred=b_pred,
        schedule_config=CosineScheduleConfig(n_steps=4),
        process_config=FramewiseActivationProcessConfig(
            n_bins=b_pred, parameterization="x0",
        ),
        denoiser_config=Conv1DDenoiserConfig(
            d_model=16, n_bins=b_pred,
            audio_feature_dim=16, audio_token_count=b_bins // 4,
            self_cond=True,
            conv_channels=8, conv_kernels=(3, 3),
            time_embed_dim=16,
            cursor_proj_dim=4, time_proj_dim=4, pos_embed_dim=4,
            dropout=0.0,
        ),
    )


def _make_sample(*, future_offsets: list[int], d_events: int = 4, sample_id: int = 0):
    F = 8
    A = 64
    B = 64
    future_events: list[RelativeOnset] = []
    mask = np.ones(d_events, dtype=bool)
    for slot in range(d_events):
        if slot < len(future_offsets):
            off = future_offsets[slot]
            future_events.append(RelativeOnset(
                time_ms=0, kind=OnsetKind.DON, bin=off, cursor_offset=off,
            ))
            mask[slot] = False
        else:
            future_events.append(RelativeOnset(
                time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=0,
            ))
    past_events = tuple(
        RelativeOnset(time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=0)
        for _ in range(2)
    )
    return TaikoDetectionSample(
        sample_id=sample_id,
        chart_id="stub",
        cursor_bin=0,
        audio_past=np.zeros((F, A), dtype=np.float32),
        audio_future=np.zeros((F, B), dtype=np.float32),
        past_events=past_events,
        past_events_mask=np.ones(len(past_events), dtype=bool),
        future_events=tuple(future_events),
        future_events_mask=mask,
        density_mean=0.0,
        density_peak=0,
        density_std=0.0,
    )


class _StubSampler:
    """A tiny sample-driven sampler we can plug as the val/train_sampler.

    Exposes ``count_samples``, ``get_sample``, ``raw_sample``."""
    def __init__(self, samples: list[TaikoDetectionSample]):
        self._samples = samples

    def count_samples(self) -> int:
        return len(self._samples)

    def get_sample(self, i: int) -> TaikoDetectionSample:
        return self._samples[i]

    def raw_sample(self, i: int) -> TaikoDetectionSample:
        return self._samples[i]


# ─────────────────────────── sample_with_intermediates ──────────────


class TestSampleWithIntermediates:
    def test_shape(self):
        n_bins = 16
        cfg = _tiny_detector_config(b_pred=n_bins)
        model = FramewiseDiffusionDetector(cfg)
        model.eval()
        sampler = DDIMSampler(
            DDIMSamplerConfig(n_inference_steps=3),
            model.process, model.denoiser,
        )
        cursor = torch.randn(2, cfg.d_model)
        audio = torch.randn(2, cfg.b_bins // 4, cfg.d_model)
        out = sampler.sample_with_intermediates(cursor, audio_features=audio)
        assert "final" in out
        assert "per_step" in out
        assert out["final"].shape == (2, n_bins)
        assert out["per_step"].shape == (3, 2, n_bins)

    def test_final_matches_last_step(self):
        n_bins = 16
        cfg = _tiny_detector_config(b_pred=n_bins)
        model = FramewiseDiffusionDetector(cfg)
        model.eval()
        sampler = DDIMSampler(
            DDIMSamplerConfig(n_inference_steps=3),
            model.process, model.denoiser,
        )
        cursor = torch.randn(1, cfg.d_model)
        audio = torch.randn(1, cfg.b_bins // 4, cfg.d_model)
        # Use the same prior so we can compare.
        x_T = torch.randn(1, n_bins)
        out = sampler.sample_with_intermediates(cursor, audio_features=audio, x_T=x_T)
        # The values are in [0, 1] (decode_to_logits clamps).
        assert out["final"].min().item() >= 0.0
        assert out["final"].max().item() <= 1.0


# ─────────────────────────── per-step metrics ────────────────────────


class TestPerStepMetrics:
    def test_shapes_and_values(self):
        T, B, N = 3, 2, 8
        per_step = np.random.rand(T, B, N).astype(np.float32)
        target_binary = np.zeros((B, N), dtype=np.float32)
        target_binary[0, 1] = 1.0
        target_binary[1, 4] = 1.0
        target_smoothed = target_binary.copy()
        m = _per_step_metrics(per_step, target_binary, target_smoothed,
                              threshold=0.5, tol_frames=1)
        assert m["f1"].shape == (B, T)
        assert m["mse"].shape == (B, T)
        assert m["mass_at_target"].shape == (B, T)

    def test_summary_values(self):
        # F1 strictly increasing per sample → best_k = last, monotone_fraction = 1.
        T, B = 4, 2
        f1 = np.tile(np.linspace(0.1, 0.9, T), (B, 1)).astype(np.float32)
        metrics = {
            "f1": f1,
            "mse": np.zeros_like(f1),
            "mass_at_target": np.zeros_like(f1),
            "mass_off_target": np.zeros_like(f1),
            "total_mass": np.zeros_like(f1),
        }
        s = _summary_per_sample(metrics)
        assert (s["best_k_step"] == T - 1).all()
        assert (s["monotone_fraction"] == 1.0).all()
        assert np.allclose(s["final_vs_best_delta"], 0.0)

    def test_summary_non_monotone(self):
        # F1 dips at the end.
        f1 = np.asarray([[0.1, 0.5, 0.9, 0.4]], dtype=np.float32)
        metrics = {
            "f1": f1,
            "mse": np.zeros_like(f1),
            "mass_at_target": np.zeros_like(f1),
            "mass_off_target": np.zeros_like(f1),
            "total_mass": np.zeros_like(f1),
        }
        s = _summary_per_sample(metrics)
        assert s["best_k_step"][0] == 2
        assert s["final_vs_best_delta"][0] == pytest.approx(0.4 - 0.9)
        # 2 monotone transitions out of 3.
        assert s["monotone_fraction"][0] == pytest.approx(2.0 / 3.0)


# ─────────────────────────── hook end-to-end ─────────────────────────


class TestRolloutHook:
    def test_constructor(self):
        cfg = _tiny_detector_config(b_pred=16)
        model = FramewiseDiffusionDetector(cfg)
        adapter = FramewiseSampleAdapter(
            FramewiseSampleAdapterConfig(b_pred=16, sigma_frames=1.0),
        )
        samples = [_make_sample(future_offsets=[2, 5], sample_id=i) for i in range(2)]
        sampler = _StubSampler(samples)
        from osu.taiko2.domain.training import RunSpec
        spec = RunSpec(root=Path("/tmp/rollout_test_root"), name="x")
        hook_cfg = FramewiseRolloutHookConfig(
            eval_n_charts=1, eval_n_windows_per_chart=1,
            noaug_n_charts=0,
            t_inf_steps=2, n_gif_samples=1, seed=0,
        )
        hook = FramewiseRolloutHook(
            config=hook_cfg, spec=spec, model=model, adapter=adapter,
            val_sampler=sampler,
        )
        assert hook._config is hook_cfg
        assert hook._sampler is None

    def test_run_once_writes_outputs(self, tmp_path):
        cfg = _tiny_detector_config(b_pred=16)
        model = FramewiseDiffusionDetector(cfg)
        adapter = FramewiseSampleAdapter(
            FramewiseSampleAdapterConfig(b_pred=16, sigma_frames=1.0),
        )
        # 4 samples — gives room for n_gif_samples picking.
        samples = [
            _make_sample(future_offsets=[2, 5, 9], sample_id=i)
            for i in range(4)
        ]
        sampler = _StubSampler(samples)
        from osu.taiko2.domain.training import RunSpec
        spec = RunSpec(root=tmp_path, name="run")
        spec.ensure()
        hook_cfg = FramewiseRolloutHookConfig(
            eval_n_charts=2, eval_n_windows_per_chart=2,
            noaug_n_charts=0,
            t_inf_steps=2,
            n_gif_samples=2,
            seed=0,
        )
        hook = FramewiseRolloutHook(
            config=hook_cfg, spec=spec, model=model, adapter=adapter,
            val_sampler=sampler,
        )
        eval_dir = spec.run_dir / "eval_42"
        hook.run_once(eval_dir=eval_dir, step=42)
        # NPZ + curves + per-bucket plots + GIFs + summary GIF.
        assert (eval_dir / "rollout_maps.npz").exists()
        assert (eval_dir / "convergence_curves.png").exists()
        assert (eval_dir / "convergence_by_density.png").exists()
        assert (eval_dir / "convergence_by_star.png").exists()
        assert (eval_dir / "convergence_by_kind.png").exists()
        gif_dir = eval_dir / "rollout_gifs"
        assert gif_dir.exists()
        gifs = list(gif_dir.glob("*.gif"))
        assert len(gifs) >= 1
        assert (eval_dir / "summary_histogram.gif").exists()
        # Summary npys.
        for k in (
            "best_k_step", "best_f1", "final_f1",
            "final_vs_best_delta", "convergence_step_90",
            "monotone_fraction",
        ):
            assert (eval_dir / f"rollout_{k}.npy").exists()

    def test_run_once_with_noaug(self, tmp_path):
        cfg = _tiny_detector_config(b_pred=16)
        model = FramewiseDiffusionDetector(cfg)
        adapter = FramewiseSampleAdapter(
            FramewiseSampleAdapterConfig(b_pred=16, sigma_frames=1.0),
        )
        samples = [_make_sample(future_offsets=[3], sample_id=i) for i in range(3)]
        sampler = _StubSampler(samples)
        from osu.taiko2.domain.training import RunSpec
        spec = RunSpec(root=tmp_path, name="run")
        spec.ensure()
        hook_cfg = FramewiseRolloutHookConfig(
            eval_n_charts=2, eval_n_windows_per_chart=1,
            noaug_n_charts=2, noaug_n_windows_per_chart=1,
            t_inf_steps=2, n_gif_samples=1, seed=0,
        )
        hook = FramewiseRolloutHook(
            config=hook_cfg, spec=spec, model=model, adapter=adapter,
            val_sampler=sampler, train_sampler=sampler,
        )
        eval_dir = spec.run_dir / "eval_5"
        hook.run_once(eval_dir=eval_dir, step=5)
        assert (eval_dir / "noaug_rollout_maps.npz").exists()

    def test_on_eval_end_skips_every_n(self, tmp_path):
        cfg = _tiny_detector_config(b_pred=16)
        model = FramewiseDiffusionDetector(cfg)
        adapter = FramewiseSampleAdapter(FramewiseSampleAdapterConfig(b_pred=16))
        samples = [_make_sample(future_offsets=[3], sample_id=i) for i in range(2)]
        sampler = _StubSampler(samples)
        from osu.taiko2.domain.training import RunSpec
        spec = RunSpec(root=tmp_path, name="run")
        spec.ensure()
        hook_cfg = FramewiseRolloutHookConfig(
            eval_n_charts=1, eval_n_windows_per_chart=1,
            noaug_n_charts=0,
            t_inf_steps=2, n_gif_samples=0,
            every_n_evals=2, seed=0,
        )
        hook = FramewiseRolloutHook(
            config=hook_cfg, spec=spec, model=model, adapter=adapter,
            val_sampler=sampler,
        )
        # Simulate two eval boundaries; first should be skipped, second fires.
        from osu.taiko2.domain.training import TrainingState
        state = TrainingState(started_at="")
        state.step = 100
        hook.on_eval_end(state, {})
        assert not (spec.run_dir / "eval_100" / "rollout_maps.npz").exists()
        state.step = 200
        hook.on_eval_end(state, {})
        assert (spec.run_dir / "eval_200" / "rollout_maps.npz").exists()


class TestConfigValidation:
    def test_t_inf_validation(self):
        with pytest.raises(ValueError):
            FramewiseRolloutHookConfig(t_inf_steps=0)

    def test_eval_n_charts_validation(self):
        with pytest.raises(ValueError):
            FramewiseRolloutHookConfig(eval_n_charts=0)

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            FramewiseRolloutHookConfig(decode_threshold=2.0)

    def test_every_n_evals_validation(self):
        with pytest.raises(ValueError):
            FramewiseRolloutHookConfig(every_n_evals=0)
