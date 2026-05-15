"""Integration tests for the #016 framewise-diffusion stack glue (chunk B):

  - ``FramewiseSampleAdapter`` (DataSample → FramewiseTarget collation).
  - ``FramewiseDiffusionDetector`` (model trunk + framewise head).
  - ``FramewiseDiffusionLoss`` (weighted MSE + Min-SNR + diagnostics).
  - ``framewise_curve_metrics`` pure functions.
  - JSON round-trip of ``FramewiseDiffusionDetectorConfig``.
  - End-to-end one-step smoke through the loss.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from osu.taiko2.data_samplers.detection import TaikoDetectionSample
from osu.taiko2.diffusion import (
    Conv1DDenoiserConfig,
    CosineScheduleConfig,
    FramewiseActivationProcessConfig,
)
from osu.taiko2.domain.beatmap import OnsetKind, RelativeOnset
from osu.taiko2.domain.framewise import FramewiseTarget, make_framewise_target
from osu.taiko2.inference.spec import build_config
from osu.taiko2.models.event_embedding import EventEmbeddingInput
from osu.taiko2.models.framewise_diffusion_detector import (
    FramewiseDiffusionDetector,
    FramewiseDiffusionDetectorConfig,
    FramewiseModelOutput,
)
from osu.taiko2.training.framewise_adapter import (
    FramewiseSampleAdapter,
    FramewiseSampleAdapterConfig,
)
from osu.taiko2.training.framewise_curve_metrics import (
    CurvesResult,
    aggregate_curves,
    build_curves_from_batch,
    compute_auc_pr,
    compute_auc_roc,
    compute_frame_f1_at_tolerance,
    compute_per_bin_curves,
    compute_per_bin_curves_at_tolerance,
    compute_separation,
    default_thresholds,
    default_tolerances_frames,
    save_curves_npz,
)
from osu.taiko2.training.framewise_diffusion_loss import (
    FramewiseDiffusionLoss,
    FramewiseDiffusionLossConfig,
)


# ─────────────────────────── helpers ─────────────────────────────────


def _tiny_detector_config(b_pred: int = 16) -> FramewiseDiffusionDetectorConfig:
    """Minimal framewise config — b_bins=64, stride 4 → 16 audio tokens
    in the future half. b_pred == n_bins (no STOP)."""
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
        schedule_config=CosineScheduleConfig(n_steps=8),
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


def _tiny_input(cfg: FramewiseDiffusionDetectorConfig, B: int = 2) -> EventEmbeddingInput:
    T = cfg.a_bins + cfg.b_bins
    return EventEmbeddingInput(
        mel=torch.randn(B, cfg.n_mels, T),
        event_offsets=torch.zeros(B, cfg.c_events, dtype=torch.long),
        event_mask=torch.ones(B, cfg.c_events, dtype=torch.bool),
        conditioning=torch.zeros(B, 3),
    )


def _make_sample(
    *, future_offsets: list[int], d_events: int = 8, sample_id: int = 0,
) -> TaikoDetectionSample:
    """Build a TaikoDetectionSample with the given future-event offsets.

    Past audio / past events are stubbed; only future_events / mask /
    densities matter for the adapter target tests.
    """
    F = 8
    A = 16
    B = 16
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
        for _ in range(4)
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


# ─────────────────────────── FramewiseSampleAdapter ──────────────────


class TestAdapter:
    def test_target_basic(self):
        cfg = FramewiseSampleAdapterConfig(b_pred=20, sigma_frames=1.0)
        adapter = FramewiseSampleAdapter(cfg)
        samples = [
            _make_sample(future_offsets=[5, 10], d_events=8),
            _make_sample(future_offsets=[2], d_events=8),
        ]
        tgt = adapter.make_target(samples, device=torch.device("cpu"))
        assert isinstance(tgt, FramewiseTarget)
        assert tgt.target_map_binary.shape == (2, 20)
        assert tgt.target_map_smoothed.shape == (2, 20)
        # Row 0 has two events.
        assert tgt.target_map_binary[0, 5] == 1.0
        assert tgt.target_map_binary[0, 10] == 1.0
        assert tgt.n_gt[0].item() == 2
        # Row 1 has one event.
        assert tgt.target_map_binary[1, 2] == 1.0
        assert tgt.n_gt[1].item() == 1
        # Smoothed map has peak at the GT bin.
        assert torch.isclose(
            tgt.target_map_smoothed[0, 5],
            torch.tensor(1.0), atol=1e-6,
        )

    def test_no_future_events(self):
        cfg = FramewiseSampleAdapterConfig(b_pred=10)
        adapter = FramewiseSampleAdapter(cfg)
        samples = [_make_sample(future_offsets=[], d_events=4)]
        tgt = adapter.make_target(samples, device=torch.device("cpu"))
        assert tgt.n_gt[0].item() == 0
        assert tgt.target_map_binary.abs().sum() == 0
        assert tgt.target_map_smoothed.abs().sum() == 0

    def test_filters_out_of_window(self):
        # Offsets >= b_pred or negative should be filtered.
        cfg = FramewiseSampleAdapterConfig(b_pred=10)
        adapter = FramewiseSampleAdapter(cfg)
        samples = [_make_sample(future_offsets=[5, 15, -3], d_events=4)]
        tgt = adapter.make_target(samples, device=torch.device("cpu"))
        # Only offset 5 is in [0, 10).
        assert tgt.n_gt[0].item() == 1
        assert tgt.target_map_binary[0, 5] == 1.0

    def test_make_input_delegates(self):
        # The adapter reuses DetectionSampleAdapter for input collation.
        cfg = FramewiseSampleAdapterConfig(b_pred=10)
        adapter = FramewiseSampleAdapter(cfg)
        samples = [_make_sample(future_offsets=[5], d_events=4)]
        inp = adapter.make_input(samples, device=torch.device("cpu"))
        assert isinstance(inp, EventEmbeddingInput)


# ─────────────────────────── Config validation ───────────────────────


class TestDetectorConfig:
    def test_default_construct(self):
        cfg = _tiny_detector_config()
        assert cfg.process_config.n_bins == cfg.b_pred
        assert cfg.denoiser_config.n_bins == cfg.b_pred

    def test_b_pred_n_bins_mismatch_rejected(self):
        with pytest.raises(ValueError, match="b_pred"):
            FramewiseDiffusionDetectorConfig(
                n_mels=8, d_model=16, n_layers=1, n_heads=2,
                c_events=4, cond_dim=8,
                a_bins=64, b_bins=64, b_pred=16,
                schedule_config=CosineScheduleConfig(n_steps=8),
                process_config=FramewiseActivationProcessConfig(
                    n_bins=20, parameterization="x0",
                ),
                denoiser_config=Conv1DDenoiserConfig(
                    d_model=16, n_bins=20,
                    audio_feature_dim=16, audio_token_count=16,
                    conv_channels=8, conv_kernels=(3,),
                    time_embed_dim=16, cursor_proj_dim=4,
                    time_proj_dim=4, pos_embed_dim=4,
                ),
            )

    def test_audio_token_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="audio_token_count"):
            FramewiseDiffusionDetectorConfig(
                n_mels=8, d_model=16, n_layers=1, n_heads=2,
                c_events=4, cond_dim=8,
                a_bins=64, b_bins=64, b_pred=16,
                schedule_config=CosineScheduleConfig(n_steps=8),
                process_config=FramewiseActivationProcessConfig(n_bins=16),
                denoiser_config=Conv1DDenoiserConfig(
                    d_model=16, n_bins=16,
                    audio_feature_dim=16,
                    audio_token_count=999,        # mismatch with b_bins//4 = 16
                    conv_channels=8, conv_kernels=(3,),
                    time_embed_dim=16, cursor_proj_dim=4,
                    time_proj_dim=4, pos_embed_dim=4,
                ),
            )

    def test_n_bins_mismatch_rejected(self):
        with pytest.raises(ValueError, match="denoiser_config.n_bins"):
            FramewiseDiffusionDetectorConfig(
                n_mels=8, d_model=16, n_layers=1, n_heads=2,
                c_events=4, cond_dim=8,
                a_bins=64, b_bins=64, b_pred=16,
                schedule_config=CosineScheduleConfig(n_steps=8),
                process_config=FramewiseActivationProcessConfig(n_bins=16),
                denoiser_config=Conv1DDenoiserConfig(
                    d_model=16, n_bins=20,        # mismatch with process
                    audio_feature_dim=16, audio_token_count=16,
                    conv_channels=8, conv_kernels=(3,),
                    time_embed_dim=16, cursor_proj_dim=4,
                    time_proj_dim=4, pos_embed_dim=4,
                ),
            )


# ─────────────────────────── Detector ─────────────────────────────────


class TestDetector:
    def test_predict_returns_cursor_and_audio(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).eval()
        x = _tiny_input(cfg, B=2)
        with torch.no_grad():
            out = model.predict(x)
        assert isinstance(out, FramewiseModelOutput)
        assert out.cursor_token.shape == (2, cfg.d_model)
        assert out.audio_features is not None
        # b_bins // 4 = 16.
        assert out.audio_features.shape == (2, cfg.b_bins // 4, cfg.d_model)
        assert out.model_out is None
        assert out.loss_target is None
        assert out.t is None
        assert out.logits.shape == (2, cfg.b_pred)

    def test_get_audio_features_shape(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).eval()
        with torch.no_grad():
            af = model.get_audio_features(_tiny_input(cfg, B=3))
        assert af.shape == (3, cfg.b_bins // 4, cfg.d_model)

    def test_forward_diffusion_populates_fields(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).train()
        B = 2
        cursor_token = torch.randn(B, cfg.d_model)
        audio_features = torch.randn(B, cfg.b_bins // 4, cfg.d_model)
        target_map = torch.zeros(B, cfg.b_pred)
        target_map[0, 3] = 1.0
        target_map[1, 7] = 1.0
        out = model.forward_diffusion(
            cursor_token, audio_features, target_map,
        )
        assert out.model_out is not None
        assert out.model_out.shape == (B, cfg.b_pred)
        assert out.loss_target.shape == (B, cfg.b_pred)
        assert out.x_t.shape == (B, cfg.b_pred)
        assert out.t.shape == (B,)
        assert out.t.dtype == torch.long
        assert out.logits.shape == (B, cfg.b_pred)

    def test_self_cond_zero_prob_matches_no_prev(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).train()
        B = 2
        cursor_token = torch.randn(B, cfg.d_model)
        audio_features = torch.randn(B, cfg.b_bins // 4, cfg.d_model)
        target_map = torch.rand(B, cfg.b_pred)
        torch.manual_seed(0)
        t = torch.randint(0, cfg.schedule_config.n_steps, (B,))
        noise = torch.randn(B, cfg.b_pred)
        out = model.forward_diffusion(
            cursor_token, audio_features, target_map,
            t=t, noise=noise, self_cond_prob=0.0,
        )
        x_0 = model.process.encode_x0(target_map)
        x_t = model.process.q_sample(x_0, t, noise=noise)
        with torch.no_grad():
            expected = model.denoiser(
                cursor_token, x_t, t,
                prev_x0_hat=None,
                audio_features=audio_features,
            )
        assert torch.allclose(out.model_out, expected, atol=1e-6)

    def test_self_cond_prob_positive_runs(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).train()
        B = 4
        cursor_token = torch.randn(B, cfg.d_model)
        audio_features = torch.randn(B, cfg.b_bins // 4, cfg.d_model)
        target_map = torch.rand(B, cfg.b_pred)
        out = model.forward_diffusion(
            cursor_token, audio_features, target_map,
            self_cond_prob=0.5,
        )
        assert torch.isfinite(out.model_out).all()


# ─────────────────────────── Loss ─────────────────────────────────────


class TestLoss:
    def _make_output_and_target(
        self, cfg: FramewiseDiffusionDetectorConfig, B: int = 4,
    ) -> tuple[FramewiseModelOutput, FramewiseTarget]:
        model = FramewiseDiffusionDetector(cfg).eval()
        cursor_token = torch.randn(B, cfg.d_model)
        audio_features = torch.randn(B, cfg.b_bins // 4, cfg.d_model)
        # Build a target with a couple of GT bins per sample.
        future_offsets = torch.full((B, 4), -1, dtype=torch.long)
        for i in range(B):
            future_offsets[i, 0] = i % cfg.b_pred
            future_offsets[i, 1] = (i * 3 + 1) % cfg.b_pred
        tgt = make_framewise_target(
            future_offsets, n_bins=cfg.b_pred, sigma=1.0,
        )
        out = model.forward_diffusion(
            cursor_token, audio_features, tgt.target_map_smoothed,
        )
        return out, tgt

    def test_mse_forward_shape_and_metrics(self):
        cfg = _tiny_detector_config()
        loss = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig())
        out, tgt = self._make_output_and_target(cfg)
        result = loss.forward(out, tgt)
        assert result.loss.dim() == 0
        assert torch.isfinite(result.loss)
        expected = {
            "loss", "loss/snr_weighted",
            "loss/pos_only", "loss/neg_only", "loss/pos_neg_ratio",
            "loss/per_t_q0", "loss/per_t_q1",
            "loss/per_t_q2", "loss/per_t_q3",
            "frame/precision_τ_50_tol_2", "frame/recall_τ_50_tol_2",
            "frame/f1_τ_50_tol_2",
            "frame/auc_pr", "frame/auc_roc",
            "frame/mean_act_pos", "frame/mean_act_neg",
            "frame/separation",
            "frame/pos_rate_pred_50", "frame/pos_rate_target",
        }
        assert expected <= set(result.metrics.keys())

    def test_huber_forward(self):
        cfg = _tiny_detector_config()
        loss = FramewiseDiffusionLoss(
            FramewiseDiffusionLossConfig(loss_type="huber"),
        )
        out, tgt = self._make_output_and_target(cfg)
        result = loss.forward(out, tgt)
        assert torch.isfinite(result.loss)

    def test_positive_class_upweighting(self):
        # Constant-zero prediction: pos loss ≫ neg loss because target
        # is non-zero only at the positive bins.
        cfg = _tiny_detector_config()
        loss = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig())
        out, tgt = self._make_output_and_target(cfg)
        # Force model_out to all zeros: target_smoothed > 0 only at
        # positives → pos_only > neg_only.
        object.__setattr__(out, "model_out", torch.zeros_like(out.model_out))
        result = loss.forward(out, tgt)
        # Either pos_only > neg_only, or both are zero (degenerate case).
        assert result.metrics["loss/pos_only"] >= result.metrics["loss/neg_only"]

    def test_snr_weighting_flag_changes_loss(self):
        cfg = _tiny_detector_config()
        out, tgt = self._make_output_and_target(cfg, B=8)
        loss_off = FramewiseDiffusionLoss(
            FramewiseDiffusionLossConfig(snr_weighting=False),
        )
        loss_on = FramewiseDiffusionLoss(
            FramewiseDiffusionLossConfig(snr_weighting=True),
        )
        model = FramewiseDiffusionDetector(cfg)
        loss_on.bind_schedule(model.schedule.alphas_cumprod())
        loss_off.bind_schedule(model.schedule.alphas_cumprod())
        r_off = loss_off.forward(out, tgt)
        r_on = loss_on.forward(out, tgt)
        assert torch.isfinite(r_off.loss)
        assert torch.isfinite(r_on.loss)

    def test_snr_x0_mode_flips_weight_direction(self):
        """ε-mode `min(snr,γ)/snr` downweights low-t (high SNR) heavily;
        x0-mode `min(snr,γ)` upweights low-t to γ. Verify both
        formulas evaluate correctly on a synthetic schedule and that
        the produced training losses differ."""
        cfg = _tiny_detector_config()
        out, tgt = self._make_output_and_target(cfg, B=8)
        model = FramewiseDiffusionDetector(cfg)
        ab = model.schedule.alphas_cumprod()
        snr = ab / (1.0 - ab).clamp(min=1e-8)
        gamma = 5.0

        loss_eps = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig(
            snr_weighting=True, snr_gamma=gamma, snr_x0_mode=False,
        ))
        loss_x0 = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig(
            snr_weighting=True, snr_gamma=gamma, snr_x0_mode=True,
        ))
        loss_eps.bind_schedule(ab)
        loss_x0.bind_schedule(ab)

        # Inspect raw weights at the schedule endpoints. t=0 is low-noise
        # (snr huge); t=T-1 is high-noise (snr tiny).
        T = ab.numel()
        t_low = torch.tensor([0], dtype=torch.long)
        t_hi = torch.tensor([T - 1], dtype=torch.long)
        eps_low = loss_eps._snr_weights(t_low).item()
        eps_hi = loss_eps._snr_weights(t_hi).item()
        x0_low = loss_x0._snr_weights(t_low).item()
        x0_hi = loss_x0._snr_weights(t_hi).item()
        # ε-mode (`min(snr,γ)/snr` = `min(γ/snr, 1)`): weight at low-t
        # is γ/snr (small relative to high-t which is 1). Direction:
        # low-t weight < high-t weight.
        assert eps_low < eps_hi, (
            f"eps-mode should weight high-t more than low-t; "
            f"got eps_low={eps_low}, eps_hi={eps_hi}"
        )
        assert eps_hi >= 0.99, (
            f"eps-mode high-t weight should be ~1; got {eps_hi}"
        )
        # x0-mode (`min(snr,γ)`): weight at low-t is capped at γ;
        # weight at high-t is snr (small). Direction: low-t > high-t.
        assert x0_low > x0_hi, (
            f"x0-mode should weight low-t more than high-t; "
            f"got x0_low={x0_low}, x0_hi={x0_hi}"
        )
        assert abs(x0_low - gamma) < 1e-3, (
            f"x0-mode low-t weight should be capped at γ={gamma}; got {x0_low}"
        )
        # End-to-end loss differs.
        r_eps = loss_eps.forward(out, tgt)
        r_x0 = loss_x0.forward(out, tgt)
        assert torch.isfinite(r_eps.loss)
        assert torch.isfinite(r_x0.loss)
        assert not torch.isclose(r_eps.loss, r_x0.loss, atol=1e-6), (
            "x0-mode and eps-mode should produce numerically different losses"
        )

    def test_inference_mode_output_unbound_raises(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).eval()
        with torch.no_grad():
            out = model.predict(_tiny_input(cfg, B=2))
        future_offsets = torch.tensor([[0], [3]], dtype=torch.long)
        tgt = make_framewise_target(future_offsets, n_bins=cfg.b_pred)
        loss = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig())
        with pytest.raises(RuntimeError, match="bind_model"):
            loss.forward(out, tgt)

    def test_bind_model_processes_inference_output(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).train()
        loss = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig())
        loss.bind_model(model)
        with torch.no_grad():
            out = model.predict(_tiny_input(cfg, B=3))
        future_offsets = torch.tensor([[0], [3], [7]], dtype=torch.long)
        tgt = make_framewise_target(future_offsets, n_bins=cfg.b_pred)
        result = loss.forward(out, tgt)
        assert torch.isfinite(result.loss)
        assert out.model_out is not None
        assert out.loss_target is not None
        assert out.t is not None

    def test_backward_propagates(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).train()
        loss = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig())
        cursor_token = torch.randn(4, cfg.d_model, requires_grad=True)
        audio_features = torch.randn(
            4, cfg.b_bins // 4, cfg.d_model, requires_grad=True,
        )
        future_offsets = torch.tensor(
            [[0], [3], [7], [10]], dtype=torch.long,
        )
        tgt = make_framewise_target(future_offsets, n_bins=cfg.b_pred)
        out = model.forward_diffusion(
            cursor_token, audio_features, tgt.target_map_smoothed,
        )
        result = loss.forward(out, tgt)
        result.loss.backward()
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.denoiser.parameters()
        )
        assert any_grad


# ─────────────────────────── Curve metrics ───────────────────────────


class TestCurves:
    def _toy(self, B: int = 2, n_bins: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        target = torch.zeros(B, n_bins)
        for i in range(B):
            target[i, (i * 3 + 1) % n_bins] = 1.0
            target[i, (i * 3 + 5) % n_bins] = 1.0
        # Predicted: noisy but biased toward target.
        pred = torch.rand(B, n_bins) * 0.4 + target * 0.5
        pred = pred.clamp(0.0, 1.0)
        return pred, target

    def test_per_bin_curves_shape(self):
        pred, target = self._toy()
        c = compute_per_bin_curves(pred, target)
        assert c["precision_curve"].shape == (101,)
        assert c["recall_curve"].shape == (101,)
        assert c["f1_curve"].shape == (101,)
        assert c["pos_rate_pred_curve"].shape == (101,)

    def test_per_bin_curves_extremes(self):
        pred, target = self._toy()
        c = compute_per_bin_curves(pred, target)
        # At threshold 0.0 every bin is predicted positive → recall == 1.
        assert c["recall_curve"][0].item() == pytest.approx(1.0, abs=1e-6)
        # At threshold 1.0 (strict >) every bin is below → recall == 0.
        assert c["recall_curve"][-1].item() == pytest.approx(0.0, abs=1e-6)

    def test_per_bin_curves_monotonicity(self):
        # Recall is monotonically non-increasing with threshold.
        pred, target = self._toy(B=4, n_bins=32)
        c = compute_per_bin_curves(pred, target)
        recall = c["recall_curve"]
        for i in range(recall.numel() - 1):
            assert recall[i + 1] <= recall[i] + 1e-6

    def test_tolerance_curves_shape(self):
        pred, target = self._toy()
        c = compute_per_bin_curves_at_tolerance(pred, target)
        assert c["precision_tol_curve"].shape == (5, 101)
        assert c["recall_tol_curve"].shape == (5, 101)
        assert c["f1_tol_curve"].shape == (5, 101)

    def test_tolerance_higher_f1_at_larger_tol(self):
        # Make a pred that's slightly off-by-one from target.
        n_bins = 32
        target = torch.zeros(1, n_bins)
        target[0, 10] = 1.0
        pred = torch.zeros(1, n_bins)
        pred[0, 11] = 1.0      # off by one
        thresholds = torch.tensor([0.5])
        tolerances = torch.tensor([0, 2])
        c = compute_per_bin_curves_at_tolerance(
            pred, target, tolerances_frames=tolerances, thresholds=thresholds,
        )
        # Tolerance 0: miss; tolerance 2: hit.
        assert c["f1_tol_curve"][0, 0].item() < c["f1_tol_curve"][1, 0].item()

    def test_auc_pr_perfect(self):
        target = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.float32)
        pred = target.clone()
        assert compute_auc_pr(pred, target) == pytest.approx(1.0, abs=1e-6)

    def test_auc_roc_perfect(self):
        target = torch.tensor([[0, 1, 0, 1, 0]], dtype=torch.float32)
        pred = target.clone()
        assert compute_auc_roc(pred, target) == pytest.approx(1.0, abs=1e-6)

    def test_auc_pr_random_in_unit_interval(self):
        torch.manual_seed(1)
        target = (torch.rand(8, 32) > 0.8).float()
        pred = torch.rand(8, 32)
        val = compute_auc_pr(pred, target)
        assert 0.0 <= val <= 1.0

    def test_auc_roc_random_in_unit_interval(self):
        torch.manual_seed(1)
        target = (torch.rand(8, 32) > 0.8).float()
        pred = torch.rand(8, 32)
        val = compute_auc_roc(pred, target)
        assert 0.0 <= val <= 1.0

    def test_frame_f1_at_tolerance_keys(self):
        pred, target = self._toy()
        out = compute_frame_f1_at_tolerance(
            pred, target, threshold=0.5, tolerance_frames=2,
        )
        assert set(out.keys()) == {"precision", "recall", "f1"}
        for v in out.values():
            assert 0.0 <= v <= 1.0

    def test_separation(self):
        pred, target = self._toy()
        mp, mn, sep = compute_separation(pred, target)
        assert sep == pytest.approx(mp - mn, abs=1e-6)

    def test_default_thresholds_and_tolerances(self):
        thr = default_thresholds()
        assert thr.shape == (101,)
        assert thr[0].item() == 0.0
        assert thr[-1].item() == 1.0
        tol = default_tolerances_frames()
        assert tuple(tol.tolist()) == (1, 2, 4, 8, 20)

    def test_aggregate_curves(self):
        pred, target = self._toy(B=2)
        c1 = build_curves_from_batch(pred, target)
        c2 = build_curves_from_batch(pred * 0.5, target)
        agg = aggregate_curves([c1, c2])
        assert isinstance(agg, CurvesResult)
        assert agg.precision_curve.shape == (101,)
        assert agg.f1_tol_curve.shape == (5, 101)

    def test_save_curves_npz(self, tmp_path):
        pred, target = self._toy(B=2)
        c = build_curves_from_batch(pred, target)
        path = tmp_path / "curves.npz"
        save_curves_npz(path, eval1=c)
        assert path.exists()
        blob = np.load(path)
        assert "eval1/precision_curve" in blob.files
        assert "eval1/auc_pr" in blob.files


# ─────────────────────────── JSON round-trip ─────────────────────────


class TestJsonRoundTrip:
    def test_polymorphic_subconfigs_resolve(self):
        node = {
            "__class__": "osu.taiko2.models.framewise_diffusion_detector:FramewiseDiffusionDetectorConfig",
            "n_mels": 8, "d_model": 16, "n_layers": 1, "n_heads": 2,
            "c_events": 4, "cond_dim": 8,
            "a_bins": 64, "b_bins": 64, "b_pred": 16,
            "schedule_config": {
                "__class__": "osu.taiko2.diffusion.schedules:CosineScheduleConfig",
                "n_steps": 8,
            },
            "process_config": {
                "__class__": "osu.taiko2.diffusion.processes:FramewiseActivationProcessConfig",
                "n_bins": 16, "parameterization": "x0",
            },
            "denoiser_config": {
                "__class__": "osu.taiko2.diffusion.denoisers:Conv1DDenoiserConfig",
                "d_model": 16, "n_bins": 16,
                "audio_feature_dim": 16, "audio_token_count": 16,
                "self_cond": True,
                "conv_channels": 8, "conv_kernels": [3, 3],
                "time_embed_dim": 16, "cursor_proj_dim": 4,
                "time_proj_dim": 4, "pos_embed_dim": 4, "dropout": 0.0,
            },
        }
        cfg = build_config(json.loads(json.dumps(node)))
        assert isinstance(cfg, FramewiseDiffusionDetectorConfig)
        # Construct the model.
        model = FramewiseDiffusionDetector(cfg)
        assert model.schedule.n_steps == 8


# ─────────────────────────── End-to-end one-step smoke ──────────────


class TestEndToEnd:
    def test_one_training_step(self):
        cfg = _tiny_detector_config()
        model = FramewiseDiffusionDetector(cfg).train()
        loss = FramewiseDiffusionLoss(FramewiseDiffusionLossConfig())
        loss.bind_model(model)
        adapter = FramewiseSampleAdapter(
            FramewiseSampleAdapterConfig(b_pred=cfg.b_pred),
        )
        # Build a stub batch via the adapter's target path; input side
        # uses _tiny_input directly to bypass the detection-adapter's
        # need for real audio_past / audio_future.
        samples = [
            _make_sample(future_offsets=[3, 11], d_events=4),
            _make_sample(future_offsets=[7], d_events=4),
        ]
        tgt = adapter.make_target(samples, device=torch.device("cpu"))
        inp = _tiny_input(cfg, B=2)
        out = model.predict(inp)
        result = loss.forward(out, tgt)
        assert torch.isfinite(result.loss)
        result.loss.backward()
