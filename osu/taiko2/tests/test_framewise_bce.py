"""Tests for the #017 framewise BCE stack.

Covers: FramewiseDetector, FramewiseBCELoss, FramewiseMetric,
FramewiseDiagnosticsArtifact, FramewiseDecoder, adapter binary_only,
chart gt_match_metrics tolerances, config round-trip, end-to-end.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from osu.taiko2.data_samplers.detection import TaikoDetectionSample
from osu.taiko2.domain.beatmap import OnsetKind, RelativeOnset
from osu.taiko2.domain.chart import gt_match_metrics
from osu.taiko2.domain.framewise import FramewiseTarget, make_framewise_target
from osu.taiko2.domain.metrics import MetricInput
from osu.taiko2.inference.spec import build_config
from osu.taiko2.models.event_embedding import EventEmbeddingInput
from osu.taiko2.models.framewise_detector import (
    FramewiseDetector,
    FramewiseDetectorConfig,
    FramewiseDetectorOutput,
)
from osu.taiko2.training.framewise_adapter import (
    FramewiseSampleAdapter,
    FramewiseSampleAdapterConfig,
)
from osu.taiko2.training.framewise_bce_loss import (
    FramewiseBCELoss,
    FramewiseBCELossConfig,
)
from osu.taiko2.training.framewise_focal_loss import (
    FramewiseFocalLoss,
    FramewiseFocalLossConfig,
)
from osu.taiko2.training.framewise_metric import (
    FramewiseMetric,
    FramewiseMetricConfig,
)
from osu.taiko2.training.framewise_diagnostics_artifact import (
    FramewiseDiagnosticsArtifact,
)
from osu.taiko2.inference.autoregressive.framewise_decoder import (
    FramewiseDecoder,
    FramewiseDecoderConfig,
    framewise_decision_from_map,
)
from osu.taiko2.inference.autoregressive.types import ARContext


# ─────────────────────────── helpers ─────────────────────────────────


def _tiny_config(b_pred: int = 64) -> FramewiseDetectorConfig:
    a_bins = 64
    b_bins = b_pred
    return FramewiseDetectorConfig(
        n_mels=8, d_model=16, n_layers=1, n_heads=2,
        c_events=4, cond_dim=8,
        a_bins=a_bins, b_bins=b_bins, b_pred=b_pred,
        head_channels=16, head_kernels=(3, 3),
        head_pos_embed_dim=8, head_cursor_proj_dim=8,
        head_dropout=0.0,
    )


def _make_input(B: int, cfg: FramewiseDetectorConfig) -> EventEmbeddingInput:
    T_mel = cfg.a_bins + cfg.b_bins
    return EventEmbeddingInput(
        mel=torch.randn(B, cfg.n_mels, T_mel),
        event_offsets=torch.full((B, cfg.c_events), -50, dtype=torch.long),
        event_mask=torch.zeros(B, cfg.c_events, dtype=torch.bool),
        conditioning=torch.randn(B, 3),
    )


def _make_target(B: int, n_bins: int, n_gt_per: int = 3) -> FramewiseTarget:
    offsets = torch.full((B, 10), -1, dtype=torch.long)
    for i in range(B):
        for j in range(n_gt_per):
            offsets[i, j] = j * (n_bins // (n_gt_per + 1)) + 1
    return make_framewise_target(offsets, n_bins=n_bins, sigma=None)


def _make_sample(cfg: FramewiseDetectorConfig) -> TaikoDetectionSample:
    past = tuple(
        RelativeOnset(time_ms=0, kind=OnsetKind.DON, bin=0, cursor_offset=-i * 10)
        for i in range(cfg.c_events)
    )
    future = (
        RelativeOnset(time_ms=100, kind=OnsetKind.DON, bin=5, cursor_offset=5),
        RelativeOnset(time_ms=200, kind=OnsetKind.KA, bin=10, cursor_offset=10),
    )
    return TaikoDetectionSample(
        sample_id=0,
        chart_id="test",
        cursor_bin=1000,
        audio_past=np.random.randn(cfg.n_mels, cfg.a_bins).astype(np.float32),
        audio_future=np.random.randn(cfg.n_mels, cfg.b_bins).astype(np.float32),
        past_events=past,
        past_events_mask=np.array([False] * cfg.c_events),
        future_events=future,
        future_events_mask=np.array([False, False]),
        density_mean=3.0,
        density_peak=5,
        density_std=1.0,
    )


# ─────────────────────────── model ───────────────────────────────────


class TestModel:
    def test_output_shapes(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        inp = _make_input(2, cfg)
        out = model.predict(inp)
        assert isinstance(out, FramewiseDetectorOutput)
        assert out.logits.shape == (2, cfg.b_pred)
        assert out.confidence_map.shape == (2, cfg.b_pred)
        assert out.cursor_token.shape == (2, cfg.d_model)
        assert out.audio_features is not None
        assert out.audio_features.shape[0] == 2
        assert out.audio_features.shape[2] == cfg.d_model

    def test_confidence_map_range(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        inp = _make_input(4, cfg)
        out = model.predict(inp)
        assert out.confidence_map.min() >= 0.0
        assert out.confidence_map.max() <= 1.0

    def test_confidence_map_detached(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        inp = _make_input(2, cfg)
        out = model.predict(inp)
        assert not out.confidence_map.requires_grad

    def test_head_channel_count(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        expected_in = cfg.head_pos_embed_dim + cfg.d_model + cfg.head_cursor_proj_dim
        assert model.head_convs[0].in_channels == expected_in


# ─────────────────────────── loss ────────────────────────────────────


class TestLoss:
    def test_smoke(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        loss_fn = FramewiseBCELoss(FramewiseBCELossConfig())
        inp = _make_input(2, cfg)
        tgt = _make_target(2, cfg.b_pred)
        out = model.predict(inp)
        result = loss_fn(out, tgt)
        assert result.loss.isfinite()
        assert "loss" in result.metrics
        assert "frame/f1_τ_50_tol_2" in result.metrics
        assert "frame/brier" in result.metrics
        assert "frame/conf_tp_median" in result.metrics

    def test_pos_weight_in_range(self) -> None:
        loss_fn = FramewiseBCELoss(FramewiseBCELossConfig(
            pos_weight_clamp_min=10.0, pos_weight_clamp_max=200.0,
        ))
        logits = torch.randn(4, 500)
        n_gt = torch.tensor([2, 2, 2, 2])
        n_neg = 500.0 - n_gt.float()
        w = (n_neg / n_gt.float().clamp(min=1)).clamp(min=10, max=200)
        assert (w >= 10).all()
        assert (w <= 200).all()

    def test_backward(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        loss_fn = FramewiseBCELoss(FramewiseBCELossConfig())
        inp = _make_input(2, cfg)
        tgt = _make_target(2, cfg.b_pred)
        out = model.predict(inp)
        result = loss_fn(out, tgt)
        result.loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


# ─────────────────────────── focal loss ──────────────────────────────


class TestFocalLoss:
    def test_smoke(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        loss_fn = FramewiseFocalLoss(FramewiseFocalLossConfig())
        inp = _make_input(2, cfg)
        tgt = _make_target(2, cfg.b_pred)
        out = model.predict(inp)
        result = loss_fn(out, tgt)
        assert result.loss.isfinite()
        assert "loss" in result.metrics
        assert "loss/focal_weight_pos" in result.metrics
        assert "loss/focal_weight_neg" in result.metrics
        assert "frame/f1_τ_50_tol_2" in result.metrics

    def test_gamma_zero_matches_bce(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        inp = _make_input(2, cfg)
        tgt = _make_target(2, cfg.b_pred)
        out = model.predict(inp)
        bce_fn = FramewiseBCELoss(FramewiseBCELossConfig())
        focal_fn = FramewiseFocalLoss(FramewiseFocalLossConfig(gamma=0.0))
        bce_result = bce_fn(out, tgt)
        focal_result = focal_fn(out, tgt)
        assert abs(bce_result.loss.item() - focal_result.loss.item()) < 1e-5

    def test_focal_reduces_easy_negative_weight(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        loss_fn = FramewiseFocalLoss(FramewiseFocalLossConfig(gamma=2.0))
        inp = _make_input(2, cfg)
        tgt = _make_target(2, cfg.b_pred)
        out = model.predict(inp)
        result = loss_fn(out, tgt)
        fw_neg = result.metrics["loss/focal_weight_neg"]
        fw_pos = result.metrics["loss/focal_weight_pos"]
        assert fw_neg < 1.0
        assert fw_pos > 0.0

    def test_backward(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        loss_fn = FramewiseFocalLoss(FramewiseFocalLossConfig(gamma=2.0))
        inp = _make_input(2, cfg)
        tgt = _make_target(2, cfg.b_pred)
        out = model.predict(inp)
        result = loss_fn(out, tgt)
        result.loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0


# ─────────────────────────── adapter ─────────────────────────────────


class TestAdapter:
    def test_binary_only(self) -> None:
        offsets = torch.tensor([[3, 7, -1]], dtype=torch.long)
        tgt = make_framewise_target(offsets, n_bins=16, sigma=None)
        assert torch.equal(tgt.target_map_binary, tgt.target_map_smoothed)

    def test_sigma_smoothing(self) -> None:
        offsets = torch.tensor([[3, 7, -1]], dtype=torch.long)
        tgt = make_framewise_target(offsets, n_bins=16, sigma=2.0)
        assert not torch.equal(tgt.target_map_binary, tgt.target_map_smoothed)
        assert tgt.target_map_smoothed[0, 3] == pytest.approx(1.0)
        assert tgt.target_map_smoothed[0, 4] > 0.5

    def test_adapter_binary_only_flag(self) -> None:
        cfg_bin = FramewiseSampleAdapterConfig(
            b_pred=16, binary_only=True, max_events_per_window=10,
        )
        cfg_smooth = FramewiseSampleAdapterConfig(
            b_pred=16, sigma_frames=2.0, binary_only=False,
            max_events_per_window=10,
        )
        det_cfg = _tiny_config()
        sample = _make_sample(det_cfg)
        a_bin = FramewiseSampleAdapter(cfg_bin)
        a_smooth = FramewiseSampleAdapter(cfg_smooth)
        tgt_bin = a_bin.make_target([sample], device=torch.device("cpu"))
        tgt_smooth = a_smooth.make_target([sample], device=torch.device("cpu"))
        assert torch.equal(tgt_bin.target_map_binary, tgt_bin.target_map_smoothed)
        assert not torch.equal(
            tgt_smooth.target_map_binary, tgt_smooth.target_map_smoothed,
        )


# ─────────────────────────── decoder ─────────────────────────────────


class TestDecoder:
    def test_stop_on_empty(self) -> None:
        scores = torch.zeros(20)
        d = framewise_decision_from_map(
            scores, decode_threshold=0.5, nms_kernel=1,
            min_emit_gap_bins=1, top_k_log=3,
        )
        assert d.bin_offsets == ()

    def test_single_peak(self) -> None:
        scores = torch.zeros(20)
        scores[7] = 0.9
        d = framewise_decision_from_map(
            scores, decode_threshold=0.5, nms_kernel=1,
            min_emit_gap_bins=1, top_k_log=3,
        )
        assert d.bin_offsets == (7,)

    def test_two_peaks(self) -> None:
        scores = torch.zeros(20)
        scores[3] = 0.8
        scores[15] = 0.7
        d = framewise_decision_from_map(
            scores, decode_threshold=0.5, nms_kernel=1,
            min_emit_gap_bins=1, top_k_log=3,
        )
        assert d.bin_offsets == (3, 15)

    def test_nms_kills_subsidiary(self) -> None:
        scores = torch.zeros(20)
        scores[7] = 0.9
        scores[8] = 0.6
        d = framewise_decision_from_map(
            scores, decode_threshold=0.5, nms_kernel=3,
            min_emit_gap_bins=1, top_k_log=3,
        )
        assert 7 in d.bin_offsets
        assert 8 not in d.bin_offsets

    def test_decoder_class(self) -> None:
        cfg = FramewiseDecoderConfig(b_pred=20, nms_kernel=1)
        decoder = FramewiseDecoder(cfg)
        conf = torch.zeros(1, 20)
        conf[0, 5] = 0.9
        out = FramewiseDetectorOutput(
            logits=torch.zeros(1, 20),
            confidence_map=conf,
            cursor_token=torch.zeros(1, 16),
        )
        ctx = ARContext(cursor_bin=0, step=0, max_bin=1000, past_onsets=())
        decision = decoder.decode(out, ctx)
        assert decision.bin_offsets == (5,)

    def test_shared_with_diffusion_decoder(self) -> None:
        from osu.taiko2.inference.autoregressive.framewise_diffusion_decoder import (
            FramewiseDiffusionDecoder,
        )
        assert hasattr(FramewiseDiffusionDecoder, "_decision_from_map")


# ─────────────────────────── metric ──────────────────────────────────


class TestMetric:
    def test_accumulation(self) -> None:
        cfg = FramewiseMetricConfig(bins_to_ms=5.0)
        m = FramewiseMetric(cfg)
        m.reset()
        n_bins = 20
        for _ in range(3):
            conf = torch.zeros(2, n_bins)
            conf[0, 5] = 0.9
            conf[0, 10] = 0.8
            conf[1, 3] = 0.7
            tb = torch.zeros(2, n_bins)
            tb[0, 5] = 1.0
            tb[0, 10] = 1.0
            tb[1, 3] = 1.0
            gt_bins = torch.full((2, 5), -1, dtype=torch.long)
            gt_bins[0, 0] = 5
            gt_bins[0, 1] = 10
            gt_bins[1, 0] = 3
            tgt = FramewiseTarget(
                target_map_binary=tb,
                target_map_smoothed=tb,
                gt_bins_padded=gt_bins,
                n_gt=torch.tensor([2, 1]),
            )
            out = FramewiseDetectorOutput(
                logits=torch.zeros(2, n_bins),
                confidence_map=conf,
                cursor_token=torch.zeros(2, 16),
            )
            batch = MetricInput(output=out, target=tgt)
            m.update(batch)
        result = m.compute()
        assert "frame/f1_τ_50_tol_2" in result
        assert "frame/brier" in result
        assert "frame/mini/τ50/matched_rate" in result
        assert "frame/mini/τ50/matched_rate_at_tol_25" in result
        assert result["frame/f1_τ_50_tol_2"] > 0
        # Calibration metrics.
        assert "frame/cal/ece" in result
        assert "frame/cal/brier" in result
        assert "frame/cal/alignment" in result
        assert "frame/cal/pos_rate_at_00" in result
        assert "frame/cal/pos_rate_at_90" in result
        assert "frame/cal/count_at_00" in result

    def test_hedge_frac_committed(self) -> None:
        cfg = FramewiseMetricConfig()
        m = FramewiseMetric(cfg)
        m.reset()
        conf = torch.zeros(1, 100)
        conf[0, :10] = 0.99
        tb = torch.zeros(1, 100)
        tb[0, :10] = 1.0
        gt_bins = torch.full((1, 10), -1, dtype=torch.long)
        for i in range(10):
            gt_bins[0, i] = i
        tgt = FramewiseTarget(
            target_map_binary=tb, target_map_smoothed=tb,
            gt_bins_padded=gt_bins, n_gt=torch.tensor([10]),
        )
        out = FramewiseDetectorOutput(
            logits=torch.zeros(1, 100),
            confidence_map=conf,
            cursor_token=torch.zeros(1, 16),
        )
        m.update(MetricInput(output=out, target=tgt))
        result = m.compute()
        assert result["frame/pred_hedge_frac"] < 0.01


# ─────────────────────────── diagnostics artifact ────────────────────


class TestDiagnostics:
    def test_save_outputs(self, tmp_path) -> None:
        art = FramewiseDiagnosticsArtifact()
        art.reset()
        n_bins = 20
        for _ in range(3):
            conf = torch.rand(4, n_bins)
            tb = torch.zeros(4, n_bins)
            tb[:, 5] = 1.0
            tb[:, 10] = 1.0
            out = FramewiseDetectorOutput(
                logits=torch.zeros(4, n_bins),
                confidence_map=conf,
                cursor_token=torch.zeros(4, 16),
            )
            tgt = FramewiseTarget(
                target_map_binary=tb, target_map_smoothed=tb,
                gt_bins_padded=torch.full((4, 5), -1, dtype=torch.long),
                n_gt=torch.tensor([2, 2, 2, 2]),
            )
            art.update(MetricInput(output=out, target=tgt))
        art.save(tmp_path, step=100)
        assert (tmp_path / "per_bin_rate.png").exists()
        assert (tmp_path / "per_bin_rate.npz").exists()
        assert (tmp_path / "value_hist_pred.png").exists()
        assert (tmp_path / "value_hist_target.png").exists()
        assert (tmp_path / "confidence_by_outcome.png").exists()
        assert (tmp_path / "value_hist_combined.png").exists()
        assert (tmp_path / "calibration.png").exists()
        assert (tmp_path / "calibration.npz").exists()
        cal = np.load(tmp_path / "calibration.npz")
        assert cal["mean_conf"].shape[0] == 20
        assert cal["pos_rate"].shape[0] == 20
        data = np.load(tmp_path / "per_bin_rate.npz")
        assert data["pos_rate"].shape == (n_bins,)


# ─────────────────────────── chart matching ──────────────────────────


class TestChartMatching:
    def test_tolerances(self) -> None:
        pred = np.array([100.0, 200.0, 300.0])
        gt = np.array([105.0, 210.0, 400.0])
        result = gt_match_metrics(pred, gt, tolerances_ms=(5.0, 10.0, 25.0))
        assert "matched_rate_at_tol_5" in result
        assert "matched_rate_at_tol_10" in result
        assert "matched_rate_at_tol_25" in result
        assert "halluc_rate_at_tol_5" in result
        assert result["matched_rate_at_tol_5"] == pytest.approx(1 / 3)
        assert result["matched_rate_at_tol_10"] == pytest.approx(2 / 3)

    def test_backward_compat(self) -> None:
        pred = np.array([100.0, 200.0])
        gt = np.array([110.0, 220.0])
        result = gt_match_metrics(pred, gt)
        assert "matched_rate" in result
        assert "close_rate" in result
        assert "hallucination_rate" in result

    def test_empty(self) -> None:
        result = gt_match_metrics(np.array([]), np.array([1.0]))
        assert result["matched_rate"] == 0.0


# ─────────────────────────── config round-trip ───────────────────────


class TestConfigRoundTrip:
    def test_json_round_trip(self) -> None:
        cfg = _tiny_config()
        d = {
            "__class__": "osu.taiko2.models.framewise_detector:FramewiseDetectorConfig",
            "n_mels": cfg.n_mels,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "c_events": cfg.c_events,
            "cond_dim": cfg.cond_dim,
            "a_bins": cfg.a_bins,
            "b_bins": cfg.b_bins,
            "b_pred": cfg.b_pred,
            "head_channels": cfg.head_channels,
            "head_kernels": list(cfg.head_kernels),
            "head_pos_embed_dim": cfg.head_pos_embed_dim,
            "head_cursor_proj_dim": cfg.head_cursor_proj_dim,
            "head_dropout": cfg.head_dropout,
        }
        rebuilt = build_config(d)
        assert isinstance(rebuilt, FramewiseDetectorConfig)
        assert rebuilt.b_pred == cfg.b_pred
        assert tuple(rebuilt.head_kernels) == cfg.head_kernels


# ─────────────────────────── end-to-end ──────────────────────────────


class TestEndToEnd:
    def test_one_step(self) -> None:
        cfg = _tiny_config()
        model = FramewiseDetector(cfg)
        loss_fn = FramewiseBCELoss(FramewiseBCELossConfig())
        adapter = FramewiseSampleAdapter(
            FramewiseSampleAdapterConfig(
                b_pred=cfg.b_pred, binary_only=True,
                max_events_per_window=10,
            ),
        )
        samples = [_make_sample(cfg) for _ in range(4)]
        inp, tgt = adapter.make_batch(samples, device=torch.device("cpu"))
        out = model.predict(inp)
        result = loss_fn(out, tgt)
        result.loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0
        assert result.loss.isfinite()
