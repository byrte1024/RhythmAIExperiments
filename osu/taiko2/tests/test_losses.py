"""Tests for `OnsetLoss`.

Covers: validation, soft-target shape + plateau + ramp + frame-floor,
hard/soft mix, STOP weighting, metrics reporting, differentiability.
"""
from __future__ import annotations

import math

import pytest
import torch

from osu.taiko2.domain.loss import LossResult
from osu.taiko2.models.event_embedding import (
    EventEmbeddingOutput,
    EventEmbeddingTarget,
)
from osu.taiko2.training.losses import OnsetLoss, OnsetLossConfig


# ─────────────────────────── helpers ──────────────────────────────────

def _call(loss: OnsetLoss, logits: torch.Tensor, target_bin: torch.Tensor) -> LossResult:
    return loss(
        EventEmbeddingOutput(logits=logits),
        EventEmbeddingTarget(target_bin=target_bin),
    )


# ─────────────────────────── config validation ────────────────────────

class TestOnsetLossConfig:
    def test_defaults_match_exp45(self):
        c = OnsetLossConfig()
        assert c.hard_alpha == 0.5
        assert c.good_pct == 0.03
        assert c.fail_pct == 0.20
        assert c.frame_tolerance == 2
        assert c.stop_weight == 1.5

    def test_hard_alpha_range(self):
        with pytest.raises(ValueError, match="hard_alpha"):
            OnsetLoss(OnsetLossConfig(hard_alpha=-0.1))
        with pytest.raises(ValueError, match="hard_alpha"):
            OnsetLoss(OnsetLossConfig(hard_alpha=1.1))

    def test_good_vs_fail_ordering(self):
        with pytest.raises(ValueError, match="good_pct"):
            OnsetLoss(OnsetLossConfig(good_pct=0.5, fail_pct=0.1))

    def test_negative_frame_tolerance(self):
        with pytest.raises(ValueError, match="frame_tolerance"):
            OnsetLoss(OnsetLossConfig(frame_tolerance=-1))


# ─────────────────────────── soft-target shape ────────────────────────

class TestSoftTargets:
    def test_stop_gets_one_hot(self):
        loss = OnsetLoss(OnsetLossConfig())
        n_classes = 501
        targets = torch.tensor([n_classes - 1, n_classes - 1])
        soft = loss._make_soft_targets(targets, n_classes)
        assert soft.shape == (2, n_classes)
        assert torch.allclose(soft[:, -1], torch.ones(2))
        assert torch.allclose(soft[:, :-1], torch.zeros(2, n_classes - 1))

    def test_bin_target_sums_to_one(self):
        loss = OnsetLoss(OnsetLossConfig())
        targets = torch.tensor([100, 42, 5, 300])
        soft = loss._make_soft_targets(targets, 501)
        # Each row sums to 1.
        row_sums = soft.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-6)

    def test_bin_target_zero_at_stop_class(self):
        """Non-STOP targets put zero mass on the STOP class."""
        loss = OnsetLoss(OnsetLossConfig())
        targets = torch.tensor([100, 250])
        soft = loss._make_soft_targets(targets, 501)
        assert torch.all(soft[:, -1] == 0)

    def test_plateau_width_matches_good_pct(self):
        """Bins inside the log(1+good_pct) window all get equal weight."""
        cfg = OnsetLossConfig(good_pct=0.10, fail_pct=0.20, frame_tolerance=0)
        loss = OnsetLoss(cfg)
        target = 100
        targets = torch.tensor([target])
        soft = loss._make_soft_targets(targets, 501)

        row = soft[0, :500]
        # A bin at distance ≤ log(1.10) from `target` in ratio space is
        # inside the plateau: |log((i+1)/(target+1))| ≤ log(1.10).
        bins = torch.arange(500, dtype=torch.float32)
        log_ratio = torch.abs(torch.log((bins + 1) / (target + 1)))
        in_plateau = log_ratio <= math.log(1.10)

        plateau_values = row[in_plateau]
        # All plateau entries should have the same normalized value
        # (there's also the frame floor contribution, but at
        # frame_tolerance=0 it only gives bin `target` a tiny extra —
        # clamped by `max` to the plateau's 1.0).
        assert plateau_values.std() < 1e-6
        assert plateau_values[0] > 0

    def test_ramp_decays_between_good_and_fail(self):
        """Weight strictly decreases as you move from good_pct to fail_pct."""
        cfg = OnsetLossConfig(good_pct=0.03, fail_pct=0.30, frame_tolerance=0)
        loss = OnsetLoss(cfg)
        target = 100
        targets = torch.tensor([target])
        soft = loss._make_soft_targets(targets, 501)
        row = soft[0, :500]

        bins = torch.arange(500, dtype=torch.float32)
        log_ratio = torch.abs(torch.log((bins + 1) / (target + 1)))
        ramp_mask = (log_ratio > math.log(1.03)) & (log_ratio <= math.log(1.30))
        # Sort ramp entries by log_ratio; values should be monotonically
        # non-increasing.
        if ramp_mask.sum() >= 2:
            ramp_rows = torch.stack([log_ratio[ramp_mask], row[ramp_mask]], dim=0)
            order = torch.argsort(ramp_rows[0])
            sorted_vals = ramp_rows[1, order]
            # Allow equal values (plateau boundary); weight never increases.
            assert torch.all(sorted_vals[1:] <= sorted_vals[:-1] + 1e-6)

    def test_zero_outside_fail(self):
        cfg = OnsetLossConfig(good_pct=0.03, fail_pct=0.20, frame_tolerance=0)
        loss = OnsetLoss(cfg)
        # target=100, fail_pct=0.20 → log cutoff ≈ 0.182.
        # bin=500 means ratio ~5.0 → log(5) ≈ 1.6, way past cutoff.
        targets = torch.tensor([100])
        soft = loss._make_soft_targets(targets, 501)
        assert soft[0, 499] < 1e-6  # well outside the ramp

    def test_frame_tolerance_provides_floor(self):
        """For very small targets, the ±frame window keeps a plateau
        even when the ratio window is narrower than a bin."""
        # target=0 → the log-ratio window is centered on bin 0; a ±2
        # frame floor means bins 0..2 get guaranteed credit.
        cfg = OnsetLossConfig(good_pct=0.03, fail_pct=0.20, frame_tolerance=2)
        loss = OnsetLoss(cfg)
        targets = torch.tensor([0])
        soft = loss._make_soft_targets(targets, 501)
        # Bins 0, 1, 2 should all have > 0 weight.
        assert soft[0, 0] > 0
        assert soft[0, 1] > 0
        assert soft[0, 2] > 0
        # Bin 10 is far outside both windows → should be 0.
        assert soft[0, 10] < 1e-6


# ─────────────────────────── mix behavior ─────────────────────────────

class TestMixBehavior:
    def _setup_logits(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 501, requires_grad=True)
        targets = torch.tensor([10, 100, 250, 500])  # last is STOP
        return logits, targets

    def test_pure_hard_alpha_one(self):
        """hard_alpha=1 → total == hard_ce, soft_ce unused."""
        logits, targets = self._setup_logits()
        loss = OnsetLoss(OnsetLossConfig(hard_alpha=1.0, stop_weight=1.0))
        r = _call(loss, logits, targets)
        # With stop_weight=1, the total should equal mean(hard_ce).
        assert r.metrics["loss"] == pytest.approx(r.metrics["hard_ce"])

    def test_pure_hard_alpha_zero(self):
        """hard_alpha=0 → total == soft_ce."""
        logits, targets = self._setup_logits()
        loss = OnsetLoss(OnsetLossConfig(hard_alpha=0.0, stop_weight=1.0))
        r = _call(loss, logits, targets)
        assert r.metrics["loss"] == pytest.approx(r.metrics["soft_ce"])

    def test_stop_weight_amplifies_stop_samples(self):
        """stop_weight=3 should yield a larger total than stop_weight=1."""
        logits, targets = self._setup_logits()
        a = OnsetLoss(OnsetLossConfig(hard_alpha=0.5, stop_weight=1.0))
        b = OnsetLoss(OnsetLossConfig(hard_alpha=0.5, stop_weight=3.0))
        ra = _call(a, logits, targets)
        rb = _call(b, logits, targets)
        assert rb.metrics["loss"] > ra.metrics["loss"]


# ─────────────────────────── metrics + differentiability ──────────────

class TestLossResult:
    def test_reports_all_sub_metrics(self):
        loss = OnsetLoss(OnsetLossConfig())
        logits = torch.randn(4, 501, requires_grad=True)
        targets = torch.tensor([10, 100, 250, 500])
        r = _call(loss, logits, targets)
        assert isinstance(r, LossResult)
        assert set(r.metrics) >= {"loss", "hard_ce", "soft_ce", "stop_rate"}

    def test_stop_rate_is_batch_fraction(self):
        loss = OnsetLoss(OnsetLossConfig())
        logits = torch.randn(4, 501)
        # 1 STOP out of 4 → 0.25
        targets = torch.tensor([10, 100, 250, 500])
        r = _call(loss, logits, targets)
        assert r.metrics["stop_rate"] == pytest.approx(0.25)
        # All STOPs
        targets_all_stop = torch.tensor([500, 500, 500, 500])
        r = _call(loss, logits, targets_all_stop)
        assert r.metrics["stop_rate"] == pytest.approx(1.0)

    def test_differentiable(self):
        """Backward through the loss produces finite gradients."""
        loss = OnsetLoss(OnsetLossConfig())
        logits = torch.randn(4, 501, requires_grad=True)
        targets = torch.tensor([10, 100, 250, 500])
        r = _call(loss, logits, targets)
        r.loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_metrics_detached(self):
        """Metric dict values are plain floats, not graph-connected."""
        loss = OnsetLoss(OnsetLossConfig())
        logits = torch.randn(4, 501, requires_grad=True)
        targets = torch.tensor([10, 100, 250, 500])
        r = _call(loss, logits, targets)
        for k, v in r.metrics.items():
            assert isinstance(v, float), f"{k} is {type(v)}"
