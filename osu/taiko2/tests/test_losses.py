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
from osu.taiko2.training.losses import (
    GaussianCELoss,
    GaussianCELossConfig,
    LogEmdLoss,
    LogEmdLossConfig,
    MdnLoss,
    MdnLossConfig,
    OnsetLoss,
    OnsetLossConfig,
    parse_mdn_params,
)


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


# ─────────────────────────── GaussianCELoss ───────────────────────────

def _gcall(
    loss: GaussianCELoss, logits: torch.Tensor, target_bin: torch.Tensor,
) -> LossResult:
    return loss(
        EventEmbeddingOutput(logits=logits),
        EventEmbeddingTarget(target_bin=target_bin),
    )


class TestGaussianCELossConfig:
    def test_default_sigma(self):
        assert GaussianCELossConfig().sigma_bins == 2.0

    def test_sigma_must_be_positive(self):
        with pytest.raises(ValueError, match="sigma_bins"):
            GaussianCELoss(GaussianCELossConfig(sigma_bins=0.0))
        with pytest.raises(ValueError, match="sigma_bins"):
            GaussianCELoss(GaussianCELossConfig(sigma_bins=-1.0))


class TestGaussianCESoftTargets:
    def test_gaussian_is_unimodal_around_target(self):
        loss = GaussianCELoss(GaussianCELossConfig(sigma_bins=2.0))
        t = torch.tensor([50])
        w = loss._gaussian_bin_targets(t, n_bins=500)[0]
        # Peaked at the target and monotonically non-increasing as we
        # walk away on either side.
        assert int(w.argmax()) == 50
        assert (w[:50].diff() >= -1e-6).all()
        assert (w[50:].diff() <= 1e-6).all()

    def test_gaussian_sums_to_one(self):
        loss = GaussianCELoss(GaussianCELossConfig(sigma_bins=3.0))
        t = torch.tensor([5, 250, 499])
        w = loss._gaussian_bin_targets(t, n_bins=500)
        for row in w:
            assert math.isclose(float(row.sum()), 1.0, rel_tol=1e-5)

    def test_smaller_sigma_is_sharper(self):
        t = torch.tensor([100])
        w_narrow = GaussianCELoss(
            GaussianCELossConfig(sigma_bins=1.0),
        )._gaussian_bin_targets(t, n_bins=500)[0]
        w_wide = GaussianCELoss(
            GaussianCELossConfig(sigma_bins=5.0),
        )._gaussian_bin_targets(t, n_bins=500)[0]
        assert float(w_narrow[100]) > float(w_wide[100])


class TestGaussianCEForward:
    def test_stop_routes_to_bce_only(self):
        loss = GaussianCELoss(GaussianCELossConfig())
        logits = torch.randn(3, 501)
        # All STOP targets → bin_ce must be exactly 0.
        r = _gcall(loss, logits, torch.tensor([500, 500, 500]))
        assert r.metrics["bin_ce"] == 0.0
        assert r.metrics["stop_bce"] > 0.0
        assert r.metrics["stop_rate"] == 1.0

    def test_bin_only_batch_has_nonzero_bin_ce(self):
        loss = GaussianCELoss(GaussianCELossConfig())
        logits = torch.randn(3, 501)
        r = _gcall(loss, logits, torch.tensor([50, 100, 250]))
        assert r.metrics["bin_ce"] > 0.0
        assert r.metrics["stop_rate"] == 0.0

    def test_loss_is_sum_of_components(self):
        loss = GaussianCELoss(GaussianCELossConfig())
        logits = torch.randn(4, 501)
        r = _gcall(loss, logits, torch.tensor([10, 100, 500, 250]))
        assert math.isclose(
            r.metrics["loss"],
            r.metrics["stop_bce"] + r.metrics["bin_ce"],
            rel_tol=1e-5, abs_tol=1e-6,
        )

    def test_stop_bce_ignores_bin_logits(self):
        """Perturbing bin logits must not change `stop_bce` — the
        STOP head is a pure binary readout of logit[-1]."""
        loss = GaussianCELoss(GaussianCELossConfig())
        torch.manual_seed(0)
        logits_a = torch.randn(4, 501)
        logits_b = logits_a.clone()
        logits_b[:, :500] += 5.0
        t = torch.tensor([10, 100, 500, 250])
        r_a = _gcall(loss, logits_a, t)
        r_b = _gcall(loss, logits_b, t)
        assert math.isclose(
            r_a.metrics["stop_bce"], r_b.metrics["stop_bce"], rel_tol=1e-5,
        )

    def test_bin_ce_ignores_stop_logit(self):
        """Perturbing the STOP logit must not change `bin_ce` — bin
        CE runs softmax over only the first 500 logits."""
        loss = GaussianCELoss(GaussianCELossConfig())
        torch.manual_seed(0)
        logits_a = torch.randn(4, 501)
        logits_b = logits_a.clone()
        logits_b[:, 500] += 10.0
        t = torch.tensor([10, 100, 250, 400])
        r_a = _gcall(loss, logits_a, t)
        r_b = _gcall(loss, logits_b, t)
        assert math.isclose(
            r_a.metrics["bin_ce"], r_b.metrics["bin_ce"], rel_tol=1e-5,
        )

    def test_grad_flows(self):
        loss = GaussianCELoss(GaussianCELossConfig())
        logits = torch.randn(4, 501, requires_grad=True)
        r = _gcall(loss, logits, torch.tensor([10, 100, 500, 250]))
        r.loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_metrics_are_floats(self):
        loss = GaussianCELoss(GaussianCELossConfig())
        logits = torch.randn(3, 501, requires_grad=True)
        r = _gcall(loss, logits, torch.tensor([10, 100, 500]))
        for k, v in r.metrics.items():
            assert isinstance(v, float), f"{k} is {type(v)}"


# ─────────────────────────── LogEmdLoss ───────────────────────────────

def _ecall(
    loss: LogEmdLoss, logits: torch.Tensor, target_bin: torch.Tensor,
) -> LossResult:
    return loss(
        EventEmbeddingOutput(logits=logits),
        EventEmbeddingTarget(target_bin=target_bin),
    )


def _spike_logits(B: int, n_classes: int, spike_idx: int | list[int],
                  height: float = 50.0) -> torch.Tensor:
    """Logits whose softmax is essentially a delta at `spike_idx` per row."""
    logits = torch.full((B, n_classes), -height, dtype=torch.float32)
    if isinstance(spike_idx, int):
        spike_idx = [spike_idx] * B
    for b, j in enumerate(spike_idx):
        logits[b, j] = height
    return logits


def _bimodal_logits(B: int, n_classes: int, idxs: list[tuple[int, int]],
                    height: float = 5.0) -> torch.Tensor:
    """Logits whose softmax has roughly equal mass on two indices per row.
    Lower height than `_spike_logits` so two-mode mass-split is real."""
    logits = torch.full((B, n_classes), -height, dtype=torch.float32)
    for b, (i, j) in enumerate(idxs):
        logits[b, i] = height
        logits[b, j] = height
    return logits


class TestLogEmdLossConfig:
    def test_defaults(self):
        c = LogEmdLossConfig()
        assert c.hard_alpha == 0.5
        assert c.exponent == 1
        assert c.stop_weight == 1.5

    def test_hard_alpha_range(self):
        with pytest.raises(ValueError, match="hard_alpha"):
            LogEmdLoss(LogEmdLossConfig(hard_alpha=-0.1))
        with pytest.raises(ValueError, match="hard_alpha"):
            LogEmdLoss(LogEmdLossConfig(hard_alpha=1.1))

    def test_exponent_must_be_1_or_2(self):
        for bad in (0, 3, -1):
            with pytest.raises(ValueError, match="exponent"):
                LogEmdLoss(LogEmdLossConfig(exponent=bad))

    def test_stop_weight_positive(self):
        with pytest.raises(ValueError, match="stop_weight"):
            LogEmdLoss(LogEmdLossConfig(stop_weight=0.0))
        with pytest.raises(ValueError, match="stop_weight"):
            LogEmdLoss(LogEmdLossConfig(stop_weight=-1.0))


class TestLogEmdValueProperties:
    """Properties of the log_emd term that the loss landscape analysis
    promised, computed exactly here."""

    def test_log_emd_zero_on_perfect_spike(self):
        """Sharp prediction at p=t ⇒ log_emd ≈ 0."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0))
        logits = _spike_logits(1, 501, spike_idx=100, height=80.0)
        r = _ecall(loss, logits, torch.tensor([100]))
        # log_emd is the only contribution at hard_alpha=0; should be tiny.
        assert r.metrics["log_emd"] < 1e-3

    def test_log_emd_octave_costs_log2(self):
        """Sharp prediction at p=2t (delta at 200, target 100) costs
        approximately log(2) ≈ 0.693."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0))
        logits = _spike_logits(1, 501, spike_idx=200, height=80.0)
        r = _ecall(loss, logits, torch.tensor([100]))
        expected = math.log((200 + 1) / (100 + 1))  # ≈ 0.689
        assert math.isclose(r.metrics["log_emd"], expected, abs_tol=1e-2)

    def test_log_emd_symmetric_log_space(self):
        """p=2t and p=t/2 cost equal in log-EMD (octave-symmetric)."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0))
        # target 100; predictions at 200 (doubling) and 50 (halving).
        # |log(201/101)| ≈ 0.689; |log(51/101)| ≈ 0.683 (off by 1 from
        # exact symmetry because of the +1 offset on tiny indices).
        l_double = _ecall(
            loss, _spike_logits(1, 501, 200, 80.0), torch.tensor([100]),
        ).metrics["log_emd"]
        l_halve = _ecall(
            loss, _spike_logits(1, 501, 50, 80.0), torch.tensor([100]),
        ).metrics["log_emd"]
        # They should be close — exactly equal up to the +1 in (i+1)/(t+1).
        assert math.isclose(l_double, l_halve, rel_tol=0.02)

    def test_log_emd_punishes_bimodal_octave_hedging(self):
        """A 50/50 mass split between t/2 and 2t scores ≈ log(2),
        which is HIGHER than a sharp 1.5×t prediction's penalty."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0))
        # Bimodal at 50 and 200 for target 100.
        bimodal = _bimodal_logits(1, 501, [(50, 200)], height=15.0)
        r_bimodal = _ecall(loss, bimodal, torch.tensor([100])).metrics["log_emd"]
        # Sharp at 150 for target 100 — a moderate single-bin miss.
        sharp_close = _spike_logits(1, 501, 150, 80.0)
        r_close = _ecall(loss, sharp_close, torch.tensor([100])).metrics["log_emd"]
        # bimodal-octave penalty dominates the moderate-miss penalty.
        assert r_bimodal > r_close * 1.5, (
            f"bimodal {r_bimodal:.3f} should be >> close-miss {r_close:.3f}"
        )

    def test_exponent_2_amplifies_far_errors(self):
        """exponent=2 makes far-off mass cost much more than linear."""
        # Sharp prediction at 200 for target 100.
        logits = _spike_logits(1, 501, 200, 80.0)
        l1 = _ecall(
            LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0, exponent=1)),
            logits, torch.tensor([100]),
        ).metrics["log_emd"]
        l2 = _ecall(
            LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0, exponent=2)),
            logits, torch.tensor([100]),
        ).metrics["log_emd"]
        # log(2)^2 ≈ 0.481 < log(2) ≈ 0.689 — the squared variant is
        # smaller for octave but larger for very-far errors. Verify
        # that l2 ≈ l1^2 here:
        assert math.isclose(l2, l1 ** 2, rel_tol=0.05)


class TestLogEmdStopBehaviour:
    def test_stop_target_skips_log_emd(self):
        """log_emd is 0 on STOP-target samples regardless of where
        prediction mass sits."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0, stop_weight=1.0))
        logits = _spike_logits(1, 501, 100, 80.0)  # mass on bin 100
        r = _ecall(loss, logits, torch.tensor([500]))             # STOP target
        # log_emd is a per-batch mean over all samples; for this single
        # STOP sample it should be exactly 0.
        assert r.metrics["log_emd"] == 0.0

    def test_stop_weight_multiplies_stop_loss(self):
        """STOP samples pay stop_weight * (their per-sample loss).
        Compare two losses identical except stop_weight."""
        logits = _spike_logits(1, 501, 100, 5.0)  # mass on bin 100
        # Use hard_alpha=1.0 (pure hard CE) so STOP behaviour mirrors
        # OnsetLoss exactly when target is STOP.
        l_a = _ecall(
            LogEmdLoss(LogEmdLossConfig(hard_alpha=1.0, stop_weight=1.0)),
            logits, torch.tensor([500]),
        ).loss
        l_b = _ecall(
            LogEmdLoss(LogEmdLossConfig(hard_alpha=1.0, stop_weight=2.0)),
            logits, torch.tensor([500]),
        ).loss
        # stop_weight=2 should double the per-sample loss vs stop_weight=1.
        assert math.isclose(float(l_b), 2.0 * float(l_a), rel_tol=1e-5)

    def test_mixed_batch_with_stop_runs_cleanly(self):
        loss = LogEmdLoss(LogEmdLossConfig())
        logits = torch.randn(4, 501)
        targets = torch.tensor([10, 100, 500, 250])
        r = _ecall(loss, logits, targets)
        assert math.isfinite(r.metrics["loss"])
        assert math.isfinite(r.metrics["log_emd"])
        assert math.isfinite(r.metrics["hard_ce"])


class TestLogEmdMixing:
    def test_hard_alpha_zero_pure_log_emd(self):
        """At hard_alpha=0 the loss equals log_emd (over batch mean) for
        a bin-only batch."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0, stop_weight=1.0))
        logits = torch.randn(3, 501)
        r = _ecall(loss, logits, torch.tensor([10, 100, 250]))
        assert math.isclose(r.metrics["loss"], r.metrics["log_emd"], rel_tol=1e-5)

    def test_hard_alpha_one_pure_hard_ce(self):
        """At hard_alpha=1 the loss equals hard_CE (over batch mean) for
        a bin-only batch."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=1.0, stop_weight=1.0))
        logits = torch.randn(3, 501)
        r = _ecall(loss, logits, torch.tensor([10, 100, 250]))
        assert math.isclose(r.metrics["loss"], r.metrics["hard_ce"], rel_tol=1e-5)

    def test_total_is_alpha_blend(self):
        """For a bin-only batch with stop_weight=1, the total per-sample
        loss equals α·hard_ce + (1−α)·log_emd up to numerical precision."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.3, stop_weight=1.0))
        logits = torch.randn(4, 501)
        r = _ecall(loss, logits, torch.tensor([10, 100, 250, 400]))
        expected = 0.3 * r.metrics["hard_ce"] + 0.7 * r.metrics["log_emd"]
        assert math.isclose(r.metrics["loss"], expected, rel_tol=1e-5)


class TestLogEmdGradient:
    def test_grad_flows_through_logits(self):
        loss = LogEmdLoss(LogEmdLossConfig())
        logits = torch.randn(4, 501, requires_grad=True)
        r = _ecall(loss, logits, torch.tensor([10, 100, 500, 250]))
        r.loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0

    def test_grad_pushes_mass_toward_target(self):
        """One step of gradient descent on the logits should push the
        log-EMD term DOWN — i.e. probability mass moves toward `t`."""
        loss = LogEmdLoss(LogEmdLossConfig(hard_alpha=0.0, stop_weight=1.0))
        torch.manual_seed(0)
        logits = torch.randn(1, 501, requires_grad=True)
        target = torch.tensor([100])
        before = _ecall(loss, logits, target).metrics["log_emd"]
        # one SGD step
        r = _ecall(loss, logits, target)
        r.loss.backward()
        with torch.no_grad():
            logits = logits - 0.5 * logits.grad
        logits.requires_grad_(True)
        after = _ecall(loss, logits, target).metrics["log_emd"]
        assert after < before, f"log_emd did not decrease: {before:.4f} -> {after:.4f}"

    def test_metrics_are_floats(self):
        loss = LogEmdLoss(LogEmdLossConfig())
        logits = torch.randn(3, 501, requires_grad=True)
        r = _ecall(loss, logits, torch.tensor([10, 100, 500]))
        for k, v in r.metrics.items():
            assert isinstance(v, float), f"{k} is {type(v)}"


# ─────────────────────────── MdnLoss ──────────────────────────────────

def _mdn_raw(B: int, K: int = 3) -> torch.Tensor:
    """Random MDN output tensor of shape (B, K*3+1)."""
    return torch.randn(B, K * 3 + 1)


def _mcall(
    loss: MdnLoss, raw: torch.Tensor, target_bin: torch.Tensor,
) -> LossResult:
    return loss(
        EventEmbeddingOutput(logits=raw),
        EventEmbeddingTarget(target_bin=target_bin),
    )


class TestMdnParsing:
    def test_shapes(self):
        raw = _mdn_raw(4, K=3)
        stop_logit, mu, sigma, pi = parse_mdn_params(raw, 3, 500)
        assert stop_logit.shape == (4,)
        assert mu.shape == (4, 3)
        assert sigma.shape == (4, 3)
        assert pi.shape == (4, 3)

    def test_mu_in_range(self):
        raw = torch.randn(10, 10)
        _, mu, _, _ = parse_mdn_params(raw, 3, 500)
        assert (mu >= 0).all()
        assert (mu <= 500).all()

    def test_sigma_above_floor(self):
        raw = torch.randn(10, 10)
        _, _, sigma, _ = parse_mdn_params(raw, 3, 500)
        assert (sigma >= 1.0).all()

    def test_pi_sums_to_one(self):
        raw = torch.randn(5, 10)
        _, _, _, pi = parse_mdn_params(raw, 3, 500)
        np_sums = pi.sum(dim=-1).detach().numpy()
        for s in np_sums:
            assert math.isclose(s, 1.0, rel_tol=1e-5)


class TestMdnLossConfig:
    def test_defaults(self):
        c = MdnLossConfig()
        assert c.n_components == 3
        assert c.b_pred == 500
        assert c.stop_weight == 1.5

    def test_n_components_positive(self):
        with pytest.raises(ValueError, match="n_components"):
            MdnLoss(MdnLossConfig(n_components=0))


class TestMdnLossForward:
    def test_runs_on_mixed_batch(self):
        loss = MdnLoss(MdnLossConfig())
        raw = _mdn_raw(4)
        r = _mcall(loss, raw, torch.tensor([10, 100, 500, 250]))
        assert math.isfinite(r.metrics["loss"])
        assert math.isfinite(r.metrics["mixture_nll"])
        assert math.isfinite(r.metrics["stop_bce"])

    def test_stop_only_batch(self):
        loss = MdnLoss(MdnLossConfig())
        raw = _mdn_raw(3)
        r = _mcall(loss, raw, torch.tensor([500, 500, 500]))
        # mixture_nll should be 0 — no bin samples.
        assert r.metrics["mixture_nll"] == 0.0
        assert r.metrics["stop_bce"] > 0.0
        assert r.metrics["stop_rate"] == 1.0

    def test_bin_only_batch(self):
        loss = MdnLoss(MdnLossConfig())
        raw = _mdn_raw(3)
        r = _mcall(loss, raw, torch.tensor([10, 100, 250]))
        assert r.metrics["mixture_nll"] > 0.0
        assert r.metrics["stop_rate"] == 0.0

    def test_stop_weight_multiplies(self):
        raw = _mdn_raw(1)
        t = torch.tensor([500])
        l1 = _mcall(
            MdnLoss(MdnLossConfig(stop_weight=1.0)), raw, t,
        ).loss
        l2 = _mcall(
            MdnLoss(MdnLossConfig(stop_weight=2.0)), raw, t,
        ).loss
        assert math.isclose(float(l2), 2.0 * float(l1), rel_tol=1e-4)

    def test_grad_flows(self):
        loss = MdnLoss(MdnLossConfig())
        raw = _mdn_raw(4).requires_grad_(True)
        r = _mcall(loss, raw, torch.tensor([10, 100, 500, 250]))
        r.loss.backward()
        assert raw.grad is not None
        assert torch.isfinite(raw.grad).all()
        assert raw.grad.abs().sum() > 0

    def test_loss_decreases_when_component_matches_target(self):
        """Manually place a component's μ near the target — loss
        should be lower than with random params."""
        loss = MdnLoss(MdnLossConfig(n_components=1, b_pred=500))
        # 1 component: raw = (B, 4) = [stop_logit, mu_raw, log_sigma, log_pi]
        # Place mu at target 100: sigmoid(mu_raw)*500 = 100 → mu_raw = logit(0.2)
        mu_raw = math.log(0.2 / 0.8)
        raw_good = torch.tensor([[
            -5.0,           # stop_logit (low → not STOP)
            mu_raw,         # mu_raw → sigmoid → 0.2 → *500 = 100
            -1.0,           # log_sigma → softplus(-1)+1 ≈ 1.31 (tight)
            0.0,            # log_pi (only 1 component, doesn't matter)
        ]])
        raw_bad = torch.tensor([[
            -5.0, 5.0, -1.0, 0.0,  # mu_raw=5 → sigmoid→0.993 → *500≈497
        ]])
        t = torch.tensor([100])
        l_good = float(_mcall(loss, raw_good, t).loss)
        l_bad = float(_mcall(loss, raw_bad, t).loss)
        assert l_good < l_bad, f"matched {l_good:.2f} should be < mismatched {l_bad:.2f}"


class TestMdnDiagnostics:
    def test_coverage_metrics_present(self):
        loss = MdnLoss(MdnLossConfig())
        raw = _mdn_raw(4)
        r = _mcall(loss, raw, torch.tensor([10, 100, 500, 250]))
        assert "mdn/coverage_2bin" in r.metrics
        assert "mdn/coverage_5bin" in r.metrics
        assert "mdn/dominant_weight" in r.metrics
        assert "mdn/n_active_components" in r.metrics
        assert "mdn/mean_sigma" in r.metrics
        assert "mdn/correct_component_weight" in r.metrics

    def test_all_metrics_are_floats(self):
        loss = MdnLoss(MdnLossConfig())
        raw = _mdn_raw(4).requires_grad_(True)
        r = _mcall(loss, raw, torch.tensor([10, 100, 500, 250]))
        for k, v in r.metrics.items():
            assert isinstance(v, float), f"{k} is {type(v)}"

    def test_coverage_is_one_when_component_exact(self):
        """Place K=1 component exactly at target → coverage = 1.0."""
        loss = MdnLoss(MdnLossConfig(n_components=1, b_pred=500))
        mu_raw = math.log(0.2 / 0.8)  # → mu ≈ 100
        raw = torch.tensor([[
            -5.0, mu_raw, -1.0, 0.0,
        ]])
        r = _mcall(loss, raw, torch.tensor([100]))
        assert r.metrics["mdn/coverage_2bin"] == 1.0
        assert r.metrics["mdn/coverage_5bin"] == 1.0
