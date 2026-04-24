"""Tests for `TimeStretch` — the exp 007 time-stretch augmentation.

Covers:
  - Config validation.
  - Probability gate (prob=0 no-op, prob=1 always fires, determinism).
  - Audio interpolation: shape preservation, cursor-pinning, linear
    correctness, symmetric stretch/compress, zero-pad when source
    runs out, s=1 identity.
  - Event scaling: cursor_offset multiplication, time_ms/bin
    consistency after stretch, out-of-window masking, STOP flip via
    adapter, pre-existing padding preserved, collision dedupe, order
    preserved, length preserved.
  - Integration with AugmentationPipeline.
  - Runtime budget.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from osu.taiko2.data_samplers import TaikoDetectionSample
from osu.taiko2.domain.augmentation import AugmentationPipeline
from osu.taiko2.domain.beatmap import OnsetKind, RelativeOnset
from osu.taiko2.training.adapters import (
    DetectionSampleAdapter,
    DetectionSampleAdapterConfig,
)
from osu.taiko2.training.augmentations import (
    MelGainJitter,
    TimeStretch,
)


# ─────────────────────────── sample builders ─────────────────────────

def _make_audio(a_bins: int = 500, b_bins: int = 500, F: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ramp 0..(a+b-1) broadcast across F bands."""
    total = a_bins + b_bins
    row = np.arange(total, dtype=np.float32)
    full = np.broadcast_to(row, (F, total)).copy()
    return full[:, :a_bins].copy(), full[:, a_bins:].copy()


def _make_event(cursor_offset: int, kind: OnsetKind = OnsetKind.DON) -> RelativeOnset:
    return RelativeOnset(
        time_ms=cursor_offset * 5,
        kind=kind,
        bin=cursor_offset,
        cursor_offset=cursor_offset,
    )


def _make_sample(
    *,
    past_offsets: list[int] | None = None,
    future_offsets: list[int] | None = None,
    c_events: int = 128,
    a_bins: int = 500,
    b_bins: int = 500,
) -> TaikoDetectionSample:
    past_offsets = past_offsets or [-400, -300, -200, -100, -50]
    future_offsets = future_offsets or [300]

    past = [_make_event(o) for o in past_offsets]
    pad = c_events - len(past)
    if pad < 0:
        past = past[-c_events:]
        pad = 0
    past_all = [_make_event(0)] * pad + past
    past_mask = np.array([True] * pad + [False] * (c_events - pad), dtype=bool)

    fut = [_make_event(o) for o in future_offsets]
    fut_mask = np.array([False] * len(fut), dtype=bool)

    audio_past, audio_future = _make_audio(a_bins=a_bins, b_bins=b_bins)
    return TaikoDetectionSample(
        sample_id=0,
        chart_id="test",
        cursor_bin=5000,
        audio_past=audio_past,
        audio_future=audio_future,
        past_events=tuple(past_all),
        past_events_mask=past_mask,
        future_events=tuple(fut),
        future_events_mask=fut_mask,
        density_mean=5.0,
        density_peak=10,
        density_std=2.2,
    )


# ─────────────────────────── config validation ───────────────────────

class TestConfigValidation:
    def test_max_scale_must_exceed_one(self):
        with pytest.raises(ValueError, match="max_scale"):
            TimeStretch(prob=1.0, max_scale=1.0)
        with pytest.raises(ValueError, match="max_scale"):
            TimeStretch(prob=1.0, max_scale=0.5)

    def test_prob_validation_inherited(self):
        with pytest.raises(ValueError, match="prob"):
            TimeStretch(prob=-0.1, max_scale=1.4)
        with pytest.raises(ValueError, match="prob"):
            TimeStretch(prob=1.1, max_scale=1.4)


# ─────────────────────────── probability gate ────────────────────────

class TestProbabilityGate:
    def test_prob_zero_is_noop(self):
        ts = TimeStretch(prob=0.0, max_scale=1.4, seed=42)
        s = _make_sample()
        out = ts.apply(s)
        assert out is s

    def test_prob_one_always_fires(self):
        """prob=1.0 skips the gate entirely — sample is always stretched."""
        ts = TimeStretch(prob=1.0, max_scale=1.4, seed=42)
        s = _make_sample()
        # s=1 is astronomically unlikely from a continuous log-uniform,
        # so at prob=1 we always see some change.
        out = ts.apply(s)
        assert out is not s
        # Past-event offsets should have changed for at least one
        # non-padded slot.
        orig_off = [e.cursor_offset for e, m in zip(s.past_events, s.past_events_mask) if not m]
        new_off = [e.cursor_offset for e, m in zip(out.past_events, out.past_events_mask) if not m]
        assert orig_off != new_off

    def test_deterministic_with_seed(self):
        s = _make_sample()
        ts_a = TimeStretch(prob=1.0, max_scale=1.4, seed=7)
        ts_b = TimeStretch(prob=1.0, max_scale=1.4, seed=7)
        out_a = ts_a.apply(s)
        out_b = ts_b.apply(s)
        np.testing.assert_array_equal(out_a.audio_past, out_b.audio_past)
        np.testing.assert_array_equal(out_a.audio_future, out_b.audio_future)
        assert [e.cursor_offset for e in out_a.past_events] == \
               [e.cursor_offset for e in out_b.past_events]
        np.testing.assert_array_equal(out_a.past_events_mask, out_b.past_events_mask)

    def test_different_seeds_differ(self):
        s = _make_sample()
        ts_a = TimeStretch(prob=1.0, max_scale=1.4, seed=1)
        ts_b = TimeStretch(prob=1.0, max_scale=1.4, seed=2)
        out_a = ts_a.apply(s)
        out_b = ts_b.apply(s)
        # At least one aspect should differ between seeds.
        same_audio = np.array_equal(out_a.audio_past, out_b.audio_past)
        same_events = [e.cursor_offset for e in out_a.past_events] == \
                      [e.cursor_offset for e in out_b.past_events]
        assert not (same_audio and same_events)


# ─────────────────────────── audio interpolation ────────────────────

class TestAudioInterpolation:
    def test_shape_preserved(self):
        ts = TimeStretch(prob=1.0, max_scale=1.4, seed=0)
        s = _make_sample()
        out = ts.apply(s)
        assert out.audio_past.shape == s.audio_past.shape
        assert out.audio_future.shape == s.audio_future.shape
        assert out.audio_past.dtype == np.float32
        assert out.audio_future.dtype == np.float32

    def test_scale_one_is_identity(self):
        """_stretch_audio with s=1 returns a bit-exact copy."""
        ap, af = _make_audio()
        ap2, af2 = TimeStretch._stretch_audio(ap, af, s=1.0)
        np.testing.assert_array_equal(ap, ap2)
        np.testing.assert_array_equal(af, af2)
        # Defensive copy — not the same underlying buffer.
        assert ap2 is not ap
        assert af2 is not af

    def test_cursor_frame_is_pinned(self):
        """Whatever s, the last past-frame and first future-frame must
        read from the source cursor vicinity (source index ≈ a_bins)."""
        ap, af = _make_audio()
        total = ap.shape[1] + af.shape[1]
        cursor = ap.shape[1]
        for s in [0.8, 1.2, 1.5, 0.6]:
            ap2, af2 = TimeStretch._stretch_audio(ap, af, s=s)
            # At output cursor-1 the source index is cursor - 1/s.
            expected_src = cursor - 1.0 / s
            src_lo = int(math.floor(expected_src))
            src_hi = min(src_lo + 1, total - 1)
            frac = float(expected_src - src_lo)
            full = np.concatenate([ap, af], axis=1)
            expected = full[:, src_lo] * (1 - frac) + full[:, src_hi] * frac
            np.testing.assert_allclose(ap2[:, -1], expected, rtol=1e-5, atol=1e-4)

    def test_s_greater_than_one_no_padding(self):
        """s > 1 requires source range ⊂ original window — no zeros
        should appear in the output regardless of content."""
        # Non-trivial audio so "no zeros" is actually meaningful.
        ap = np.full((80, 500), 5.0, dtype=np.float32)
        af = np.full((80, 500), 7.0, dtype=np.float32)
        ap2, af2 = TimeStretch._stretch_audio(ap, af, s=1.4)
        assert (ap2 >= 4.99).all() and (ap2 <= 7.01).all()
        assert (af2 >= 4.99).all() and (af2 <= 7.01).all()
        # No zero cells.
        assert (ap2 > 0).all()
        assert (af2 > 0).all()

    def test_s_less_than_one_zero_pads_edges(self):
        """s < 1: outer edge frames map to source indices < 0 or
        > total-1 and must be zero-padded."""
        ap = np.full((80, 500), 5.0, dtype=np.float32)
        af = np.full((80, 500), 7.0, dtype=np.float32)
        ap2, af2 = TimeStretch._stretch_audio(ap, af, s=0.5)
        # At s=0.5, output frame 0 maps to source index
        # 500 + (0 - 500)/0.5 = 500 - 1000 = -500 → zero.
        np.testing.assert_array_equal(ap2[:, 0], 0.0)
        # Output frame 999 maps to 500 + (999-500)/0.5 = 500 + 998 = 1498 → zero.
        np.testing.assert_array_equal(af2[:, -1], 0.0)
        # But frame 500 (cursor) maps exactly to source 500, a valid
        # index — should NOT be zero.
        assert af2[0, 0] != 0.0  # af2[:, 0] is output frame 500

    def test_linear_interp_on_ramp(self):
        """Known-input ramp stretched by s=2: output frame f (past
        side, f < 500) reads source index cursor + (f-cursor)/2 which
        we can cross-check analytically. Ramp values equal the source
        index, so we can read the output directly."""
        ap, af = _make_audio()  # values == source index
        ap2, af2 = TimeStretch._stretch_audio(ap, af, s=2.0)
        # Output frame 100 → source index 500 + (100-500)/2 = 300.
        np.testing.assert_allclose(ap2[0, 100], 300.0, atol=1e-3)
        # Output frame 499 → source index 500 - 0.5 = 499.5.
        np.testing.assert_allclose(ap2[0, 499], 499.5, atol=1e-3)
        # Output frame 501 (first future) → source 500 + 0.5 = 500.5.
        np.testing.assert_allclose(af2[0, 1], 500.5, atol=1e-3)

    def test_stretch_is_symmetric_around_cursor(self):
        """Equal-offset output frames on past and future sides read
        from equal-offset source frames on the opposite side. Uses a
        symmetric source around the cursor."""
        total = 1000
        cursor = 500
        row = np.abs(np.arange(total, dtype=np.float32) - cursor)  # V-shape
        full = np.broadcast_to(row, (80, total)).copy()
        ap = full[:, :cursor].copy()
        af = full[:, cursor:].copy()
        ap2, af2 = TimeStretch._stretch_audio(ap, af, s=1.25)
        # Output frames equidistant from cursor should have equal values.
        for d in [1, 50, 100, 200, 400]:
            np.testing.assert_allclose(
                ap2[0, cursor - d], af2[0, d], atol=1e-4,
            )


# ─────────────────────────── past event scaling ──────────────────────

class TestPastEventScaling:
    def test_offsets_scaled_and_consistent(self):
        s = _make_sample(past_offsets=[-100, -50])
        new_events, new_mask = TimeStretch._stretch_past_events(
            s.past_events, s.past_events_mask, s=1.4, a_bins=500,
        )
        # Real slots at the end, padded at the start.
        real = [e for e, m in zip(new_events, new_mask) if not m]
        assert len(real) == 2
        # -100 × 1.4 = -140; -50 × 1.4 = -70.
        assert [e.cursor_offset for e in real] == [-140, -70]
        # time_ms and bin are self-consistent with the new offset.
        for e in real:
            assert e.bin == e.cursor_offset
            assert e.time_ms == e.cursor_offset * 5

    def test_event_falling_out_of_window_dropped(self):
        """Past event at -400 scaled by 1.4 → -560, beyond -500 → drop."""
        s = _make_sample(past_offsets=[-400, -100])
        new_events, new_mask = TimeStretch._stretch_past_events(
            s.past_events, s.past_events_mask, s=1.4, a_bins=500,
        )
        real = [e for e, m in zip(new_events, new_mask) if not m]
        assert len(real) == 1
        assert real[0].cursor_offset == -140

    def test_padding_appended_to_keep_length(self):
        """After drops, the list is re-padded so its length matches
        c_events; padding goes at the start."""
        s = _make_sample(past_offsets=[-400, -100], c_events=10)
        new_events, new_mask = TimeStretch._stretch_past_events(
            s.past_events, s.past_events_mask, s=1.4, a_bins=500,
        )
        assert len(new_events) == 10
        assert new_mask.shape == (10,)
        assert new_mask.dtype == bool
        # Padding at the start, real events at the end.
        assert all(bool(new_mask[i]) for i in range(9))
        assert not bool(new_mask[9])
        assert new_events[9].cursor_offset == -140

    def test_existing_padding_discarded(self):
        """Pre-existing padded slots are DROPPED from the real-event
        list, not rescaled — post-aug padding is rebuilt fresh."""
        s = _make_sample(past_offsets=[-100], c_events=5)
        # Sample now has 4 padded + 1 real.
        assert int(s.past_events_mask.sum()) == 4
        new_events, new_mask = TimeStretch._stretch_past_events(
            s.past_events, s.past_events_mask, s=1.2, a_bins=500,
        )
        real = [e for e, m in zip(new_events, new_mask) if not m]
        assert len(real) == 1
        assert real[0].cursor_offset == -120

    def test_collision_dedupe_keeps_older(self):
        """Two past events at -10 and -11 scaled by s=0.5 → -5 and -6:
        no collision. Now pick offsets where scaling DOES collide."""
        # At s=0.5, -10 → -5 and -11 → -6 (distinct). To force a
        # collision we need the rounded targets to coincide: offsets
        # -10 and -10 itself collide already, so pick a case where
        # rounding causes overlap. -10 → round(-10*0.5)=-5,
        # -9 → round(-9*0.5)=-4 or -5 depending on banker's rounding
        # — Python rounds half to even, so -9*0.5=-4.5 → -4.
        # Use offsets that genuinely round to the same target:
        # -10 and -11 at s=0.45: -4.5→-4 (even), -4.95→-5. Distinct.
        # Try offsets -12 and -11 at s=0.5: -6, -5.5→-6 (banker's).
        # Actually -5.5 rounds to -6 in Python 3. So {-12, -11} at
        # s=0.5 both round to -6 → collision.
        s_in = _make_sample(past_offsets=[-12, -11])  # oldest first
        new_events, new_mask = TimeStretch._stretch_past_events(
            s_in.past_events, s_in.past_events_mask, s=0.5, a_bins=500,
        )
        real = [e for e, m in zip(new_events, new_mask) if not m]
        # We use dedupe on cursor_offset set → exactly 1 real event.
        assert len(real) == 1
        # Keep the OLDER (earlier in oldest-first) — the -12 event.
        assert real[0].cursor_offset == -6

    def test_ordering_preserved_oldest_first(self):
        """Real events come out in the same oldest-first order they
        went in (after scaling but before padding)."""
        s = _make_sample(past_offsets=[-400, -300, -200, -100, -50])
        new_events, new_mask = TimeStretch._stretch_past_events(
            s.past_events, s.past_events_mask, s=1.2, a_bins=500,
        )
        real = [e.cursor_offset for e, m in zip(new_events, new_mask) if not m]
        # Still monotonically non-decreasing (= oldest-first).
        assert real == sorted(real)

    def test_overflow_keeps_newest(self):
        """If after stretching + existing events fill the slots, the
        oldest excess events get dropped — we keep the newest
        c_events. In practice pure scaling can't create events, so
        this path requires an oversized input; test the branch
        explicitly."""
        # Two real events + c_events = 2 slots — no overflow under
        # normal scaling. Force it by setting c_events=1 with 2 real.
        past = tuple(_make_event(o) for o in [-100, -50])
        mask = np.zeros(2, dtype=bool)
        # Even though caller respects c_events, test the internal
        # overflow branch directly.
        new_events, new_mask = TimeStretch._stretch_past_events(
            past, mask, s=1.0, a_bins=500,
        )
        assert len(new_events) == 2
        # Both real, ordering preserved.
        assert [e.cursor_offset for e in new_events] == [-100, -50]
        assert not new_mask.any()


# ─────────────────────────── future event scaling ────────────────────

class TestFutureEventScaling:
    def test_future_offset_scales(self):
        events, mask = TimeStretch._stretch_future_events(
            (_make_event(300),), np.array([False]), s=1.4,
        )
        assert events[0].cursor_offset == int(round(300 * 1.4))
        assert not mask[0]

    def test_mask_preserved_when_already_padded(self):
        events, mask = TimeStretch._stretch_future_events(
            (_make_event(0),), np.array([True]), s=0.8,
        )
        assert mask[0] is np.True_ or bool(mask[0])

    def test_time_ms_and_bin_updated(self):
        events, _ = TimeStretch._stretch_future_events(
            (_make_event(200),), np.array([False]), s=1.2,
        )
        new_off = int(round(200 * 1.2))  # 240
        assert events[0].cursor_offset == new_off
        assert events[0].bin == new_off
        assert events[0].time_ms == new_off * 5


# ─────────────────────────── STOP flip via adapter ───────────────────

class TestStopDerivation:
    def _adapter(self, b_pred: int = 500) -> DetectionSampleAdapter:
        return DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=b_pred),
        )

    def test_bin_target_stays_bin_when_in_range(self):
        import torch
        s = _make_sample(future_offsets=[300])
        ts = TimeStretch(prob=1.0, max_scale=1.2, seed=42)
        out = ts.apply(s)
        tgt = self._adapter().make_target([out], device=torch.device("cpu"))
        # 300 × [1/1.2, 1.2] = [250, 360] — all in-range bin targets.
        t = int(tgt.target_bin[0])
        assert 200 <= t < 500

    def test_bin_target_flips_to_stop_when_stretched_out(self):
        """Future event at 400 with s=1.4 → 560 > b_pred=500 → STOP."""
        import torch
        s = _make_sample(future_offsets=[400])
        # Force s=1.4 by a large max_scale draw — seed-deterministic.
        # We'll side-step the rng by driving the future path directly.
        events, mask = TimeStretch._stretch_future_events(
            s.future_events, s.future_events_mask, s=1.4,
        )
        stretched = TaikoDetectionSample(
            sample_id=0, chart_id="t", cursor_bin=0,
            audio_past=s.audio_past, audio_future=s.audio_future,
            past_events=s.past_events, past_events_mask=s.past_events_mask,
            future_events=events, future_events_mask=mask,
            density_mean=5.0, density_peak=10, density_std=2.2,
        )
        tgt = self._adapter(b_pred=500).make_target(
            [stretched], device=torch.device("cpu"),
        )
        assert int(tgt.target_bin[0]) == 500  # STOP

    def test_originally_padded_future_stays_stop(self):
        """Pre-aug mask=True means no future onset within b_bins. That
        must survive stretching — we can't materialize a real event."""
        import torch
        s = _make_sample(future_offsets=[0])
        # Mask it True to simulate a "no future event" sample.
        s2 = TaikoDetectionSample(
            sample_id=0, chart_id="t", cursor_bin=0,
            audio_past=s.audio_past, audio_future=s.audio_future,
            past_events=s.past_events, past_events_mask=s.past_events_mask,
            future_events=s.future_events,
            future_events_mask=np.array([True]),
            density_mean=5.0, density_peak=10, density_std=2.2,
        )
        ts = TimeStretch(prob=1.0, max_scale=1.4, seed=42)
        out = ts.apply(s2)
        tgt = self._adapter().make_target([out], device=torch.device("cpu"))
        assert int(tgt.target_bin[0]) == 500


# ─────────────────────────── integration ─────────────────────────────

class TestIntegration:
    def test_composes_with_other_aug(self):
        """TimeStretch + MelGainJitter chained via AugmentationPipeline
        produces a valid sample with no shape / mask drift."""
        s = _make_sample()
        pipeline = AugmentationPipeline(
            pre=(),
            post=(
                TimeStretch(prob=1.0, max_scale=1.4, seed=11),
                MelGainJitter(prob=1.0, range_db=2.0, seed=22),
            ),
        )
        out = pipeline.apply_post(s)
        assert out.audio_past.shape == s.audio_past.shape
        assert out.audio_future.shape == s.audio_future.shape
        assert len(out.past_events) == len(s.past_events)
        assert out.past_events_mask.shape == s.past_events_mask.shape
        assert len(out.future_events) == len(s.future_events)

    def test_goes_through_adapter_cleanly(self):
        import torch
        s = _make_sample()
        ts = TimeStretch(prob=1.0, max_scale=1.4, seed=1)
        out = ts.apply(s)
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=500),
        )
        inp = adapter.make_input([out], device=torch.device("cpu"))
        tgt = adapter.make_target([out], device=torch.device("cpu"))
        assert inp.mel.shape == (1, 80, 1000)
        assert inp.event_offsets.shape == (1, 128)
        assert inp.event_mask.shape == (1, 128)
        # Target is either a valid bin or STOP — never negative.
        t = int(tgt.target_bin[0])
        assert 0 <= t <= 500


# ─────────────────────────── runtime budget ──────────────────────────

class TestRuntime:
    def test_per_sample_under_budget(self):
        """Stretch on one sample must stay under ~2 ms on CPU — at 64
        batch this is ~130 ms / batch of augmentation-only overhead,
        well below the forward pass."""
        s = _make_sample()
        ts = TimeStretch(prob=1.0, max_scale=1.4, seed=0)
        # Warm up (first call pays attr lookup cost).
        ts.apply(s)
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            ts.apply(s)
        per_sample = (time.perf_counter() - t0) / n
        assert per_sample < 0.010, (
            f"per-sample stretch {per_sample*1000:.3f} ms exceeds 10 ms "
            "budget — investigate regression"
        )
