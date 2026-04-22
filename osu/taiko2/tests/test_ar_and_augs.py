"""Tests for the AR concretes (ArgmaxDecoder, DetectionARInputBuilder)
and the exp 45 augmentation bundle."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from osu.taiko2.data_samplers.detection import TaikoDetectionSample
from osu.taiko2.domain.beatmap import OnsetBinned, OnsetKind, RelativeOnset
from osu.taiko2.domain.inference import Conditioning
from osu.taiko2.inference.autoregressive import (
    ARContext,
    ArgmaxDecoder,
    ArgmaxDecoderConfig,
    DetectionARBuilderConfig,
    DetectionARInputBuilder,
)
from osu.taiko2.models import EventEmbeddingOutput
from osu.taiko2.training.augmentations import (
    ConditioningJitter,
    ContextTruncation,
    EventDropout,
    EventInsertion,
    EventJitter,
    LargeTimeShift,
    MelFreqJitter,
    MelGainJitter,
    MelGaussianNoise,
    PartialAdvMetronome,
    PartialMetronome,
    SpecAugFreq,
    SpecAugTime,
    build_exp45_post_augs,
)


# ─────────────────────────── fixtures ─────────────────────────────────

def _make_sample(
    *,
    n_features: int = 16,
    a_bins: int = 40,
    b_bins: int = 40,
    c_events: int = 16,
    d_events: int = 2,
    n_past_real: int = 8,
) -> TaikoDetectionSample:
    """A plausible TaikoDetectionSample for augmentation + builder tests."""
    audio_past = np.zeros((n_features, a_bins), dtype=np.float32)
    audio_future = np.zeros((n_features, b_bins), dtype=np.float32)

    # Pad at start; most recent real event at slot c_events-1 with offset 0.
    pad = c_events - n_past_real
    past_mask = np.ones(c_events, dtype=bool)
    past_mask[pad:] = False
    past_events = tuple(
        RelativeOnset(
            time_ms=0, kind=OnsetKind.DON,
            bin=1000 + i * 100, cursor_offset=-100 * (n_past_real - 1 - i),
        )
        if i >= pad else
        RelativeOnset(time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=0)
        for i in range(c_events)
    )

    future_mask = np.ones(d_events, dtype=bool)
    future_mask[0] = False
    future_events = tuple(
        RelativeOnset(
            time_ms=0, kind=OnsetKind.DON,
            bin=1000 + (n_past_real) * 100, cursor_offset=100,
        )
        if i == 0 else
        RelativeOnset(time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=0)
        for i in range(d_events)
    )

    return TaikoDetectionSample(
        sample_id=0,
        chart_id="test",
        cursor_bin=1000 + (n_past_real - 1) * 100,
        audio_past=audio_past,
        audio_future=audio_future,
        past_events=past_events,
        past_events_mask=past_mask,
        future_events=future_events,
        future_events_mask=future_mask,
        density_mean=4.0,
        density_peak=8,
        density_std=1.0,
    )


# ─────────────────────────── ArgmaxDecoder ────────────────────────────

class TestArgmaxDecoder:
    def _ctx(self) -> ARContext:
        return ARContext(cursor_bin=100, step=0, max_bin=10_000,
                         past_onsets=())

    def test_simple_argmax(self):
        dec = ArgmaxDecoder(ArgmaxDecoderConfig(b_pred=10))
        logits = torch.full((1, 11), -10.0)
        logits[0, 3] = 10.0
        out = EventEmbeddingOutput(logits=logits)
        decision = dec.decode(out, self._ctx())
        assert decision.bin_offsets == (3,)
        assert decision.confidences[0] > 0.9
        assert decision.is_stop is False

    def test_stop_class_emits_empty(self):
        dec = ArgmaxDecoder(ArgmaxDecoderConfig(b_pred=10))
        logits = torch.full((1, 11), -10.0)
        logits[0, 10] = 10.0          # STOP class
        decision = dec.decode(EventEmbeddingOutput(logits=logits), self._ctx())
        assert decision.bin_offsets == ()
        assert decision.is_stop is True

    def test_extras_include_topk_and_entropy(self):
        dec = ArgmaxDecoder(ArgmaxDecoderConfig(b_pred=10, top_k_log=3))
        logits = torch.randn(1, 11)
        decision = dec.decode(EventEmbeddingOutput(logits=logits), self._ctx())
        assert "entropy" in decision.extras
        for i in (1, 2, 3):
            assert f"top{i}_bin" in decision.extras
            assert f"top{i}_prob" in decision.extras

    def test_bad_batch_size_rejected(self):
        dec = ArgmaxDecoder(ArgmaxDecoderConfig(b_pred=10))
        # AR expects (1, n_classes); a batch > 1 is a bug.
        with pytest.raises(ValueError, match="1, n_classes"):
            dec.decode(
                EventEmbeddingOutput(logits=torch.randn(2, 11)),
                self._ctx(),
            )


# ─────────────────────────── DetectionARInputBuilder ──────────────────

class TestDetectionARInputBuilder:
    def test_build_shapes_and_padding(self):
        builder = DetectionARInputBuilder(
            DetectionARBuilderConfig(a_bins=40, b_bins=40, c_events=8),
        )
        # Full-chart feature array — mostly zeros, to test the zero-
        # padding behavior at cursor edges.
        features = np.random.randn(16, 200).astype(np.float32)
        inp = builder.build(
            cursor_bin=100,
            past_onsets=tuple(
                OnsetBinned(time_ms=0, kind=OnsetKind.DON, bin=100 - k * 20)
                for k in range(3)
            ),
            audio_features=features,
            conditioning=Conditioning(4.0, 8, 1.0),
            device=torch.device("cpu"),
        )
        assert inp.mel.shape == (1, 16, 80)       # a + b = 80
        assert inp.event_offsets.shape == (1, 8)
        assert inp.event_mask.shape == (1, 8)
        assert inp.conditioning.shape == (1, 3)

    def test_past_onsets_fill_recent_slots(self):
        builder = DetectionARInputBuilder(
            DetectionARBuilderConfig(a_bins=20, b_bins=20, c_events=4),
        )
        features = np.zeros((4, 100), dtype=np.float32)
        # Two real past onsets; 4 total slots → 2 pad slots at the start.
        past = (
            OnsetBinned(0, OnsetKind.DON, 80),
            OnsetBinned(0, OnsetKind.DON, 95),
        )
        inp = builder.build(
            cursor_bin=100,
            past_onsets=past,
            audio_features=features,
            conditioning=Conditioning(1.0, 2, 0.5),
            device=torch.device("cpu"),
        )
        mask = inp.event_mask[0]
        # First two slots padded, last two real.
        assert bool(mask[0]) and bool(mask[1])
        assert not bool(mask[2]) and not bool(mask[3])
        # Offsets are cursor-relative and non-positive.
        offsets = inp.event_offsets[0]
        assert int(offsets[2]) == 80 - 100
        assert int(offsets[3]) == 95 - 100

    def test_edge_cursor_zero_pads(self):
        """Cursor at bin 0 → past window is all zero-padding on the left."""
        builder = DetectionARInputBuilder(
            DetectionARBuilderConfig(a_bins=20, b_bins=20, c_events=4),
        )
        features = np.ones((4, 100), dtype=np.float32)
        inp = builder.build(
            cursor_bin=0,
            past_onsets=(),
            audio_features=features,
            conditioning=Conditioning(1.0, 2, 0.5),
            device=torch.device("cpu"),
        )
        # Left 20 frames are zeros, right 20 are ones.
        mel = inp.mel[0].numpy()
        assert (mel[:, :20] == 0).all()
        assert (mel[:, 20:] == 1).all()

    def test_requires_conditioning(self):
        builder = DetectionARInputBuilder(DetectionARBuilderConfig())
        with pytest.raises(ValueError, match="conditioning"):
            builder.build(
                cursor_bin=0, past_onsets=(),
                audio_features=np.zeros((16, 100), dtype=np.float32),
                conditioning=None,
                device=torch.device("cpu"),
            )


# ─────────────────────────── augmentations ────────────────────────────

class TestAudioAugs:
    def test_mel_gain_jitter_shifts(self):
        aug = MelGainJitter(prob=1.0, range_db=2.0, seed=0)
        sample = _make_sample()
        out = aug.apply(sample)
        # Every value shifted by the same constant.
        delta = (out.audio_past - sample.audio_past).flatten()
        assert np.allclose(delta, delta[0])
        assert abs(delta[0]) <= 2.0

    def test_mel_gaussian_noise_nonzero(self):
        aug = MelGaussianNoise(prob=1.0, min_std=0.5, max_std=0.5, seed=0)
        sample = _make_sample()
        out = aug.apply(sample)
        assert not np.allclose(out.audio_past, sample.audio_past)

    def test_mel_freq_jitter(self):
        aug = MelFreqJitter(prob=1.0, max_shift=3, seed=0)
        s = _make_sample(n_features=8)
        # Fill past audio with rising band indices so a roll is visible.
        s = replace(
            s,
            audio_past=np.arange(8, dtype=np.float32).reshape(8, 1).repeat(40, axis=1),
        )
        out = aug.apply(s)
        # At least one band reordered.
        assert not np.array_equal(out.audio_past, s.audio_past)

    def test_specaug_freq_zeroes_band(self):
        aug = SpecAugFreq(prob=1.0, max_bands=3, seed=0)
        sample = replace(
            _make_sample(),
            audio_past=np.ones((16, 40), dtype=np.float32),
            audio_future=np.ones((16, 40), dtype=np.float32),
        )
        out = aug.apply(sample)
        # At least one full row of zeros.
        row_sums = out.audio_past.sum(axis=1)
        assert (row_sums == 0).any()

    def test_specaug_time_zeroes_column(self):
        aug = SpecAugTime(prob=1.0, max_frames=5, seed=0)
        sample = replace(
            _make_sample(),
            audio_past=np.ones((16, 40), dtype=np.float32),
            audio_future=np.ones((16, 40), dtype=np.float32),
        )
        out = aug.apply(sample)
        # Either past or future has a zeroed column.
        col_sums_past = out.audio_past.sum(axis=0)
        col_sums_fut = out.audio_future.sum(axis=0)
        assert (col_sums_past == 0).any() or (col_sums_fut == 0).any()

    def test_prob_zero_is_noop(self):
        """prob=0 → never triggers; sample passes through unchanged."""
        aug = MelGainJitter(prob=0.0, seed=0)
        s = _make_sample()
        assert aug.apply(s) is s


class TestEventAugs:
    def test_event_jitter_preserves_count(self):
        aug = EventJitter(prob=1.0, seed=0)
        s = _make_sample(n_past_real=6, c_events=16)
        out = aug.apply(s)
        assert int((~out.past_events_mask).sum()) == 6

    def test_event_dropout_reduces_count(self):
        aug = EventDropout(prob=1.0, drop_min=2, drop_max=2, seed=0)
        s = _make_sample(n_past_real=6, c_events=16)
        out = aug.apply(s)
        assert int((~out.past_events_mask).sum()) == 4

    def test_event_insertion_grows_count(self):
        aug = EventInsertion(prob=1.0, seed=0)
        s = _make_sample(n_past_real=4, c_events=16)
        out = aug.apply(s)
        # Up to 5 now; if no gap > 1 bin it may be skipped, but our
        # fixture has 100-bin spacing so there's plenty of room.
        assert int((~out.past_events_mask).sum()) == 5

    def test_context_truncation_caps_count(self):
        aug = ContextTruncation(prob=1.0, keep_min=2, keep_max=2, seed=0)
        s = _make_sample(n_past_real=10, c_events=16)
        out = aug.apply(s)
        assert int((~out.past_events_mask).sum()) == 2

    def test_large_time_shift_moves_recent(self):
        aug = LargeTimeShift(prob=1.0, max_shift=50, n_min=2, n_max=2, seed=0)
        s = _make_sample(n_past_real=8, c_events=16)
        out = aug.apply(s)
        # Count preserved; positions changed.
        assert int((~out.past_events_mask).sum()) == 8
        # Most recent real events should differ from original.
        old_recent = [o.cursor_offset for o in s.past_events[-2:]]
        new_recent = [o.cursor_offset for o in out.past_events[-2:]]
        assert new_recent != old_recent

    def test_partial_metronome_keeps_count(self):
        aug = PartialMetronome(prob=1.0, gap_min=20, gap_max=20, seed=0)
        s = _make_sample(n_past_real=10, c_events=16)
        out = aug.apply(s)
        assert int((~out.past_events_mask).sum()) == 10

    def test_partial_adv_metronome_keeps_count(self):
        aug = PartialAdvMetronome(prob=1.0, seed=0)
        s = _make_sample(n_past_real=10, c_events=16)
        out = aug.apply(s)
        assert int((~out.past_events_mask).sum()) == 10


class TestConditioningJitter:
    def test_small_multiplier(self):
        aug = ConditioningJitter(prob=1.0, pct=0.02, seed=0)
        s = _make_sample()
        out = aug.apply(s)
        assert abs(out.density_mean - s.density_mean) / s.density_mean <= 0.02 + 1e-6
        # Peak is an int — can differ by at most round(peak*0.02).
        assert abs(out.density_peak - s.density_peak) <= 1


class TestExp45Bundle:
    def test_bundle_length(self):
        augs = build_exp45_post_augs(seed=0)
        assert len(augs) == 13

    def test_bundle_runs_end_to_end(self):
        from osu.taiko2.domain.augmentation import AugmentationPipeline
        augs = build_exp45_post_augs(seed=0)
        pipe = AugmentationPipeline(post=tuple(augs))
        s = _make_sample(n_past_real=8, c_events=16)
        # Running through every aug in the pipe shouldn't crash on our
        # fixture.
        out = pipe.apply_post(s)
        assert isinstance(out, TaikoDetectionSample)
        # Audio shape preserved; event array lengths preserved.
        assert out.audio_past.shape == s.audio_past.shape
        assert out.audio_future.shape == s.audio_future.shape
        assert len(out.past_events) == len(s.past_events)
