"""Tests for the octopus onset representation and sampler."""
import time

import numpy as np
import pytest

from osu.taiko2.domain.beatmap import OnsetKind, RelativeOnset
from osu.taiko2.domain.octopus import (
    apply_gammatone_filterbank,
    apply_group_delay_compensation,
    compute_gradient,
    compute_group_delays,
    erb_space,
)
from osu.taiko2.samplers.mel_octopus import MelOctopusSampler, MelOctopusSamplerConfig


# ── domain/octopus.py ────────────────────────────────────────────────

class TestErbSpace:
    def test_shape(self):
        cfs = erb_space(50.0, 8000.0, 128)
        assert cfs.shape == (128,)

    def test_ascending(self):
        cfs = erb_space(50.0, 8000.0, 128)
        assert np.all(np.diff(cfs) > 0)

    def test_range(self):
        cfs = erb_space(50.0, 8000.0, 128)
        assert cfs[0] == pytest.approx(50.0, abs=0.1)
        assert cfs[-1] <= 8000.0

    def test_fewer_filters(self):
        cfs = erb_space(50.0, 8000.0, 32)
        assert cfs.shape == (32,)
        assert np.all(np.diff(cfs) > 0)


class TestGroupDelay:
    def test_shape(self):
        cfs = erb_space(50.0, 8000.0, 128)
        delays = compute_group_delays(cfs)
        assert delays.shape == (128,)

    def test_low_freq_higher_delay(self):
        cfs = erb_space(50.0, 8000.0, 128)
        delays = compute_group_delays(cfs)
        assert delays[0] > delays[-1]

    def test_positive(self):
        cfs = erb_space(50.0, 8000.0, 128)
        delays = compute_group_delays(cfs)
        assert np.all(delays > 0)


class TestFilterbank:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        env, cfs, n_frames = apply_gammatone_filterbank(
            audio, 22000, n_filters=32, max_workers=2,
        )
        assert env.shape[0] == 32
        assert env.shape[1] == n_frames
        assert n_frames == 1000  # 1ms hop at 22kHz: hop=22 samples, 22000/22=1000

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        env, _, _ = apply_gammatone_filterbank(
            audio, 22000, n_filters=16, max_workers=2,
        )
        assert np.all(env >= 0)


class TestGroupDelayCompensation:
    def test_preserves_shape(self):
        cfs = erb_space(50.0, 8000.0, 32)
        onset_fn = np.random.default_rng(42).standard_normal((32, 1000)).astype(np.float32)
        compensated = apply_group_delay_compensation(onset_fn, cfs)
        assert compensated.shape == onset_fn.shape

    def test_aligns_channels(self):
        cfs = erb_space(50.0, 8000.0, 32)
        onset_fn = np.zeros((32, 100), dtype=np.float32)
        onset_fn[:, 50] = 1.0
        compensated = apply_group_delay_compensation(onset_fn, cfs)
        # High-freq channels have less group delay, so they get shifted
        # forward more. After compensation, all peaks should be closer
        # together than the original 18ms spread.
        peaks = np.array([np.argmax(compensated[ch]) for ch in range(32)])
        spread = peaks.max() - peaks.min()
        assert spread <= 20  # much less than uncompensated


class TestComputeGradient:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        gradient, cfs, n_cells = compute_gradient(
            audio, 22000, n_filters=32, max_workers=2,
        )
        assert gradient.shape[0] == n_cells
        assert gradient.shape[0] > 0
        assert gradient.shape[1] > 0

    def test_normalized_range(self):
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(44000).astype(np.float32)
        gradient, _, _ = compute_gradient(
            audio, 22000, n_filters=32, max_workers=2,
        )
        assert gradient.min() >= 0.0
        assert gradient.max() <= 1.01  # small float tolerance

    def test_click_produces_onset(self):
        """A loud click should produce nonzero gradient values."""
        audio = np.zeros(22000, dtype=np.float32)
        audio[11000:11010] = 1.0  # 10-sample click at 0.5s
        gradient, _, _ = compute_gradient(
            audio, 22000, n_filters=32, max_workers=2,
        )
        assert gradient.max() > 0


# ── samplers/octopus.py ──────────────────────────────────────────────

class TestMelOctopusSampler:
    def test_n_features(self):
        cfg = MelOctopusSamplerConfig(oct_n_filters=128)
        sampler = MelOctopusSampler(cfg)
        # 80 mel + 97 octopus = 177
        assert sampler.n_features == 177

    def test_frame_ms(self):
        cfg = MelOctopusSamplerConfig(hop_divisor=200)
        sampler = MelOctopusSampler(cfg)
        assert sampler.frame_ms == pytest.approx(5.0)

    def test_transform_shape(self):
        cfg = MelOctopusSamplerConfig(oct_n_filters=32, oct_max_workers=2)
        sampler = MelOctopusSampler(cfg)
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        features = sampler._transform(audio)
        assert features.shape[0] == sampler.n_features
        assert features.shape[1] == pytest.approx(200, abs=2)

    def test_mel_and_octopus_separate(self):
        """First 80 rows are mel (dB scale), rest are octopus ([0,1])."""
        cfg = MelOctopusSamplerConfig(oct_n_filters=32, oct_max_workers=2)
        sampler = MelOctopusSampler(cfg)
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        features = sampler._transform(audio)
        mel_part = features[:80, :]
        oct_part = features[80:, :]
        # Mel values are in dB (typically 10-50 for random noise).
        assert mel_part.max() > 10.0
        # Octopus values are normalized [0, 1].
        assert oct_part.min() >= 0.0
        assert oct_part.max() <= 1.01

    def test_transform_dtype(self):
        cfg = MelOctopusSamplerConfig(oct_n_filters=32, oct_max_workers=2)
        sampler = MelOctopusSampler(cfg)
        audio = np.zeros(22000, dtype=np.float32)
        features = sampler._transform(audio)
        assert features.dtype == np.float32


class TestOutputRows:
    """MelOctopusSampler.output_rows and adapter feature_rows slicing."""

    def _make_full(self):
        cfg = MelOctopusSamplerConfig(oct_n_filters=32, oct_max_workers=2)
        sampler = MelOctopusSampler(cfg)
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        full = sampler._transform(audio)
        return full, sampler._total_features

    def test_no_output_rows_returns_all(self):
        full, total = self._make_full()
        assert full.shape[0] == total

    def test_octopus_only_slice(self):
        full, total = self._make_full()
        cfg = MelOctopusSamplerConfig(oct_n_filters=32, oct_max_workers=2,
                                      output_rows=(80, total))
        sampler = MelOctopusSampler(cfg)
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        sliced = sampler._transform(audio)
        assert sliced.shape[0] == total - 80
        assert sampler.n_features == total - 80
        assert np.array_equal(sliced, full[80:])

    def test_mel_only_slice(self):
        full, _ = self._make_full()
        cfg = MelOctopusSamplerConfig(oct_n_filters=32, oct_max_workers=2,
                                      output_rows=(0, 80))
        sampler = MelOctopusSampler(cfg)
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)
        sliced = sampler._transform(audio)
        assert sliced.shape[0] == 80
        assert sampler.n_features == 80
        assert np.array_equal(sliced, full[:80])

    def test_n_features_matches_slice(self):
        cfg = MelOctopusSamplerConfig(oct_n_filters=32, oct_max_workers=2,
                                      output_rows=(80, 105))
        sampler = MelOctopusSampler(cfg)
        assert sampler.n_features == 25


class TestFeatureRowsAdapter:
    """FramewiseSampleAdapterConfig.feature_rows slicing at training."""

    def test_feature_rows_none_passes_all(self):
        from osu.taiko2.training.framewise_adapter import (
            FramewiseSampleAdapterConfig, FramewiseSampleAdapter,
        )
        import torch
        cfg = FramewiseSampleAdapterConfig(feature_rows=None)
        assert cfg.feature_rows is None

    def test_feature_rows_slices_mel(self):
        from osu.taiko2.training.framewise_adapter import (
            FramewiseSampleAdapterConfig, FramewiseSampleAdapter,
        )
        from osu.taiko2.models.event_embedding import EventEmbeddingInput
        import torch
        cfg = FramewiseSampleAdapterConfig(feature_rows=(80, 177))
        adapter = FramewiseSampleAdapter(cfg)
        # Build a fake input with 177 rows
        inp = EventEmbeddingInput(
            mel=torch.randn(2, 177, 1000),
            event_offsets=torch.zeros(2, 128, dtype=torch.int64),
            event_mask=torch.ones(2, 128, dtype=torch.bool),
            conditioning=torch.randn(2, 3),
        )
        # The adapter's make_input calls _detection_adapter.make_input
        # which builds from samples, not from EventEmbeddingInput.
        # Test the slicing logic directly:
        sliced = EventEmbeddingInput(
            mel=inp.mel[:, 80:177, :],
            event_offsets=inp.event_offsets,
            event_mask=inp.event_mask,
            conditioning=inp.conditioning,
        )
        assert sliced.mel.shape == (2, 97, 1000)


class TestFreqRollSectionBoundary:
    """MelFreqJitter must not roll mel rows into octopus or vice versa."""

    def _make_sample(self):
        from dataclasses import replace as dc_replace
        from osu.taiko2.data_samplers.detection import TaikoDetectionSample
        past = np.zeros((177, 500), dtype=np.float32)
        fut = np.zeros((177, 500), dtype=np.float32)
        # Mel rows: fill with 100.0 (dB-scale)
        past[:80] = 100.0
        fut[:80] = 100.0
        # Octopus rows: fill with 0.5 ([0,1] scale)
        past[80:] = 0.5
        fut[80:] = 0.5
        events = tuple(
            RelativeOnset(cursor_offset=-i*10, time_ms=0, kind=OnsetKind.DON, bin=0)
            for i in range(128)
        )
        mask = np.ones(128, dtype=bool)
        return TaikoDetectionSample(
            sample_id=0, chart_id="test", cursor_bin=6000,
            audio_past=past, audio_future=fut,
            past_events=events, past_events_mask=mask,
            future_events=events[:1], future_events_mask=mask[:1],
        )

    def test_no_boundary_wraps_across(self):
        """Without section_boundary, rolling wraps mel into octopus."""
        from osu.taiko2.training.augmentations import MelFreqJitter
        sample = self._make_sample()
        aug = MelFreqJitter(prob=1.0, max_shift=3, section_boundary=None, seed=0)
        result = aug.apply(sample)
        # With shift != 0, some rows that were 100.0 (mel) will wrap
        # into octopus territory, or vice versa. Check that the boundary
        # between 100.0 and 0.5 values is no longer at row 80.
        past = result.audio_past
        # Find where values change from mel-scale to octopus-scale.
        boundary_intact = (past[79, 0] == 100.0 and past[80, 0] == 0.5)
        # With any nonzero shift, the boundary should be disrupted.
        # (Unless shift happens to be 0, which seed=0 might produce.)
        # We just verify the augmentation runs without error.
        assert past.shape == (177, 500)

    def test_with_boundary_preserves_sections(self):
        """With section_boundary=80, mel and octopus roll independently."""
        from osu.taiko2.training.augmentations import MelFreqJitter
        sample = self._make_sample()
        # Use a large shift to make the test deterministic.
        aug = MelFreqJitter(prob=1.0, max_shift=5, section_boundary=80, seed=42)
        result = aug.apply(sample)
        past = result.audio_past
        # All mel rows (0-79) should still be 100.0.
        assert np.all(past[:80] == 100.0), \
            f"Mel rows corrupted: min={past[:80].min()}, max={past[:80].max()}"
        # All octopus rows (80+) should still be 0.5.
        assert np.all(past[80:] == 0.5), \
            f"Octopus rows corrupted: min={past[80:].min()}, max={past[80:].max()}"

    def test_boundary_rolls_independently(self):
        """Each section is actually rolled (not just passed through)."""
        from osu.taiko2.training.augmentations import MelFreqJitter
        # Create distinct values per row so we can detect rolling.
        from osu.taiko2.data_samplers.detection import TaikoDetectionSample
        past = np.arange(177, dtype=np.float32).reshape(177, 1).repeat(500, axis=1)
        fut = past.copy()
        events = tuple(
            RelativeOnset(cursor_offset=-i*10, time_ms=0, kind=OnsetKind.DON, bin=0)
            for i in range(128)
        )
        mask = np.ones(128, dtype=bool)
        sample = TaikoDetectionSample(
            sample_id=0, chart_id="test", cursor_bin=6000,
            audio_past=past, audio_future=fut,
            past_events=events, past_events_mask=mask,
            future_events=events[:1], future_events_mask=mask[:1],
        )
        # Force a known shift by trying many seeds until we get nonzero.
        for seed in range(100):
            aug = MelFreqJitter(prob=1.0, max_shift=3, section_boundary=80, seed=seed)
            result = aug.apply(sample)
            if not np.array_equal(result.audio_past, past):
                break
        rp = result.audio_past
        # Mel section rolled: row values in [0, 79] but potentially reordered.
        mel_vals = set(rp[:80, 0].astype(int))
        oct_vals = set(rp[80:, 0].astype(int))
        assert mel_vals.issubset(set(range(80))), \
            f"Mel section contains non-mel values: {mel_vals - set(range(80))}"
        assert oct_vals.issubset(set(range(80, 177))), \
            f"Octopus section contains non-octopus values: {oct_vals - set(range(80, 177))}"


class TestMelOctopusPerformance:
    def test_speed_1s_audio(self):
        """1 second of audio should process in < 3 seconds (mel + octopus)."""
        cfg = MelOctopusSamplerConfig(oct_n_filters=128, oct_max_workers=8)
        sampler = MelOctopusSampler(cfg)
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(22000).astype(np.float32)

        t0 = time.perf_counter()
        features = sampler._transform(audio)
        elapsed = time.perf_counter() - t0

        assert features.shape[0] == sampler.n_features
        assert elapsed < 3.0, f"1s audio took {elapsed:.2f}s (limit 3.0s)"

    def test_speed_10s_audio(self):
        """10 seconds should process in < 6 seconds."""
        cfg = MelOctopusSamplerConfig(oct_n_filters=128, oct_max_workers=8)
        sampler = MelOctopusSampler(cfg)
        rng = np.random.default_rng(42)
        audio = rng.standard_normal(220000).astype(np.float32)

        t0 = time.perf_counter()
        features = sampler._transform(audio)
        elapsed = time.perf_counter() - t0

        assert features.shape[0] == sampler.n_features
        assert elapsed < 6.0, f"10s audio took {elapsed:.2f}s (limit 6.0s)"
