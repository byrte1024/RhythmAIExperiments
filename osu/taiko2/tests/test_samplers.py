"""Unit tests for audio + event sampler configs and implementations."""
import numpy as np
import pytest

from osu.taiko2.domain.beatmap import Onset, OnsetKind
from osu.taiko2.domain.dataset import (
    EventSamplerConfig,
    MelSamplerConfig,
)
from osu.taiko2.samplers import FixedRateEventSampler


class TestMelSamplerConfig:
    def test_default_hop_length(self):
        cfg = MelSamplerConfig()
        assert cfg.effective_hop_length == 110
        assert cfg.sample_rate == 22000

    def test_default_aligns_at_5ms(self):
        cfg = MelSamplerConfig()
        frame_ms = cfg.effective_hop_length / cfg.sample_rate * 1000.0
        assert frame_ms == pytest.approx(5.0)

    def test_hop_ms(self):
        cfg = MelSamplerConfig(hop_ms=5.0)
        assert cfg.effective_hop_length == 110  # round(5 * 22000 / 1000) = 110

    def test_hop_divisor(self):
        cfg = MelSamplerConfig(hop_divisor=200)
        assert cfg.effective_hop_length == 110  # 22000 // 200

    def test_explicit_hop_length(self):
        cfg = MelSamplerConfig(hop_length=256)
        assert cfg.effective_hop_length == 256

    def test_multiple_hop_specs_rejected(self):
        with pytest.raises(ValueError, match="at most one"):
            MelSamplerConfig(hop_length=110, hop_ms=5.0)


class TestEventSamplerConfig:
    def test_default_is_divisor_200(self):
        cfg = EventSamplerConfig()
        assert cfg.effective_bin_ms == pytest.approx(5.0)

    def test_bins_per_second(self):
        cfg = EventSamplerConfig(bins_per_second=100.0)
        assert cfg.effective_bin_ms == pytest.approx(10.0)

    def test_bin_ms(self):
        cfg = EventSamplerConfig(bin_ms=2.5)
        assert cfg.effective_bin_ms == pytest.approx(2.5)

    def test_divisor(self):
        cfg = EventSamplerConfig(divisor=400)
        assert cfg.effective_bin_ms == pytest.approx(2.5)

    def test_multiple_specs_rejected(self):
        with pytest.raises(ValueError, match="at most one"):
            EventSamplerConfig(bin_ms=5.0, divisor=200)


class TestFixedRateEventSampler:
    def test_bin_of_matches_config(self):
        s = FixedRateEventSampler(EventSamplerConfig(divisor=200))
        assert s.bin_ms == pytest.approx(5.0)
        assert s.bin_of(0) == 0
        assert s.bin_of(5) == 1
        assert s.bin_of(2_000) == 400
        # 4.99 ms is still bin 0 (floor)
        assert s.bin_of(4.99) == 0

    def test_sample_builds_binned_onsets(self):
        s = FixedRateEventSampler(EventSamplerConfig(divisor=200))
        onsets = (
            Onset(0, OnsetKind.DON),
            Onset(500, OnsetKind.KA),
            Onset(1500, OnsetKind.BIG_DON),
        )
        binned = s.sample(onsets)
        assert len(binned) == 3
        assert [b.bin for b in binned] == [0, 100, 300]
        assert [b.kind for b in binned] == [
            OnsetKind.DON, OnsetKind.KA, OnsetKind.BIG_DON,
        ]
        # Original ms preserved for re-binning under a different framerate
        assert [b.time_ms for b in binned] == [0, 500, 1500]

    def test_exact_alignment_with_mel_defaults(self):
        """At the repo's default (sr=22000, divisor=200), audio frames
        and event bins align at exactly 5 ms."""
        mel_cfg = MelSamplerConfig()
        evt_cfg = EventSamplerConfig()
        mel_frame_ms = mel_cfg.effective_hop_length / mel_cfg.sample_rate * 1000.0
        assert mel_frame_ms == pytest.approx(evt_cfg.effective_bin_ms)
