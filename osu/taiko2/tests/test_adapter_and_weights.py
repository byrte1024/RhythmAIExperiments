"""Tests for `DetectionSampleAdapter` and the weighted-sampling
utilities on `TaikoDetectionSampler`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from osu.taiko2.data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from osu.taiko2.dataset import _safe_filename
from osu.taiko2.domain.beatmap import OnsetBinned, OnsetKind
from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.models import EventEmbeddingInput, EventEmbeddingTarget
from osu.taiko2.persistence.events import save_events
from osu.taiko2.persistence.features import save_features
from osu.taiko2.persistence.manifest import save_manifest
from osu.taiko2.training import DetectionSampleAdapter, DetectionSampleAdapterConfig


# ─────────────────────────── fixture ──────────────────────────────────

def _mk_entry(chart_id: str, bset: str, feat_rel: Path, n_frames: int,
              *, density_mean: float = 4.3, density_peak: int = 9,
              density_std: float = 1.5) -> ChartEntry:
    return ChartEntry(
        chart_id=chart_id, beatmap_id=chart_id, beatmapset_id=bset,
        artist="a", title="t",
        difficulty_version=chart_id.split("[")[-1].rstrip("]"),
        overall_difficulty=5.0, star_rating=None,
        density_mean=density_mean,
        density_peak=density_peak,
        density_std=density_std,
        duration_s=100.0, total_events=10,
        audio_filename=f"{bset}.mp3",
        features_path=feat_rel,
        n_frames=n_frames,
    )


@pytest.fixture
def small_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    feat_dir = ds / "features"
    evt_dir = ds / "events"
    feat_dir.mkdir(parents=True)
    evt_dir.mkdir(parents=True)

    songs = [("song1", "s1"), ("song2", "s2")]
    difficulties = ["Normal", "Oni"]
    entries = []
    n_frames = 10_000

    for song_name, bset in songs:
        feat_rel = Path("features") / f"{song_name}.npy"
        save_features(
            np.zeros((80, n_frames), dtype=np.float32), ds / feat_rel,
        )
        for diff in difficulties:
            chart_id = f"{song_name} [{diff}]"
            onsets = tuple(
                OnsetBinned(time_ms=int(b * 5), kind=OnsetKind.DON, bin=int(b))
                for b in range(1000, 2000, 100)   # 10 onsets @ 100 bins apart
            )
            save_events(
                onsets, evt_dir / f"{_safe_filename(chart_id)}.npz",
            )
            entries.append(_mk_entry(
                chart_id, bset, feat_rel, n_frames,
                density_mean=4.3 + ("Oni" in diff) * 2.0,
                density_peak=12 if "Oni" in diff else 8,
                density_std=1.5,
            ))

    save_manifest(
        DatasetManifest(
            name="small", created_at="t",
            sampler_config=MelSamplerConfig(),
            charts=tuple(entries),
        ),
        ds / "manifest.json",
    )
    return ds


def _build_sampler(
    ds_root: Path, *, d_events: int = 1, a_bins: int = 80, b_bins: int = 80,
) -> TaikoDetectionSampler:
    s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
        batch_size=4, dataset_root=ds_root,
        a_bins=a_bins, b_bins=b_bins, c_events=8, d_events=d_events,
        min_cursor_bin=0,
        allowed_overlap_forward=0, allowed_overlap_back=0,
    ))
    s.load_data()
    return s


# ─────────────────────────── density round-trip ───────────────────────

class TestDensityOnSample:
    def test_sample_carries_density(self, small_dataset: Path):
        s = _build_sampler(small_dataset)
        sample = s.raw_sample(0)
        # Density must match the chart's manifest entry values; stub
        # fixture uses 4.3/8 for Normal, 6.3/12 for Oni.
        assert sample.density_mean in (4.3, 6.3)
        assert sample.density_peak in (8, 12)
        assert sample.density_std == 1.5


# ─────────────────────────── adapter ──────────────────────────────────

class TestDetectionSampleAdapter:
    def test_make_input_shapes(self, small_dataset: Path):
        s = _build_sampler(small_dataset)
        batch = s.raw_batch(0)  # list of 4
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=80),
        )
        inp = adapter.make_input(batch, device=torch.device("cpu"))

        assert isinstance(inp, EventEmbeddingInput)
        assert inp.mel.shape == (4, 80, 160)   # a_bins + b_bins = 80 + 80
        assert inp.mel.dtype == torch.float32
        assert inp.event_offsets.shape == (4, 8)
        assert inp.event_offsets.dtype == torch.int64
        assert inp.event_mask.shape == (4, 8)
        assert inp.event_mask.dtype == torch.bool
        assert inp.conditioning.shape == (4, 3)
        assert inp.conditioning.dtype == torch.float32

    def test_make_target_in_window(self, small_dataset: Path):
        """Fixture has onsets 100 bins apart; cursor lands on one onset,
        next is 100 bins away. With b_pred=200 the next-onset target
        should land in range (not STOP)."""
        s = _build_sampler(small_dataset, a_bins=80, b_bins=400)
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=200),
        )
        # Find a sample whose next-onset offset lands strictly inside
        # [1, b_pred). The ei=0 pre-roll sample has offset 500 (out of
        # range for b_pred=200) so we skip those.
        for i in range(s.count_samples()):
            smp = s.raw_sample(i)
            if smp.future_events_mask[0]:
                continue
            off = smp.future_events[0].cursor_offset
            if off <= 0 or off >= 200:
                continue
            tgt = adapter.make_target([smp], device=torch.device("cpu"))
            assert tgt.target_bin.shape == (1,)
            # The fixture's onsets are 100 bins apart, so an in-range
            # offset is exactly 100.
            assert tgt.target_bin[0].item() == 100
            break
        else:
            pytest.fail("no sample with a valid next onset in the fixture")

    def test_make_target_stop_when_masked(self, small_dataset: Path):
        """Samples at the trailing ei=N have future_events_mask[0]=True
        → target must be STOP (class index b_pred)."""
        s = _build_sampler(small_dataset)
        b_pred = 80
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=b_pred),
        )
        # Find a sample whose future is fully padded.
        for i in range(s.count_samples()):
            smp = s.raw_sample(i)
            if smp.future_events_mask[0]:
                tgt = adapter.make_target([smp], device=torch.device("cpu"))
                assert tgt.target_bin.item() == b_pred
                return
        pytest.fail("no padded-future sample in fixture")

    def test_make_target_stop_when_out_of_range(self, small_dataset: Path):
        """Real future event at offset > b_pred collapses to STOP."""
        s = _build_sampler(small_dataset, a_bins=80, b_bins=200)
        # b_pred = 50: onsets 100 bins apart → out-of-range.
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=50),
        )
        for i in range(s.count_samples()):
            smp = s.raw_sample(i)
            if smp.future_events_mask[0]:
                continue
            if smp.future_events[0].cursor_offset < 50:
                continue
            tgt = adapter.make_target([smp], device=torch.device("cpu"))
            assert tgt.target_bin.item() == 50
            return
        pytest.fail("no out-of-range next-onset sample in fixture")

    def test_make_batch(self, small_dataset: Path):
        s = _build_sampler(small_dataset)
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=80),
        )
        inp, tgt = adapter.make_batch(s.raw_batch(0), device=torch.device("cpu"))
        assert isinstance(inp, EventEmbeddingInput)
        assert isinstance(tgt, EventEmbeddingTarget)

    def test_empty_batch_raises(self):
        adapter = DetectionSampleAdapter(
            DetectionSampleAdapterConfig(b_pred=80),
        )
        with pytest.raises(ValueError, match="empty batch"):
            adapter.make_input([], device=torch.device("cpu"))

    def test_bad_b_pred_rejected(self):
        with pytest.raises(ValueError, match="b_pred"):
            DetectionSampleAdapterConfig(b_pred=0)


# ─────────────────────────── weighted sampling ────────────────────────

class TestWeightedSampling:
    def test_target_bins_shape(self, small_dataset: Path):
        s = _build_sampler(small_dataset)
        targets = s.target_bins(b_pred=200)
        assert targets.shape == (s.count_samples(),)
        assert targets.dtype == np.int64
        assert targets.max() <= 200
        assert targets.min() >= 0

    def test_stop_samples_hit_stop_index(self, small_dataset: Path):
        """Fixture has 10 onsets per chart; trailing ei=10 samples
        produce the STOP target."""
        s = _build_sampler(small_dataset)
        b_pred = 200
        targets = s.target_bins(b_pred=b_pred)
        assert (targets == b_pred).any()

    def test_weights_shape_and_positive(self, small_dataset: Path):
        s = _build_sampler(small_dataset)
        w = s.compute_target_weights(b_pred=200)
        assert w.shape == (s.count_samples(),)
        assert np.all(w > 0)
        assert w.dtype == np.float64

    def test_rare_classes_weighted_higher(self):
        """Synthetic: if 99% of samples are class 0 and 1% are class 1,
        class 1's weight should be higher than class 0's."""
        # Build an artificial targets array by monkey-patching the
        # sampler's _samples + _event_bins; easier is just exercising
        # the weight formula directly via a minimal sampler subclass.
        class _StubSampler(TaikoDetectionSampler):
            def target_bins(self, *, b_pred):  # type: ignore[override]
                # 99 zeros, 1 one, 0 stops (b_pred=2 here)
                return np.array([0] * 99 + [1], dtype=np.int64)
            def count_samples(self): return 100
            @property
            def _require_loaded(self): return lambda: None

        s = _StubSampler.__new__(_StubSampler)
        w = TaikoDetectionSampler.compute_target_weights(
            s, b_pred=2, power=0.5,
        )
        assert w.shape == (100,)
        class0_weight = w[0]
        class1_weight = w[99]
        assert class1_weight > class0_weight

    def test_stop_boost_amplifies_stop_weights(self, small_dataset: Path):
        s = _build_sampler(small_dataset)
        b_pred = 200
        w_no_boost = s.compute_target_weights(b_pred=b_pred, stop_boost=1.0)
        w_boosted = s.compute_target_weights(b_pred=b_pred, stop_boost=10.0)
        targets = s.target_bins(b_pred=b_pred)
        stop_mask = targets == b_pred
        if not stop_mask.any():
            pytest.skip("no STOP samples in fixture")
        # STOP-sample weights should be larger under boost; non-STOP
        # weights should be unchanged.
        assert np.all(w_boosted[stop_mask] >= w_no_boost[stop_mask])
        assert np.all(w_boosted[stop_mask] > w_no_boost[stop_mask])
        np.testing.assert_allclose(
            w_boosted[~stop_mask], w_no_boost[~stop_mask],
        )

    def test_cap_limits_weights(self):
        s = _StubSamplerTargets([0, 0, 1])
        # With cap=0.5 no weight exceeds 0.5 regardless of how rare.
        w = TaikoDetectionSampler.compute_target_weights(
            s, b_pred=2, power=1.0, cap=0.5,
        )
        assert w.max() <= 0.5


class _StubSamplerTargets(TaikoDetectionSampler):
    """Dummy sampler that short-circuits `target_bins`. Avoids building
    a full on-disk fixture just to exercise the weight formula."""
    def __init__(self, targets: list[int]):
        self._targets_stub = np.asarray(targets, dtype=np.int64)
    def target_bins(self, *, b_pred):  # type: ignore[override]
        return self._targets_stub
