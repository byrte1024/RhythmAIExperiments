"""Round-trip tests for events / features / manifest persistence."""
from pathlib import Path

import numpy as np
import pytest

from osu.taiko2.domain.beatmap import OnsetBinned, OnsetKind
from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.persistence.events import (
    load_event_bins,
    load_events,
    save_events,
)
from osu.taiko2.persistence.features import load_features, save_features
from osu.taiko2.persistence.manifest import load_manifest, save_manifest


def test_events_round_trip(tmp_path: Path):
    onsets = (
        OnsetBinned(time_ms=0, kind=OnsetKind.DON, bin=0),
        OnsetBinned(time_ms=500, kind=OnsetKind.KA, bin=100),
        OnsetBinned(time_ms=1500, kind=OnsetKind.BIG_DON, bin=300),
    )
    path = tmp_path / "sample.npz"
    save_events(onsets, path)
    loaded = load_events(path)
    assert loaded == onsets


def test_events_empty(tmp_path: Path):
    path = tmp_path / "empty.npz"
    save_events(tuple(), path)
    assert load_events(path) == tuple()
    assert load_event_bins(path).size == 0


def test_load_event_bins_is_fast_path(tmp_path: Path):
    onsets = (
        OnsetBinned(0, OnsetKind.DON, 0),
        OnsetBinned(500, OnsetKind.KA, 100),
    )
    path = tmp_path / "ev.npz"
    save_events(onsets, path)
    bins = load_event_bins(path)
    assert bins.dtype == np.int32
    np.testing.assert_array_equal(bins, [0, 100])


def test_features_round_trip(tmp_path: Path):
    # Small values — float16 max is 65504, so avoid overflow warnings.
    features = (np.arange(80 * 1000, dtype=np.float32) / 10000.0).reshape(80, 1000)
    path = tmp_path / "f.npy"
    save_features(features, path)

    loaded = load_features(path)
    assert loaded.shape == (80, 1000)
    assert loaded.dtype == np.float16  # stored as float16
    # float16 precision loss is expected, but shape+ordering must match
    np.testing.assert_array_equal(loaded.shape, features.shape)


def test_features_wrong_shape_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="2D"):
        save_features(np.zeros(100, dtype=np.float32), tmp_path / "bad.npy")


def test_manifest_round_trip(tmp_path: Path):
    entry = ChartEntry(
        chart_id="pack [Oni]",
        beatmap_id="1",
        beatmapset_id="10",
        artist="A",
        title="T",
        difficulty_version="Oni",
        overall_difficulty=6.0,
        star_rating=4.25,
        density_mean=5.0,
        density_peak=10,
        density_std=2.0,
        duration_s=123.4,
        total_events=800,
        audio_filename="audio.mp3",
        features_path=Path("features/pack__audio.mp3.npy"),
        n_frames=25000,
    )
    manifest = DatasetManifest(
        name="test_v1",
        created_at="2026-04-22 12:00:00",
        sampler_config=MelSamplerConfig(),
        charts=(entry,),
    )
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)

    loaded = load_manifest(path)
    assert loaded.name == manifest.name
    assert loaded.created_at == manifest.created_at
    assert len(loaded.charts) == 1
    assert loaded.charts[0].chart_id == entry.chart_id
    assert loaded.charts[0].features_path == entry.features_path
    # Polymorphic sampler_config survived the round-trip
    assert isinstance(loaded.sampler_config, MelSamplerConfig)
    assert loaded.sampler_config.sample_rate == 22000


def test_manifest_missing_star_rating(tmp_path: Path):
    entry = ChartEntry(
        chart_id="c", beatmap_id="1", beatmapset_id="1",
        artist="", title="", difficulty_version="",
        overall_difficulty=0.0, star_rating=None,
        density_mean=0.0, density_peak=0, density_std=0.0,
        duration_s=0.0, total_events=0,
        audio_filename="", features_path=Path("x.npy"), n_frames=0,
    )
    manifest = DatasetManifest(
        name="x", created_at="", sampler_config=MelSamplerConfig(),
        charts=(entry,),
    )
    path = tmp_path / "m.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded.charts[0].star_rating is None
