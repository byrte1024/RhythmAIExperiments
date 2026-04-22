"""Tests for `osu.taiko2.splits` — song-based train/val split."""
from pathlib import Path

import pytest

from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.splits import chart_ids_for_split, song_based_split


def _chart(chart_id: str, bset: str) -> ChartEntry:
    return ChartEntry(
        chart_id=chart_id, beatmap_id=chart_id, beatmapset_id=bset,
        artist="", title="", difficulty_version="",
        overall_difficulty=0.0, star_rating=None,
        density_mean=0.0, density_peak=0, density_std=0.0,
        duration_s=0.0, total_events=0,
        audio_filename="", features_path=Path("x.npy"), n_frames=0,
    )


def _manifest(charts):
    return DatasetManifest(
        name="t", created_at="", sampler_config=MelSamplerConfig(),
        charts=tuple(charts),
    )


class TestSongBasedSplit:
    def test_all_difficulties_of_same_song_stay_together(self):
        charts = [
            _chart("song1_easy", "song1"),
            _chart("song1_hard", "song1"),
            _chart("song2_easy", "song2"),
            _chart("song2_hard", "song2"),
            _chart("song3_easy", "song3"),
            _chart("song3_hard", "song3"),
        ]
        m = _manifest(charts)
        train, val = song_based_split(m, val_ratio=0.5, seed=0)

        train_songs = {charts[i].beatmapset_id for i in train}
        val_songs = {charts[i].beatmapset_id for i in val}
        # No song id appears in both splits
        assert train_songs.isdisjoint(val_songs)

    def test_deterministic_by_seed(self):
        charts = [_chart(f"c{i}", f"s{i}") for i in range(20)]
        m = _manifest(charts)
        t1, v1 = song_based_split(m, val_ratio=0.2, seed=42)
        t2, v2 = song_based_split(m, val_ratio=0.2, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds_differ(self):
        charts = [_chart(f"c{i}", f"s{i}") for i in range(20)]
        m = _manifest(charts)
        _, v1 = song_based_split(m, val_ratio=0.2, seed=1)
        _, v2 = song_based_split(m, val_ratio=0.2, seed=2)
        assert set(v1) != set(v2)

    def test_val_ratio_zero_yields_empty_val(self):
        charts = [_chart(f"c{i}", f"s{i}") for i in range(10)]
        m = _manifest(charts)
        train, val = song_based_split(m, val_ratio=0.0, seed=0)
        assert len(val) == 0
        assert len(train) == 10

    def test_rejects_bad_ratio(self):
        m = _manifest([_chart("c", "s")])
        with pytest.raises(ValueError):
            song_based_split(m, val_ratio=1.5, seed=0)


class TestChartIdsForSplit:
    def test_all_returns_everything(self):
        charts = [_chart(f"c{i}", f"s{i}") for i in range(5)]
        m = _manifest(charts)
        ids = chart_ids_for_split(m, "all", 0.1, 0)
        assert ids == {f"c{i}" for i in range(5)}

    def test_train_and_val_are_disjoint_and_cover_everything(self):
        charts = [_chart(f"c{i}", f"s{i}") for i in range(20)]
        m = _manifest(charts)
        train = chart_ids_for_split(m, "train", 0.2, 0)
        val = chart_ids_for_split(m, "val", 0.2, 0)
        assert train.isdisjoint(val)
        assert train | val == {c.chart_id for c in charts}

    def test_invalid_split_name(self):
        m = _manifest([_chart("c", "s")])
        with pytest.raises(ValueError, match="split must be"):
            chart_ids_for_split(m, "bogus", 0.1, 0)
