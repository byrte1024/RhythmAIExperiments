"""Tests for `osu.taiko2.splits` — N-way song-based splitting."""
from pathlib import Path

import pytest

from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.splits import (
    chart_ids_for_split,
    named_song_splits,
    song_based_split,
)


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


def _multi_song_manifest(n_songs: int, diffs_per_song: int = 2) -> DatasetManifest:
    charts = []
    for s in range(n_songs):
        for d in range(diffs_per_song):
            charts.append(_chart(f"song{s}_diff{d}", f"s{s}"))
    return _manifest(charts)


# ─────────────────────────── multi-split ──────────────────────────────

class TestNamedSongSplits:
    def test_three_way_split(self):
        m = _multi_song_manifest(100)
        spec = (("train", 0.7), ("val", 0.15), ("test", 0.15))
        buckets = named_song_splits(m, spec, seed=0)

        assert set(buckets.keys()) == {"train", "val", "test"}
        # Every chart appears in exactly one bucket
        all_indices = set(buckets["train"]) | set(buckets["val"]) | set(buckets["test"])
        assert len(all_indices) == len(m.charts)
        assert len(all_indices) == sum(len(v) for v in buckets.values())

    def test_five_way_split(self):
        m = _multi_song_manifest(50, diffs_per_song=3)
        spec = tuple((f"fold{i}", 0.2) for i in range(5))
        buckets = named_song_splits(m, spec, seed=1)
        assert set(buckets.keys()) == {f"fold{i}" for i in range(5)}
        assert sum(len(v) for v in buckets.values()) == len(m.charts)

    def test_partial_sum_leaves_songs_unassigned(self):
        m = _multi_song_manifest(100)
        spec = (("train", 0.5), ("val", 0.1))   # sum = 0.6
        buckets = named_song_splits(m, spec, seed=0)
        placed = sum(len(v) for v in buckets.values())
        assert placed < len(m.charts)  # 40% of songs not in any bucket

    def test_song_level_isolation(self):
        m = _multi_song_manifest(60, diffs_per_song=3)
        spec = (("train", 0.6), ("val", 0.2), ("test", 0.2))
        buckets = named_song_splits(m, spec, seed=42)

        def _songs(idx: tuple[int, ...]) -> set[str]:
            return {m.charts[i].beatmapset_id for i in idx}

        # No song id appears in two buckets
        train, val, test = map(_songs, (buckets["train"], buckets["val"], buckets["test"]))
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_all_difficulties_of_song_share_bucket(self):
        m = _multi_song_manifest(20, diffs_per_song=4)
        spec = (("train", 0.75), ("val", 0.25))
        buckets = named_song_splits(m, spec, seed=7)
        # For each bucket, charts grouped by song should all belong to it
        for name, idx in buckets.items():
            songs_in_bucket = {m.charts[i].beatmapset_id for i in idx}
            for song in songs_in_bucket:
                expected = {
                    i for i, c in enumerate(m.charts)
                    if c.beatmapset_id == song
                }
                assert expected.issubset(set(idx))

    def test_deterministic_by_seed(self):
        m = _multi_song_manifest(30)
        spec = (("train", 0.8), ("val", 0.2))
        a = named_song_splits(m, spec, seed=99)
        b = named_song_splits(m, spec, seed=99)
        assert a == b

    def test_different_seeds_differ(self):
        m = _multi_song_manifest(30)
        spec = (("train", 0.8), ("val", 0.2))
        a = named_song_splits(m, spec, seed=1)
        b = named_song_splits(m, spec, seed=2)
        assert set(a["val"]) != set(b["val"])


class TestSpecValidation:
    def test_duplicate_names_rejected(self):
        m = _multi_song_manifest(10)
        with pytest.raises(ValueError, match="duplicate"):
            named_song_splits(m, (("a", 0.5), ("a", 0.5)), seed=0)

    def test_reserved_name_rejected(self):
        m = _multi_song_manifest(10)
        with pytest.raises(ValueError, match="reserved"):
            named_song_splits(m, (("all", 0.5), ("train", 0.5)), seed=0)

    def test_sum_over_one_rejected(self):
        m = _multi_song_manifest(10)
        with pytest.raises(ValueError, match="sum to"):
            named_song_splits(m, (("a", 0.6), ("b", 0.6)), seed=0)

    def test_negative_ratio_rejected(self):
        m = _multi_song_manifest(10)
        with pytest.raises(ValueError, match="negative"):
            named_song_splits(m, (("a", -0.1), ("b", 0.5)), seed=0)

    def test_empty_name_rejected(self):
        m = _multi_song_manifest(10)
        with pytest.raises(ValueError, match="name"):
            named_song_splits(m, (("", 0.5),), seed=0)

    def test_empty_spec_rejected(self):
        m = _multi_song_manifest(10)
        with pytest.raises(ValueError, match="empty"):
            named_song_splits(m, (), seed=0)


# ─────────────────────────── chart_ids_for_split ──────────────────────

class TestChartIdsForSplit:
    SPEC = (("train", 0.6), ("val", 0.2), ("test", 0.2))

    def test_all_returns_everything(self):
        m = _multi_song_manifest(10)
        ids = chart_ids_for_split(m, "all", self.SPEC, seed=0)
        assert ids == {c.chart_id for c in m.charts}

    def test_named_split_returns_bucket(self):
        m = _multi_song_manifest(20)
        train = chart_ids_for_split(m, "train", self.SPEC, seed=0)
        val = chart_ids_for_split(m, "val", self.SPEC, seed=0)
        test = chart_ids_for_split(m, "test", self.SPEC, seed=0)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == {c.chart_id for c in m.charts}

    def test_unknown_split_rejected(self):
        m = _multi_song_manifest(5)
        with pytest.raises(ValueError, match="unknown split"):
            chart_ids_for_split(m, "bogus", self.SPEC, seed=0)


# ─────────────────────────── legacy wrapper ───────────────────────────

class TestSongBasedSplit:
    def test_wraps_two_way(self):
        m = _multi_song_manifest(20)
        train, val = song_based_split(m, val_ratio=0.2, seed=0)
        assert set(train).isdisjoint(set(val))
        assert len(train) + len(val) == len(m.charts)

    def test_val_ratio_zero(self):
        m = _multi_song_manifest(10)
        train, val = song_based_split(m, val_ratio=0.0, seed=0)
        assert len(val) == 0
        assert len(train) == len(m.charts)

    def test_rejects_bad_ratio(self):
        m = _multi_song_manifest(5)
        with pytest.raises(ValueError):
            song_based_split(m, val_ratio=1.5, seed=0)
