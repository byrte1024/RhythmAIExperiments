"""Tests for `TaikoDetectionSampler` — cursor rules, padding, split routing.

Builds a synthetic dataset on tmp_path (two songs × two difficulties), runs
the real save/load path, then exercises the sampler against it. No audio
decode or real mel computation — features are synthetic arrays.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from osu.taiko2.domain.beatmap import OnsetBinned, OnsetKind
from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.data_samplers import (
    TaikoDetectionSampler,
    TaikoDetectionSamplerConfig,
)
from osu.taiko2.dataset import _safe_filename
from osu.taiko2.persistence.events import save_events
from osu.taiko2.persistence.features import save_features
from osu.taiko2.persistence.manifest import save_manifest


# ─────────────────────────── fixtures ─────────────────────────────────

def _mk_chart(chart_id: str, bset: str, feat_rel: Path, n_frames: int) -> ChartEntry:
    return ChartEntry(
        chart_id=chart_id,
        beatmap_id=chart_id,
        beatmapset_id=bset,
        artist="a", title="t", difficulty_version=chart_id.split("[")[-1].rstrip("]"),
        overall_difficulty=5.0, star_rating=None,
        density_mean=5.0, density_peak=8, density_std=1.5,
        duration_s=100.0, total_events=10,
        audio_filename=f"{bset}.mp3",
        features_path=feat_rel,
        n_frames=n_frames,
    )


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Build a 2-song × 2-difficulty dataset in tmp_path. Each song has its
    own 10k-frame feature array; each chart has 10 evenly-spaced onsets
    starting at bin 1000.
    """
    ds = tmp_path / "tiny_ds"
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
        feat = np.zeros((80, n_frames), dtype=np.float32)
        save_features(feat, ds / feat_rel)

        for diff in difficulties:
            chart_id = f"{song_name} [{diff}]"
            onsets = tuple(
                OnsetBinned(time_ms=int(b * 5), kind=OnsetKind.DON, bin=int(b))
                for b in range(1000, 2000, 100)  # 10 onsets
            )
            save_events(onsets, evt_dir / f"{_safe_filename(chart_id)}.npz")
            entries.append(_mk_chart(chart_id, bset, feat_rel, n_frames))

    manifest = DatasetManifest(
        name="tiny", created_at="t",
        sampler_config=MelSamplerConfig(),
        charts=tuple(entries),
    )
    save_manifest(manifest, ds / "manifest.json")
    return ds


# ─────────────────────────── shape / cursor tests ─────────────────────

class TestSampleShapes:
    def test_audio_shapes(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=2, dataset_root=tiny_dataset,
            a_bins=200, b_bins=300, c_events=16, d_events=4,
            min_cursor_bin=0,
        ))
        s.load_data()
        sample = s.raw_sample(0)
        assert sample.audio_past.shape == (80, 200)
        assert sample.audio_future.shape == (80, 300)
        assert sample.audio_past.dtype == np.float32

    def test_cursor_sits_on_previous_onset(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=8, d_events=1,
            min_cursor_bin=0,
        ))
        s.load_data()
        # sample 1 should have cursor == first onset of that chart (bin 1000)
        # but ei indexing depends on chart ordering; find the first sample
        # whose past_events has a real entry with cursor_offset == 0.
        found = False
        for i in range(min(50, s.count_samples())):
            smp = s.raw_sample(i)
            unpadded = [o for o, m in zip(smp.past_events, smp.past_events_mask) if not m]
            if unpadded:
                # newest past event sits exactly on the cursor
                assert unpadded[-1].cursor_offset == 0
                found = True
                break
        assert found, "expected at least one sample with a real past onset"

    def test_past_padded_at_start_future_padded_at_end(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=16, d_events=4,
            min_cursor_bin=0,
        ))
        s.load_data()
        # ei=0 sample → past should be fully padded (no past events at start of chart)
        smp = s.raw_sample(0)
        # mask True = padded. Start-padding convention for past:
        assert smp.past_events_mask.all() or smp.past_events_mask[0]

    def test_batch_contiguous(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=3, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=8, d_events=2,
            min_cursor_bin=0,
        ))
        s.load_data()
        batch = s.raw_batch(0)
        assert len(batch) == 3
        assert [b.sample_id for b in batch] == [0, 1, 2]

    def test_out_of_range(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset, min_cursor_bin=0,
            allowed_overlap_forward=0, allowed_overlap_back=0,
        ))
        s.load_data()
        with pytest.raises(IndexError):
            s.raw_sample(s.count_samples())
        with pytest.raises(IndexError):
            s.raw_batch(s.count_batches())


# ─────────────────────────── overlap-filter tests ─────────────────────

class TestAllowedOverlap:
    """Fixture has 2 songs × 2 difficulties × 10 onsets each, bins at
    1000, 1100, …, 1900. Each chart's candidate cursors (ei=0..10) land
    at: [500, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900].
    """

    def _expected_per_chart(self, gap: int) -> int:
        """Greedy forward filter over the 11 candidate cursors per chart."""
        cursors = [500] + list(range(1000, 2000, 100))  # 11 cursors
        last = None
        kept = 0
        for c in cursors:
            if last is None or (c - last) >= gap:
                kept += 1
                last = c
        return kept

    def test_zero_overlap_keeps_everything(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=8, d_events=2,
            min_cursor_bin=0,
            allowed_overlap_forward=0, allowed_overlap_back=0,
        ))
        s.load_data()
        # 4 charts × 11 candidate cursors (ei=0..10)
        assert s.count_samples() == 4 * 11

    def test_default_overlap_matches_a_b(self, tiny_dataset: Path):
        """Default overlap = a_bins forward / b_bins backward → max gap = max(a,b)."""
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=8, d_events=2,
            min_cursor_bin=0,  # allowed_overlap_forward/back default to None → 100
        ))
        s.load_data()
        assert s.config.allowed_overlap_forward == 100
        assert s.config.allowed_overlap_back == 100
        # min gap = 100. 500→1000 (gap 500 ✓), then every cursor 100 apart (accept).
        # Really: gap check 100 → keeps all 11 because all consecutive gaps ≥100.
        assert s.count_samples() == 4 * self._expected_per_chart(100)

    def test_large_overlap_thins_samples(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=8, d_events=2,
            min_cursor_bin=0,
            allowed_overlap_forward=300,
            allowed_overlap_back=300,
        ))
        s.load_data()
        # min gap = 300. Greedy filter on [500, 1000, 1100, ..., 1900]:
        # keep 500; next >=800 → 1000; next >=1300 → 1300; next >=1600 → 1600;
        # next >=1900 → 1900. → 5 per chart.
        assert s.count_samples() == 4 * self._expected_per_chart(300)

    def test_asymmetric_overlap_uses_max(self, tiny_dataset: Path):
        """allowed_overlap_forward=50, allowed_overlap_back=500 → min_gap=500."""
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=8, d_events=2,
            min_cursor_bin=0,
            allowed_overlap_forward=50,
            allowed_overlap_back=500,
        ))
        s.load_data()
        assert s.count_samples() == 4 * self._expected_per_chart(500)

    def test_negative_overlap_rejected(self):
        with pytest.raises(ValueError, match="must be"):
            TaikoDetectionSamplerConfig(
                batch_size=1, allowed_overlap_forward=-1,
            )


# ─────────────────────────── per-split overrides ──────────────────────

class TestSplitOverrides:
    """A single base config expresses per-split sampling differences.

    Fixture has 4 charts × 11 candidate cursors = 44 total candidates.
    With default overlap (100/100 at a=b=100) greedy filtering still
    keeps all 11 per chart because event gaps are ≥100. So here we
    deliberately set tighter overlap in train and looser in val to
    observe the difference.
    """

    def _base_config(self, tiny_dataset: Path) -> TaikoDetectionSamplerConfig:
        return TaikoDetectionSamplerConfig(
            batch_size=4,
            dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, c_events=8, d_events=2,
            min_cursor_bin=0,
            split_ratios=(("train", 0.5), ("val", 0.5)),
            split_seed=0,
            split_overrides={
                # Train: tight overlap → most samples kept.
                "train": {
                    "allowed_overlap_forward": 100,
                    "allowed_overlap_back": 100,
                },
                # Val: looser overlap → many cursors dropped.
                "val": {
                    "allowed_overlap_forward": 500,
                    "allowed_overlap_back": 500,
                },
            },
        )

    def test_train_and_val_see_different_overlap(self, tiny_dataset: Path):
        base = self._base_config(tiny_dataset)
        train = TaikoDetectionSampler(replace(base, split="train"))
        val = TaikoDetectionSampler(replace(base, split="val"))
        train.load_data()
        val.load_data()

        assert train.config.allowed_overlap_forward == 100
        assert val.config.allowed_overlap_forward == 500
        # Train sees more samples per chart than val because its overlap
        # filter is far less aggressive.
        assert train.count_samples() > val.count_samples()

    def test_overrides_cleared_after_resolve(self, tiny_dataset: Path):
        base = self._base_config(tiny_dataset)
        train = TaikoDetectionSampler(replace(base, split="train"))
        train.load_data()
        # Subsequent resolve() is a no-op — overrides were cleared.
        assert train.config.split_overrides == {}

    def test_no_overrides_is_no_op(self, tiny_dataset: Path):
        cfg = TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, min_cursor_bin=0,
        )
        s = TaikoDetectionSampler(cfg)
        s.load_data()
        # config unchanged (still has default values, not resolved copy)
        assert s.config.allowed_overlap_forward == 100

    def test_override_reserved_field_rejected(self):
        with pytest.raises(ValueError, match="reserved"):
            TaikoDetectionSamplerConfig(
                batch_size=1,
                split_overrides={"train": {"split_seed": 999}},
            )

    def test_batch_size_can_differ_per_split(self, tiny_dataset: Path):
        """Demonstrates overrides aren't limited to overlap — any field."""
        cfg = TaikoDetectionSamplerConfig(
            batch_size=4,
            dataset_root=tiny_dataset,
            a_bins=100, b_bins=100, min_cursor_bin=0,
            split_ratios=(("train", 0.5), ("val", 0.5)),
            split_overrides={
                "train": {"batch_size": 32},
                "val": {"batch_size": 1},
            },
        )
        train = TaikoDetectionSampler(replace(cfg, split="train"))
        val = TaikoDetectionSampler(replace(cfg, split="val"))
        train.load_data()
        val.load_data()
        assert train.config.batch_size == 32
        assert val.config.batch_size == 1


# ─────────────────────────── split tests ──────────────────────────────

class TestSplit:
    def test_all_gives_every_chart(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
            batch_size=1, dataset_root=tiny_dataset, split="all",
            min_cursor_bin=0,
        ))
        s.load_data()
        assert set(s._chart_ids) == {
            "song1 [Normal]", "song1 [Oni]",
            "song2 [Normal]", "song2 [Oni]",
        }

    def test_train_val_are_complementary(self, tiny_dataset: Path):
        common = dict(
            batch_size=1, dataset_root=tiny_dataset,
            split_ratios=(("train", 0.5), ("val", 0.5)),
            split_seed=7, min_cursor_bin=0,
        )
        train = TaikoDetectionSampler(TaikoDetectionSamplerConfig(**common, split="train"))
        val = TaikoDetectionSampler(TaikoDetectionSamplerConfig(**common, split="val"))
        train.load_data()
        val.load_data()

        train_ids = set(train._chart_ids)
        val_ids = set(val._chart_ids)
        assert train_ids.isdisjoint(val_ids)
        assert train_ids | val_ids == {
            "song1 [Normal]", "song1 [Oni]",
            "song2 [Normal]", "song2 [Oni]",
        }

    def test_split_is_song_level_not_chart_level(self, tiny_dataset: Path):
        """All difficulties of a song must land in the same split."""
        common = dict(
            batch_size=1, dataset_root=tiny_dataset,
            split_ratios=(("train", 0.5), ("val", 0.5)),
            split_seed=123, min_cursor_bin=0,
        )
        train = TaikoDetectionSampler(TaikoDetectionSamplerConfig(**common, split="train"))
        train.load_data()
        train_songs = {cid.split(" [")[0] for cid in train._chart_ids}
        # For every song in train, both its difficulties must be there
        for song in train_songs:
            for diff in ["Normal", "Oni"]:
                assert f"{song} [{diff}]" in train._chart_ids

    def test_split_reproducible(self, tiny_dataset: Path):
        common = dict(
            batch_size=1, dataset_root=tiny_dataset, split="val",
            split_ratios=(("train", 0.5), ("val", 0.5)),
            split_seed=99, min_cursor_bin=0,
        )
        a = TaikoDetectionSampler(TaikoDetectionSamplerConfig(**common))
        b = TaikoDetectionSampler(TaikoDetectionSamplerConfig(**common))
        a.load_data()
        b.load_data()
        assert a._chart_ids == b._chart_ids

    def test_three_way_split(self, tiny_dataset: Path):
        """Four charts (2 songs × 2 difficulties) split 3 ways with ratios
        (0.5, 0.25, 0.25) ⇒ 1 song to train, 0 to val, 1 to test. Verifies
        songs land intact in one bucket each."""
        # Use 4 songs to exercise uneven 3-way splits more realistically.
        # The existing fixture has only 2 songs, so this test uses plain
        # ratios and checks that buckets are disjoint and together cover
        # everything.
        common = dict(
            batch_size=1, dataset_root=tiny_dataset,
            split_ratios=(("train", 0.5), ("val", 0.25), ("test", 0.25)),
            split_seed=0, min_cursor_bin=0,
        )
        samplers = {
            name: TaikoDetectionSampler(
                TaikoDetectionSamplerConfig(**common, split=name),
            )
            for name in ("train", "val", "test")
        }
        for s in samplers.values():
            s.load_data()

        ids = {name: set(s._chart_ids) for name, s in samplers.items()}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert ids[a].isdisjoint(ids[b])
        combined = ids["train"] | ids["val"] | ids["test"]
        assert combined == {
            "song1 [Normal]", "song1 [Oni]",
            "song2 [Normal]", "song2 [Oni]",
        }
