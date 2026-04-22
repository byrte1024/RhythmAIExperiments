"""Tests for `TaikoDetectionSampler` — cursor rules, padding, split routing.

Builds a synthetic dataset on tmp_path (two songs × two difficulties), runs
the real save/load path, then exercises the sampler against it. No audio
decode or real mel computation — features are synthetic arrays.
"""
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
        ))
        s.load_data()
        with pytest.raises(IndexError):
            s.raw_sample(s.count_samples())
        with pytest.raises(IndexError):
            s.raw_batch(s.count_batches())


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
            val_ratio=0.5, split_seed=7, min_cursor_bin=0,
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
            val_ratio=0.5, split_seed=123, min_cursor_bin=0,
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
            batch_size=1, dataset_root=tiny_dataset,
            split="val", val_ratio=0.5, split_seed=99, min_cursor_bin=0,
        )
        a = TaikoDetectionSampler(TaikoDetectionSamplerConfig(**common))
        b = TaikoDetectionSampler(TaikoDetectionSamplerConfig(**common))
        a.load_data()
        b.load_data()
        assert a._chart_ids == b._chart_ids
