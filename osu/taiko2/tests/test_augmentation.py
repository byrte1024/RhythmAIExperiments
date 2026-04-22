"""Tests for the abstract augmentation framework + wiring into the
TaikoDetectionSampler."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from osu.taiko2.domain.augmentation import (
    AugmentationPipeline,
    PostSampleAugmentation,
    PreSampleAugmentation,
)
from osu.taiko2.domain.beatmap import OnsetBinned, OnsetKind
from osu.taiko2.domain.dataset import (
    ChartEntry,
    DatasetManifest,
    MelSamplerConfig,
)
from osu.taiko2.data_samplers import (
    TaikoDetectionPreContext,
    TaikoDetectionSample,
    TaikoDetectionSampler,
    TaikoDetectionSamplerConfig,
)
from osu.taiko2.dataset import _safe_filename
from osu.taiko2.persistence.events import save_events
from osu.taiko2.persistence.features import save_features
from osu.taiko2.persistence.manifest import save_manifest


# ─────────────────────────── pipeline unit tests ──────────────────────

class _RecordingPreAug(PreSampleAugmentation[str]):
    def __init__(self, tag: str, log: list[str]):
        self.tag = tag
        self.log = log

    def apply(self, context: str) -> str:
        self.log.append(self.tag)
        return context + f":{self.tag}"


class _RecordingPostAug(PostSampleAugmentation[str]):
    def __init__(self, tag: str, log: list[str]):
        self.tag = tag
        self.log = log

    def apply(self, sample: str) -> str:
        self.log.append(self.tag)
        return sample + f":{self.tag}"


class TestPipelineOrdering:
    def test_empty_is_identity(self):
        pipe = AugmentationPipeline()
        assert pipe.is_empty
        assert pipe.apply_pre("ctx") == "ctx"
        assert pipe.apply_post("sample") == "sample"

    def test_pre_runs_in_tuple_order(self):
        log = []
        pipe = AugmentationPipeline(
            pre=(_RecordingPreAug("a", log), _RecordingPreAug("b", log)),
        )
        out = pipe.apply_pre("")
        assert log == ["a", "b"]
        assert out == ":a:b"

    def test_post_runs_in_tuple_order(self):
        log = []
        pipe = AugmentationPipeline(
            post=(_RecordingPostAug("x", log), _RecordingPostAug("y", log)),
        )
        out = pipe.apply_post("")
        assert log == ["x", "y"]
        assert out == ":x:y"

    def test_pre_and_post_are_independent(self):
        """apply_post should not invoke pre, and vice versa."""
        log = []
        pipe = AugmentationPipeline(
            pre=(_RecordingPreAug("pre", log),),
            post=(_RecordingPostAug("post", log),),
        )
        pipe.apply_pre("x")
        assert log == ["pre"]
        log.clear()
        pipe.apply_post("x")
        assert log == ["post"]


# ─────────────────────────── integration: sampler wiring ──────────────

def _mk_entry(chart_id: str, bset: str, feat_rel: Path, n_frames: int) -> ChartEntry:
    return ChartEntry(
        chart_id=chart_id, beatmap_id=chart_id, beatmapset_id=bset,
        artist="a", title="t",
        difficulty_version=chart_id.split("[")[-1].rstrip("]"),
        overall_difficulty=5.0, star_rating=None,
        density_mean=5.0, density_peak=8, density_std=1.5,
        duration_s=100.0, total_events=10,
        audio_filename=f"{bset}.mp3",
        features_path=feat_rel,
        n_frames=n_frames,
    )


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "ds"
    feat_dir = ds / "features"
    evt_dir = ds / "events"
    feat_dir.mkdir(parents=True)
    evt_dir.mkdir(parents=True)

    chart_id = "song1 [Oni]"
    feat_rel = Path("features") / "song1.npy"
    save_features(np.zeros((80, 10_000), dtype=np.float32), ds / feat_rel)
    onsets = tuple(
        OnsetBinned(time_ms=int(b * 5), kind=OnsetKind.DON, bin=int(b))
        for b in range(1000, 2000, 100)
    )
    save_events(onsets, evt_dir / f"{_safe_filename(chart_id)}.npz")

    save_manifest(DatasetManifest(
        name="ds", created_at="t",
        sampler_config=MelSamplerConfig(),
        charts=(_mk_entry(chart_id, "s1", feat_rel, 10_000),),
    ), ds / "manifest.json")
    return ds


class _ShiftCursor(PreSampleAugmentation[TaikoDetectionPreContext]):
    """Concrete pre-aug: shift cursor by a fixed amount."""
    def __init__(self, delta: int):
        self.delta = delta

    def apply(self, context: TaikoDetectionPreContext) -> TaikoDetectionPreContext:
        return replace(context, cursor_bin=context.cursor_bin + self.delta)


class _ZeroPastAudio(PostSampleAugmentation[TaikoDetectionSample]):
    """Concrete post-aug: zero out the past-audio window."""
    def apply(self, sample: TaikoDetectionSample) -> TaikoDetectionSample:
        return replace(sample, audio_past=np.zeros_like(sample.audio_past))


class TestSamplerPipelineWiring:
    def test_raw_bypasses_pipeline(self, tiny_dataset: Path):
        # Pipeline that shifts cursor by a large amount, but raw_sample
        # should ignore it entirely.
        pipe = AugmentationPipeline(pre=(_ShiftCursor(delta=9999),))
        s = TaikoDetectionSampler(
            TaikoDetectionSamplerConfig(
                batch_size=1, dataset_root=tiny_dataset,
                a_bins=100, b_bins=100, c_events=4, d_events=2,
                min_cursor_bin=0,
            ),
            pipeline=pipe,
        )
        s.load_data()
        raw = s.raw_sample(1)          # ei=1 → cursor sits on bin 1000
        assert raw.cursor_bin == 1000  # unaffected by pre-aug

    def test_pre_aug_shifts_cursor(self, tiny_dataset: Path):
        pipe = AugmentationPipeline(pre=(_ShiftCursor(delta=50),))
        s = TaikoDetectionSampler(
            TaikoDetectionSamplerConfig(
                batch_size=1, dataset_root=tiny_dataset,
                a_bins=100, b_bins=100, c_events=4, d_events=2,
                min_cursor_bin=0,
            ),
            pipeline=pipe,
        )
        s.load_data()
        aug = s.augment_sample(1)
        assert aug.cursor_bin == 1050

    def test_post_aug_mutates_sample(self, tiny_dataset: Path):
        pipe = AugmentationPipeline(post=(_ZeroPastAudio(),))
        s = TaikoDetectionSampler(
            TaikoDetectionSamplerConfig(
                batch_size=1, dataset_root=tiny_dataset,
                a_bins=100, b_bins=100, c_events=4, d_events=2,
                min_cursor_bin=0,
            ),
            pipeline=pipe,
        )
        s.load_data()
        aug = s.augment_sample(1)
        assert np.all(aug.audio_past == 0)
        # past_events were NOT touched by the post-aug → still reflect cursor
        assert len(aug.past_events) == 4

    def test_empty_pipeline_matches_raw(self, tiny_dataset: Path):
        s = TaikoDetectionSampler(
            TaikoDetectionSamplerConfig(
                batch_size=1, dataset_root=tiny_dataset,
                a_bins=100, b_bins=100, c_events=4, d_events=2,
                min_cursor_bin=0,
            ),
        )
        s.load_data()
        raw = s.raw_sample(2)
        aug = s.augment_sample(2)
        assert raw.sample_id == aug.sample_id
        assert raw.cursor_bin == aug.cursor_bin
        np.testing.assert_array_equal(raw.audio_past, aug.audio_past)
