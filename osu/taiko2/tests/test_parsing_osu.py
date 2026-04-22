"""Unit tests for `osu.taiko2.parsing.osu`."""
import pytest

from osu.taiko2.domain.beatmap import OnsetKind
from osu.taiko2.parsing.osu import (
    _classify_hit_object,
    compute_density,
    parse_osu_text,
)
from osu.taiko2.domain.beatmap import Onset


_MINIMAL_TAIKO = """osu file format v14
[General]
Mode: 1
AudioFilename: track.mp3
[Metadata]
Title: My Song
Artist: Some Artist
Version: Oni
BeatmapID: 111
BeatmapSetID: 222
[Difficulty]
OverallDifficulty: 6.5
[HitObjects]
0,0,1000,1,0,0:0:0:0:
0,0,1500,1,2,0:0:0:0:
0,0,2000,1,4,0:0:0:0:
0,0,2500,1,6,0:0:0:0:
0,0,3000,2,0,0:0:0:0:
0,0,3500,8,0,0:0:0:0:
"""


class TestClassifyHitObject:
    def test_circle_don(self):
        assert _classify_hit_object(1, 0) == OnsetKind.DON

    def test_circle_ka_via_whistle(self):
        assert _classify_hit_object(1, 2) == OnsetKind.KA

    def test_circle_ka_via_clap(self):
        assert _classify_hit_object(1, 8) == OnsetKind.KA

    def test_circle_big_don(self):
        assert _classify_hit_object(1, 4) == OnsetKind.BIG_DON

    def test_circle_big_ka(self):
        assert _classify_hit_object(1, 6) == OnsetKind.BIG_KA  # finish + whistle

    def test_drumroll(self):
        assert _classify_hit_object(2, 0) == OnsetKind.DRUMROLL

    def test_spinner(self):
        assert _classify_hit_object(8, 0) == OnsetKind.SPINNER

    def test_unknown(self):
        assert _classify_hit_object(0, 0) == OnsetKind.UNKNOWN


class TestParseOsuText:
    def test_valid_taiko_round_trip(self):
        track = parse_osu_text(_MINIMAL_TAIKO)
        assert track is not None
        assert track.beatmap_id == "111"
        assert track.beatmapset_id == "222"
        assert track.artist == "Some Artist"
        assert track.title == "My Song"
        assert track.difficulty.version == "Oni"
        assert track.difficulty.overall_difficulty == pytest.approx(6.5)
        assert track.difficulty.star_rating is None
        assert track.audio.filename == "track.mp3"
        assert track.audio.format == "mp3"

        # 4 circles + 1 drumroll + 1 spinner, in order
        assert len(track.onsets) == 6
        kinds = [o.kind for o in track.onsets]
        assert kinds == [
            OnsetKind.DON, OnsetKind.KA, OnsetKind.BIG_DON, OnsetKind.BIG_KA,
            OnsetKind.DRUMROLL, OnsetKind.SPINNER,
        ]

    def test_returns_none_for_non_taiko(self):
        txt = _MINIMAL_TAIKO.replace("Mode: 1", "Mode: 0")
        assert parse_osu_text(txt) is None

    def test_returns_none_for_empty_onsets(self):
        txt = _MINIMAL_TAIKO.split("[HitObjects]")[0] + "[HitObjects]\n"
        assert parse_osu_text(txt) is None

    def test_returns_none_when_audio_missing(self):
        txt = _MINIMAL_TAIKO.replace("AudioFilename: track.mp3", "AudioFilename:")
        assert parse_osu_text(txt) is None

    def test_malformed_hit_object_is_skipped(self):
        txt = _MINIMAL_TAIKO + "bad,line\n"
        t = parse_osu_text(txt)
        assert t is not None
        assert len(t.onsets) == 6  # malformed line silently ignored


class TestComputeDensity:
    def test_empty_onsets(self):
        d = compute_density(tuple())
        assert d.total_events == 0
        assert d.duration_s == 0

    def test_single_onset(self):
        d = compute_density((Onset(1000, OnsetKind.DON),))
        assert d.total_events == 1
        assert d.duration_s == 0

    def test_uniform_spacing(self):
        onsets = tuple(Onset(i * 500, OnsetKind.DON) for i in range(1, 5))
        d = compute_density(onsets)
        assert d.total_events == 4
        assert d.duration_s == pytest.approx((2000 - 500) / 1000.0)
        # 4 events in 1.5s
        assert d.mean == pytest.approx(4 / 1.5, rel=1e-3)
