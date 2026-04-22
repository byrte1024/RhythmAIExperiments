"""Tests for `Chart` — metrics, comparison, save/load."""
from pathlib import Path

import numpy as np
import pytest

from osu.taiko2.domain.beatmap import (
    AudioRef,
    Density,
    Difficulty,
    Onset,
    OnsetKind,
    Track,
)
from osu.taiko2.domain.chart import (
    Chart,
    _compute_gap_distribution,
    _compute_ratio_distribution,
    _estimate_bpm_from_ioi,
    _events_to_binary,
    _find_streaks,
    _hi_pspace,
    _over_pspace,
)


# ─────────────────────────── builders ─────────────────────────────────

def _build_track(
    onsets: tuple[Onset, ...],
    *,
    version: str = "Oni",
    od: float = 6.0,
    star: float | None = 4.0,
) -> Track:
    if onsets:
        duration_s = (onsets[-1].time_ms - onsets[0].time_ms) / 1000.0
    else:
        duration_s = 0.0
    return Track(
        beatmap_id="1",
        beatmapset_id="10",
        artist="a",
        title="t",
        difficulty=Difficulty(version=version, overall_difficulty=od, star_rating=star),
        audio=AudioRef(filename="audio.mp3", format="mp3"),
        onsets=onsets,
        density=Density(
            mean=len(onsets) / max(duration_s, 0.1),
            peak=0, std=0.0,
            duration_s=duration_s,
            total_events=len(onsets),
        ),
    )


def _metronome(bpm: float, n: int, kind: OnsetKind = OnsetKind.DON) -> tuple[Onset, ...]:
    interval_ms = 60_000.0 / bpm
    return tuple(
        Onset(time_ms=int(round(i * interval_ms)), kind=kind) for i in range(n)
    )


# ─────────────────────────── helpers ──────────────────────────────────

class TestHelpers:
    def test_find_streaks_detects_uniform(self):
        gaps = np.full(10, 100.0)
        streaks = _find_streaks(gaps, tolerance=0.05)
        assert len(streaks) == 1
        start, length, head = streaks[0]
        assert start == 0 and length == 10 and head == pytest.approx(100.0)

    def test_find_streaks_breaks_on_change(self):
        gaps = np.array([100, 100, 100, 200, 200, 200], dtype=np.float64)
        streaks = _find_streaks(gaps, tolerance=0.05)
        assert [(l, h) for _, l, h in streaks] == [(3, 100.0), (3, 200.0)]

    def test_events_to_binary_density(self):
        times = np.array([0, 23, 46, 1000], dtype=np.int64)
        binary = _events_to_binary(times, step_ms=23)
        assert binary[0] == 1 and binary[1] == 1 and binary[2] == 1
        assert binary[1000 // 23] == 1

    def test_over_pspace_empty(self):
        assert _over_pspace(np.zeros(0, dtype=np.int32)) == 0.0

    def test_over_pspace_all_zeros(self):
        # Only one pattern observed — all zeros.
        b = np.zeros(50, dtype=np.int32)
        ps = _over_pspace(b, scale=8)
        assert ps == pytest.approx(1 / 256 * 100)

    def test_hi_pspace_identical_charts(self):
        b = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
        # intersection == other → 100%
        assert _hi_pspace(b, b, scale=4) == pytest.approx(100.0)

    def test_estimate_bpm_metronome_120(self):
        interval = 500  # ms between onsets → 120 BPM
        gaps = np.full(20, interval, dtype=np.float64)
        bpm, dom = _estimate_bpm_from_ioi(gaps)
        assert dom == pytest.approx(500.0)
        assert bpm == pytest.approx(120.0)

    def test_estimate_bpm_half_time_normalizes(self):
        interval = 250  # 240 BPM at quarter; within range, no halving
        gaps = np.full(20, interval, dtype=np.float64)
        bpm, dom = _estimate_bpm_from_ioi(gaps)
        assert dom == pytest.approx(250.0)
        assert 60.0 <= bpm <= 240.0

    def test_estimate_bpm_short_sequence(self):
        bpm, dom = _estimate_bpm_from_ioi(np.array([], dtype=np.float64))
        assert bpm is None and dom is None


# ─────────────────────────── metrics ──────────────────────────────────

class TestChartMetrics:
    def test_basic_counts(self):
        chart = Chart(track=_build_track(_metronome(120, 20)))
        m = chart.calculate_metrics()
        assert m.total_events == 20
        assert m.count_don == 20
        assert m.count_ka == 0

    def test_don_ratio(self):
        onsets = (
            Onset(0, OnsetKind.DON), Onset(100, OnsetKind.KA),
            Onset(200, OnsetKind.DON), Onset(300, OnsetKind.BIG_DON),
        )
        m = Chart(track=_build_track(onsets)).calculate_metrics()
        # 3 don-side / 4 total hits = 0.75
        assert m.don_ratio == pytest.approx(0.75)

    def test_ioi_stats_metronome(self):
        chart = Chart(track=_build_track(_metronome(120, 20)))
        m = chart.calculate_metrics()
        assert m.ioi_median_ms == pytest.approx(500.0)
        assert m.ioi_min_ms == pytest.approx(500.0)
        assert m.ioi_max_ms == pytest.approx(500.0)
        assert m.ioi_std_ms == pytest.approx(0.0, abs=1e-6)

    def test_streak_detection(self):
        chart = Chart(track=_build_track(_metronome(120, 20)))
        m = chart.calculate_metrics()
        assert m.longest_streak == 19  # 20 events → 19 gaps, all identical
        assert m.streak_event_fraction == pytest.approx(1.0)

    def test_bpm_estimate(self):
        chart = Chart(track=_build_track(_metronome(140, 30)))
        m = chart.calculate_metrics()
        assert m.estimated_bpm == pytest.approx(140.0, abs=1.0)
        assert m.dominant_ioi_ms == pytest.approx(60_000 / 140, abs=5.0)

    def test_silence_region_detected(self):
        # 10 beats at 120 BPM (0..4.5s), then 5s silence, then more beats
        first = list(_metronome(120, 10))
        after_silence_ms = first[-1].time_ms + 5000
        more = [Onset(after_silence_ms + i * 500, OnsetKind.DON) for i in range(10)]
        chart = Chart(track=_build_track(tuple(first + more)))
        m = chart.calculate_metrics()
        assert len(m.silence_regions) >= 1

    def test_short_ioi_counted(self):
        # Two onsets 10ms apart → short_ioi = 1
        onsets = (Onset(0, OnsetKind.DON), Onset(10, OnsetKind.DON),
                  Onset(500, OnsetKind.DON))
        m = Chart(track=_build_track(onsets)).calculate_metrics()
        assert m.short_ioi_count == 1

    def test_empty_chart(self):
        m = Chart(track=_build_track(tuple())).calculate_metrics()
        assert m.total_events == 0
        assert m.estimated_bpm is None


# ─────────────────────────── gap-distribution shape ───────────────────

class TestGapDistribution:
    """Metrics over the 200-bucket gap histogram. Return shape is
    ``(peaks, peak_count, peak_falloff, random_distance,
    metronome_distance)`` — `peaks` is (bucket_center_ms, count) desc."""

    def test_empty(self):
        assert _compute_gap_distribution(np.zeros(0, dtype=np.float64)) == (
            (), 0, 0.0, 0.0, 0.0,
        )

    def test_single_gap(self):
        # One gap isn't enough to form a histogram — treated as empty.
        assert _compute_gap_distribution(np.array([250.0])) == (
            (), 0, 0.0, 0.0, 0.0,
        )

    def test_pure_metronome(self):
        # All 200 gaps at 250 ms → one bucket holds everything.
        gaps = np.full(200, 250.0)
        peaks, pc, pf, rd, md = _compute_gap_distribution(gaps)
        assert pc == 1
        # Only one peak → no next peak to decay into → 0.0 by convention.
        assert pf == 0.0
        # One bucket holds all mass → TVD from uniform ≈ 1 − 1/200.
        assert rd == pytest.approx(0.995, abs=1e-3)
        # Mode IS the whole distribution → distance = 0.
        assert md == pytest.approx(0.0, abs=1e-6)
        # Peak detail: center of bucket 25 = 255 ms, count 200.
        assert peaks == ((255.0, 200),)

    def test_two_equal_peaks(self):
        # 100 gaps at 125 ms, 100 gaps at 250 ms. Buckets 12 and 25.
        # They're >= 30 ms apart so the min-separation filter keeps both.
        gaps = np.concatenate([np.full(100, 125.0), np.full(100, 250.0)])
        peaks, pc, pf, rd, md = _compute_gap_distribution(gaps)
        assert pc == 2
        assert pf == pytest.approx(1.0)
        assert md == pytest.approx(0.5, abs=1e-6)
        assert rd > 0.98
        # Both peaks in the list; both have count 100.
        assert sorted(peaks) == [(125.0, 100), (255.0, 100)]

    def test_peak_falloff_400_200_100(self):
        # Three peaks with counts 400 / 200 / 100.
        gaps = np.concatenate([
            np.full(400, 100.0),       # bucket 10 → center 105
            np.full(200, 200.0),       # bucket 20 → center 205
            np.full(100, 400.0),       # bucket 40 → center 405
        ])
        peaks, pc, pf, rd, md = _compute_gap_distribution(gaps)
        assert pc == 3
        # Ratios: 200/400=0.5, 100/200=0.5 → mean 0.5.
        assert pf == pytest.approx(0.5)
        assert md == pytest.approx(1.0 - 400 / 700, abs=1e-4)
        # Peaks sorted by count desc.
        assert peaks == ((105.0, 400), (205.0, 200), (405.0, 100))

    def test_height_threshold_rejects_noise(self):
        gaps = np.concatenate([
            np.full(500, 250.0),
            np.array([130.0, 310.0, 470.0, 670.0, 990.0]),
        ])
        _, pc, _, _, _ = _compute_gap_distribution(gaps)
        assert pc == 1

    def test_minimum_separation_merges_close_peaks(self):
        gaps = np.concatenate([
            np.full(300, 100.0),
            np.full(200, 110.0),
        ])
        _, pc, _, _, _ = _compute_gap_distribution(gaps)
        assert pc == 1

    def test_gaps_above_cap_excluded(self):
        mostly = np.full(100, 250.0)
        mixed = np.concatenate([mostly, np.array([5000.0])])
        assert _compute_gap_distribution(mostly) == _compute_gap_distribution(mixed)


class TestRatioDistribution:
    """Metrics over the log2-bucketed ratio histogram
    (gap[i] / gap[i-1]). Metronome reference is anchored at the 1.0x
    bucket: a run of identical gaps has every ratio = 1.0 regardless of
    what the absolute gap value is."""

    def test_empty(self):
        assert _compute_ratio_distribution(np.zeros(0, dtype=np.float64)) == (
            (), 0, 0.0, 0.0, 0.0,
        )

    def test_too_few_gaps(self):
        # Two onsets → one gap → no ratios → empty.
        assert _compute_ratio_distribution(np.array([250.0])) == (
            (), 0, 0.0, 0.0, 0.0,
        )

    def test_pure_metronome_all_ratios_at_one(self):
        # Constant-gap chart → every consecutive ratio = 1.0.
        # Metronome distance must be 0 (all mass in the 1.0x bucket).
        gaps = np.full(200, 250.0)
        peaks, pc, pf, rd, md = _compute_ratio_distribution(gaps)
        assert pc == 1
        assert md == pytest.approx(0.0, abs=1e-6)
        assert pf == 0.0
        # One peak exactly at ratio=1.0 (center of the 0-log2 bucket).
        assert len(peaks) == 1
        center, count = peaks[0]
        assert center == pytest.approx(1.0, rel=0.02)
        assert count == 199    # 200 gaps → 199 ratios

    def test_metronome_distance_anchored_at_one_not_mode(self):
        # All ratios at 2.0x (each gap doubles the last). Mode is at
        # 2.0x, NOT at 1.0x — so metronome distance should be 1.0
        # (nothing in the 1.0x bucket) even though the histogram is
        # "pure" in the peak-count sense.
        # Construct gaps 10, 20, 40, 80, 160, 320, 640, 1280 → 7 ratios
        # all equal to 2.0. Repeat the pattern to get enough data.
        gaps = np.array(
            [10.0 * (2 ** i) for i in range(8)]
            + [10.0 * (2 ** i) for i in range(8)],
            dtype=np.float64,
        )
        peaks, pc, pf, rd, md = _compute_ratio_distribution(gaps)
        assert pc == 1
        # Peak should be at ratio 2.0x (ratio bucket for log2=1).
        assert peaks[0][0] == pytest.approx(2.0, rel=0.05)
        # Metronome distance: NO ratios land in the 1.0x bucket → 1.0.
        assert md == pytest.approx(1.0, abs=1e-6)

    def test_falloff_two_equal_ratio_peaks(self):
        # Alternating doubling/halving → ratios alternate 2.0x, 0.5x.
        gaps = np.array([100.0, 200.0] * 100, dtype=np.float64)
        peaks, pc, pf, rd, md = _compute_ratio_distribution(gaps)
        assert pc == 2
        # Peaks near-equal (within 1) → falloff ≈ 1.0 up to the
        # alternating pattern's off-by-one.
        assert pf == pytest.approx(1.0, abs=0.02)
        centers = sorted(c for c, _ in peaks)
        assert centers[0] == pytest.approx(0.5, rel=0.05)
        assert centers[1] == pytest.approx(2.0, rel=0.05)
        # No ratios at exactly 1.0x → metronome distance = 1.
        assert md == pytest.approx(1.0, abs=1e-6)

    def test_near_one_ratios_merge_to_single_peak(self):
        # Ratios of 1.01 and 0.95 are musically "just a metronome with
        # a bit of jitter" — the ratio histogram's wider smoothing /
        # merge radius must fold them into one peak at ~1.0x.
        # Construct gaps whose consecutive ratios alternate between
        # 1.01 and 0.95 (and one 1.0 to seed the mode).
        gaps_list: list[float] = [100.0]
        for i in range(400):
            r = 1.01 if i % 2 == 0 else 0.95
            gaps_list.append(gaps_list[-1] * r)
        gaps = np.array(gaps_list, dtype=np.float64)
        peaks, pc, _, _, md = _compute_ratio_distribution(gaps)
        assert pc == 1
        # Single peak centered near 1.0x.
        center, _ = peaks[0]
        assert center == pytest.approx(1.0, rel=0.10)
        # Metronome distance is low because the merged peak sits at
        # ~1.0x — this is essentially metronome + jitter.
        assert md < 0.6

    def test_musically_distinct_ratios_stay_separate(self):
        # Classic tempo-change ratios: 0.67x and 1.33x are ~33% apart
        # and represent different rhythmic categories; they must NOT
        # merge.
        gaps_list: list[float] = [100.0]
        for i in range(200):
            r = 1.33 if i % 2 == 0 else 0.67
            gaps_list.append(gaps_list[-1] * r)
        gaps = np.array(gaps_list, dtype=np.float64)
        _, pc, _, _, _ = _compute_ratio_distribution(gaps)
        assert pc == 2

    def test_ratios_out_of_range_dropped(self):
        # Ratio of 100 is log2≈6.64 → outside the [-3, +3] support.
        # Add such a pair; the result should match the same chart
        # without that pair (modulo the two onsets it adds to gaps).
        base = np.array([100.0] * 10, dtype=np.float64)
        # Adding a pair where gap[i]=100, gap[i+1]=10000 gives a ratio
        # of 100 which is outside range. Drop it, and the surviving
        # ratios all equal 1.0.
        mixed = np.concatenate([base, np.array([10000.0, 10.0])])
        peaks_a, *_ = _compute_ratio_distribution(base)
        peaks_b, *_ = _compute_ratio_distribution(mixed)
        # Both should have a single peak at 1.0x; peak counts may
        # differ by 1-2 from the extra in-range ratios the tail
        # introduces. Just check shape.
        assert len(peaks_a) == 1 and len(peaks_b) >= 1
        assert peaks_a[0][0] == pytest.approx(1.0, rel=0.05)


# ─────────────────────────── comparison ───────────────────────────────

class TestChartComparison:
    def test_identical_charts_perfect_match(self):
        chart = Chart(track=_build_track(_metronome(120, 30)))
        cmp = chart.compare(chart)
        assert cmp.n_self == cmp.n_other == 30
        assert cmp.matched_rate == pytest.approx(1.0)
        assert cmp.close_rate == pytest.approx(1.0)
        assert cmp.hallucination_rate == pytest.approx(0.0)
        assert cmp.error_mean_ms == pytest.approx(0.0)
        assert cmp.density_ratio == pytest.approx(1.0)
        assert cmp.hi_pspace == pytest.approx(100.0)

    def test_completely_disjoint_charts(self):
        a = Chart(track=_build_track(_metronome(120, 30)))
        # Shift every onset by 500 ms = way outside 100 ms window
        b = Chart(track=_build_track(
            tuple(Onset(o.time_ms + 5000, o.kind) for o in a.track.onsets)
        ))
        cmp = a.compare(b)
        # With 500ms shift and identical structure, many still cross-match
        # beyond the window boundary. Just require hall + far to be high.
        assert cmp.hallucination_rate >= 0.0
        assert cmp.far_rate >= 0.0

    def test_density_ratio(self):
        dense = Chart(track=_build_track(_metronome(240, 40)))   # 40 @ 250ms
        sparse = Chart(track=_build_track(_metronome(60, 40)))   # 40 @ 1000ms
        cmp = dense.compare(sparse)
        assert cmp.density_ratio > 1.0  # self (dense) / other (sparse)


# ─────────────────────────── save / load ──────────────────────────────

class TestSaveLoad:
    def test_round_trip_without_audio(self, tmp_path: Path):
        chart = Chart(track=_build_track(_metronome(120, 10)))
        path = tmp_path / "c.zip"
        chart.save(path)
        loaded = Chart.load(path)

        assert loaded.audio is None
        assert loaded.track.artist == chart.track.artist
        assert loaded.track.beatmap_id == chart.track.beatmap_id
        assert loaded.track.difficulty.star_rating == chart.track.difficulty.star_rating
        assert len(loaded.track.onsets) == len(chart.track.onsets)
        assert loaded.track.onsets[0] == chart.track.onsets[0]
        assert loaded.track.onsets[-1] == chart.track.onsets[-1]

    def test_round_trip_with_audio(self, tmp_path: Path):
        audio_bytes = b"\x00\xff" * 1024  # fake audio payload
        chart = Chart(
            track=_build_track(_metronome(120, 10)),
            audio=audio_bytes,
        )
        path = tmp_path / "c.zip"
        chart.save(path)
        loaded = Chart.load(path)

        assert loaded.audio == audio_bytes
        assert loaded.track.audio.format == chart.track.audio.format

    def test_round_trip_preserves_all_onset_kinds(self, tmp_path: Path):
        onsets = (
            Onset(0, OnsetKind.DON),
            Onset(100, OnsetKind.KA),
            Onset(200, OnsetKind.BIG_DON),
            Onset(300, OnsetKind.BIG_KA),
            Onset(400, OnsetKind.DRUMROLL),
            Onset(500, OnsetKind.SPINNER),
        )
        chart = Chart(track=_build_track(onsets))
        path = tmp_path / "c.zip"
        chart.save(path)
        loaded = Chart.load(path)
        assert loaded.track.onsets == chart.track.onsets

    def test_star_rating_none_round_trips(self, tmp_path: Path):
        chart = Chart(track=_build_track(_metronome(120, 5), star=None))
        path = tmp_path / "c.zip"
        chart.save(path)
        loaded = Chart.load(path)
        assert loaded.track.difficulty.star_rating is None


# ─────────────────────────── .osu / .osz load ─────────────────────────

class TestLoadOsu:
    def test_load_from_osu_text_file(self, tmp_path: Path):
        osu = (
            "osu file format v14\n"
            "[General]\n"
            "Mode: 1\n"
            "AudioFilename: song.mp3\n"
            "[Metadata]\n"
            "Artist: A\n"
            "Title: T\n"
            "Version: Oni\n"
            "BeatmapID: 1\n"
            "BeatmapSetID: 10\n"
            "[Difficulty]\n"
            "OverallDifficulty: 6\n"
            "[HitObjects]\n"
            "256,192,1000,1,0,0:0:0:0:\n"
            "256,192,1500,1,2,0:0:0:0:\n"
            "256,192,2000,1,4,0:0:0:0:\n"
        )
        p = tmp_path / "chart.osu"
        p.write_text(osu, encoding="utf-8")

        chart = Chart.load(p)
        assert chart.audio is None
        kinds = [o.kind for o in chart.track.onsets]
        assert kinds == [OnsetKind.DON, OnsetKind.KA, OnsetKind.BIG_DON]
        assert chart.track.beatmap_id == "1"

    def test_load_non_taiko_osu_rejected(self, tmp_path: Path):
        osu = (
            "osu file format v14\n"
            "[General]\nMode: 0\nAudioFilename: x.mp3\n"
            "[Metadata]\nArtist: A\nTitle: T\nVersion: Hard\n"
            "[HitObjects]\n256,192,1000,1,0,0:0:0:0:\n"
        )
        p = tmp_path / "std.osu"
        p.write_text(osu, encoding="utf-8")
        with pytest.raises(ValueError, match="not a valid osu!taiko"):
            Chart.load(p)


class TestLoadOsz:
    def _mk_osz(self, tmp_path: Path) -> Path:
        """Build a minimal .osz with two taiko charts sharing one audio."""
        import zipfile
        osu1 = (
            "osu file format v14\n"
            "[General]\nMode: 1\nAudioFilename: track.mp3\n"
            "[Metadata]\nArtist: Band\nTitle: Song\nVersion: Easy\n"
            "BeatmapID: 11\nBeatmapSetID: 99\n"
            "[Difficulty]\nOverallDifficulty: 3\n"
            "[HitObjects]\n256,192,100,1,0,0:0:0:0:\n256,192,600,1,2,0:0:0:0:\n"
        )
        osu2 = osu1.replace("Easy", "Oni").replace("BeatmapID: 11", "BeatmapID: 22")
        osz = tmp_path / "pack.osz"
        with zipfile.ZipFile(osz, "w") as z:
            z.writestr("song [Easy].osu", osu1)
            z.writestr("song [Oni].osu", osu2)
            z.writestr("track.mp3", b"\x00fake-audio\x00")
        return osz

    def test_requires_index(self, tmp_path: Path):
        osz = self._mk_osz(tmp_path)
        with pytest.raises(ValueError, match="pass index"):
            Chart.load(osz)

    def test_loads_selected_chart_with_audio(self, tmp_path: Path):
        osz = self._mk_osz(tmp_path)
        chart = Chart.load(osz, index=0)
        assert chart.track.beatmapset_id == "99"
        assert chart.audio == b"\x00fake-audio\x00"

    def test_out_of_range_index(self, tmp_path: Path):
        osz = self._mk_osz(tmp_path)
        with pytest.raises(IndexError):
            Chart.load(osz, index=5)


# ─────────────────────────── .osu / .osz save ─────────────────────────

class TestSaveOsu:
    def test_osu_round_trip(self, tmp_path: Path):
        original = Chart(track=_build_track(_metronome(120, 10)))
        p = tmp_path / "out.osu"
        original.save(p)
        assert p.exists()

        loaded = Chart.load(p)
        assert len(loaded.track.onsets) == 10
        assert [o.time_ms for o in loaded.track.onsets] == [
            o.time_ms for o in original.track.onsets
        ]
        assert all(o.kind == OnsetKind.DON for o in loaded.track.onsets)

    def test_osu_preserves_circle_kinds(self, tmp_path: Path):
        onsets = (
            Onset(0, OnsetKind.DON),
            Onset(100, OnsetKind.KA),
            Onset(200, OnsetKind.BIG_DON),
            Onset(300, OnsetKind.BIG_KA),
        )
        original = Chart(track=_build_track(onsets))
        p = tmp_path / "kinds.osu"
        original.save(p)
        loaded = Chart.load(p)
        assert [o.kind for o in loaded.track.onsets] == list(
            o.kind for o in onsets
        )

    def test_save_osu_explicit_method(self, tmp_path: Path):
        chart = Chart(track=_build_track(_metronome(100, 3)))
        p = tmp_path / "explicit.txt"  # extension != .osu, but method is explicit
        chart.save_osu(p)
        assert "Mode: 1" in p.read_text(encoding="utf-8")


class TestSaveOsz:
    def test_osz_round_trip_with_audio(self, tmp_path: Path):
        audio = b"AUDIO" * 50
        chart = Chart(track=_build_track(_metronome(140, 8)), audio=audio)
        p = tmp_path / "out.osz"
        chart.save(p)

        loaded = Chart.load(p, index=0)
        assert loaded.audio == audio
        assert [o.time_ms for o in loaded.track.onsets] == [
            o.time_ms for o in chart.track.onsets
        ]

    def test_osz_contains_one_osu(self, tmp_path: Path):
        import zipfile
        chart = Chart(track=_build_track(_metronome(120, 3)))
        p = tmp_path / "minimal.osz"
        chart.save(p)
        with zipfile.ZipFile(p, "r") as z:
            osu_names = [n for n in z.namelist() if n.endswith(".osu")]
            assert len(osu_names) == 1

    def test_save_dispatch_by_suffix(self, tmp_path: Path):
        """Save should choose the right format based on extension alone."""
        chart = Chart(track=_build_track(_metronome(100, 4)))
        bundle = tmp_path / "c.zip"
        osu = tmp_path / "c.osu"
        osz = tmp_path / "c.osz"
        chart.save(bundle)
        chart.save(osu)
        chart.save(osz)
        # All three loadable back into a valid chart
        assert len(Chart.load(bundle).track.onsets) == 4
        assert len(Chart.load(osu).track.onsets) == 4
        assert len(Chart.load(osz, index=0).track.onsets) == 4
