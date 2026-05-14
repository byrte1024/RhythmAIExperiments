"""Tests for `inference.multi_tolerance_compare`."""
from __future__ import annotations

import pytest

from osu.taiko2.domain.beatmap import (
    AudioRef,
    Density,
    Difficulty,
    Onset,
    OnsetKind,
    Track,
)
from osu.taiko2.domain.chart import Chart, ChartComparison
from osu.taiko2.inference.multi_tolerance_compare import (
    aggregate_multi_tolerance_summaries,
    compare_at_tolerances,
)


def _mk_chart(times_ms: list[int]) -> Chart:
    onsets = tuple(
        Onset(time_ms=t, kind=OnsetKind.DON) for t in times_ms
    )
    track = Track(
        beatmap_id="bm", beatmapset_id="bms",
        artist="a", title="t",
        difficulty=Difficulty(version="v", overall_difficulty=0.0),
        audio=AudioRef(filename="x", format="mp3"),
        onsets=onsets,
        density=Density(mean=0.0, peak=0, std=0.0, duration_s=10.0, total_events=len(onsets)),
    )
    return Chart(track=track, audio=None)


def _aligned_pair() -> tuple[Chart, Chart]:
    times = [100, 200, 300, 400, 500]
    return _mk_chart(times), _mk_chart(times)


class TestCompareAtTolerances:
    def test_one_per_tolerance(self):
        a, b = _aligned_pair()
        out = compare_at_tolerances(a, b, tolerances_ms=(5, 10, 25, 50))
        assert set(out.keys()) == {5, 10, 25, 50}
        for v in out.values():
            assert isinstance(v, ChartComparison)

    def test_aligned_perfect_match(self):
        a, b = _aligned_pair()
        out = compare_at_tolerances(a, b, tolerances_ms=(5, 100))
        assert out[5].matched_rate == pytest.approx(1.0)
        assert out[100].matched_rate == pytest.approx(1.0)

    def test_matched_rate_monotone_in_tol(self):
        # 5 ms aligned, 5 ms off-by-15, 5 ms off-by-30 → matched_rate
        # grows with tolerance.
        a = _mk_chart([100, 200, 300, 400, 500])
        b = _mk_chart([100, 215, 300, 430, 500])  # two off by 15, 30
        out = compare_at_tolerances(a, b, tolerances_ms=(5, 20, 50))
        assert out[5].matched_rate <= out[20].matched_rate
        assert out[20].matched_rate <= out[50].matched_rate

    def test_empty_input_returns_zeros(self):
        a = _mk_chart([100])
        b = _mk_chart([])
        out = compare_at_tolerances(a, b, tolerances_ms=(25,))
        assert out[25].matched_rate == 0.0

    def test_empty_tolerances_raises(self):
        a, b = _aligned_pair()
        with pytest.raises(ValueError):
            compare_at_tolerances(a, b, tolerances_ms=())

    def test_pattern_metrics_consistent(self):
        # The TN metrics should be the same regardless of tolerance.
        a, b = _aligned_pair()
        out = compare_at_tolerances(a, b, tolerances_ms=(5, 100))
        assert out[5].over_pspace_self == out[100].over_pspace_self
        assert out[5].hi_pspace == out[100].hi_pspace


class TestAggregate:
    def test_basic_shape(self):
        a, b = _aligned_pair()
        per_chart = [
            compare_at_tolerances(a, b, tolerances_ms=(5, 25, 50)),
            compare_at_tolerances(a, b, tolerances_ms=(5, 25, 50)),
        ]
        summary = aggregate_multi_tolerance_summaries(per_chart, (5, 25, 50))
        assert "tolerances_ms" in summary
        assert summary["tolerances_ms"] == [5, 25, 50]
        assert "fields" in summary
        assert "matched_rate_at_tol_5" in summary["fields"]
        assert "matched_rate_at_tol_25" in summary["fields"]
        assert "matched_rate_at_tol_50" in summary["fields"]

    def test_stats_keys_present(self):
        a, b = _aligned_pair()
        per_chart = [
            compare_at_tolerances(a, b, tolerances_ms=(25,)),
        ]
        summary = aggregate_multi_tolerance_summaries(per_chart, (25,))
        stats = summary["fields"]["matched_rate_at_tol_25"]
        for k in ("median", "p25", "p75", "mean", "min", "max", "n"):
            assert k in stats

    def test_median_correct(self):
        # Three charts with matched_rate=1.0 should yield median=1.0.
        a, b = _aligned_pair()
        per_chart = [compare_at_tolerances(a, b, tolerances_ms=(25,)) for _ in range(3)]
        summary = aggregate_multi_tolerance_summaries(per_chart, (25,))
        assert summary["fields"]["matched_rate_at_tol_25"]["median"] == pytest.approx(1.0)
