"""Tests for `fetch.engagement` — response parsing, batching, sidecar IO.

No live API calls: we inject a fake client that records requested ids
and returns scripted beatmap payloads. Covers the happy path plus
missing-field tolerance so the parser doesn't crash on sparse /
older beatmaps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osu.taiko2.fetch.engagement import (
    EngagementRow,
    _parse_one,
    fetch_engagement,
    load_engagement_sidecar,
    write_engagement_sidecar,
)


class _FakeClient:
    """Records requested id batches and returns scripted responses keyed
    by string id. Unknown ids are absent from the response, mimicking
    the real API's behavior on deleted / private beatmaps."""

    def __init__(self, responses: dict[str, dict[str, Any]]):
        self._responses = responses
        self.calls: list[list[str]] = []

    def get_beatmaps(self, ids: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(ids))
        return [self._responses[i] for i in ids if i in self._responses]


def _beatmap_payload(
    bid: str, *,
    set_id: str = "4242",
    playcount: int = 10_000,
    passcount: int = 3_000,
    favourite_count: int = 500,
    play_count_set: int = 40_000,
    ratings: list[int] | None = None,
    bpm: float | None = 180.0,
    genre: str | None = "Anime",
    language: str | None = "Japanese",
    user_id: int | None = 1234,
    status: str = "ranked",
    nsfw: bool = False,
    nominations_current: int | None = 2,
) -> dict[str, Any]:
    """Build a minimal but realistic `/beatmaps` response payload."""
    if ratings is None:
        ratings = [0, 1, 0, 2, 5, 10, 20, 40, 80, 150, 200]
    return {
        "id": bid,
        "beatmapset_id": set_id,
        "playcount": playcount,
        "passcount": passcount,
        "status": status,
        "last_updated": "2024-01-15T12:00:00Z",
        "beatmapset": {
            "id": set_id,
            "play_count": play_count_set,
            "favourite_count": favourite_count,
            "ratings": ratings,
            "bpm": bpm,
            "genre": {"id": 3, "name": genre} if genre else None,
            "language": {"id": 3, "name": language} if language else None,
            "user_id": user_id,
            "nsfw": nsfw,
            "nominations_summary": {
                "current": nominations_current, "required": 2,
            } if nominations_current is not None else None,
        },
    }


class TestParseOne:
    def test_full_payload(self):
        row = _parse_one(_beatmap_payload("100"))
        assert row is not None
        assert row.beatmap_id == "100"
        assert row.beatmapset_id == "4242"
        assert row.playcount == 10_000
        assert row.passcount == 3_000
        assert row.pass_rate == 0.3
        assert row.status == "ranked"
        assert row.play_count_set == 40_000
        assert row.favourite_count == 500
        # rating_mean = sum(i * ratings[i]) / total
        expected_total = 0 + 1 + 0 + 2 + 5 + 10 + 20 + 40 + 80 + 150 + 200
        expected_num = sum(i * r for i, r in enumerate(
            [0, 1, 0, 2, 5, 10, 20, 40, 80, 150, 200],
        ))
        assert row.rating_count == expected_total
        assert row.rating_mean == pytest.approx(
            expected_num / expected_total, abs=1e-4,
        )
        assert row.bpm_set == 180.0
        assert row.genre == "Anime"
        assert row.language == "Japanese"
        assert row.user_id == "1234"
        assert row.nominations_current == 2
        assert row.nsfw is False

    def test_zero_playcount_gives_zero_pass_rate(self):
        row = _parse_one(_beatmap_payload("200", playcount=0, passcount=0))
        assert row.pass_rate == 0.0

    def test_empty_ratings_gives_none_mean(self):
        row = _parse_one(_beatmap_payload("300", ratings=[]))
        assert row.rating_buckets == ()
        assert row.rating_count == 0
        assert row.rating_mean is None

    def test_missing_optional_fields(self):
        # Sparse payload — older beatmaps don't always carry every
        # field; parser must not crash or invent defaults.
        row = _parse_one({
            "id": "999",
            "beatmapset_id": "1000",
            "playcount": 500,
            "passcount": 200,
            "beatmapset": {
                "id": "1000",
                "ratings": [],
            },
        })
        assert row is not None
        assert row.beatmap_id == "999"
        assert row.bpm_set is None
        assert row.genre is None
        assert row.language is None
        assert row.user_id is None
        assert row.nominations_current is None
        assert row.nsfw is None
        assert row.rating_mean is None

    def test_missing_id_returns_none(self):
        assert _parse_one({"beatmapset_id": "x"}) is None


class TestFetchEngagement:
    def test_returns_dict_keyed_by_beatmap_id(self):
        responses = {
            "1": _beatmap_payload("1"),
            "2": _beatmap_payload("2", playcount=99, passcount=11),
        }
        client = _FakeClient(responses)
        out = fetch_engagement(["1", "2"], client=client, progress=False)
        assert set(out) == {"1", "2"}
        assert out["2"].playcount == 99
        assert out["2"].passcount == 11

    def test_deduplicates_input_ids(self):
        client = _FakeClient({"5": _beatmap_payload("5")})
        fetch_engagement(["5", "5", "5"], client=client, progress=False)
        # Only one batch, requesting "5" exactly once.
        assert client.calls == [["5"]]

    def test_missing_ids_absent_from_result(self):
        client = _FakeClient({"10": _beatmap_payload("10")})
        out = fetch_engagement(["10", "unknown"], client=client, progress=False)
        assert set(out) == {"10"}


class TestSidecarIO:
    def _fake_manifest(
        self, tmp_path: Path, beatmap_ids: list[str],
    ) -> Path:
        """Write a minimal valid manifest.json so `load_manifest`
        succeeds. Uses the same serialization path the real code does."""
        from dataclasses import replace
        from osu.taiko2.domain.dataset import (
            ChartEntry, DatasetManifest, MelSamplerConfig,
        )
        from osu.taiko2.persistence.manifest import save_manifest
        cfg = MelSamplerConfig(
            sample_rate=22000, n_fft=2048, hop_divisor=200, n_mels=80,
        )
        charts = tuple(
            ChartEntry(
                chart_id=f"chart-{bid}",
                beatmap_id=bid,
                beatmapset_id="set",
                artist="a", title="t", difficulty_version="Oni",
                overall_difficulty=5.0, star_rating=None,
                density_mean=0.0, density_peak=0, density_std=0.0,
                duration_s=0.0, total_events=0,
                audio_filename="audio.mp3",
                features_path=tmp_path / f"{bid}.npy",
                n_frames=100,
            )
            for bid in beatmap_ids
        )
        m = DatasetManifest(
            name="test_ds", created_at="2024-01-01", sampler_config=cfg,
            charts=charts,
        )
        path = tmp_path / "manifest.json"
        save_manifest(m, path)
        return path

    def test_write_then_load_roundtrip(self, tmp_path: Path):
        manifest_path = self._fake_manifest(tmp_path, ["1", "2", "3"])
        responses = {
            "1": _beatmap_payload("1"),
            "2": _beatmap_payload("2", playcount=99, ratings=[0]*11),
            # "3" is missing from responses on purpose.
        }
        client = _FakeClient(responses)
        json_path, csv_path = write_engagement_sidecar(
            manifest_path, client=client, progress=False,
        )
        assert json_path.exists() and csv_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["n_charts_in_manifest"] == 3
        assert payload["n_rows_fetched"] == 2

        loaded = load_engagement_sidecar(manifest_path)
        assert set(loaded) == {"1", "2"}
        # Round-trip preserves scalar fields.
        assert loaded["1"].playcount == 10_000
        assert loaded["2"].playcount == 99
        assert loaded["2"].rating_mean is None  # all-zero ratings

    def test_csv_schema(self, tmp_path: Path):
        manifest_path = self._fake_manifest(tmp_path, ["7"])
        client = _FakeClient({"7": _beatmap_payload("7")})
        _, csv_path = write_engagement_sidecar(
            manifest_path, client=client, progress=False,
        )
        import csv
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        assert len(rows) == 1
        r = rows[0]
        # Raw rating buckets list must NOT be in the CSV (non-scalar).
        assert "rating_buckets" not in r
        # Every declared CSV field is present.
        for key in (
            "beatmap_id", "playcount", "passcount", "pass_rate",
            "favourite_count", "rating_mean", "rating_count",
            "bpm_set", "genre", "language", "user_id",
        ):
            assert key in r
