"""Fetch per-chart engagement / popularity metrics from osu! API v2.

These fields don't live on the training manifest — they're used by
corpus analysis to weight charts by popularity, slice by mapper /
genre / language, and answer "what's IDEAL, not just AVERAGE" from
experiment #003.

Output lands as a JSON sidecar next to the manifest at
``manifest_engagement.json`` and a flat CSV at
``manifest_engagement.csv``. Both are keyed by ``beatmap_id``.

Fields pulled from `/api/v2/beatmaps?ids[]=...` (batched 50 at a time
by the shared `OsuV2Client.get_beatmaps`):

Per-beatmap (difficulty-specific):
  - playcount / passcount     → `pass_rate = passcount / playcount`
  - status                    ranked / loved / qualified / graveyard …
  - last_updated              iso date

Per-beatmapset (same for every diff in a set — each row carries a copy):
  - play_count_set            aggregate plays across the set
  - favourite_count
  - ratings buckets (11-elem 0..10) → `rating_mean`, `rating_count`
  - bpm_set                   mapper-declared BPM
  - genre, language
  - user_id                   mapper id
  - nominations_current
  - nsfw

Missing / unreachable beatmaps are simply absent from the result —
callers decide how to handle absence (same policy as `fetch_stars`).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..persistence.manifest import load_manifest
from .client import BATCH_SIZE, OsuV2Client


@dataclass(frozen=True, slots=True)
class EngagementRow:
    """Engagement / popularity metrics for one difficulty."""
    beatmap_id: str
    beatmapset_id: str

    # Per-difficulty
    playcount: int = 0
    passcount: int = 0
    pass_rate: float = 0.0
    status: str = ""
    last_updated: str = ""

    # Per-beatmapset (copied per row)
    play_count_set: int = 0
    favourite_count: int = 0
    rating_buckets: tuple[int, ...] = ()
    rating_mean: float | None = None
    rating_count: int = 0
    bpm_set: float | None = None
    genre: str | None = None
    language: str | None = None
    user_id: str | None = None
    nominations_current: int | None = None
    nsfw: bool | None = None


def _parse_one(beatmap: dict[str, Any]) -> EngagementRow | None:
    """Convert one `/api/v2/beatmaps` response dict into an `EngagementRow`.
    Returns None if the payload is missing required ids."""
    bid = beatmap.get("id")
    if bid is None:
        return None
    beatmapset = beatmap.get("beatmapset") or {}
    bsid = beatmap.get("beatmapset_id") or beatmapset.get("id")

    playcount = int(beatmap.get("playcount") or 0)
    passcount = int(beatmap.get("passcount") or 0)
    pass_rate = passcount / playcount if playcount > 0 else 0.0

    # Rating buckets are a length-11 list (ratings[0] through ratings[10]).
    # Some sets return empty / None. `rating_mean` = sum(i * ratings[i]) /
    # total; `rating_count` = total — both useful as popularity weights.
    raw_ratings = beatmapset.get("ratings") or []
    buckets = tuple(int(x) for x in raw_ratings)
    rating_count = sum(buckets)
    rating_mean: float | None = None
    if rating_count > 0:
        rating_mean = round(
            sum(i * c for i, c in enumerate(buckets)) / rating_count, 4,
        )

    def _genre() -> str | None:
        g = beatmapset.get("genre")
        return g.get("name") if isinstance(g, dict) else None

    def _language() -> str | None:
        lang = beatmapset.get("language")
        return lang.get("name") if isinstance(lang, dict) else None

    nominations = beatmapset.get("nominations_summary") or {}
    nominations_current = (
        int(nominations["current"])
        if isinstance(nominations, dict) and "current" in nominations
        else None
    )

    return EngagementRow(
        beatmap_id=str(bid),
        beatmapset_id=str(bsid) if bsid is not None else "",
        playcount=playcount,
        passcount=passcount,
        pass_rate=round(pass_rate, 4),
        status=str(beatmap.get("status") or ""),
        last_updated=str(beatmap.get("last_updated") or ""),
        play_count_set=int(beatmapset.get("play_count") or 0),
        favourite_count=int(beatmapset.get("favourite_count") or 0),
        rating_buckets=buckets,
        rating_mean=rating_mean,
        rating_count=rating_count,
        bpm_set=(
            float(beatmapset["bpm"])
            if beatmapset.get("bpm") is not None else None
        ),
        genre=_genre(),
        language=_language(),
        user_id=(
            str(beatmapset["user_id"])
            if beatmapset.get("user_id") is not None else None
        ),
        nominations_current=nominations_current,
        nsfw=(
            bool(beatmapset["nsfw"])
            if "nsfw" in beatmapset else None
        ),
    )


def fetch_engagement(
    beatmap_ids: list[str],
    client: OsuV2Client | None = None,
    *,
    progress: bool = True,
) -> dict[str, EngagementRow]:
    """Return ``{beatmap_id: EngagementRow}`` for the given ids. Ids
    that fail to return a beatmap (unknown, deleted, API error) are
    silently absent from the result."""
    client = client or OsuV2Client()
    unique = sorted({str(b) for b in beatmap_ids if b})

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(
                total=len(unique), desc="Fetching engagement", unit="map",
            )
        except ImportError:
            pass

    out: dict[str, EngagementRow] = {}
    for i in range(0, len(unique), BATCH_SIZE):
        batch = unique[i: i + BATCH_SIZE]
        try:
            beatmaps = client.get_beatmaps(batch)
        except Exception as exc:  # pragma: no cover — best-effort fetch
            print(f"  batch {i // BATCH_SIZE + 1} failed: {exc}")
            if bar is not None:
                bar.update(len(batch))
            continue
        for bm in beatmaps:
            row = _parse_one(bm)
            if row is not None:
                out[row.beatmap_id] = row
        if bar is not None:
            bar.update(len(batch))
    if bar is not None:
        bar.close()
    return out


# ─────────────────────────── sidecar IO ──────────────────────────────

_JSON_NAME = "manifest_engagement.json"
_CSV_NAME = "manifest_engagement.csv"


def _row_to_json(row: EngagementRow) -> dict[str, Any]:
    d = asdict(row)
    d["rating_buckets"] = list(d["rating_buckets"])
    return d


_CSV_FIELDS: tuple[str, ...] = (
    "beatmap_id", "beatmapset_id",
    "playcount", "passcount", "pass_rate", "status", "last_updated",
    "play_count_set", "favourite_count",
    "rating_mean", "rating_count",
    "bpm_set", "genre", "language", "user_id",
    "nominations_current", "nsfw",
)


def write_engagement_sidecar(
    manifest_path: Path,
    client: OsuV2Client | None = None,
    *,
    progress: bool = True,
) -> tuple[Path, Path]:
    """Fetch engagement rows for every beatmap_id in the manifest and
    write them to `manifest_engagement.{json,csv}` alongside the
    manifest. Returns the two output paths.
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    ids = [c.beatmap_id for c in manifest.charts if c.beatmap_id]
    rows = fetch_engagement(ids, client=client, progress=progress)

    out_dir = manifest_path.parent
    json_path = out_dir / _JSON_NAME
    csv_path = out_dir / _CSV_NAME

    payload = {
        "dataset": manifest.name,
        "n_charts_in_manifest": len(manifest.charts),
        "n_rows_fetched": len(rows),
        "rows": [_row_to_json(r) for r in rows.values()],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows.values():
            d = asdict(row)
            d.pop("rating_buckets", None)  # not a scalar
            for k, v in d.items():
                if v is None:
                    d[k] = ""
            writer.writerow({k: d.get(k, "") for k in _CSV_FIELDS})

    return json_path, csv_path


def load_engagement_sidecar(
    manifest_path: Path,
) -> dict[str, EngagementRow]:
    """Reverse of `write_engagement_sidecar`. Returns `{beatmap_id: row}`
    read from the JSON sidecar, or an empty dict if no sidecar exists."""
    path = Path(manifest_path).parent / _JSON_NAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, EngagementRow] = {}
    for d in data.get("rows", []):
        buckets = tuple(int(x) for x in d.get("rating_buckets") or [])
        row = EngagementRow(
            beatmap_id=str(d["beatmap_id"]),
            beatmapset_id=str(d.get("beatmapset_id", "")),
            playcount=int(d.get("playcount", 0)),
            passcount=int(d.get("passcount", 0)),
            pass_rate=float(d.get("pass_rate", 0.0)),
            status=str(d.get("status", "")),
            last_updated=str(d.get("last_updated", "")),
            play_count_set=int(d.get("play_count_set", 0)),
            favourite_count=int(d.get("favourite_count", 0)),
            rating_buckets=buckets,
            rating_mean=d.get("rating_mean"),
            rating_count=int(d.get("rating_count", 0)),
            bpm_set=d.get("bpm_set"),
            genre=d.get("genre"),
            language=d.get("language"),
            user_id=d.get("user_id"),
            nominations_current=d.get("nominations_current"),
            nsfw=d.get("nsfw"),
        )
        out[row.beatmap_id] = row
    return out
