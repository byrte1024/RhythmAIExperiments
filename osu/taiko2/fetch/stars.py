"""Fetch star ratings from osu! API v2 and merge them into a manifest."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..persistence.manifest import load_manifest, save_manifest
from ..types.dataset import DatasetManifest
from .client import BATCH_SIZE, OsuV2Client


def fetch_star_ratings(
    beatmap_ids: list[str],
    client: OsuV2Client | None = None,
    *,
    progress: bool = True,
) -> dict[str, float]:
    """Return `{beatmap_id: star_rating}` for the given ids.

    Missing/unknown ids are omitted from the result rather than carrying a
    sentinel — callers decide how to handle absence.
    """
    client = client or OsuV2Client()
    unique = sorted({str(b) for b in beatmap_ids if b})

    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(unique), desc="Fetching star ratings", unit="map")
        except ImportError:
            bar = None
    else:
        bar = None

    result: dict[str, float] = {}
    for i in range(0, len(unique), BATCH_SIZE):
        batch = unique[i : i + BATCH_SIZE]
        try:
            beatmaps = client.get_beatmaps(batch)
        except Exception as e:
            print(f"  batch {i // BATCH_SIZE + 1} failed: {e}")
            if bar is not None:
                bar.update(len(batch))
            continue

        for bm in beatmaps:
            bid = str(bm.get("id", ""))
            sr = bm.get("difficulty_rating")
            if bid and sr is not None:
                result[bid] = float(sr)
        if bar is not None:
            bar.update(len(batch))

    if bar is not None:
        bar.close()
    return result


def update_manifest_stars(
    manifest_path: Path,
    client: OsuV2Client | None = None,
    *,
    progress: bool = True,
) -> DatasetManifest:
    """Load a manifest, fetch star ratings for every chart, save it back.

    Charts whose beatmap_id is blank or unreachable keep their existing
    `star_rating` (which may be None). Returns the updated manifest.
    """
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)

    ids = [c.beatmap_id for c in manifest.charts if c.beatmap_id]
    stars = fetch_star_ratings(ids, client=client, progress=progress)

    updated_charts = tuple(
        replace(c, star_rating=stars[c.beatmap_id])
        if c.beatmap_id in stars
        else c
        for c in manifest.charts
    )
    updated = replace(manifest, charts=updated_charts)
    save_manifest(updated, manifest_path)
    return updated
