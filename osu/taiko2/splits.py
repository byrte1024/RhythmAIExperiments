"""Deterministic train/val splitting by song.

Charts that share a `beatmapset_id` (same song, different difficulties)
always land in the same split. Prevents audio leakage where the model
has heard a song during training under a different difficulty label.
"""
from __future__ import annotations

import random

from .domain.dataset import ChartEntry, DatasetManifest


def song_based_split(
    manifest: DatasetManifest,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return `(train_indices, val_indices)` into `manifest.charts`.

    Grouping key is `beatmapset_id`; charts with an empty id fall back
    to their own index (so they're treated as unique songs). Shuffling
    is seeded — identical `(manifest, val_ratio, seed)` triples produce
    identical splits on every run.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    song_to_charts: dict[str, list[int]] = {}
    for i, chart in enumerate(manifest.charts):
        key = chart.beatmapset_id if chart.beatmapset_id else f"__solo_{i}"
        song_to_charts.setdefault(key, []).append(i)

    songs = list(song_to_charts.keys())
    rng = random.Random(seed)
    rng.shuffle(songs)

    n_val_songs = max(1, int(len(songs) * val_ratio)) if val_ratio > 0 else 0
    val_songs = set(songs[:n_val_songs])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for song, charts in song_to_charts.items():
        target = val_idx if song in val_songs else train_idx
        target.extend(charts)

    return tuple(train_idx), tuple(val_idx)


def chart_ids_for_split(
    manifest: DatasetManifest,
    split: str,
    val_ratio: float,
    seed: int,
) -> set[str]:
    """Return the set of `chart_id` values belonging to `split`.

    `split` ∈ {"all", "train", "val"}. `"all"` ignores the ratio/seed and
    returns every chart_id.
    """
    if split == "all":
        return {c.chart_id for c in manifest.charts}

    train_idx, val_idx = song_based_split(manifest, val_ratio, seed)
    if split == "train":
        idx = train_idx
    elif split == "val":
        idx = val_idx
    else:
        raise ValueError(
            f"split must be 'all', 'train', or 'val'; got {split!r}"
        )
    return {manifest.charts[i].chart_id for i in idx}
