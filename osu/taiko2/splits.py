"""Deterministic N-way splitting by song.

Charts that share a `beatmapset_id` (same song, different difficulties)
always land in the same split. Prevents audio leakage where the model
has heard a song during training under a different difficulty label.

The primary entry point is `named_song_splits`, which accepts an ordered
sequence of `(name, ratio)` pairs and returns a dict of chart indices
per split. Ratios must sum to ≤ 1.0; any shortfall leaves those songs
unassigned.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .domain.dataset import DatasetManifest

SplitSpec = tuple[tuple[str, float], ...]

_RESERVED_NAMES: frozenset[str] = frozenset({"all"})
_SUM_TOLERANCE: float = 1e-6


def _validate_spec(spec: SplitSpec) -> None:
    if not spec:
        raise ValueError("split spec is empty")
    names = [n for n, _ in spec]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate split names in {names}")
    bad = [n for n in names if n in _RESERVED_NAMES]
    if bad:
        raise ValueError(f"reserved split names not allowed: {bad}")
    for name, ratio in spec:
        if not isinstance(name, str) or not name:
            raise ValueError(f"split name must be a non-empty string, got {name!r}")
        if ratio < 0:
            raise ValueError(f"split ratio for {name!r} is negative: {ratio}")
    total = sum(r for _, r in spec)
    if total > 1.0 + _SUM_TOLERANCE:
        raise ValueError(
            f"split ratios sum to {total:.6f}, must be ≤ 1.0"
        )


def named_song_splits(
    manifest: DatasetManifest,
    spec: SplitSpec,
    seed: int = 42,
) -> dict[str, tuple[int, ...]]:
    """Partition `manifest.charts` by `beatmapset_id` into N named buckets.

    Returns `{split_name: tuple[chart_index, ...]}`. Order within each
    tuple is the order songs were shuffled — not manifest order — so
    consumers should treat it as an unordered set unless they care about
    the (deterministic) shuffle position.

    If `sum(ratios) < 1.0`, the leftover songs are excluded from every
    returned bucket. If `sum(ratios) == 1.0` (within tolerance), the last
    bucket gets the remainder so no song is accidentally dropped to
    rounding.
    """
    _validate_spec(spec)

    song_to_charts: dict[str, list[int]] = {}
    for i, chart in enumerate(manifest.charts):
        key = chart.beatmapset_id if chart.beatmapset_id else f"__solo_{i}"
        song_to_charts.setdefault(key, []).append(i)

    songs = list(song_to_charts.keys())
    rng = random.Random(seed)
    rng.shuffle(songs)
    n_songs = len(songs)

    buckets: dict[str, list[int]] = {name: [] for name, _ in spec}
    total = sum(r for _, r in spec)
    sum_is_one = abs(total - 1.0) < _SUM_TOLERANCE

    cursor = 0
    for i, (name, ratio) in enumerate(spec):
        is_last = (i == len(spec) - 1)
        if is_last and sum_is_one:
            take_songs = songs[cursor:]
        else:
            n_take = int(round(ratio * n_songs))
            take_songs = songs[cursor:cursor + n_take]
            cursor += len(take_songs)
        for s in take_songs:
            buckets[name].extend(song_to_charts[s])

    return {name: tuple(idx) for name, idx in buckets.items()}


def chart_ids_for_split(
    manifest: DatasetManifest,
    split: str,
    spec: SplitSpec,
    seed: int,
) -> set[str]:
    """Return the set of `chart_id` values for the named `split`.

    `split="all"` ignores the spec/seed and returns every chart_id.
    Otherwise `split` must match one of the names in `spec`.
    """
    if split == "all":
        return {c.chart_id for c in manifest.charts}

    buckets = named_song_splits(manifest, spec, seed)
    if split not in buckets:
        raise ValueError(
            f"unknown split {split!r}; expected 'all' or one of "
            f"{sorted(buckets.keys())}"
        )
    idx = buckets[split]
    return {manifest.charts[i].chart_id for i in idx}


# ─────────────────────────── back-compat helper ────────────────────────

def song_based_split(
    manifest: DatasetManifest,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Convenience wrapper for the common train/val case.

    Equivalent to ``named_song_splits(manifest, spec, seed)`` with
    ``spec = (("train", 1 - val_ratio), ("val", val_ratio))`` — kept as a
    dedicated helper because 2-way splits are overwhelmingly the common
    case and this signature is easier to read.
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    if val_ratio == 0.0:
        buckets = named_song_splits(manifest, (("train", 1.0),), seed)
        return buckets["train"], ()

    spec: SplitSpec = (
        ("train", 1.0 - val_ratio),
        ("val", val_ratio),
    )
    buckets = named_song_splits(manifest, spec, seed)
    return buckets["train"], buckets["val"]
