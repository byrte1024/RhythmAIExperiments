"""Save/load binned onset sequences.

Stored as .npz with three parallel arrays: `bins` (int32), `times_ms`
(int32), `kind_ids` (uint8 enum index). Keeping all three preserves enough
information to re-bin under a different EventSampler framerate without
losing kind data, while still allowing a fast `bins`-only load for models
that don't need kind.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.beatmap import OnsetBinned, OnsetKind

_KIND_ORDER: tuple[OnsetKind, ...] = (
    OnsetKind.DON,
    OnsetKind.KA,
    OnsetKind.BIG_DON,
    OnsetKind.BIG_KA,
    OnsetKind.DRUMROLL,
    OnsetKind.SPINNER,
    OnsetKind.UNKNOWN,
)
_KIND_TO_ID: dict[OnsetKind, int] = {k: i for i, k in enumerate(_KIND_ORDER)}


def save_events(onsets: tuple[OnsetBinned, ...], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not onsets:
        bins = np.zeros((0,), dtype=np.int32)
        times = np.zeros((0,), dtype=np.int32)
        kinds = np.zeros((0,), dtype=np.uint8)
    else:
        bins = np.asarray([o.bin for o in onsets], dtype=np.int32)
        times = np.asarray([o.time_ms for o in onsets], dtype=np.int32)
        kinds = np.asarray([_KIND_TO_ID[o.kind] for o in onsets], dtype=np.uint8)
    np.savez(path, bins=bins, times_ms=times, kind_ids=kinds)


def load_events(path: Path) -> tuple[OnsetBinned, ...]:
    with np.load(Path(path)) as data:
        bins = data["bins"]
        times = data["times_ms"]
        kinds = data["kind_ids"]
    return tuple(
        OnsetBinned(
            time_ms=int(t),
            kind=_KIND_ORDER[int(k)],
            bin=int(b),
        )
        for b, t, k in zip(bins, times, kinds)
    )


def load_event_bins(path: Path) -> np.ndarray:
    """Fast path for training: only return the bin array (int32)."""
    with np.load(Path(path)) as data:
        return data["bins"]
