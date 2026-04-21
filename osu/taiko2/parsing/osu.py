"""Parse a single .osu file's text into a Track.

Referenced against the working logic in osu/taiko/create_dataset.py:
same section parser, same hit-object bitmask rules, same density formula.
"""
from __future__ import annotations

import os

from ..domain.beatmap import (
    AudioRef,
    Density,
    Difficulty,
    Onset,
    OnsetKind,
    Track,
)

TAIKO_MODE = 1


def _classify_hit_object(obj_type: int, hit_sound: int) -> OnsetKind:
    """Decode an osu! hit object's type+sound bits into an OnsetKind.

    Type bitmask: bit0=circle, bit1=slider (drumroll), bit3=spinner (denden).
    For circles, the hit-sound bitmask distinguishes don/ka and big (finish).
    """
    if obj_type & 1:
        is_ka = (hit_sound & 0x0A) != 0  # whistle (2) or clap (8)
        is_big = bool(hit_sound & 4)      # finish
        if is_big and is_ka:
            return OnsetKind.BIG_KA
        if is_big:
            return OnsetKind.BIG_DON
        return OnsetKind.KA if is_ka else OnsetKind.DON
    if obj_type & 2:
        return OnsetKind.DRUMROLL
    if obj_type & 8:
        return OnsetKind.SPINNER
    return OnsetKind.UNKNOWN


def compute_density(onsets: tuple[Onset, ...]) -> Density:
    """Per-second bucket density summary. Matches create_dataset.compute_density_stats.

    Only circle-type onsets (don/ka/big) contribute to density; drumrolls and
    spinners are included in `total_events` but not used to shape peak/std.
    Zero-duration or empty inputs return an all-zero Density.
    """
    if len(onsets) < 2:
        return Density(mean=0.0, peak=0, std=0.0, duration_s=0.0,
                       total_events=len(onsets))

    first_ms = onsets[0].time_ms
    last_ms = onsets[-1].time_ms
    duration_s = (last_ms - first_ms) / 1000.0
    if duration_s <= 0:
        return Density(mean=0.0, peak=0, std=0.0, duration_s=0.0,
                       total_events=len(onsets))

    n = len(onsets)
    mean = n / duration_s

    n_buckets = last_ms // 1000 + 1
    buckets = [0] * n_buckets
    for o in onsets:
        buckets[o.time_ms // 1000] += 1
    active = [b for b in buckets if b > 0]
    peak = max(buckets)
    if active:
        avg_active = sum(active) / len(active)
        var = sum((v - avg_active) ** 2 for v in active) / len(active)
        std = var ** 0.5
    else:
        std = 0.0

    return Density(
        mean=round(mean, 3),
        peak=peak,
        std=round(std, 3),
        duration_s=round(duration_s, 2),
        total_events=n,
    )


def parse_osu_text(text: str) -> Track | None:
    """Parse one .osu file's text.

    Returns a `Track` when the file is osu!taiko (mode=1) with at least one
    onset. Returns `None` for any other mode, malformed files, or empty
    charts — matching the filtering already used upstream.

    `Track.difficulty.star_rating` is `None`; star ratings come from a
    separate scraper and are merged in later.
    """
    meta: dict[str, str | float | int] = {}
    raw_onsets: list[Onset] = []
    section: str | None = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue

        if section == "[General]":
            if line.startswith("Mode:"):
                try:
                    meta["mode"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("AudioFilename:"):
                meta["audio"] = line.split(":", 1)[1].strip()

        elif section == "[Metadata]":
            if line.startswith("Title:"):
                meta["title"] = line.split(":", 1)[1].strip()
            elif line.startswith("Artist:"):
                meta["artist"] = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                meta["difficulty"] = line.split(":", 1)[1].strip()
            elif line.startswith("BeatmapID:"):
                meta["beatmap_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("BeatmapSetID:"):
                meta["beatmapset_id"] = line.split(":", 1)[1].strip()

        elif section == "[Difficulty]":
            if line.startswith("OverallDifficulty:"):
                try:
                    meta["od"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

        elif section == "[HitObjects]":
            parts = line.split(",")
            if len(parts) < 5:
                continue
            try:
                time_ms = int(parts[2])
                obj_type = int(parts[3])
                hit_sound = int(parts[4])
            except ValueError:
                continue
            raw_onsets.append(Onset(
                time_ms=time_ms,
                kind=_classify_hit_object(obj_type, hit_sound),
            ))

    if meta.get("mode") != TAIKO_MODE or not raw_onsets:
        return None

    audio_filename = str(meta.get("audio", "")).strip()
    if not audio_filename:
        return None

    onsets = tuple(raw_onsets)
    density = compute_density(onsets)

    audio_ext = os.path.splitext(audio_filename)[1].lstrip(".").lower()
    audio = AudioRef(filename=audio_filename, format=audio_ext)

    difficulty = Difficulty(
        version=str(meta.get("difficulty", "")),
        overall_difficulty=float(meta.get("od", 0.0)),
        star_rating=None,
    )

    return Track(
        beatmap_id=str(meta.get("beatmap_id", "")),
        artist=str(meta.get("artist", "")),
        title=str(meta.get("title", "")),
        difficulty=difficulty,
        audio=audio,
        onsets=onsets,
        density=density,
    )
