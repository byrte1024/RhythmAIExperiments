"""Load an .osz archive into a Pack; extract audio bytes or waveforms.

Referenced against osu/taiko/create_dataset.py's `scan_all_osz` and
`load_audio_worker` — same zip iteration, same tempfile-based decode.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ..types.beatmap import AudioRef, Pack, Track
from .osu import parse_osu_text


def _pack_beatmapset_id(tracks: tuple[Track, ...]) -> str:
    """Most-common beatmapset_id across a pack's tracks (first wins on tie)."""
    counts: dict[str, int] = {}
    for t in tracks:
        counts[t.beatmapset_id] = counts.get(t.beatmapset_id, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _unique_audios(tracks: tuple[Track, ...]) -> tuple[AudioRef, ...]:
    seen: dict[str, AudioRef] = {}
    for t in tracks:
        if t.audio.filename not in seen:
            seen[t.audio.filename] = t.audio
    return tuple(seen.values())


def load_pack(osz_path: Path) -> Pack | None:
    """Parse every taiko chart in an .osz into a Pack.

    Returns None if the archive contains no taiko charts, is unreadable,
    or is not a valid zip.
    """
    osz_path = Path(osz_path)
    basename = osz_path.stem

    tracks: list[Track] = []
    try:
        with zipfile.ZipFile(osz_path) as z:
            for name in z.namelist():
                if not name.endswith(".osu"):
                    continue
                text = z.read(name).decode("utf-8", errors="replace")
                track = parse_osu_text(text)
                if track is not None:
                    tracks.append(track)
    except (zipfile.BadZipFile, OSError):
        return None

    if not tracks:
        return None

    tracks_t = tuple(tracks)
    return Pack(
        source_path=osz_path,
        basename=basename,
        beatmapset_id=_pack_beatmapset_id(tracks_t),
        tracks=tracks_t,
        audio_files=_unique_audios(tracks_t),
    )


def extract_audio_bytes(osz_path: Path, audio_filename: str) -> bytes:
    """Raw audio bytes for `audio_filename` inside `osz_path`.

    Raises `FileNotFoundError` if the entry is missing, `zipfile.BadZipFile`
    on a corrupt archive.
    """
    with zipfile.ZipFile(osz_path) as z:
        if audio_filename not in z.namelist():
            raise FileNotFoundError(
                f"{audio_filename!r} not found inside {osz_path}"
            )
        return z.read(audio_filename)


def load_audio_waveform(
    osz_path: Path, audio_filename: str, target_sr: int,
) -> tuple[np.ndarray, int]:
    """Extract `audio_filename` from `osz_path` and decode to a mono waveform.

    Returns `(waveform, sample_rate)`. Uses a tempfile because librosa's
    decode backends (soundfile, audioread) expect a filesystem path for
    most formats, particularly MP3.
    """
    import librosa

    audio_bytes = extract_audio_bytes(osz_path, audio_filename)
    ext = os.path.splitext(audio_filename)[1] or ".mp3"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return y.astype(np.float32), int(sr)
