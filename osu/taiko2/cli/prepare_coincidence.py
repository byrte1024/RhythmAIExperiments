"""Precompute coincidence map summaries for a dataset.

Walks the .osz packs, extracts audio, runs the coincidence map
pipeline, and saves 13-row summaries as .npy alongside the existing
mel features.

Optimizations:
  - Pre-builds an osz index (beatmapset_id -> path) once upfront.
  - Groups all charts per audio file — each audio decoded once.
  - Parallel workers via multiprocessing for CPU-bound coincidence
    computation.
  - Skips already-computed files.
  - Imports once at module level.

Usage::

    osu/taiko2/.venv/bin/python -m osu.taiko2.cli.prepare_coincidence \
        --dataset taiko2_v1 \
        --charts-dir /home/drore/charts/repos/BeatDetector/osu/taiko/charts/ \
        --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from ..domain.coincidence import compute_summary


def _compute_summary_from_waveform(
    y: np.ndarray, sr: int, hop_length: int,
) -> np.ndarray:
    """Coincidence map → 13-row summary from a pre-loaded waveform."""
    return compute_summary(y, sr, hop_length=hop_length).astype(np.float32)


def _extract_audio_bytes(osz_path: Path, audio_filename: str) -> bytes | None:
    """Read audio bytes from an .osz without writing to disk."""
    try:
        with zipfile.ZipFile(osz_path, "r") as zf:
            # Exact match.
            for name in zf.namelist():
                if name == audio_filename or name.lower() == audio_filename.lower():
                    return zf.read(name)
            # Fallback: any audio.
            for name in zf.namelist():
                if name.lower().endswith((".mp3", ".ogg", ".wav", ".flac")):
                    return zf.read(name)
    except (zipfile.BadZipFile, OSError):
        pass
    return None


def _process_one_set(
    bms_id: str,
    osz_path: Path,
    audio_filename: str,
    stems_and_mel_T: list[tuple[str, int]],
    coin_dir: Path,
    sr: int,
    hop_length: int,
) -> tuple[int, int, str | None]:
    """Process one beatmapset. Returns (n_saved, n_skipped, error_msg)."""
    import librosa
    import io

    audio_bytes = _extract_audio_bytes(osz_path, audio_filename)
    if audio_bytes is None:
        return 0, len(stems_and_mel_T), None

    # Decode audio in memory — no temp file.
    try:
        import soundfile as sf
        with io.BytesIO(audio_bytes) as buf:
            try:
                y, sr_out = sf.read(buf, dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr_out != sr:
                    y = librosa.resample(y, orig_sr=sr_out, target_sr=sr)
            except Exception:
                # soundfile can't handle mp3 — fall back to librosa via temp file.
                buf.seek(0)
                ext = Path(audio_filename).suffix or ".mp3"
                fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="coin_")
                os.close(fd)
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(audio_bytes)
                    y, _ = librosa.load(tmp_path, sr=sr, mono=True)
                finally:
                    os.unlink(tmp_path)
    except Exception as e:
        return 0, len(stems_and_mel_T), f"{bms_id}: audio decode failed: {e}"

    try:
        summary = _compute_summary_from_waveform(y, sr, hop_length)
    except Exception as e:
        return 0, len(stems_and_mel_T), f"{bms_id}: coincidence failed: {e}"

    saved = 0
    skipped = 0
    coin_T = summary.shape[1]

    for stem, mel_T in stems_and_mel_T:
        out_path = coin_dir / f"{stem}.npy"
        if out_path.exists():
            skipped += 1
            continue
        if coin_T >= mel_T:
            matched = summary[:, :mel_T]
        else:
            matched = np.zeros((summary.shape[0], mel_T), dtype=np.float32)
            matched[:, :coin_T] = summary
        np.save(out_path, matched.astype(np.float16))
        saved += 1

    return saved, skipped, None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Precompute coincidence summaries for a dataset.",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--charts-dir", type=Path, required=True,
                   help="Directory containing .osz files.")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--hop-length", type=int, default=110)
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for coincidence computation.")
    args = p.parse_args(argv)

    ds_root = args.datasets_dir / args.dataset
    if not ds_root.is_dir():
        print(f"ERROR: dataset not found: {ds_root}", file=sys.stderr)
        return 2

    coin_dir = ds_root / "coincidence"
    coin_dir.mkdir(exist_ok=True)

    from ..persistence.manifest import load_manifest
    manifest = load_manifest(ds_root / "manifest.json")

    # Pre-build osz index: beatmapset_id -> osz path.
    print("Building .osz index...")
    osz_index: dict[str, Path] = {}
    for f in args.charts_dir.iterdir():
        if f.suffix.lower() == ".osz":
            bms_id = f.name.split(" ", 1)[0]
            osz_index[bms_id] = f
    print(f"  {len(osz_index)} .osz files indexed")

    # Group charts by beatmapset_id and collect (stem, mel_T) per set.
    by_set: dict[str, dict[str, Any]] = {}
    features_dir = ds_root / "features"
    for entry in manifest.charts:
        bms_id = entry.beatmapset_id
        stem = PureWindowsPath(entry.features_path).stem
        mel_path = features_dir / f"{stem}.npy"
        if not mel_path.exists():
            continue
        if bms_id not in by_set:
            by_set[bms_id] = {
                "audio_filename": entry.audio_filename,
                "stems": [],
            }
        # Get mel T without loading the full array.
        mel_header = np.load(mel_path, mmap_mode="r")
        mel_T = mel_header.shape[1]
        by_set[bms_id]["stems"].append((stem, mel_T))

    # Filter to sets that have an osz and aren't fully done.
    jobs: list[tuple[str, Path, str, list[tuple[str, int]]]] = []
    already_done = 0
    no_osz = 0
    for bms_id, info in by_set.items():
        if bms_id not in osz_index:
            no_osz += len(info["stems"])
            continue
        stems = info["stems"]
        pending = [(s, t) for s, t in stems if not (coin_dir / f"{s}.npy").exists()]
        if not pending:
            already_done += len(stems)
            continue
        jobs.append((bms_id, osz_index[bms_id], info["audio_filename"], pending))

    total_pending = sum(len(j[3]) for j in jobs)
    print(f"  {len(manifest.charts)} charts in manifest")
    print(f"  {already_done} already computed")
    print(f"  {no_osz} no .osz found")
    print(f"  {len(jobs)} sets to process ({total_pending} charts)")

    if not jobs:
        print("Nothing to do.")
        return 0

    total_saved = 0
    total_skipped = 0
    total_errors = 0

    if args.workers <= 1:
        # Serial — simpler, better error messages.
        try:
            from tqdm.auto import tqdm
            job_iter = tqdm(jobs, desc="coincidence", unit="set")
        except ImportError:
            job_iter = jobs

        for bms_id, osz_path, audio_fn, stems in job_iter:
            saved, skipped, err = _process_one_set(
                bms_id, osz_path, audio_fn, stems,
                coin_dir, args.sr, args.hop_length,
            )
            total_saved += saved
            total_skipped += skipped
            if err:
                total_errors += 1
                if total_errors <= 10:
                    print(f"  ERROR: {err}")
    else:
        # Parallel.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_one_set,
                    bms_id, osz_path, audio_fn, stems,
                    coin_dir, args.sr, args.hop_length,
                ): bms_id
                for bms_id, osz_path, audio_fn, stems in jobs
            }
            try:
                from tqdm.auto import tqdm
                fut_iter = tqdm(
                    as_completed(futures), total=len(futures),
                    desc="coincidence", unit="set",
                )
            except ImportError:
                fut_iter = as_completed(futures)

            for fut in fut_iter:
                try:
                    saved, skipped, err = fut.result()
                    total_saved += saved
                    total_skipped += skipped
                    if err:
                        total_errors += 1
                        if total_errors <= 10:
                            print(f"  ERROR: {err}")
                except Exception as e:
                    total_errors += 1
                    if total_errors <= 10:
                        print(f"  WORKER ERROR: {e}")

    print(f"\nDone. Saved: {total_saved}, Skipped: {total_skipped}, Errors: {total_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
