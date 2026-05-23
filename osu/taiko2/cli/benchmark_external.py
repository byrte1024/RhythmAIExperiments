"""Benchmark external onset/chart generators against our GT dataset.

Runs an external model on a fraction of the val split, compares
generated onsets against GT charts using ``Chart.compare``, and
reports the same metrics as our AR corpus runner.

Supported backends:
  - ``mapperatorinator``: Runs Mapperatorinator/Mapperatorinator2 via
    its ``inference.py`` CLI. Requires the repo cloned and set up.
  - ``librosa``: librosa.onset.onset_detect (classical baseline).
  - ``madmom``: madmom CNN onset processor (neural baseline).

Usage::

    # Mapperatorinator2
    osu/taiko2/.venv/bin/python -m osu.taiko2.cli.benchmark_external \
        --backend mapperatorinator \
        --backend-path /home/drore/repos/Mapperatorinator2 \
        --dataset taiko2_v1 \
        --fraction 0.05 \
        --device cuda \
        --experiment-dir osu/taiko2/experiments/018-baselines

    # librosa (built-in)
    osu/taiko2/.venv/bin/python -m osu.taiko2.cli.benchmark_external \
        --backend librosa \
        --dataset taiko2_v1 \
        --fraction 0.05 \
        --experiment-dir osu/taiko2/experiments/018-baselines
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from ..domain.beatmap import OnsetKind
from ..domain.chart import Chart, gt_match_metrics
from ..persistence.manifest import load_manifest


def _find_audio_from_osz(
    chart_entry: "Any",
    charts_dir: Path,
) -> Path | None:
    """Find and extract audio from the .osz file matching a chart entry."""
    import zipfile

    # The .osz filename starts with the beatmapset_id.
    bms_id = chart_entry.beatmapset_id
    candidates = list(charts_dir.glob(f"{bms_id} *.osz"))
    if not candidates:
        return None

    osz_path = candidates[0]
    try:
        with zipfile.ZipFile(osz_path, "r") as zf:
            audio_name = chart_entry.audio_filename
            # Try exact match first, then case-insensitive.
            for name in zf.namelist():
                if name == audio_name or name.lower() == audio_name.lower():
                    data = zf.read(name)
                    ext = Path(name).suffix or ".mp3"
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=ext, delete=False, prefix="bench_audio_",
                    )
                    tmp.write(data)
                    tmp.close()
                    return Path(tmp.name)
            # Fallback: grab any audio file.
            for name in zf.namelist():
                low = name.lower()
                if low.endswith((".mp3", ".ogg", ".wav", ".flac")):
                    data = zf.read(name)
                    ext = Path(name).suffix
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=ext, delete=False, prefix="bench_audio_",
                    )
                    tmp.write(data)
                    tmp.close()
                    return Path(tmp.name)
    except (zipfile.BadZipFile, OSError):
        pass
    return None


def _pick_charts(total: int, fraction: float, seed: int) -> list[int]:
    n = max(1, int(round(total * fraction)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total, size=min(n, total), replace=False).tolist())


# ─────────────────────────── backends ────────────────────────────────


def _run_mapperatorinator(
    audio_path: Path,
    backend_path: Path,
    difficulty: float,
    gamemode: int,
    device: str,
) -> list[float]:
    """Run Mapperatorinator inference and return onset times in ms."""
    with tempfile.TemporaryDirectory(prefix="mapper_") as tmpdir:
        out_dir = Path(tmpdir) / "output"
        out_dir.mkdir()

        # Use Mapperatorinator's own venv Python, not ours.
        mapper_python = backend_path / ".venv" / "bin" / "python"
        if not mapper_python.exists():
            mapper_python = backend_path / ".venv" / "Scripts" / "python.exe"
        if not mapper_python.exists():
            mapper_python = Path(sys.executable)  # fallback

        cmd = [
            str(mapper_python), str(backend_path / "inference.py"),
            f"audio_path={audio_path}",
            f"output_path={out_dir}",
            f"gamemode={gamemode}",
            f"difficulty={difficulty}",
            "super_timing=false",
            "output_type=[TIMING,MAP]",
            "device=cpu",
        ]
        env = os.environ.copy()
        # Force CPU — Mapperatorinator's PyTorch build may not support
        # newer GPUs (sm_120 / RTX 5070).
        env["CUDA_VISIBLE_DEVICES"] = ""

        try:
            result = subprocess.run(
                cmd, cwd=str(backend_path), env=env,
                capture_output=True, text=True, timeout=600,
            )
        except subprocess.TimeoutExpired:
            print("    WARN: mapperatorinator timed out (600s)")
            return []
        if result.returncode != 0:
            # Only warn if stderr has something beyond the CUDA compat warning.
            err = result.stderr or ""
            non_warn = [l for l in err.splitlines()
                        if l.strip() and "UserWarning" not in l
                        and "NVIDIA" not in l and "sm_" not in l
                        and "warnings.warn" not in l]
            if non_warn:
                print(f"    WARN: mapperatorinator failed (rc={result.returncode}): {non_warn[0][:200]}")

        # Parse generated .osu files for onset times regardless of returncode.
        osu_files = list(out_dir.glob("*.osu"))
        if not osu_files:
            return []

        onsets_ms: list[float] = []
        for osu_file in osu_files:
            for line in osu_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line and line[0].isdigit() and "," in line:
                    parts = line.split(",")
                    if len(parts) >= 4:
                        try:
                            t = float(parts[2])
                            onsets_ms.append(t)
                        except ValueError:
                            pass
        return sorted(set(onsets_ms))


def _run_librosa(audio_path: Path, sr: int = 22050) -> list[float]:
    """Run librosa onset detection and return onset times in ms."""
    import librosa
    y, sr_out = librosa.load(str(audio_path), sr=sr, mono=True)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr_out, hop_length=512, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr_out, hop_length=512)
    return [float(t * 1000) for t in onset_times]


def _run_madmom(audio_path: Path) -> list[float]:
    """Run madmom CNN onset detection and return onset times in ms."""
    try:
        import madmom
    except ImportError:
        print("ERROR: madmom not installed. pip install madmom", file=sys.stderr)
        return []
    proc = madmom.features.onsets.CNNOnsetProcessor()
    act = proc(str(audio_path))
    pp = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=0.3)
    onsets_s = pp(act)
    return [float(t * 1000) for t in onsets_s]


def _run_beatthis(
    audio_path: Path,
    _cache: dict[str, object] = {},
) -> list[float]:
    """Run BeatThis! beat tracker and return beat times in ms."""
    try:
        from beat_this.inference import File2Beats
    except ImportError:
        print("ERROR: beat-this not installed. pip install beat-this", file=sys.stderr)
        return []
    if "model" not in _cache:
        _cache["model"] = File2Beats(
            checkpoint_path="final0", device="cuda", dbn=False,
        )
    file2beats = _cache["model"]
    beats, downbeats = file2beats(str(audio_path))
    # beats is an array of times in seconds.
    # Return all beats (not just downbeats) as onset candidates.
    all_times = sorted(set(
        [float(t * 1000) for t in beats]
        + [float(t * 1000) for t in downbeats]
    ))
    return all_times


# ─────────────────────────── main ────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Benchmark external onset detectors against GT.",
    )
    p.add_argument("--backend", required=True,
                   choices=["mapperatorinator", "librosa", "madmom", "beatthis"],
                   help="Which external model to benchmark.")
    p.add_argument("--backend-path", type=Path, default=None,
                   help="Path to cloned repo (for mapperatorinator).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--difficulty", type=float, default=4.0,
                   help="Target difficulty for mapperatorinator.")
    p.add_argument("--charts-dir", type=Path, default=None,
                   help="Directory containing .osz files (for audio extraction).")
    p.add_argument("--experiment-dir", type=Path, default=None)
    args = p.parse_args(argv)

    ds_root = args.datasets_dir / args.dataset
    if not ds_root.is_dir():
        print(f"ERROR: dataset not found: {ds_root}", file=sys.stderr)
        return 2

    # Load val sampler.
    data_cfg = TaikoDetectionSamplerConfig(
        batch_size=64, a_bins=500, b_bins=500, c_events=128, d_events=100,
        min_cursor_bin=6000,
        split_ratios=(("train", 0.9), ("val", 0.1)),
        split_seed=42,
    )
    data_cfg = replace(data_cfg, split="val", dataset_root=ds_root)
    sampler = TaikoDetectionSampler(data_cfg)
    sampler.load_data(progress=False)
    total = sampler.count_charts()
    selected = _pick_charts(total, args.fraction, args.seed)
    print(f"Backend: {args.backend}")
    print(f"Val charts: {total}, selected: {len(selected)}")

    # We need audio files. The sampler has features but not audio.
    # We need to reconstruct audio paths from the manifest.
    manifest = load_manifest(ds_root / "manifest.json")

    results: list[dict[str, Any]] = []
    all_comparisons: list[dict[str, float]] = []

    try:
        from tqdm.auto import tqdm
        chart_iter = tqdm(selected, desc=args.backend, unit="chart")
    except ImportError:
        chart_iter = selected

    if args.backend == "mapperatorinator" and args.backend_path is None:
        print("ERROR: --backend-path required for mapperatorinator", file=sys.stderr)
        return 2

    charts_dir = args.charts_dir
    skipped_no_audio = 0

    for idx in chart_iter:
        gt_chart = sampler.get_chart(idx)
        gt_ms = np.array([o.time_ms for o in gt_chart.track.onsets], dtype=np.float64)

        if len(gt_ms) < 5:
            continue

        # Extract audio: try chart.audio first, fall back to .osz packs.
        audio_path: Path | None = None
        needs_cleanup = False

        if gt_chart.audio is not None:
            ext = getattr(gt_chart.track, "audio", None)
            ext = getattr(ext, "format", "mp3") if ext else "mp3"
            tmp = tempfile.NamedTemporaryFile(
                suffix=f".{ext}", delete=False, prefix="bench_",
            )
            tmp.write(gt_chart.audio)
            tmp.close()
            audio_path = Path(tmp.name)
            needs_cleanup = True
        elif charts_dir is not None:
            entry = sampler._chart_entries[idx]
            audio_path = _find_audio_from_osz(entry, charts_dir)
            needs_cleanup = audio_path is not None

        if audio_path is None:
            skipped_no_audio += 1
            continue

        try:
            if args.backend == "mapperatorinator":
                pred_ms = _run_mapperatorinator(
                    audio_path, args.backend_path,
                    difficulty=args.difficulty,
                    gamemode=1, device=args.device,
                )
            elif args.backend == "librosa":
                pred_ms = _run_librosa(audio_path)
            elif args.backend == "madmom":
                pred_ms = _run_madmom(audio_path)
            elif args.backend == "beatthis":
                pred_ms = _run_beatthis(audio_path)
            else:
                pred_ms = []
        finally:
            if needs_cleanup and audio_path is not None:
                audio_path.unlink(missing_ok=True)

        if not pred_ms:
            continue

        pred_arr = np.array(pred_ms, dtype=np.float64)

        # Build a Chart from the predicted onsets for full comparison.
        from ..domain.beatmap import AudioRef, Onset, OnsetKind, Track
        from ..domain.chart import Chart
        from ..parsing.osu import compute_density

        pred_onsets = tuple(
            Onset(time_ms=int(round(t)), kind=OnsetKind.DON)
            for t in sorted(set(pred_ms))
        )
        gt_onsets = gt_chart.track.onsets

        if pred_onsets:
            pred_density = compute_density(pred_onsets)
            pred_track = Track(
                beatmap_id=gt_chart.track.beatmap_id,
                beatmapset_id=gt_chart.track.beatmapset_id,
                artist=gt_chart.track.artist,
                title=gt_chart.track.title,
                difficulty=gt_chart.track.difficulty,
                audio=gt_chart.track.audio,
                onsets=pred_onsets,
                density=pred_density,
            )
            pred_chart = Chart(track=pred_track, audio=None)
            comparison = pred_chart.compare(gt_chart)
            pred_metrics = pred_chart.calculate_metrics()

            chart_result: dict[str, Any] = {
                "chart_id": gt_chart.track.beatmap_id,
                "artist": gt_chart.track.artist,
                "title": gt_chart.track.title,
                "n_gt": len(gt_onsets),
                "n_pred": len(pred_onsets),
            }
            # Comparison metrics.
            for f in comparison.__dataclass_fields__:
                v = getattr(comparison, f)
                if isinstance(v, (int, float)):
                    chart_result[f"cmp/{f}"] = v
            # Predicted chart's own metrics.
            for f in pred_metrics.__dataclass_fields__:
                v = getattr(pred_metrics, f)
                if isinstance(v, (int, float)):
                    chart_result[f"pred/{f}"] = v
            # GT chart metrics for reference.
            gt_metrics = gt_chart.calculate_metrics()
            for f in gt_metrics.__dataclass_fields__:
                v = getattr(gt_metrics, f)
                if isinstance(v, (int, float)):
                    chart_result[f"gt/{f}"] = v

            all_comparisons.append(chart_result)

            # Save predicted chart as .osu for inspection.
            if args.experiment_dir:
                charts_out = args.experiment_dir / f"charts_{args.backend}"
                charts_out.mkdir(parents=True, exist_ok=True)
                safe_name = (
                    f"{gt_chart.track.beatmapset_id}_"
                    f"{gt_chart.track.difficulty.version}"
                ).replace("/", "_").replace("\\", "_")[:80]
                # Save as .osu (the chart carries the GT audio ref
                # but no audio bytes, so .osu is the right format).
                osu_path = charts_out / f"{safe_name}.osu"
                try:
                    pred_chart.save_osu(osu_path)
                except Exception:
                    pass
        else:
            continue

    # Aggregate.
    if skipped_no_audio:
        print(f"  Skipped {skipped_no_audio} charts (no audio found)")

    if not all_comparisons:
        print("No charts processed successfully.")
        if skipped_no_audio:
            print("All charts skipped due to missing audio. Use --charts-dir to point at .osz packs.")
        return 1

    # Aggregate numeric fields.
    agg: dict[str, float] = {}
    all_keys: set[str] = set()
    for c in all_comparisons:
        all_keys.update(c.keys())
    for k in sorted(all_keys):
        vals = [c[k] for c in all_comparisons if k in c and isinstance(c[k], (int, float))]
        if vals:
            agg[k] = float(np.mean(vals))

    print()
    print(f"=== {args.backend} results ({len(all_comparisons)} charts) ===")
    print(f"  matched_rate:      {agg.get('cmp/matched_rate', 0):.4f}")
    print(f"  hallucination_rate:{agg.get('cmp/hallucination_rate', 0):.4f}")
    print(f"  density_ratio:     {agg.get('cmp/density_ratio', 0):.4f}")
    print(f"  error_median_ms:   {agg.get('cmp/error_median_ms', 0):.1f}")
    print(f"  dc_human:          {agg.get('cmp/dc_human', 0):.1f}")
    print(f"  oc_human:          {agg.get('cmp/oc_human', 0):.1f}")
    print(f"  precision:         {agg.get('cmp/precision', 0):.4f}")
    print(f"  recall:            {agg.get('cmp/recall', 0):.4f}")
    print(f"  f1:                {agg.get('cmp/f1', 0):.4f}")
    print(f"  events_per_sec:    {agg.get('pred/events_per_sec', 0):.2f}")
    print(f"  gap_metro_dist:    {agg.get('pred/gap_metronome_distance', 0):.4f}")
    print(f"  over_pspace_self:  {agg.get('cmp/over_pspace_self', 0):.2f}")

    result = {
        "backend": args.backend,
        "n_charts": len(all_comparisons),
        "fraction": args.fraction,
        "aggregate": {k: round(v, 4) for k, v in agg.items()},
        "per_chart": all_comparisons,
    }
    if args.backend == "mapperatorinator":
        result["difficulty"] = args.difficulty
        result["backend_path"] = str(args.backend_path)

    if args.experiment_dir:
        out_path = args.experiment_dir / f"benchmark_{args.backend}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
