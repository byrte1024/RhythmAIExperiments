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

        cmd = [
            sys.executable, str(backend_path / "inference.py"),
            f"audio_path={audio_path}",
            f"output_path={out_dir}",
            f"gamemode={gamemode}",
            f"difficulty={difficulty}",
            "super_timing=false",
            "output_type=[TIMING,MAP]",
        ]
        env = os.environ.copy()
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        result = subprocess.run(
            cmd, cwd=str(backend_path), env=env,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"    WARN: mapperatorinator failed: {result.stderr[:200]}")
            return []

        # Parse generated .osu files for onset times.
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


# ─────────────────────────── main ────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Benchmark external onset detectors against GT.",
    )
    p.add_argument("--backend", required=True,
                   choices=["mapperatorinator", "librosa", "madmom"],
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
            else:
                pred_ms = []
        finally:
            if needs_cleanup and audio_path is not None:
                audio_path.unlink(missing_ok=True)

        if not pred_ms:
            continue

        pred_arr = np.array(pred_ms, dtype=np.float64)
        metrics = gt_match_metrics(pred_arr, gt_ms)
        all_comparisons.append(metrics)

    # Aggregate.
    if skipped_no_audio:
        print(f"  Skipped {skipped_no_audio} charts (no audio found)")

    if not all_comparisons:
        print("No charts processed successfully.")
        if skipped_no_audio:
            print("All charts skipped due to missing audio. Use --charts-dir to point at .osz packs.")
        return 1

    agg: dict[str, float] = {}
    keys = all_comparisons[0].keys()
    for k in keys:
        vals = [c[k] for c in all_comparisons if k in c and isinstance(c[k], (int, float))]
        if vals:
            agg[k] = float(np.mean(vals))

    print()
    print(f"=== {args.backend} results ({len(all_comparisons)} charts) ===")
    print(f"  matched_rate:      {agg.get('matched_rate', 0):.4f}")
    print(f"  hallucination_rate:{agg.get('hallucination_rate', 0):.4f}")
    print(f"  density_ratio:     {agg.get('density_ratio', 0):.4f}")
    print(f"  error_median_ms:   {agg.get('error_median_ms', 0):.1f}")
    print(f"  precision:         {agg.get('precision', 0):.4f}")
    print(f"  recall:            {agg.get('recall', 0):.4f}")
    print(f"  f1:                {agg.get('f1', 0):.4f}")
    print(f"  close_rate:        {agg.get('close_rate', 0):.4f}")
    print(f"  far_rate:          {agg.get('far_rate', 0):.4f}")

    result = {
        "backend": args.backend,
        "n_charts": len(all_comparisons),
        "fraction": args.fraction,
        **{k: round(v, 4) for k, v in agg.items()},
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
