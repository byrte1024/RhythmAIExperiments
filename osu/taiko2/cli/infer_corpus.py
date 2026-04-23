"""Run inference over a fraction of a dataset's charts, in TWO modes
per chart (GT-conditioning + fixed mean conditioning), save every
generated chart + its metrics + comparisons against the GT chart.

Outputs
=======

Under ``{analysis-dir}/{dataset}/infer_corpus/{run_stem}/``:

  index.json
    Manifest of what was run — spec snapshot, chart ids selected, per-
    mode directories, fixed-conditioning values, seed.
  gt_cond/
    generated/{chart_id}.zip       Chart.save bundle of the generated chart
    metrics/{chart_id}.json        ChartMetrics (full, via calculate_metrics)
    comparisons/{chart_id}.json    ChartComparison (full, via compare(GT))
    metrics.csv                    flat scalar per-chart summary
    comparisons.csv                flat scalar per-chart comparison
    metrics_summary.json           per-field p25/median/p75/p95 across charts
    comparisons_summary.json       same for comparison fields
  fixed_cond/
    ... identical layout with conditioning = Conditioning(mean=5.0,
        peak=10, std=2.2) — peak is int-typed in the dataclass so 9.5
        is rounded up to 10.

Pipeline
========

1. Load the dataset's specified split via `TaikoDetectionSampler`.
2. Deterministically select ``--fraction`` of the split's charts
   (default 1/10, seed 42 — reproducible across runs).
3. Assemble predictor once from the spec JSON (same shape as
   `cli.infer`). If checkpoint exists it's loaded once and reused.
4. For each selected chart, for each of the two conditioning modes:
   a. Load the pre-computed mel features from disk.
   b. `predictor.predict_from_features(chart, conditioning, features)`.
   c. Save the generated chart as a bundle + compute metrics +
      compute comparison against the GT chart.
5. Aggregate flat CSVs + per-field summary JSONs per mode.

No graphs in this CLI — raw data only. Use `cli.analyze_charts` style
tooling downstream to visualise the generated-vs-GT differences.

Example
=======

::

    osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.infer_corpus \\
        --config osu/taiko2/experiments/002-exp45-full/config/infer.json \\
        --dataset taiko2_v1 \\
        --split val
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from ..dataset import _safe_filename
from ..domain.chart import Chart, ChartComparison, ChartMetrics
from ..domain.inference import Conditioning
from ..inference.autoregressive.predictor import AutoregressivePredictor
from ._infer_common import assemble_predictor, load_spec


# ─────────────────────────── fixed conditioning ──────────────────────
# User-specified "dataset-mean baseline" — same value used for every
# chart in the fixed-cond run so any variance in the generated outputs
# reflects ONLY the audio, not the conditioning signal.
FIXED_DENSITY_MEAN: float = 5.0
FIXED_DENSITY_PEAK: int = 10         # user wrote 9.5; Conditioning.peak is int
FIXED_DENSITY_STD: float = 2.2


# ─────────────────────────── output shape ────────────────────────────

# Scalar fields promoted from ChartMetrics into the flat CSV. Mirrors
# `cli/analyze_charts._CSV_FIELDS` so downstream consumers can diff.
_METRIC_CSV_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(ChartMetrics)
    if not isinstance(f.default, (tuple, dict))
    and f.name not in {"density_timeline", "silence_regions", "dense_regions",
                       "ioi_histogram_10ms", "gap_peaks", "ratio_peaks"}
)
_COMPARISON_CSV_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(ChartComparison)
)


# ─────────────────────────── helpers ─────────────────────────────────

def _pick_chart_indices(
    total: int, fraction: float, seed: int,
) -> list[int]:
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    n = max(1, int(round(total * fraction)))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:n])


def _features_for(
    sampler: TaikoDetectionSampler,
    chart_index: int,
    ds_root: Path,
) -> np.ndarray:
    """Pull the mel feature file the dataset already has on disk for
    this chart. `features_path` in the manifest is stored RELATIVE to
    the dataset root (see `dataset.py:features_path.relative_to(
    features_root)`), so we resolve against `ds_root`. Float16 on
    disk → float32 in memory (predictor needs float32)."""
    manifest = sampler._manifest
    if manifest is None:
        raise RuntimeError("sampler not loaded")
    chart_id = sampler.chart_ids()[chart_index]
    entry = next(
        (c for c in manifest.charts if c.chart_id == chart_id), None,
    )
    if entry is None:
        raise KeyError(f"chart id {chart_id} not in manifest")
    feat_path = ds_root / entry.features_path
    return np.load(feat_path).astype(np.float32)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # calculate_metrics() returns nested dataclass + numpy scalars;
    # `_json_safe` is the same helper analyze_charts uses.
    def _safe(v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): _safe(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    path.write_text(
        json.dumps(_safe(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _metrics_to_csv_row(
    chart_id: str, metrics: ChartMetrics,
) -> dict[str, Any]:
    row: dict[str, Any] = {"chart_id": chart_id}
    for name in _METRIC_CSV_FIELDS:
        v = getattr(metrics, name)
        row[name] = "" if v is None else v
    return row


def _comparison_to_csv_row(
    chart_id: str, cmp_: ChartComparison,
) -> dict[str, Any]:
    row: dict[str, Any] = {"chart_id": chart_id}
    for name in _COMPARISON_CSV_FIELDS:
        v = getattr(cmp_, name)
        row[name] = "" if v is None else v
    return row


def _write_csv(
    rows: list[dict[str, Any]], fieldnames: tuple[str, ...], path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chart_id", *fieldnames])
        writer.writeheader()
        writer.writerows(rows)


def _compute_summary(
    rows: list[dict[str, Any]], fieldnames: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows), "fields": {}}
    for name in fieldnames:
        vals = []
        for r in rows:
            v = r.get(name, "")
            if v == "" or v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        q = np.percentile(arr, [0, 25, 50, 75, 95, 100])
        out["fields"][name] = {
            "n": int(arr.size),
            "min":    round(float(q[0]), 4),
            "p25":    round(float(q[1]), 4),
            "median": round(float(q[2]), 4),
            "p75":    round(float(q[3]), 4),
            "p95":    round(float(q[4]), 4),
            "max":    round(float(q[5]), 4),
            "mean":   round(float(arr.mean()), 4),
        }
    return out


# ─────────────────────────── per-mode runner ──────────────────────────

def _run_one_mode(
    *,
    predictor: AutoregressivePredictor,
    mode_dir: Path,
    selected: list[int],
    sampler: TaikoDetectionSampler,
    features_cache: dict[int, np.ndarray],
    gt_charts: dict[int, Chart],
    conditioning_for: "callable[[Chart], Conditioning]",
    mode_label: str,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Infer every selected chart under one conditioning regime and
    write its per-chart artifacts. Returns (metric rows, comparison
    rows) for the CSV aggregator."""
    gen_dir = mode_dir / "generated"
    metrics_dir = mode_dir / "metrics"
    cmp_dir = mode_dir / "comparisons"
    for d in (gen_dir, metrics_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    cmp_rows: list[dict[str, Any]] = []

    it = selected
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(selected, desc=f"infer [{mode_label}]", unit="chart")
        except ImportError:
            pass

    for i in it:
        gt_chart = gt_charts[i]
        features = features_cache[i]
        cond = conditioning_for(gt_chart)
        generated = predictor.predict_from_features(
            gt_chart, conditioning=cond, features=features,
        )
        chart_id = sampler.chart_ids()[i]
        stem = _safe_filename(chart_id) or f"chart_{i}"

        # 1) Save the chart itself (bundle; audio not stored so .zip is
        # smaller and lossless for the onset data).
        generated.save(gen_dir / f"{stem}.zip")

        # 2) Metrics.
        metrics = generated.calculate_metrics()
        _dump_json(metrics_dir / f"{stem}.json", asdict(metrics))
        metric_rows.append(_metrics_to_csv_row(chart_id, metrics))

        # 3) Comparison against GT onsets.
        cmp_ = generated.compare(gt_chart)
        _dump_json(cmp_dir / f"{stem}.json", asdict(cmp_))
        cmp_rows.append(_comparison_to_csv_row(chart_id, cmp_))

    return metric_rows, cmp_rows


# ─────────────────────────── CLI ─────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate charts for a fraction of a dataset via "
            "ChartPredictor, in two conditioning modes, and save "
            "per-chart metrics + comparisons against GT."
        ),
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Predictor spec JSON (same shape as cli.infer).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Override the spec's `checkpoint` field.")
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under --datasets-dir) or path.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--analysis-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "analysis")
    p.add_argument("--split", default="val",
                   help="Dataset split to iterate (default: val).")
    p.add_argument("--split-ratios", default="train:0.9,val:0.1")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=0.1,
                   help="Fraction of charts in the split to infer.")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed for chart selection.")
    p.add_argument("--max-charts", type=int, default=None,
                   help="Smoke-test override — cap after fraction selection.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_split_ratios(raw: str) -> tuple[tuple[str, float], ...]:
    parts: list[tuple[str, float]] = []
    for frag in raw.split(","):
        name, _, ratio = frag.strip().partition(":")
        if not name or not ratio:
            raise ValueError(f"bad split-ratios fragment {frag!r}")
        parts.append((name.strip(), float(ratio)))
    return tuple(parts)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ds_root = Path(args.dataset)
    if not ds_root.is_absolute() and not ds_root.exists():
        ds_root = args.datasets_dir / args.dataset
    ds_root = ds_root.resolve()
    if not ds_root.is_dir():
        print(f"ERROR: dataset root not found: {ds_root}", file=sys.stderr)
        return 2

    spec = load_spec(config=args.config, config_json=None)
    if args.checkpoint is not None:
        spec["checkpoint"] = str(args.checkpoint)

    # Assemble predictor once — reused across every chart.
    device = torch.device(args.device)
    print(f"[infer_corpus] checkpoint: {spec['checkpoint']}")
    print(f"[infer_corpus] device:     {device}")
    predictor = assemble_predictor(spec=spec, device=device)
    if not isinstance(predictor, AutoregressivePredictor):
        print(
            "ERROR: infer_corpus currently only supports "
            "AutoregressivePredictor (needs predict_from_features).",
            file=sys.stderr,
        )
        return 2

    # Build the sampler for the requested split.
    sampler = TaikoDetectionSampler(TaikoDetectionSamplerConfig(
        batch_size=1,
        dataset_root=ds_root,
        split=args.split,
        split_ratios=_parse_split_ratios(args.split_ratios),
        split_seed=args.split_seed,
        min_cursor_bin=0,
        allowed_overlap_forward=0,
        allowed_overlap_back=0,
    ))
    sampler.load_data(progress=not args.no_progress)
    total_charts = sampler.count_charts()
    selected = _pick_chart_indices(total_charts, args.fraction, args.seed)
    if args.max_charts is not None:
        selected = selected[: args.max_charts]
    print(
        f"[infer_corpus] split '{args.split}': {total_charts:,} charts, "
        f"infer {len(selected)} ({args.fraction * 100:.1f}%, seed={args.seed})"
    )

    # Preload GT charts + features once — each chart gets inferred
    # twice (GT + fixed cond), so caching saves disk I/O.
    gt_charts: dict[int, Chart] = {}
    features_cache: dict[int, np.ndarray] = {}
    load_iter = selected
    if not args.no_progress:
        try:
            from tqdm.auto import tqdm
            load_iter = tqdm(selected, desc="preload", unit="chart")
        except ImportError:
            pass
    for i in load_iter:
        gt_charts[i] = sampler.get_chart(i)
        features_cache[i] = _features_for(sampler, i, ds_root)

    # Derive a per-run output dir from the checkpoint path so repeated
    # runs against different evals don't clobber each other.
    ckpt_path = Path(spec["checkpoint"])
    run_stem = f"{ckpt_path.parent.name}_{ckpt_path.stem}"
    dataset_name = (
        sampler._manifest.name if sampler._manifest else ds_root.name
    )
    out_root = (args.analysis_dir / dataset_name / "infer_corpus" / run_stem).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[infer_corpus] out_root:   {out_root}")

    # Mode A: GT conditioning (each chart gets its own density trio).
    def _cond_gt(chart: Chart) -> Conditioning:
        d = chart.track.density
        return Conditioning(
            density_mean=float(d.mean),
            density_peak=int(d.peak),
            density_std=float(d.std),
        )

    # Mode B: fixed conditioning — one constant trio for every chart.
    fixed_cond = Conditioning(
        density_mean=FIXED_DENSITY_MEAN,
        density_peak=FIXED_DENSITY_PEAK,
        density_std=FIXED_DENSITY_STD,
    )
    def _cond_fixed(_chart: Chart) -> Conditioning:
        return fixed_cond

    # Per-mode run.
    started = time.time()
    for mode_label, subdir, cond_fn in (
        ("gt_cond",    out_root / "gt_cond",    _cond_gt),
        ("fixed_cond", out_root / "fixed_cond", _cond_fixed),
    ):
        subdir.mkdir(parents=True, exist_ok=True)
        metric_rows, cmp_rows = _run_one_mode(
            predictor=predictor, mode_dir=subdir,
            selected=selected, sampler=sampler,
            features_cache=features_cache, gt_charts=gt_charts,
            conditioning_for=cond_fn,
            mode_label=mode_label,
            progress=not args.no_progress,
        )
        _write_csv(metric_rows, _METRIC_CSV_FIELDS, subdir / "metrics.csv")
        _write_csv(cmp_rows, _COMPARISON_CSV_FIELDS, subdir / "comparisons.csv")
        _dump_json(
            subdir / "metrics_summary.json",
            _compute_summary(metric_rows, _METRIC_CSV_FIELDS),
        )
        _dump_json(
            subdir / "comparisons_summary.json",
            _compute_summary(cmp_rows, _COMPARISON_CSV_FIELDS),
        )
        print(
            f"[infer_corpus] {mode_label}: wrote "
            f"{len(metric_rows)} metrics, {len(cmp_rows)} comparisons"
        )

    # Run-level index for downstream tools.
    elapsed_s = round(time.time() - started, 1)
    _dump_json(out_root / "index.json", {
        "spec": spec,
        "dataset": dataset_name,
        "split": args.split,
        "fraction": args.fraction,
        "seed": args.seed,
        "n_selected": len(selected),
        "chart_ids": [sampler.chart_ids()[i] for i in selected],
        "fixed_conditioning": {
            "density_mean": FIXED_DENSITY_MEAN,
            "density_peak": FIXED_DENSITY_PEAK,
            "density_std":  FIXED_DENSITY_STD,
        },
        "device": str(device),
        "elapsed_s": elapsed_s,
    })
    print(f"[infer_corpus] done in {elapsed_s}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
