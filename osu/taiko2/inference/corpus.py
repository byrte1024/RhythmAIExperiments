"""Per-corpus inference driver — runs a `ChartPredictor` over a
fraction of a dataset split, computes per-chart metrics +
comparisons against the GT chart, and emits:

  1. Rich per-chart artifacts on disk (JSONs, optional bundles, CSVs).
  2. A small "averaged summary" dict suitable for merging into the
     training `val_metrics` stream so per-eval corpus numbers get
     tracked alongside HIT/MISS/etc. and auto-graphed by the curves
     hook.

Sharing this as a library function makes the exact same work callable
from two places: the `cli.infer_corpus` command and the training loop's
`InferCorpusHook`. Same output shape both ways.
"""
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..data_samplers import TaikoDetectionSampler
from ..domain.chart import Chart, ChartComparison, ChartMetrics
from ..domain.inference import ChartPredictor, Conditioning
from .autoregressive.predictor import AutoregressivePredictor


# ─────────────────────────── config ──────────────────────────────────

@dataclass(frozen=True, slots=True)
class InferCorpusConfig:
    """Knobs for `run_infer_corpus`. Intended to live in a JSON config
    file loadable by the same `build_config` helper the predictor
    spec uses.

    - ``fraction`` of the split's charts to infer (default 0.1).
    - ``seed`` shuffles the chart order; same seed across runs → same
       charts, so per-eval comparisons are meaningful.
    - ``max_charts`` caps the selection after `fraction` (for smoke
       tests).
    - ``conditioning_modes`` is any non-empty subset of
       ``("gt", "fixed")``. "gt" = conditioning copied from each chart's
       own density; "fixed" = constant conditioning from
       `fixed_mean/peak/std`.
    - ``save_bundles`` controls whether each generated chart is
       persisted as a `.zip`. Per-eval hook runs should leave this False
       to avoid thousands of small files; CLI runs default to True.
    """
    fraction: float = 0.1
    seed: int = 42
    max_charts: int | None = None
    conditioning_modes: tuple[str, ...] = ("gt", "fixed")
    fixed_density_mean: float = 5.0
    fixed_density_peak: int = 10
    fixed_density_std: float = 2.2
    save_bundles: bool = True
    save_per_chart_jsons: bool = True


# ─────────────────────────── metric averaging ────────────────────────

# Which ChartMetrics scalars get averaged into the per-eval summary.
# Kept small on purpose — every key here becomes a metric column the
# training loop logs + auto-graphs per eval.
_SUMMARY_METRIC_FIELDS: tuple[str, ...] = (
    "total_events", "events_per_sec", "density_mean",
    "ioi_mean_ms", "ioi_median_ms",
    "don_ratio",
    "gap_peak_count", "gap_peak_mass_total",
    "gap_metronome_distance", "gap_random_distance",
    "ratio_peak_count", "ratio_peak_mass_total",
    "ratio_metronome_distance", "ratio_random_distance",
)

# Which ChartComparison fields get averaged.
_SUMMARY_COMPARISON_FIELDS: tuple[str, ...] = (
    "matched_rate", "close_rate", "far_rate", "hallucination_rate",
    "error_mean_ms", "error_median_ms",
    "density_ratio",
    "over_pspace_self", "over_pspace_other", "hi_pspace",
    "dc_human", "oc_human",
)

# Everything in ChartMetrics that the CSV should carry — same as the
# CLI uses, minus non-scalar fields.
_METRIC_CSV_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(ChartMetrics)
    if f.name not in {
        "density_timeline", "silence_regions", "dense_regions",
        "ioi_histogram_10ms", "gap_peaks", "ratio_peaks",
    }
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
    sampler: TaikoDetectionSampler, chart_index: int, ds_root: Path,
) -> np.ndarray:
    """Load the mel features the dataset stores on disk for this chart.
    `features_path` in the manifest is relative to the dataset root."""
    manifest = sampler._manifest
    if manifest is None:
        raise RuntimeError("sampler not loaded")
    chart_id = sampler.chart_ids()[chart_index]
    entry = next(
        (c for c in manifest.charts if c.chart_id == chart_id), None,
    )
    if entry is None:
        raise KeyError(f"chart id {chart_id} not in manifest")
    return np.load(ds_root / entry.features_path).astype(np.float32)


def _safe_chart_stem(chart_id: str) -> str:
    allowed = []
    for ch in chart_id:
        if ch.isalnum() or ch in " _-.[](),":
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip() or "chart"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(obj), ensure_ascii=False, indent=2),
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


def _mean_of_field(rows: list[dict[str, Any]], name: str) -> float | None:
    vals: list[float] = []
    for r in rows:
        v = r.get(name, "")
        if v == "" or v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return float(np.mean(vals))


def _full_summary(
    rows: list[dict[str, Any]], fieldnames: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows), "fields": {}}
    for name in fieldnames:
        vals: list[float] = []
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


# ─────────────────────────── per-mode runner ─────────────────────────

def _run_one_mode(
    *,
    predictor: AutoregressivePredictor,
    mode_dir: Path,
    mode_label: str,
    selected: list[int],
    sampler: TaikoDetectionSampler,
    features_cache: dict[int, np.ndarray],
    gt_charts: dict[int, Chart],
    conditioning_for: Callable[[Chart], Conditioning],
    save_bundles: bool,
    save_per_chart_jsons: bool,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gen_dir = mode_dir / "generated"
    metrics_dir = mode_dir / "metrics"
    cmp_dir = mode_dir / "comparisons"
    for d in (gen_dir, metrics_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    cmp_rows: list[dict[str, Any]] = []

    it: Any = selected
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
        stem = _safe_chart_stem(chart_id) or f"chart_{i}"

        if save_bundles:
            generated.save(gen_dir / f"{stem}.zip")

        metrics = generated.calculate_metrics()
        if save_per_chart_jsons:
            _dump_json(metrics_dir / f"{stem}.json", asdict(metrics))
        metric_rows.append(_metrics_to_csv_row(chart_id, metrics))

        cmp_ = generated.compare(gt_chart)
        if save_per_chart_jsons:
            _dump_json(cmp_dir / f"{stem}.json", asdict(cmp_))
        cmp_rows.append(_comparison_to_csv_row(chart_id, cmp_))

    return metric_rows, cmp_rows


# ─────────────────────────── entry point ─────────────────────────────

def run_infer_corpus(
    *,
    predictor: ChartPredictor,
    val_sampler: TaikoDetectionSampler,
    ds_root: Path,
    out_dir: Path,
    config: InferCorpusConfig,
    step: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run `predictor` over a fraction of `val_sampler`'s split, under
    every conditioning mode in `config.conditioning_modes`. Write
    per-chart JSONs + CSVs + summary JSONs under ``out_dir``, and
    return a FLAT dict of averaged metrics suitable for merging into a
    training-loop `val_metrics` dict.

    The returned dict has keys of the form
    ``corpus/{mode}/{metric_name}_mean`` and
    ``corpus/{mode}_cmp/{metric_name}_mean`` so downstream curve
    generation treats each as its own time series.
    """
    if not isinstance(predictor, AutoregressivePredictor):
        raise TypeError(
            "run_infer_corpus currently requires AutoregressivePredictor "
            "(needs predict_from_features)."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_charts = val_sampler.count_charts()
    selected = _pick_chart_indices(
        total_charts, config.fraction, config.seed,
    )
    if config.max_charts is not None:
        selected = selected[: config.max_charts]

    # Preload charts + features once — each chart is inferred once per
    # mode, so caching saves disk I/O on multi-mode runs.
    gt_charts: dict[int, Chart] = {}
    features_cache: dict[int, np.ndarray] = {}
    load_iter: Any = selected
    if progress:
        try:
            from tqdm.auto import tqdm
            load_iter = tqdm(selected, desc="preload", unit="chart")
        except ImportError:
            pass
    for i in load_iter:
        gt_charts[i] = val_sampler.get_chart(i)
        features_cache[i] = _features_for(val_sampler, i, ds_root)

    # Conditioning builders per mode.
    fixed_cond = Conditioning(
        density_mean=config.fixed_density_mean,
        density_peak=config.fixed_density_peak,
        density_std=config.fixed_density_std,
    )

    def _cond_gt(chart: Chart) -> Conditioning:
        d = chart.track.density
        return Conditioning(
            density_mean=float(d.mean),
            density_peak=int(d.peak),
            density_std=float(d.std),
        )

    def _cond_fixed(_chart: Chart) -> Conditioning:
        return fixed_cond

    mode_cond_map: dict[str, Callable[[Chart], Conditioning]] = {
        "gt": _cond_gt,
        "fixed": _cond_fixed,
    }

    started = time.time()
    flat_summary: dict[str, float] = {}
    per_mode_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

    for mode in config.conditioning_modes:
        if mode not in mode_cond_map:
            raise ValueError(
                f"unknown conditioning mode {mode!r}; "
                f"known: {sorted(mode_cond_map)!r}"
            )
        mode_dir = out_dir / f"{mode}_cond"
        metric_rows, cmp_rows = _run_one_mode(
            predictor=predictor, mode_dir=mode_dir, mode_label=mode,
            selected=selected, sampler=val_sampler,
            features_cache=features_cache, gt_charts=gt_charts,
            conditioning_for=mode_cond_map[mode],
            save_bundles=config.save_bundles,
            save_per_chart_jsons=config.save_per_chart_jsons,
            progress=progress,
        )
        per_mode_rows[mode] = (metric_rows, cmp_rows)

        _write_csv(metric_rows, _METRIC_CSV_FIELDS, mode_dir / "metrics.csv")
        _write_csv(cmp_rows, _COMPARISON_CSV_FIELDS, mode_dir / "comparisons.csv")
        _dump_json(
            mode_dir / "metrics_summary.json",
            _full_summary(metric_rows, _METRIC_CSV_FIELDS),
        )
        _dump_json(
            mode_dir / "comparisons_summary.json",
            _full_summary(cmp_rows, _COMPARISON_CSV_FIELDS),
        )

        # Flat summary — one mean per tracked field, keyed for direct
        # merging into the training val_metrics dict.
        for name in _SUMMARY_METRIC_FIELDS:
            v = _mean_of_field(metric_rows, name)
            if v is not None:
                flat_summary[f"corpus/{mode}_cond/{name}_mean"] = v
        for name in _SUMMARY_COMPARISON_FIELDS:
            v = _mean_of_field(cmp_rows, name)
            if v is not None:
                flat_summary[f"corpus/{mode}_cond_cmp/{name}_mean"] = v

    elapsed_s = round(time.time() - started, 2)

    # Small top-level summary of just the averaged values — separate
    # from the full percentile summary JSONs inside each mode dir.
    _dump_json(out_dir / "summary.json", {
        "step": step,
        "n_charts": len(selected),
        "modes": list(config.conditioning_modes),
        "elapsed_s": elapsed_s,
        "means": flat_summary,
    })

    return flat_summary
