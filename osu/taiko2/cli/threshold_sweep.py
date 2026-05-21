"""Threshold sweep for framewise models.

Loads a model once, runs AR inference at multiple decode thresholds,
and reports per-threshold AR corpus metrics. Much faster than running
the full training pipeline — no loss, no metrics, no artifacts.

Usage::

    osu/taiko2/.venv/bin/python -m osu.taiko2.cli.threshold_sweep \
        --config osu/taiko2/experiments/017d-framewise-bce-noweight/config/infer.json \
        --checkpoint osu/taiko2/runs/exp_017d_framewise_bce_noweight/checkpoints/best.pt \
        --dataset taiko2_v1 \
        --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
        --fraction 0.1 \
        --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from ..domain.chart import Chart, gt_match_metrics
from ..domain.inference import Conditioning
from ..inference.spec import (
    assemble_predictor,
    build_config,
    build_component,
    load_spec,
    resolve_class,
)
from ..inference.autoregressive.predictor import AutoregressivePredictor


def _pick_charts(
    total: int, fraction: float, seed: int,
) -> list[int]:
    n = max(1, int(round(total * fraction)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total, size=min(n, total), replace=False).tolist())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Threshold sweep for framewise AR inference.",
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Predictor spec JSON (infer.json).")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Override checkpoint path (single). Mutually exclusive with --checkpoints.")
    p.add_argument("--checkpoints", default=None,
                   help="Comma-separated checkpoint paths or 'all' to sweep all eval_*/checkpoint.pt in the run dir.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7",
                   help="Comma-separated thresholds to sweep.")
    p.add_argument("--nms-kernels", default=None,
                   help="Comma-separated NMS kernels to sweep (optional, cross-product with thresholds).")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--experiment-dir", type=Path, default=None,
                   help="Experiment directory — saves sweep results to {dir}/threshold_sweep.json.")
    args = p.parse_args(argv)

    ds_root = args.datasets_dir / args.dataset
    if not ds_root.is_dir():
        print(f"ERROR: dataset not found: {ds_root}", file=sys.stderr)
        return 2

    thresholds = [float(t) for t in args.thresholds.split(",")]
    nms_kernels = [3]  # default
    if args.nms_kernels:
        nms_kernels = [int(k) for k in args.nms_kernels.split(",")]

    device = torch.device(args.device)
    spec = json.loads(args.config.read_text(encoding="utf-8"))

    # Resolve checkpoint list.
    from ..inference.loader import load_model_from_checkpoint
    ckpt_paths: list[Path] = []
    if args.checkpoints:
        if args.checkpoints == "all":
            run_dir = Path(spec["checkpoint"]).parent.parent
            eval_dirs = sorted(run_dir.glob("eval_*/checkpoint.pt"))
            ckpt_paths = eval_dirs
            best = run_dir / "checkpoints" / "best.pt"
            if best.exists() and best not in ckpt_paths:
                ckpt_paths.append(best)
        else:
            ckpt_paths = [Path(p.strip()) for p in args.checkpoints.split(",")]
    elif args.checkpoint:
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = [Path(spec["checkpoint"])]

    print(f"Checkpoints to sweep: {len(ckpt_paths)}")
    for cp in ckpt_paths:
        print(f"  {cp}")

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
    print(f"Val charts: {total}, selected: {len(selected)}")

    # Preload charts + features.
    gt_charts: dict[int, Chart] = {}
    features_cache: dict[int, np.ndarray] = {}
    for i in selected:
        gt_charts[i] = sampler.get_chart(i)
        entry = sampler._chart_entries[i]
        feat_path = ds_root / entry.features_path
        features_cache[i] = np.load(feat_path, mmap_mode="r")
    print(f"Preloaded {len(gt_charts)} charts")

    results: list[dict[str, Any]] = []

    for ckpt_path in ckpt_paths:
        # Extract a label from the path.
        if "eval_" in str(ckpt_path):
            ckpt_label = ckpt_path.parent.name  # eval_82696
        else:
            ckpt_label = ckpt_path.stem  # best / latest

        print(f"\n=== Checkpoint: {ckpt_label} ({ckpt_path}) ===")
        model, _loss, meta = load_model_from_checkpoint(ckpt_path, device=device)
        model.eval()

        for nms_k in nms_kernels:
          for tau in thresholds:
            # Build a fresh predictor with this threshold.
            spec_copy = copy.deepcopy(spec)
            decoder_cfg = spec_copy["decoder"]["config"]
            decoder_cfg["decode_threshold"] = tau
            decoder_cfg["nms_kernel"] = nms_k

            from ..inference.spec import assemble_predictor_with_model
            predictor = assemble_predictor_with_model(
                spec=spec_copy, model=model, device=device,
            )

            t0 = time.time()
            all_comparisons: list[dict[str, float]] = []

            try:
                from tqdm.auto import tqdm
                chart_iter = tqdm(
                    selected,
                    desc=f"tau={tau:.2f} nms={nms_k}",
                    unit="chart",
                    leave=False,
                )
            except ImportError:
                chart_iter = selected

            for idx in chart_iter:
                gt = gt_charts[idx]
                features = features_cache[idx]
                gt_cond = Conditioning(
                    density_mean=gt.track.density.mean,
                    density_peak=gt.track.density.peak,
                    density_std=gt.track.density.std,
                )
                pred_chart = predictor.predict_from_features(
                    gt, conditioning=gt_cond, features=features,
                )
                comparison = gt.compare(pred_chart)
                comp_dict = {
                    "cmp/" + f.name: getattr(comparison, f.name)
                    for f in comparison.__dataclass_fields__.values()
                }
                pred_metrics = pred_chart.calculate_metrics()
                metrics_dict = {
                    "pred/" + f.name: getattr(pred_metrics, f.name)
                    for f in pred_metrics.__dataclass_fields__.values()
                    if isinstance(getattr(pred_metrics, f.name), (int, float))
                }
                comp_dict.update(metrics_dict)
                all_comparisons.append(comp_dict)

            elapsed = time.time() - t0

            # Aggregate.
            agg: dict[str, float] = {}
            if all_comparisons:
                keys = all_comparisons[0].keys()
                for k in keys:
                    vals = [c[k] for c in all_comparisons if isinstance(c[k], (int, float))]
                    if vals:
                        agg[k] = float(np.mean(vals))

            # Estimated AR P/R/F1.
            mr = agg.get("cmp/matched_rate", 0)
            dr = agg.get("cmp/density_ratio", 1)
            ar_prec = mr / dr if dr > 0 else 0
            ar_f1 = 2 * ar_prec * mr / (ar_prec + mr) if (ar_prec + mr) > 0 else 0

            row = {
                "checkpoint": ckpt_label,
                "threshold": tau,
                "nms_kernel": nms_k,
                "n_charts": len(all_comparisons),
                "elapsed_s": round(elapsed, 1),
                "ar_precision": round(ar_prec, 4),
                "ar_recall": round(mr, 4),
                "ar_f1": round(ar_f1, 4),
                **{k: round(v, 4) for k, v in agg.items()},
            }
            results.append(row)

            print(
                f"  tau={tau:.2f}  nms={nms_k}  "
                f"AR P={ar_prec:.4f}  R={mr:.4f}  F1={ar_f1:.4f}  "
                f"dr={dr:.3f}  hr={agg.get('cmp/hallucination_rate',0):.4f}  "
                f"dc={agg.get('cmp/dc_human',0):.1f}  "
                f"err={agg.get('cmp/error_median_ms',0):.1f}ms  "
                f"eps={agg.get('pred/events_per_sec',0):.2f}  "
                f"({elapsed:.1f}s)"
            )

    # Summary table.
    print()
    print(f"{'ckpt':>12s}  {'tau':>5s}  {'nms':>4s}  {'AR P':>7s}  {'AR R':>7s}  {'AR F1':>7s}  "
          f"{'dr':>6s}  {'hr':>7s}  {'dc':>6s}  {'err_ms':>7s}  {'eps':>5s}  {'gmet':>6s}  {'gpk':>4s}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['checkpoint']:>12s}  {r['threshold']:5.2f}  {r['nms_kernel']:4d}  "
            f"{r['ar_precision']:7.4f}  {r['ar_recall']:7.4f}  {r['ar_f1']:7.4f}  "
            f"{r.get('cmp/density_ratio',0):6.3f}  "
            f"{r.get('cmp/hallucination_rate',0):7.4f}  "
            f"{r.get('cmp/dc_human',0):6.1f}  "
            f"{r.get('cmp/error_median_ms',0):7.1f}  "
            f"{r.get('pred/events_per_sec',0):5.2f}  "
            f"{r.get('pred/gap_metronome_distance',0):6.4f}  "
            f"{r.get('pred/gap_peak_count',0):4.1f}"
        )

    if args.experiment_dir:
        out_path = args.experiment_dir / "threshold_sweep.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
