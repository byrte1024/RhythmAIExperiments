"""Training entry point.

Takes four JSON configs (model / loss / trainer / data) plus a run
name, wires everything to the `train()` loop, and runs until
completion (or until the run-dir's `latest.pt` says otherwise — the
loop resumes automatically).

Config JSONs use the ``__class__`` tag convention the checkpoint
loader already speaks; this CLI reuses that reconstruction. The idea
is one folder of configs per experiment, checked in alongside the
experiment's README — a deterministic description of the run.

Usage::

    python -m osu.taiko2.cli.train \\
        --run-name exp_001_exp45 \\
        --config-dir osu/taiko2/experiments/001-exp45-port/config \\
        --dataset taiko2_v1

Output lands under ``osu/taiko2/runs/{run_name}/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from ..data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from ..domain.augmentation import AugmentationPipeline
from ..domain.metrics import MetricSet
from ..domain.training import RunSpec, TrainerConfig
from ..models import EventEmbeddingConfig, EventEmbeddingDetector
from ..persistence.checkpoint import _resolve_class  # reuse the loader's resolver
from ..training import (
    DetectionSampleAdapter,
    DetectionSampleAdapterConfig,
    DistributionArtifact,
    ErrorHistogramArtifact,
    OnsetLoss,
    OnsetLossConfig,
    OnsetMetric,
    OnsetMetricConfig,
    PredictionScatterArtifact,
    RatioErrorScatterArtifact,
    build_exp45_post_augs,
    train,
)


# ─────────────────────────── config loading ──────────────────────────

def _load_config_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_from_json(path: Path) -> Any:
    """Read a JSON file shaped like ``{"__class__": "...", ...}`` and
    instantiate the config class, stripping any underscore-prefixed
    comment keys the file may carry for human notes.
    """
    data = _load_config_json(path)
    cls_name = data.pop("__class__")
    # Strip comment-only keys and unknown fields.
    for k in list(data):
        if k.startswith("_"):
            data.pop(k)
    cls = _resolve_class(cls_name)
    known = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    # tuple restoration for fields that store tuples-of-tuples in JSON.
    for key, val in list(filtered.items()):
        if isinstance(val, list):
            # list-of-lists → tuple-of-tuples (split_ratios, etc).
            if val and isinstance(val[0], list):
                filtered[key] = tuple(tuple(x) for x in val)
            else:
                filtered[key] = tuple(val)
    return cls(**filtered)


# ─────────────────────────── CLI ─────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a taiko2 model from config JSONs.",
    )
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--runs-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "runs",
    )
    p.add_argument(
        "--config-dir", type=Path, required=True,
        help="Directory containing model.json, loss.json, trainer.json, "
             "data.json, adapter.json.",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under osu/taiko2/datasets/).")
    p.add_argument(
        "--datasets-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--weighted-sampling", action="store_true", default=True,
        help="Use per-target-class balanced sampling (exp 45 default).",
    )
    p.add_argument(
        "--no-weighted-sampling", dest="weighted_sampling",
        action="store_false",
    )
    p.add_argument(
        "--weighted-power", type=float, default=0.5,
        help="power in 1/(count+1)^power weighting (default 0.5 = sqrt).",
    )
    p.add_argument(
        "--no-augmentation", action="store_true",
        help="Skip the exp 45 augmentation set (bare training).",
    )
    return p.parse_args(argv)


# ─────────────────────────── main ────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg_dir = args.config_dir.resolve()
    if not cfg_dir.is_dir():
        print(f"ERROR: --config-dir {cfg_dir} not a directory", file=sys.stderr)
        return 2

    # 1. Load configs.
    model_cfg: EventEmbeddingConfig = _build_from_json(cfg_dir / "model.json")
    loss_cfg: OnsetLossConfig = _build_from_json(cfg_dir / "loss.json")
    trainer_cfg: TrainerConfig = _build_from_json(cfg_dir / "trainer.json")
    data_cfg: TaikoDetectionSamplerConfig = _build_from_json(cfg_dir / "data.json")
    adapter_cfg: DetectionSampleAdapterConfig = _build_from_json(
        cfg_dir / "adapter.json",
    )

    # Sanity: adapter/model/data must agree on b_pred + window geometry.
    if adapter_cfg.b_pred != model_cfg.b_pred:
        raise ValueError(
            f"adapter.b_pred ({adapter_cfg.b_pred}) != "
            f"model.b_pred ({model_cfg.b_pred})"
        )
    if data_cfg.a_bins != model_cfg.a_bins or data_cfg.b_bins != model_cfg.b_bins:
        raise ValueError(
            f"data a/b_bins ({data_cfg.a_bins}/{data_cfg.b_bins}) != "
            f"model a/b_bins ({model_cfg.a_bins}/{model_cfg.b_bins})"
        )

    # 2. Resolve dataset root and wire train / val samplers. Splits
    # come from `data_cfg`; we only override `split` here so the same
    # config file works for both.
    ds_root = args.datasets_dir / args.dataset
    if not ds_root.is_dir():
        print(f"ERROR: dataset not found: {ds_root}", file=sys.stderr)
        return 2
    data_cfg_train = _with(data_cfg, split="train", dataset_root=ds_root)
    data_cfg_val = _with(data_cfg, split="val", dataset_root=ds_root)

    pipeline: AugmentationPipeline | None = None
    if not args.no_augmentation:
        pipeline = AugmentationPipeline(
            pre=(),
            post=tuple(build_exp45_post_augs(seed=trainer_cfg.seed)),
        )

    train_sampler = TaikoDetectionSampler(data_cfg_train, pipeline=pipeline)
    val_sampler = TaikoDetectionSampler(data_cfg_val)
    train_sampler.load_data(progress=True)
    val_sampler.load_data(progress=True)

    # 3. Model + loss + adapter + metrics + artifacts.
    model = EventEmbeddingDetector(model_cfg)
    loss = OnsetLoss(loss_cfg)
    adapter = DetectionSampleAdapter(adapter_cfg)

    # Train metrics are per-epoch running means (reset each epoch) — the
    # loop uses them for the tqdm postfix so "good_avg / bad_avg" show
    # during training, not just at eval boundaries.
    train_metrics = MetricSet(
        OnsetMetric(OnsetMetricConfig(b_pred=model_cfg.b_pred)),
    )
    val_metrics = MetricSet(
        OnsetMetric(OnsetMetricConfig(b_pred=model_cfg.b_pred)),
    )
    eval_artifacts = [
        PredictionScatterArtifact(b_pred=model_cfg.b_pred),
        DistributionArtifact(b_pred=model_cfg.b_pred),
        RatioErrorScatterArtifact(b_pred=model_cfg.b_pred),
        ErrorHistogramArtifact(b_pred=model_cfg.b_pred),
    ]

    # 4. Weighted sampling (exp 45 default).
    train_weights = None
    if args.weighted_sampling:
        train_weights = train_sampler.compute_target_weights(
            b_pred=model_cfg.b_pred, power=args.weighted_power,
        )

    # 5. Run.
    spec = RunSpec(root=args.runs_dir, name=args.run_name)
    print(f"run dir: {spec.run_dir}")
    state = train(
        spec=spec,
        trainer_config=trainer_cfg,
        model=model,
        loss=loss,
        adapter=adapter,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        eval_artifacts=eval_artifacts,
        train_weights=train_weights,
        device=args.device,
    )
    print(
        f"done. final step={state.step:,} epoch={state.epoch} "
        f"best_metric={state.best_metric} at step={state.best_metric_step}",
    )
    return 0


def _with(
    cfg: TaikoDetectionSamplerConfig, **overrides,
) -> TaikoDetectionSamplerConfig:
    """`dataclasses.replace` but works with the sampler's `split_overrides`
    resolving path — normalize overrides to a clean copy."""
    from dataclasses import replace as _replace
    return _replace(cfg, **overrides)


if __name__ == "__main__":
    raise SystemExit(main())
