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
    MetronomeHitArtifact,
    GaussianCELoss,
    GaussianCELossConfig,
    LogEmdLoss,
    LogEmdLossConfig,
    MdnLoss,
    MdnLossConfig,
    OnsetLoss,
    OnsetLossConfig,
    OnsetMetric,
    OnsetMetricConfig,
    PredictionHeatmapArtifact,
    RatioErrorHeatmapArtifact,
    RatioHitArtifact,
    TimeStretch,
    build_exp45_post_augs,
    train,
)
from ..training.ratio_loss import RatioLoss, RatioLossConfig
from ..training.augmentations import CursorShift
from ..models.ratio_detector import RatioDetector, RatioDetectorConfig


# ─────────────────────────── config loading ──────────────────────────

def _load_config_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_from_json(path: Path) -> Any:
    """Read a JSON file shaped like ``{"__class__": "...", ...}`` and
    instantiate the config class, stripping any underscore-prefixed
    comment keys the file may carry for human notes.

    Recurses into nested dict fields that themselves carry
    ``__class__`` (polymorphic sub-configs, e.g. #014's diffusion
    schedule / process / denoiser), instantiating each as its declared
    concrete class.
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
    for key, val in list(filtered.items()):
        if isinstance(val, dict) and "__class__" in val:
            # Polymorphic sub-config — defer to inference.spec.build_config
            # which knows how to recurse + coerce enums + tuples.
            from ..inference.spec import build_config
            filtered[key] = build_config(val)
        elif isinstance(val, list):
            # list-of-lists → tuple-of-tuples (split_ratios, etc).
            if val and isinstance(val[0], list):
                filtered[key] = tuple(tuple(x) for x in val)
            else:
                filtered[key] = tuple(val)
    return cls(**filtered)


# ─────────────────────────── CLI ─────────────────────────────────────

def _resolve_benchmarks(names_csv: str) -> "list[Any]":
    """`--benchmarks` CSV → list of BenchmarkMode. Empty string → []."""
    names_csv = (names_csv or "").strip()
    if not names_csv:
        return []
    from ..training.benchmarks import (
        DEFAULT_BENCHMARKS, benchmarks_by_name,
    )
    if names_csv == "all":
        return list(DEFAULT_BENCHMARKS)
    return benchmarks_by_name(
        [n.strip() for n in names_csv.split(",") if n.strip()]
    )


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
    p.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume the SAME run from the last finished eval. Loads "
            "`{run_dir}/eval_{step}/checkpoint.pt` with the largest step, "
            "restores model / optimizer / scheduler / training-state, and "
            "truncates metrics.jsonl + later eval_{N}/ dirs so stats "
            "match the checkpoint exactly. Without this flag, the loop "
            "auto-resumes from latest.pt if present (legacy behavior)."
        ),
    )
    p.add_argument(
        "--infer-corpus-spec", type=Path, default=None,
        help=(
            "Predictor spec JSON for per-eval corpus inference (same "
            "shape as cli.infer). When both --infer-corpus-spec AND "
            "--infer-corpus-config are provided, `InferCorpusHook` is "
            "installed and runs after every eval, merging its "
            "averaged summary into val_metrics so the corpus scalars "
            "are tracked + auto-graphed alongside onset metrics."
        ),
    )
    p.add_argument(
        "--infer-corpus-config", type=Path, default=None,
        help="InferCorpusConfig JSON (fraction / seed / conditioning_modes / etc).",
    )
    p.add_argument(
        "--infer-corpus-every", type=int, default=1,
        help="Run the corpus hook every N evals (default 1 = every eval).",
    )
    p.add_argument(
        "--train-noaug-fraction", type=float, default=0.0,
        help=(
            "Fraction of train split to evaluate WITHOUT augmentations "
            "after each eval. Metrics land under `train_noaug/*` in the "
            "val_metrics stream. Set to 0 (default) to disable. "
            "Distinguishes overfitting from data-ceiling."
        ),
    )
    p.add_argument(
        "--benchmarks", default="",
        help=(
            "Comma-separated list of benchmark mode names (or 'all'). "
            "See training/benchmarks.py for available modes. Each mode "
            "runs over `--benchmark-fraction` of the val split per eval "
            "and lands metrics under `bench/{mode}/*`."
        ),
    )
    p.add_argument(
        "--benchmark-fraction", type=float, default=0.05,
        help="Fraction of val split per benchmark pass (default 0.05 = 5%).",
    )
    p.add_argument(
        "--benchmark-seed", type=int, default=42,
        help="Seed for benchmark sample selection + per-mode rng.",
    )
    p.add_argument(
        "--time-stretch-prob", type=float, default=0.0,
        help=(
            "Probability of applying the TimeStretch augmentation per "
            "sample. 0.0 (default) disables it. When > 0, the aug is "
            "prepended to the post-augmentation list so downstream augs "
            "operate on the already-stretched sample."
        ),
    )
    p.add_argument(
        "--time-stretch-max-scale", type=float, default=1.4,
        help=(
            "Maximum stretch factor; per-call draw is log-uniform in "
            "[1/max_scale, max_scale]. Default 1.4 corresponds to "
            "about +/-40%% around normal speed."
        ),
    )
    p.add_argument(
        "--rollout-eval-n-charts", type=int, default=32,
        help=(
            "Framewise rollout hook: number of val samples drawn per "
            "eval boundary for the full DDIM convergence rollout."
        ),
    )
    p.add_argument(
        "--rollout-noaug-n-charts", type=int, default=32,
        help="Framewise rollout hook: same on the train-noaug subset.",
    )
    p.add_argument(
        "--rollout-t-inf-steps", type=int, default=16,
        help="Framewise rollout hook: T_inf for the DDIM sampler.",
    )
    p.add_argument(
        "--rollout-every-n-evals", type=int, default=1,
        help="Run the framewise rollout hook every Nth eval.",
    )
    p.add_argument(
        "--rollout-n-gif-samples", type=int, default=5,
        help="Framewise rollout hook: number of representative GIF samples.",
    )
    p.add_argument(
        "--no-rollout-hook", action="store_true",
        help="Disable the framewise rollout hook even in framewise mode.",
    )
    p.add_argument(
        "--cursor-shift-prob", type=float, default=0.0,
        help=(
            "Probability of shifting the cursor forward between the "
            "current event and the next (pre-sample augmentation). "
            "Creates training examples with non-zero offset for the "
            "ratio detector's offset head. 0.0 (default) disables."
        ),
    )
    p.add_argument(
        "--compile", action="store_true",
        help=(
            "Apply torch.compile to the model for faster training. "
            "Requires triton. First few batches are slow (compilation); "
            "subsequent batches are 20-40%% faster on modern GPUs."
        ),
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
    model_cfg = _build_from_json(cfg_dir / "model.json")
    loss_cfg = _build_from_json(cfg_dir / "loss.json")
    trainer_cfg: TrainerConfig = _build_from_json(cfg_dir / "trainer.json")
    data_cfg = _build_from_json(cfg_dir / "data.json")
    adapter_cfg = _build_from_json(cfg_dir / "adapter.json")

    # Sanity: adapter/model/data must agree on b_pred + window geometry
    # (onset-detector modes only; typing mode has different config shapes).
    from ..domain.typing import TypingModelConfig
    if not isinstance(model_cfg, TypingModelConfig):
        if hasattr(adapter_cfg, "b_pred") and hasattr(model_cfg, "b_pred"):
            if adapter_cfg.b_pred != model_cfg.b_pred:
                raise ValueError(
                    f"adapter.b_pred ({adapter_cfg.b_pred}) != "
                    f"model.b_pred ({model_cfg.b_pred})"
                )
        if hasattr(data_cfg, "a_bins") and hasattr(model_cfg, "a_bins"):
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
    # Typing mode builds its own samplers later; onset modes use
    # TaikoDetectionSampler with the audio augmentation pipeline.
    if not isinstance(model_cfg, TypingModelConfig):
        data_cfg_train = _with(data_cfg, split="train", dataset_root=ds_root)
        data_cfg_val = _with(data_cfg, split="val", dataset_root=ds_root)

        pipeline: AugmentationPipeline | None = None
        if not args.no_augmentation:
            freq_boundary = getattr(adapter_cfg, "freq_roll_boundary", None)
            if freq_boundary is None:
                feat_rows = getattr(adapter_cfg, "feature_rows", None)
                freq_boundary = feat_rows[0] if feat_rows else None
            post_augs = build_exp45_post_augs(
                seed=trainer_cfg.seed,
                freq_roll_section_boundary=freq_boundary,
            )
            if args.time_stretch_prob > 0.0:
                post_augs = [
                    TimeStretch(
                        prob=args.time_stretch_prob,
                        max_scale=args.time_stretch_max_scale,
                        seed=trainer_cfg.seed,
                    ),
                ] + post_augs
            pre_augs: list = []
            if args.cursor_shift_prob > 0.0:
                pre_augs.append(
                    CursorShift(prob=args.cursor_shift_prob, seed=trainer_cfg.seed),
                )
            pipeline = AugmentationPipeline(
                pre=tuple(pre_augs), post=tuple(post_augs),
            )

        train_sampler = TaikoDetectionSampler(data_cfg_train, pipeline=pipeline)
        val_sampler = TaikoDetectionSampler(data_cfg_val)
        train_sampler.load_data(progress=True)
        val_sampler.load_data(progress=True)

    # 3. Model + loss + adapter + metrics + artifacts.
    is_ratio_mode = isinstance(model_cfg, RatioDetectorConfig)
    from ..models.diffusion_detector import (
        DiffusionDetector, DiffusionDetectorConfig,
    )
    from ..models.framewise_diffusion_detector import (
        FramewiseDiffusionDetector, FramewiseDiffusionDetectorConfig,
    )
    from ..models.framewise_detector import (
        FramewiseDetector, FramewiseDetectorConfig,
    )
    is_diffusion_mode = isinstance(model_cfg, DiffusionDetectorConfig)
    is_framewise_diffusion = isinstance(
        model_cfg, FramewiseDiffusionDetectorConfig,
    )
    is_framewise_bce = isinstance(model_cfg, FramewiseDetectorConfig)
    from ..domain.typing import TypingModelConfig
    from ..models.typing_model import TypingTransformer
    is_typing_mode = isinstance(model_cfg, TypingModelConfig)
    if is_typing_mode:
        model = TypingTransformer(model_cfg)
    elif is_ratio_mode:
        model = RatioDetector(model_cfg)
    elif is_framewise_bce:
        model = FramewiseDetector(model_cfg)
    elif is_framewise_diffusion:
        model = FramewiseDiffusionDetector(model_cfg)
    elif is_diffusion_mode:
        model = DiffusionDetector(model_cfg)
    else:
        model = EventEmbeddingDetector(model_cfg)
    from ..training.typing_loss import TypingLoss, TypingLossConfig
    if isinstance(loss_cfg, TypingLossConfig):
        loss = TypingLoss(loss_cfg)
    elif isinstance(loss_cfg, GaussianCELossConfig):
        loss = GaussianCELoss(loss_cfg)
    elif isinstance(loss_cfg, LogEmdLossConfig):
        loss = LogEmdLoss(loss_cfg)
    elif isinstance(loss_cfg, MdnLossConfig):
        loss = MdnLoss(loss_cfg)
    elif isinstance(loss_cfg, RatioLossConfig):
        loss = RatioLoss(loss_cfg)
    elif isinstance(loss_cfg, OnsetLossConfig):
        loss = OnsetLoss(loss_cfg)
    else:
        from ..training.framewise_bce_loss import (
            FramewiseBCELoss, FramewiseBCELossConfig,
        )
        from ..training.framewise_focal_loss import (
            FramewiseFocalLoss, FramewiseFocalLossConfig,
        )
        from ..training.diffusion_loss import DiffusionLoss, DiffusionLossConfig
        from ..training.framewise_diffusion_loss import (
            FramewiseDiffusionLoss, FramewiseDiffusionLossConfig,
        )
        if isinstance(loss_cfg, FramewiseFocalLossConfig):
            loss = FramewiseFocalLoss(loss_cfg)
        elif isinstance(loss_cfg, FramewiseBCELossConfig):
            loss = FramewiseBCELoss(loss_cfg)
        elif isinstance(loss_cfg, FramewiseDiffusionLossConfig):
            loss = FramewiseDiffusionLoss(loss_cfg)
        elif isinstance(loss_cfg, DiffusionLossConfig):
            loss = DiffusionLoss(loss_cfg)
        else:
            raise TypeError(f"unsupported loss config: {type(loss_cfg).__name__}")
    if (is_diffusion_mode or is_framewise_diffusion) and hasattr(loss, "bind_model"):
        loss.bind_model(model)
    if is_typing_mode:
        from ..data_samplers.typing import TypingSampler, TypingSamplerConfig
        from ..training.typing_adapter import TypingSampleAdapter, TypingAdapterConfig
        from ..training.metrics_typing import TypingMetric
        from ..training.typing_artifacts import TypingConfusionArtifact

        typing_adapter_cfg = adapter_cfg if isinstance(adapter_cfg, TypingAdapterConfig) else TypingAdapterConfig()
        adapter = TypingSampleAdapter(typing_adapter_cfg, training=True)
        train_metrics = MetricSet(TypingMetric(prefix="typing"))
        val_metrics = MetricSet(TypingMetric(prefix="typing"))
        eval_artifacts: list = [TypingConfusionArtifact()]
        # Override samplers with typing-specific ones
        typing_data_cfg = data_cfg if isinstance(data_cfg, TypingSamplerConfig) else TypingSamplerConfig(
            dataset_root=ds_root, batch_size=trainer_cfg.batch_size,
        )
        data_cfg_train = _with(typing_data_cfg, split="train", dataset_root=ds_root)
        data_cfg_val = _with(typing_data_cfg, split="val", dataset_root=ds_root)
        train_sampler = TypingSampler(data_cfg_train)
        val_sampler = TypingSampler(data_cfg_val)
        train_sampler.load_data(progress=True)
        val_sampler.load_data(progress=True)
        train_weights = None
    else:
        if is_ratio_mode:
            adapter_cfg = _with(adapter_cfg, ratio_mode=True)
        if is_framewise_bce or is_framewise_diffusion:
            from ..training.framewise_adapter import (
                FramewiseSampleAdapter, FramewiseSampleAdapterConfig,
            )
            if isinstance(adapter_cfg, FramewiseSampleAdapterConfig):
                fw_cfg = adapter_cfg
            else:
                fw_cfg = FramewiseSampleAdapterConfig(
                    b_pred=model_cfg.b_pred,
                    binary_only=is_framewise_bce,
                )
            adapter = FramewiseSampleAdapter(fw_cfg)
        else:
            adapter = DetectionSampleAdapter(adapter_cfg)

        if is_framewise_bce:
            from ..training.framewise_metric import (
                FramewiseMetric, FramewiseMetricConfig,
            )
            train_metrics = MetricSet()
            val_metrics = MetricSet(FramewiseMetric(FramewiseMetricConfig()))
            from ..training.framewise_artifacts import (
                FramewiseDistributionArtifact,
                FramewiseHeatmapArtifact,
            )
            from ..training.framewise_diagnostics_artifact import (
                FramewiseDiagnosticsArtifact,
            )
            eval_artifacts = [
                FramewiseHeatmapArtifact(),
                FramewiseDistributionArtifact(),
                FramewiseDiagnosticsArtifact(),
            ]
        elif is_framewise_diffusion:
            train_metrics = MetricSet()
            val_metrics = MetricSet()
            from ..training.framewise_artifacts import (
                FramewiseDistributionArtifact,
                FramewiseHeatmapArtifact,
            )
            eval_artifacts = [
                FramewiseHeatmapArtifact(),
                FramewiseDistributionArtifact(),
            ]
        else:
            train_metrics = MetricSet(
                OnsetMetric(OnsetMetricConfig(b_pred=model_cfg.b_pred)),
            )
            val_metrics = MetricSet(
                OnsetMetric(OnsetMetricConfig(b_pred=model_cfg.b_pred)),
            )
            eval_artifacts = [
                PredictionHeatmapArtifact(b_pred=model_cfg.b_pred),
                DistributionArtifact(b_pred=model_cfg.b_pred),
                RatioErrorHeatmapArtifact(b_pred=model_cfg.b_pred),
                ErrorHistogramArtifact(b_pred=model_cfg.b_pred),
                RatioHitArtifact(b_pred=model_cfg.b_pred),
                MetronomeHitArtifact(b_pred=model_cfg.b_pred),
            ]
    if not is_typing_mode:
        if getattr(model_cfg, "n_mdn_components", 0) > 0:
            from ..training.artifacts import MdnComponentArtifact
            eval_artifacts.append(
                MdnComponentArtifact(
                    b_pred=model_cfg.b_pred,
                    n_components=model_cfg.n_mdn_components,
                ),
            )
        if is_ratio_mode:
            from ..training.artifacts import RatioDecompositionArtifact
            eval_artifacts.append(
                RatioDecompositionArtifact(
                    b_pred=model_cfg.b_pred,
                    offset_bins=model_cfg.offset_bins,
                    ratio_bins=model_cfg.ratio_bins,
                ),
            )

    # 4. Weighted sampling (exp 45 default).
    # Typing mode sets train_weights=None above; onset modes use
    # per-target-class balanced sampling.
    if not is_typing_mode:
        train_weights = None
        if args.weighted_sampling and not (is_framewise_diffusion or is_framewise_bce):
            train_weights = train_sampler.compute_target_weights(
                b_pred=model_cfg.b_pred, power=args.weighted_power,
            )

    # 5. Run.
    spec = RunSpec(root=args.runs_dir, name=args.run_name)
    print(f"run dir: {spec.run_dir}")

    # Optional per-eval corpus inference hook.
    pre_hooks: list = []
    if args.infer_corpus_spec is not None and args.infer_corpus_config is not None:
        import json as _json
        from ..inference.spec import build_config, load_spec
        from ..inference.corpus import InferCorpusConfig
        from ..training.hooks import InferCorpusHook

        predictor_spec = load_spec(
            config=args.infer_corpus_spec, config_json=None,
        )
        corpus_node = _json.loads(
            args.infer_corpus_config.read_text(encoding="utf-8"),
        )
        if "__class__" not in corpus_node:
            corpus_node = {
                "__class__": "osu.taiko2.inference.corpus:InferCorpusConfig",
                **corpus_node,
            }
        corpus_cfg = build_config(corpus_node)
        if not isinstance(corpus_cfg, InferCorpusConfig):
            raise TypeError("--infer-corpus-config must resolve to InferCorpusConfig")
        pre_hooks.append(InferCorpusHook(
            spec=spec,
            model=model,
            corpus_config=corpus_cfg,
            predictor_spec=predictor_spec,
            val_sampler=val_sampler,
            ds_root=ds_root,
            device=args.device,
            every_n_evals=args.infer_corpus_every,
        ))
        print(
            f"[infer-corpus] hook enabled — every {args.infer_corpus_every} "
            f"eval(s), fraction={corpus_cfg.fraction}, "
            f"modes={corpus_cfg.conditioning_modes}"
        )

    # Ratio-mode warmup: translate freeze_evals to freeze_steps.
    if isinstance(loss, RatioLoss):
        import math as _math
        _n_train = train_sampler.count_samples()
        _steps_per_epoch = _math.ceil(_n_train / trainer_cfg.batch_size)
        _eval_every = max(1, int(_steps_per_epoch / max(1e-9, trainer_cfg.evals_per_epoch)))
        loss.set_freeze_limit(_eval_every)
        print(
            f"[ratio] freeze_evals={loss.config.ratio_freeze_evals} "
            f"-> freeze_steps={loss._freeze_step_limit} "
            f"(eval_every={_eval_every})"
        )
        if isinstance(model, RatioDetector):
            model.set_warmup_steps(
                loss._freeze_step_limit,
                freeze_aux_at_boundary=loss.config.freeze_aux_after_warmup,
            )
            print(
                f"[ratio] model warmup_steps={model._warmup_step_limit} "
                f"freeze_aux_at_boundary={model.freeze_aux_at_boundary}"
            )

    extra_hooks: list = []
    if is_framewise_diffusion and not args.no_rollout_hook:
        from ..training.framewise_rollout_hook import (
            FramewiseRolloutHook,
            FramewiseRolloutHookConfig,
        )
        rollout_cfg = FramewiseRolloutHookConfig(
            eval_n_charts=args.rollout_eval_n_charts,
            noaug_n_charts=args.rollout_noaug_n_charts,
            t_inf_steps=args.rollout_t_inf_steps,
            every_n_evals=args.rollout_every_n_evals,
            n_gif_samples=args.rollout_n_gif_samples,
        )
        extra_hooks.append(FramewiseRolloutHook(
            config=rollout_cfg,
            spec=spec,
            model=model,
            adapter=adapter,
            val_sampler=val_sampler,
            train_sampler=train_sampler,
            device=args.device,
        ))
        print(
            f"[framewise-rollout] hook enabled — every "
            f"{rollout_cfg.every_n_evals} eval(s), "
            f"{rollout_cfg.eval_n_charts}*{rollout_cfg.eval_n_windows_per_chart}"
            f" samples, T_inf={rollout_cfg.t_inf_steps}"
        )

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
        pre_hooks=pre_hooks,
        extra_hooks=extra_hooks,
        device=args.device,
        resume=args.resume,
        train_noaug_fraction=args.train_noaug_fraction,
        benchmarks=_resolve_benchmarks(args.benchmarks),
        benchmark_fraction=args.benchmark_fraction,
        benchmark_seed=args.benchmark_seed,
        compile=args.compile,
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
