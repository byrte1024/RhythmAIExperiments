"""Standalone CLI wrapper around `inference.corpus.run_infer_corpus`.

Loads a predictor spec JSON, builds the val sampler for a dataset,
and calls the shared library function. The same function also powers
the training-loop `InferCorpusHook`, so per-eval and one-shot runs
produce identical output shapes.

Outputs land under ``{run_dir}/infer_corpus/eval_{step}/`` where
``run_dir`` is derived from the checkpoint path's grandparent (e.g.
``runs/{run_name}/checkpoints/best.pt`` → ``runs/{run_name}/``) and
``step`` comes from the loaded checkpoint's ``training_state.step``.
Can be overridden with ``--out-dir``.

Example
=======

::

    osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.infer_corpus \\
        --config osu/taiko2/experiments/002-exp45-full/config/infer.json \\
        --corpus-config osu/taiko2/configs/infer_corpus_default.json \\
        --dataset taiko2_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ..data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from ..inference.corpus import InferCorpusConfig, run_infer_corpus
from ..inference.spec import assemble_predictor, build_config, load_spec


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate charts for a fraction of a dataset via "
            "ChartPredictor, save per-chart metrics + comparisons + "
            "averaged summary."
        ),
    )
    p.add_argument("--config", type=Path, required=True,
                   help="Predictor spec JSON (same shape as cli.infer).")
    p.add_argument("--corpus-config", type=Path, default=None,
                   help=(
                       "InferCorpusConfig JSON (fraction / seed / "
                       "conditioning modes). If omitted, defaults apply "
                       "(fraction=0.1, both modes, save_bundles=True)."
                   ))
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Override the spec's `checkpoint` field.")
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under --datasets-dir) or path.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--out-dir", type=Path, default=None,
                   help=(
                       "Override output directory. Default: "
                       "`{run_dir}/infer_corpus/eval_{step}/`."
                   ))
    p.add_argument("--split", default="val")
    p.add_argument("--split-ratios", default="train:0.9,val:0.1")
    p.add_argument("--split-seed", type=int, default=42)
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


def _load_corpus_config(path: Path | None) -> InferCorpusConfig:
    if path is None:
        return InferCorpusConfig()
    node = json.loads(path.read_text(encoding="utf-8"))
    # Allow a bare dict (no __class__) for backwards-friendly configs.
    if "__class__" not in node:
        node = {
            "__class__": "osu.taiko2.inference.corpus:InferCorpusConfig",
            **node,
        }
    cfg = build_config(node)
    if not isinstance(cfg, InferCorpusConfig):
        raise TypeError(
            f"corpus-config must resolve to InferCorpusConfig, got {type(cfg)}"
        )
    return cfg


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

    device = torch.device(args.device)
    print(f"[infer_corpus] checkpoint: {spec['checkpoint']}")
    print(f"[infer_corpus] device:     {device}")
    predictor, meta = assemble_predictor(spec=spec, device=device)
    ckpt_step = int(meta.training_state.step)
    print(f"[infer_corpus] ckpt step:  {ckpt_step}")

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
    print(
        f"[infer_corpus] split '{args.split}': "
        f"{sampler.count_charts():,} charts total"
    )

    corpus_cfg = _load_corpus_config(args.corpus_config)
    print(
        f"[infer_corpus] corpus cfg: fraction={corpus_cfg.fraction} "
        f"seed={corpus_cfg.seed} modes={corpus_cfg.conditioning_modes}"
    )

    if args.out_dir is not None:
        out_root = args.out_dir.resolve()
    else:
        ckpt_path = Path(spec["checkpoint"]).resolve()
        run_dir = ckpt_path.parent.parent
        out_root = (run_dir / "infer_corpus" / f"eval_{ckpt_step}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[infer_corpus] out_root:   {out_root}")

    summary = run_infer_corpus(
        predictor=predictor,
        val_sampler=sampler,
        ds_root=ds_root,
        out_dir=out_root,
        config=corpus_cfg,
        step=ckpt_step,
        progress=not args.no_progress,
    )

    print("\n[infer_corpus] averaged summary:")
    for k in sorted(summary):
        print(f"  {k:<60} {summary[k]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
