"""Run a sampler-ablation matrix against a trained DiffusionDetector
checkpoint.

For each row in the ablation matrix:
  1. Build the predictor + decoder with that sampler config.
  2. Run AR-corpus inference on a fraction of the val split.
  3. Write per-chart metrics + pooled summary to a per-variant
     subdirectory.

Reuses ``inference.corpus.run_infer_corpus`` so the metric definitions
match the in-training AR-corpus hook exactly.

Output layout::

    {output_root}/
      summary.csv           # one row per variant: pooled metrics
      summary.json          # same data + ablation matrix metadata
      {variant_name}/
        gt_cond/
          comparisons_summary.json
          metrics_summary.json
          ...
        fixed_cond/
          ...

Usage::

    osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.diffusion_sampler_ablation \\
        --checkpoint runs/exp_014_diffusion/checkpoints/best.pt \\
        --base-spec osu/taiko2/experiments/014-diffusion/config/infer.json \\
        --matrix osu/taiko2/experiments/014-diffusion/config/ablation_matrix.json \\
        --dataset taiko2_v1 \\
        --out-dir osu/taiko2/experiments/014-diffusion/ablations

The matrix JSON is a list of variants, each with a name + a partial
sampler config to merge into the base spec's decoder.sampler_config.
Example::

    [
      {"name": "ddim_16_e0_n1",  "sampler_overrides": {"n_inference_steps": 16, "eta": 0.0}, "decoder_overrides": {"n_samples": 1}},
      {"name": "ddim_4_e0_n1",   "sampler_overrides": {"n_inference_steps": 4,  "eta": 0.0}, "decoder_overrides": {"n_samples": 1}},
      {"name": "ddim_16_e1_n1",  "sampler_overrides": {"n_inference_steps": 16, "eta": 1.0}, "decoder_overrides": {"n_samples": 1}},
      {"name": "ddim_16_e0_n4",  "sampler_overrides": {"n_inference_steps": 16, "eta": 0.0}, "decoder_overrides": {"n_samples": 4}},
      {"name": "ddpm_64_e1_n1",  "sampler_overrides": {"__class__": "osu.taiko2.domain.diffusion:DiffusionSamplerConfig", "n_inference_steps": 64, "eta": 1.0}, "decoder_overrides": {"n_samples": 1}}
    ]

Sampler-overrides override fields inside ``decoder.config.sampler_config``;
decoder-overrides override fields inside ``decoder.config`` itself.
A new ``__class__`` in sampler_overrides switches sampler types
(e.g. DDIM → DDPM); otherwise the base spec's class is kept.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

from ..data_samplers import TaikoDetectionSampler, TaikoDetectionSamplerConfig
from ..inference.corpus import InferCorpusConfig, run_infer_corpus
from ..inference.spec import assemble_predictor, build_config, load_spec


# ─────────────────────────── Args ─────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run sampler-ablation matrix on a trained "
                    "DiffusionDetector checkpoint.",
    )
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Override the base spec's checkpoint field.")
    p.add_argument("--base-spec", type=Path, required=True,
                   help="JSON inference spec (model + decoder + samplers + "
                        "input_builder). Sampler/decoder fields are "
                        "overlaid per variant from --matrix.")
    p.add_argument("--matrix", type=Path, required=True,
                   help="JSON list of variants. Each variant has "
                        "name / sampler_overrides / decoder_overrides.")
    p.add_argument("--corpus-config", type=Path, default=None,
                   help="InferCorpusConfig JSON (fraction / seed / modes). "
                        "Defaults: fraction=0.1, both modes, save_bundles=False.")
    p.add_argument("--dataset", required=True,
                   help="Dataset name (under --datasets-dir) or path.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets")
    p.add_argument("--split", default="val")
    p.add_argument("--split-ratios", default="train:0.9,val:0.1")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Root directory for ablation outputs.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--variants", default=None,
                   help="Comma-separated variant names to run (default: all "
                        "in the matrix). Useful for re-running a subset.")
    return p.parse_args(argv)


def _parse_split_ratios(raw: str) -> tuple[tuple[str, float], ...]:
    parts: list[tuple[str, float]] = []
    for frag in raw.split(","):
        name, _, ratio = frag.strip().partition(":")
        if not name or not ratio:
            raise ValueError(f"bad split-ratios fragment {frag!r}")
        parts.append((name.strip(), float(ratio)))
    return tuple(parts)


def _resolve_dataset(name_or_path: str, datasets_dir: Path) -> Path:
    p = Path(name_or_path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (datasets_dir / name_or_path).resolve()


# ─────────────────────────── Variant overlay ──────────────────────────


def _apply_variant_overrides(
    base_spec: dict[str, Any],
    sampler_overrides: dict[str, Any],
    decoder_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a new spec dict with the sampler / decoder overrides
    applied. Doesn't mutate the input."""
    spec = copy.deepcopy(base_spec)
    decoder_cfg = spec["decoder"]["config"]
    # Sampler-level overrides land inside decoder.config.sampler_config.
    sampler_cfg = decoder_cfg.get("sampler_config")
    if sampler_cfg is None:
        raise ValueError(
            "base spec's decoder.config has no sampler_config field "
            "— is it a DiffusionDecoderConfig?"
        )
    for k, v in sampler_overrides.items():
        sampler_cfg[k] = v
    decoder_cfg["sampler_config"] = sampler_cfg
    # Decoder-level overrides land directly on decoder.config.
    for k, v in decoder_overrides.items():
        decoder_cfg[k] = v
    spec["decoder"]["config"] = decoder_cfg
    return spec


# ─────────────────────────── Summary aggregator ───────────────────────


# AR-corpus comparison fields we roll up into the per-variant summary
# row. Keep these stable so the resulting CSV has consistent columns.
_SUMMARY_KEYS_GT: tuple[str, ...] = (
    "matched_rate", "close_rate", "far_rate", "hallucination_rate",
    "error_mean_ms", "error_median_ms", "error_p90_ms",
    "density_ratio", "hi_pspace", "dc_human", "oc_human",
)


def _write_aggregate_summary(
    *,
    summary_rows: list[dict[str, Any]],
    out_dir: Path,
    base_spec: str,
    matrix_path: str,
    ds_root: str,
    split: str,
) -> None:
    """Write the aggregate ``summary.csv`` + ``summary.json`` from the
    rows accumulated so far. Called after every variant so the partial
    sweep is queryable on disk in real time."""
    if not summary_rows:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in summary_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with (out_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8",
    ) as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_spec": base_spec,
                "matrix": matrix_path,
                "dataset": ds_root,
                "split": split,
                "rows": summary_rows,
            },
            f, indent=2,
        )


def _read_summary(variant_dir: Path) -> dict[str, float]:
    """Read the gt_cond comparisons_summary.json and pull a flat dict
    of medians for the canonical fields. Missing fields land as None."""
    f = variant_dir / "gt_cond" / "comparisons_summary.json"
    if not f.exists():
        return {}
    data = json.load(f.open(encoding="utf-8"))
    fields = data.get("fields", {})
    out: dict[str, float] = {"n_charts": float(data.get("n", 0))}
    for k in _SUMMARY_KEYS_GT:
        if k in fields and "median" in fields[k]:
            out[f"{k}_median"] = float(fields[k]["median"])
    return out


# ─────────────────────────── Main ─────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    base_spec = load_spec(config=args.base_spec, config_json=None)
    if args.checkpoint is not None:
        base_spec["checkpoint"] = str(args.checkpoint)

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not isinstance(matrix, list) or not all(
        isinstance(v, dict) and "name" in v for v in matrix
    ):
        raise SystemExit(
            f"matrix file {args.matrix} must be a JSON list of "
            f"dicts each with a 'name' field"
        )
    selected_names: set[str] | None = None
    if args.variants:
        selected_names = {n.strip() for n in args.variants.split(",")}

    ds_root = _resolve_dataset(args.dataset, args.datasets_dir)
    split_ratios = _parse_split_ratios(args.split_ratios)
    split = args.split
    split_seed = args.split_seed
    device = torch.device(args.device)

    # Build the val sampler once; reuse across variants.
    sampler_cfg = TaikoDetectionSamplerConfig(
        batch_size=1,                      # AR inference is per-cursor; bs unused
        dataset_root=ds_root, split=split, split_ratios=split_ratios,
        split_seed=split_seed,
    )
    val_sampler = TaikoDetectionSampler(sampler_cfg)
    val_sampler.load_data(progress=not args.no_progress)

    # Default corpus config if not provided.
    if args.corpus_config is not None:
        corpus_cfg = build_config(json.loads(
            args.corpus_config.read_text(encoding="utf-8"),
        ))
    else:
        corpus_cfg = InferCorpusConfig(
            fraction=0.1, seed=42,
            conditioning_modes=("gt_cond", "fixed_cond"),
            save_bundles=False,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    # Headline metrics surfaced inline after each variant so the user can
    # decide whether the sweep is worth continuing. Pulled from the
    # variant's gt_cond comparisons_summary.json + the matched fixed_cond.
    _HEADLINE_KEYS: tuple[tuple[str, str], ...] = (
        ("matched_rate", "matched_rate_median"),
        ("error_median_ms", "error_median_ms_median"),
        ("density_ratio", "density_ratio_median"),
        ("hi_pspace", "hi_pspace_median"),
        ("hallucination_rate", "hallucination_rate_median"),
    )

    for variant in matrix:
        name = variant["name"]
        if selected_names is not None and name not in selected_names:
            continue
        sampler_overrides = variant.get("sampler_overrides", {}) or {}
        decoder_overrides = variant.get("decoder_overrides", {}) or {}
        variant_dir = args.out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ablation] running variant {name!r}")
        print(
            f"  sampler_overrides={sampler_overrides!r} "
            f"decoder_overrides={decoder_overrides!r}",
        )

        spec = _apply_variant_overrides(
            base_spec, sampler_overrides, decoder_overrides,
        )
        predictor, _meta = assemble_predictor(spec=spec, device=device)
        run_infer_corpus(
            predictor=predictor,
            val_sampler=val_sampler,
            ds_root=ds_root,
            out_dir=variant_dir,
            config=corpus_cfg,
            progress=not args.no_progress,
        )

        # Roll up per-variant summary.
        row = {"variant": name}
        row.update({"sampler_" + k: v for k, v in sampler_overrides.items()})
        row.update({"decoder_" + k: v for k, v in decoder_overrides.items()})
        row.update(_read_summary(variant_dir))
        summary_rows.append(row)

        # Streaming summary write — the aggregate CSV + JSON are
        # rewritten after every variant, so the user can `cat
        # summary.csv` mid-sweep and decide whether to ctrl-C.
        _write_aggregate_summary(
            summary_rows=summary_rows, out_dir=args.out_dir,
            base_spec=str(args.base_spec), matrix_path=str(args.matrix),
            ds_root=str(ds_root), split=split,
        )

        # Headline print — one line that summarizes the variant's gt_cond
        # result against the existing pre-run #007 baseline. Keeps the
        # operator informed without needing to grep JSON.
        print(f"[ablation]   ↳ {name} headline:")
        for label, key in _HEADLINE_KEYS:
            v = row.get(key)
            if v is None:
                print(f"      {label:24s}  (missing)")
            elif isinstance(v, float):
                print(f"      {label:24s}  {v:>10.4f}")
            else:
                print(f"      {label:24s}  {v!s:>10}")
        print(f"[ablation]   ↳ summary.csv updated "
              f"({len(summary_rows)}/{len(matrix)} variants done)")

    print(f"[ablation] done. {len(summary_rows)} variants. "
          f"output: {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
