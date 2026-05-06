# osu!taiko Onset Detection — taiko2

A re-engineered rewrite of [`osu/taiko/`](../taiko/) for predicting
beat onset timings in osu!taiko charts from audio. Same task, same
broad architectural family, but a new code base built around stable
domain ABCs, frozen-dataclass IO types, and a strict
"reproduce experiments from disk alone" discipline.

## Status

Active development. taiko1 is feature-frozen and migrated; new
experiments and architectural changes live here. taiko1's existing
experiments and trained checkpoints remain reproducible from that
directory but are not the basis for ongoing work.

## What's different from taiko1

- **Layered architecture.** Pluggable extension points are ABCs in
  [`domain/`](domain/) (audio sampler, event sampler, model, loss,
  data sampler, augmentation pipeline, trainer hook, predictor).
  Concrete implementations live in sibling packages
  ([`samplers/`](samplers/), [`models/`](models/),
  [`training/`](training/), [`inference/autoregressive/`](inference/autoregressive/)).
- **On-disk format stability.** Old git commits must continue to
  load and run. Manifest / events / features / checkpoint / chart
  bundle formats are versioned and migrate via additive optional
  fields, never silent renames.
- **Frozen dataclasses with slots** for data types; explicit
  type hints everywhere; stdlib-name shadowing prohibited (the
  `secrets`/`types` rename trap from earlier work is documented
  in [`CLAUDE.md`](CLAUDE.md)).
- **Save don't print.** Every metric, checkpoint, and analysis
  output that's worth showing a human is also written to disk
  in a parseable format. The terminal is a debugging aid; the
  log is the source of truth.
- **Experiment hygiene.** Every experiment has its own folder in
  [`experiments/`](experiments/) with `README.md` (narrative +
  results) and `ARCHITECTURE.md` (reference-free, fully
  self-contained spec). Pre-run sections are written before the
  run starts and never edited afterward (use amendments instead).

Conventions for working in this directory are documented in
[`CLAUDE.md`](CLAUDE.md). Read it before changing anything under
`osu/taiko2/`.

## Current state of the work

Direct-bin baseline ([#007](experiments/007-time-stretch/)) holds
the lowest val miss at **0.241** (best, eval 18) [exp_007_time_stretch, step 372,132, val/single/onset/miss=0.2406].

See [`experiments/README.md`](experiments/README.md) for the full
log with status, key result per experiment, and links.

## Repository layout

```
osu/taiko2/
  domain/              # ABCs + frozen data types only — no I/O, no logic
  samplers/            # AudioSampler / EventSampler implementations
  data_samplers/       # DataSampler — TaikoDetectionSampler
  parsing/             # .osu / .osz → typed Pack/Track
  persistence/         # disk I/O — features, events, manifest, checkpoint
  models/              # EventEmbeddingDetector, RatioDetector, OnsetAugmentedDetector
  training/            # losses, augmentations, training-loop helpers
  inference/           # ChartPredictor implementations
    autoregressive/    # AR loop + decoder/builder ABCs and concretes
  cli/                 # argparse wrappers; one file per command
  configs/             # JSON config presets
  analysis/            # offline analysis tools (loss landscapes, onset survey)
  tests/               # pytest suite

  fetch/               # osu! API client (star ratings, engagement)
  splits.py            # song-grouped train/val/test split helper
  dataset.py           # build_dataset orchestration (.osz packs → cached features)
  credentials.py       # env-var + .env + secrets.json loader

  experiments/         # numbered experiment folders + index README
  datasets/            # built datasets (gitignored)
  runs/                # training run outputs (gitignored)

  pyproject.toml       # deps + dev group
  uv.lock              # committed lockfile (uv sync --frozen --group dev)
  CLAUDE.md            # working conventions
  README.md            # this file
  DATA.md              # how to obtain & build datasets
  LICENSE.md           # licensing & disclaimer
```

## Quick start

### Environment

```bash
uv sync --group dev
```

Python 3.13 is pinned via `pyproject.toml` + `uv.lock`. The
`.venv` lives under `osu/taiko2/.venv/`. Run anything with
`osu/taiko2/.venv/Scripts/python.exe` rather than the ambient
Python.

### Build a dataset

See [`DATA.md`](DATA.md) for how to obtain `.osz` packs. Then:

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.prepare_dataset \
    --name taiko2_v1 \
    --charts-dir <path-to-osz-pack-root>
```

This decodes audio, computes log-mel features, parses chart
events, and writes a built dataset under
`osu/taiko2/datasets/taiko2_v1/`.

### Train a model

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
    --run-name my_run \
    --config-dir osu/taiko2/experiments/007-time-stretch/config \
    --dataset taiko2_v1 --device cuda \
    --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
    --benchmarks all --benchmark-fraction 0.05 \
    --train-noaug-fraction 0.05
```

The training CLI consumes a config directory of JSON files
(model, loss, adapter, trainer, data sampler, infer corpus).
Each experiment under [`experiments/`](experiments/) has its own
`config/` snapshot.

### Run inference

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.infer \
    --config osu/taiko2/experiments/007-time-stretch/config/infer.json \
    --checkpoint runs/my_run/checkpoints/best.pt \
    --audio path/to/song.mp3
```

### Run the test suite

```bash
osu/taiko2/.venv/Scripts/python.exe -m pytest osu/taiko2/tests -q
```

385 tests covering domain types, persistence round-trips,
samplers, augmentations, and the autoregressive predictor.
[385 tests collected, pytest --collect-only osu/taiko2/tests]

## CLI reference

| Command                    | Purpose                                                               |
| -------------------------- | --------------------------------------------------------------------- |
| `prepare_dataset`        | Build a dataset from `.osz` packs.                                  |
| `train`                  | Train a model. Owns the eval loop, checkpoints, AR-corpus hook.       |
| `infer`                  | Single-chart AR inference from a checkpoint.                          |
| `infer_corpus`           | Batch AR inference on a fraction of a dataset's val split.            |
| `analyze_dataset`        | Per-chart density / star / event-kind statistics + corpus aggregates. |
| `analyze_charts`         | Per-chart metrics for a single chart (debug aid).                     |
| `analyze_engagement`     | osu! engagement (favourites / plays) join from manifest.              |
| `fetch_stars`            | Star-rating fetch via osu! API v2 → manifest update.                 |
| `fetch_engagement`       | Engagement metric fetch.                                              |
| `viewer`                 | Pygame chart viewer (`.osu` / `.osz` / chart bundle).             |
| `onset_feature_survey`   | #011's algorithm survey.                                              |
| `onset_feature_survey_b` | #011b's disagreement / sub-analysis survey.                           |

## Performance

The headline numbers and full per-experiment progression live in
[`experiments/README.md`](experiments/README.md). At a glance:

| Metric                                       | Best taiko2 | Experiment                            | Source |
| -------------------------------------------- | ----------: | ------------------------------------- | ------ |
| val/single/onset/miss                        |      0.2406 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 372,132, val/single/onset/miss] |
| val/single/onset/hit                         |      0.7512 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 372,132, val/single/onset/hit] |
| AR `matched_rate` (gt_cond, median)          |      0.7061 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 413,480, infer_corpus/eval_413480/gt_cond/comparisons_summary.json:fields.matched_rate.median] |
| AR `error_median_ms` (gt_cond, median)       |           8 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 248,088+ (multiple late evals), infer_corpus/eval_*/gt_cond/comparisons_summary.json:fields.error_median_ms.median] |
| Per-step `ratio/rhit` (within ±3 % log-ratio)|      0.5332 | [#010e](experiments/010e-aux-frozen/) | [exp_010e_aux_frozen, step 475,502, val/single/ratio/rhit] |

#007 wins per-step *and* AR generation. The ratio-decomposition
family (#010 → #010e) introduces ratio-space metrics
(`rhit`, `rgood`, `div_acc`, `off_acc`) that don't apply to
direct-bin runs, so the #010e row above is for a metric that
doesn't exist on #007.

The val miss baseline taiko1 exp 44 sat at HIT 73.7 % / MISS
25.7 % [`osu/taiko/README.md`, "Per-sample (validation set)"
table]. #007 reaches val/single/onset/miss 0.2406 in taiko2
[exp_007_time_stretch, step 372,132, val/single/onset/miss],
beating that baseline with direct time-stretch augmentation
alone.


## Documentation

| File                                            | Purpose                                              |
| ----------------------------------------------- | ---------------------------------------------------- |
| [`README.md`](README.md)                         | This file.                                           |
| [`CLAUDE.md`](CLAUDE.md)                         | Working conventions. Read before editing.            |
| [`DATA.md`](DATA.md)                             | How to obtain `.osz` packs and build datasets.     |
| [`LICENSE.md`](LICENSE.md)                       | Licensing and disclaimer.                            |
| [`experiments/README.md`](experiments/README.md) | Per-experiment index with status + key result.       |
| `experiments/NNN-slug/README.md`              | Per-experiment narrative + results.                  |
| `experiments/NNN-slug/ARCHITECTURE.md`        | Per-experiment self-contained spec (reference-free). |

## Disclaimer

Commercial use of AI-generated rhythm game content is legally
ambiguous. You are solely responsible for your use of this
software and any data you obtain for it. See
[`LICENSE.md`](LICENSE.md) for details.
