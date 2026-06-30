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
  type hints everywhere; stdlib-name shadowing prohibited (an
  early `secrets`/`types` shadowing bug motivated this rule).
- **Save don't print.** Every metric, checkpoint, and analysis
  output that's worth showing a human is also written to disk
  in a parseable format. The terminal is a debugging aid; the
  log is the source of truth.
- **Experiment hygiene.** Every experiment has its own folder in
  [`experiments/`](experiments/) with `README.md` (narrative +
  results) and `ARCHITECTURE.md` (reference-free, fully
  self-contained spec). Pre-run sections are written before the
  run starts and never edited afterward (use amendments instead).

## Current state of the work

Two modeling tracks share the domain layer:

- **Onset detection** — predict beat onset timings from audio. The
  current models are **framewise** (per-frame activation maps decoded
  by threshold + NMS), evaluated by chart-comparison metrics against
  ground-truth charts rather than per-sample bin classification. The
  dual-channel mel+octopus model
  ([#022b](experiments/022b-dual-channel/)) is the current best — AR
  F1 **0.794** at its best decode threshold
  [022b-dual-channel/threshold_sweep.json, eval_329120, tau=0.3,
  ar_f1], beating the prior framewise best
  [#017e](experiments/017e-framewise-bce-regularized/) at every
  threshold in the sweep.
- **Typing** — given onset timings, predict each note's type
  (don/ka) and strength (big/normal). The context-64 model
  ([#024d](experiments/024d-typing-ctx64/)) holds the best
  teacher-forced type accuracy at **0.728** [024d-typing-ctx64,
  step 489,480, val/single/typing/type/accuracy=0.7279]. The 024
  context-scaling series concluded at 32/32 as the sweet spot;
  [#025](experiments/025-typing-temporal-bias/) tested temporal
  bias and regressed (0.715, miss).

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

### Onset detection

The current onset models are **framewise** — a per-frame activation
map decoded by threshold + non-max suppression, then scored by
chart-comparison metrics against the ground-truth chart over a
threshold sweep across every checkpoint. This replaced the earlier
autoregressive direct-bin line ([#007](experiments/007-time-stretch/)),
which was scored by per-sample bin classification (`val/single/onset/*`).
Per-sample metrics are no longer the tracked headline for onsets;
chart-comparison metrics are. The framewise line runs
[#017](experiments/017-framewise-bce/) (BCE control) →
[#017e](experiments/017e-framewise-bce-regularized/) (regularized;
first framewise model to beat #007 on chart comparison) →
[#022b](experiments/022b-dual-channel/) (dual-channel mel+octopus,
current best).

Best chart-comparison metrics, all from #022b's threshold sweep
[022b-dual-channel/threshold_sweep.json]:

| Metric                                  |   Best | Operating point                         | Source |
| --------------------------------------- | -----: | --------------------------------------- | ------ |
| AR F1 (25 ms, best threshold)           | 0.7944 | eval_329120, tau=0.3                    | [022b/threshold_sweep.json, eval_329120, tau=0.3, ar_f1] |
| AR F1 at matched density (DR ≈ 1.0)     | 0.7847 | eval_287980, tau=0.4, density_ratio 1.003 | [022b/threshold_sweep.json, eval_287980, tau=0.4, ar_f1] |
| `cmp/dc_human` (best)                   |  93.26 | eval_205700, tau=0.5                    | [022b/threshold_sweep.json, eval_205700, tau=0.5, cmp/dc_human] |
| `cmp/dc_human` at matched density       |  92.79 | eval_287980, tau=0.4                    | [022b/threshold_sweep.json, eval_287980, tau=0.4, cmp/dc_human] |
| `cmp/gap_hist_tvd` (best, lower=better) | 0.2865 | eval_308550, tau=0.4                    | [022b/threshold_sweep.json, eval_308550, tau=0.4, cmp/gap_hist_tvd] |
| `cmp/error_median_ms` at matched density|    8.0 | eval_287980, tau=0.4                    | [022b/threshold_sweep.json, eval_287980, tau=0.4, cmp/error_median_ms=7.98] |

#022b beats the prior framewise best #017e at every threshold (AR F1
+0.009 to +0.036 across the sweep) and on `cmp/dc_human` at matched
density (92.79 vs 92.56 [017e/threshold_sweep.json, eval_206740,
tau=0.4, cmp/dc_human]) and `cmp/error_median_ms` (8.0 vs 15.5
[same row, cmp/error_median_ms]). At matched density (density_ratio
≈ 1.0) #022b reaches AR F1 0.785, against the autoregressive line's
gt_cond `matched_rate` 0.706 [exp_007_time_stretch, step 413,480,
infer_corpus/eval_413480/gt_cond/comparisons_summary.json:fields.matched_rate.median]
— the two lines use different decode procedures and different scores,
so the framewise numbers are not a drop-in continuation of the
direct-bin progression.

### Typing

Given onset timings, predict each note's type (don/ka) and strength
(big/normal). Best values are held by the context-64 model
([#024d](experiments/024d-typing-ctx64/)) at its final eval.

| Metric                                  | Best taiko2 | Experiment                                | Source |
| --------------------------------------- | ----------: | ----------------------------------------- | ------ |
| `typing/type/accuracy` (teacher-forced) |      0.7279 | [#024d](experiments/024d-typing-ctx64/)   | [024d-typing-ctx64, step 489,480, val/single/typing/type/accuracy] |
| `typing/strength/best_f1_BIG`           |      0.7263 | [#024d](experiments/024d-typing-ctx64/)   | [024d-typing-ctx64, step 489,480, val/single/typing/strength/best_f1_BIG] |
| `typing/combined/accuracy`              |      0.6866 | [#024d](experiments/024d-typing-ctx64/)   | [024d-typing-ctx64, step 489,480, val/single/typing/combined/accuracy] |
| AR `type_accuracy_sym_mean`             |      0.5618 | [#024d](experiments/024d-typing-ctx64/)   | [024d-typing-ctx64, step 489,480, val/single/ar/type_accuracy_sym_mean] |
| AR `strength_f1_BIG_mean`               |      0.4769 | [#024d](experiments/024d-typing-ctx64/)   | [024d-typing-ctx64, step 489,480, val/single/ar/strength_f1_BIG_mean] |

Context scaling across the 024 series lifted type accuracy from
0.718 (ctx16, [#024b](experiments/024b-typing-full/)) to 0.726
(ctx32, [#024c](experiments/024c-typing-ctx32/)) to 0.728 (ctx64,
#024d); the 16→32 jump (+0.8 pp) was the only significant one,
32→64 (+0.2 pp) sits within noise. The type-accuracy ceiling holds
near 0.728 — [#025](experiments/025-typing-temporal-bias/) added a
learned temporal bias and regressed to a 0.715 peak
[025-typing-temporal-bias, step 293,688, val/single/typing/type/accuracy].


## Documentation

| File                                            | Purpose                                              |
| ----------------------------------------------- | ---------------------------------------------------- |
| [`README.md`](README.md)                         | This file.                                           |
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
