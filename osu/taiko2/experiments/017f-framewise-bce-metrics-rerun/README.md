# Experiment 017f — Framewise BCE metrics rerun

## Status

`Planned`

## Context

[#017e](../017e-framewise-bce-regularized/) is the strongest framewise
model to date, but its metrics infrastructure has significant gaps.
The AR corpus comparison uses `matched_rate` and `hallucination_rate`
as primary metrics -- both are threshold-based counts at fixed 25 ms /
100 ms tolerances that conflate timing precision with onset detection
accuracy. The evaluation also lacks standard precision/recall/F1 and
has no way to measure quality at different temporal resolutions.

This experiment is a **clean re-training** of the identical 017e
model/loss/data configuration with the upgraded metrics pipeline:

1. **Multi-resolution FPS comparisons** (1, 2, 4, 10, 20, 50, 100,
   200 FPS): binary precision/recall/F1 and count MAE/correlation/
   accuracy at each temporal resolution. This separates "did the model
   get the right 20ms window" (50 FPS) from "did it hit the exact
   5ms bin" (200 FPS).
2. **Standard precision/recall/F1** at 25 ms tolerance on
   `ChartComparison`.
3. **Dense gap/ratio histograms** saved per chart for post-hoc
   distribution analysis.
4. **Per-chart step logs** and bundles always saved (no longer gated
   by flags).

The watched metric changes from `frame/f1_tau_50_tol_2` (frame-level
F1 at threshold 0.50, 2-frame tolerance) to
`frame/mini/tau50/fps_50/binary_f1` (mini-chart binary F1 at 50 FPS
resolution, threshold 0.50). This metric directly measures "in each
20 ms window, did the model agree with GT on whether an onset exists"
-- a resolution-aware proxy for AR chart quality.

No model, loss, data, or augmentation changes. The only functional
difference is the watched metric for `best.pt` selection; all other
training dynamics are identical.

## Citations

- Direct baseline:
  - [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).
    Best sweep: E8 (step 165,392), tau=0.40, `matched_rate` 0.783,
    `hallucination_rate` 0.201, `density_ratio` 1.020, `dc_human` 92.7,
    `error_median_ms` 10.3
    [exp_017e_framewise_bce_regularized, threshold_sweep.json].
    Best F1: E11 (step 227,414), F1 0.827, Precision 0.882, Recall 0.779.
- Related priors:
  - [#017d -- framewise BCE noweight](../017d-framewise-bce-noweight/).
    `matched_rate` 0.742 at E9/tau=0.3
    [exp_017d_framewise_bce_noweight, threshold_sweep.json].
  - [#007 -- TimeStretch](../007-time-stretch/). `matched_rate` 0.703,
    `hallucination_rate` 0.172, `density_ratio` 0.865, `dc_human` 92.0
    [exp_007_time_stretch, step 413,480].

---
<!--
PRE-RUN. Do not edit after the run.
-->
---------------------------------------------------------------------

## Hypothesis

### Claim

This is a metrics-collection rerun, not a hypothesis-driven
experiment. The model, loss, and data are identical to
[#017e](../017e-framewise-bce-regularized/). The purpose is to
establish clean baselines with the upgraded metrics pipeline
(FPS-resolution comparisons, precision/recall/F1, dense histograms,
per-chart step logs) that were absent in prior runs.

### Mechanism

Training dynamics will match 017e because nothing changes except the
watched metric (`frame/mini/tau50/fps_50/binary_f1` vs
`frame/f1_tau_50_tol_2`). The new watched metric may select a
different checkpoint as `best.pt`, but the per-eval trajectory should
be statistically equivalent given the same seed, data, and
hyperparameters.

### Predicted numbers

Reference: [#017e](../017e-framewise-bce-regularized/) E8/tau=0.40
sweep [exp_017e_framewise_bce_regularized, threshold_sweep.json] and
[#007](../007-time-stretch/) step 413,480.

017e AR corpus numbers below are at `decode_threshold=0.3` (the
infer.json setting, same as 017f). The sweep result at tau=0.40
had different operating-point characteristics (`density_ratio` 1.020,
`matched_rate` 0.783
[exp_017e_framewise_bce_regularized, threshold_sweep.json]); a
post-run threshold sweep on 017f will produce comparable @0.4 numbers
with the new metrics.

| Metric | #017e @0.3 | #017e @0.4 (sweep) | Predicted (#017f @0.3) | Notes |
|---|---:|---:|---:|---|
| AR `precision` (25ms) | n/a | n/a | **new baseline** | not measured in 017e |
| AR `recall` (25ms) | n/a | n/a | **new baseline** | not measured in 017e |
| AR `f1` (25ms) | n/a | n/a | **new baseline** | not measured in 017e |
| AR `density_ratio` (median) | 1.209 | 1.020 | **~1.21** | should match 017e @0.3 |
| AR `dc_human` (median %) | 93.3 | 92.7 | **~93** | should match 017e @0.3 |
| AR `matched_rate` (median) | 0.892 | 0.783 | **~0.89** | legacy, kept for comparison |
| frame F1 (best eval) | 0.827 | -- | **0.82-0.84** | should match 017e |
| `binary_f1` @ 50 FPS | n/a | n/a | **new baseline** | watched metric |
| `binary_f1` @ 200 FPS | n/a | n/a | **new baseline** | per-bin resolution |
| `count_corr` @ 50 FPS | n/a | n/a | **new baseline** | onset count correlation |
| `gap_hist_tvd` | n/a | n/a | **new baseline** | gap distribution distance |
| `ratio_hist_tvd` | n/a | n/a | **new baseline** | ratio distribution distance |
| `density_corr` | n/a | n/a | **new baseline** | per-second density correlation |

## Success criteria

- **Must:** frame F1 trajectory matches 017e within noise (same seed,
  same config). Best frame F1 should land within 0.01 of 017e's 0.827.
- **Must:** all new metrics (FPS comparisons, precision/recall/f1,
  dense histograms, distributional comparisons) populate correctly in
  `metrics.jsonl`, per-chart JSONs, `fps_summary.json`, and
  per-chart comparison JSONs.
- **Must:** AR corpus `density_ratio` median near 1.21 (matching 017e
  at decode_threshold=0.3 [exp_017e, eval_165392, gt_cond]).
- **Fails if:** frame F1 deviates > 0.02 from 017e -- indicates a
  regression in the training infrastructure.
- **Nice-to-have:** the FPS-50 binary_f1 watched metric selects a
  comparable or better `best.pt` than 017e's frame/f1-based selection.

## Changes from baseline

Baseline: [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).

One change:

- `config/trainer.json` -- `metric_to_watch` changed from
  `frame/f1_tau_50_tol_2` to `frame/mini/tau50/fps_50/binary_f1`.

All other configs (model.json, loss.json, data.json, adapter.json,
infer.json) are identical to #017e except the checkpoint path and
run name.

Infrastructure changes (applied globally, not experiment-specific):

- `domain/chart.py` -- `ResolutionComparison` dataclass, standard
  precision/recall/F1 on `ChartComparison`, dense gap/ratio
  histograms on `ChartMetrics`.
- `inference/corpus.py` -- FPS summary aggregation, always saves
  bundles + per-chart JSONs + per-chart step logs.
- `training/framewise_metric.py` -- FPS resolution mini-chart metrics
  per threshold in eval.

## Run config

- Run name: `exp_017f_framewise_bce_metrics_rerun`
- Config snapshots: [`config/`](./config/)
- Dataset: `taiko2_v1`, split `train` / `val`
- Command:
  ```bash
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_017f_framewise_bce_metrics_rerun \
      --config-dir osu/taiko2/experiments/017f-framewise-bce-metrics-rerun/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/017f-framewise-bce-metrics-rerun/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

## Primary metrics

The following metrics are the focus of this run. Prior experiments
used `matched_rate` / `hallucination_rate` as headline numbers;
017f replaces those with FPS-resolution metrics and standard P/R/F1.

### Frame-level (per-eval)

| Metric key | Description |
|---|---|
| `frame/mini/tau50/fps_50/binary_f1` | **Watched metric.** Mini-chart binary F1 at 50 FPS (20ms windows), threshold 0.50. |
| `frame/mini/tau50/fps_50/binary_precision` | Mini-chart binary precision at 50 FPS. |
| `frame/mini/tau50/fps_50/binary_recall` | Mini-chart binary recall at 50 FPS. |
| `frame/mini/tau50/fps_50/count_mae` | Mean absolute error of onset count per 20ms frame. |
| `frame/mini/tau50/fps_50/count_corr` | Pearson correlation of onset counts per 20ms frame. |
| `frame/mini/tau50/fps_{N}/binary_f1` | Same at N = 1, 2, 4, 10, 20, 100, 200 FPS. |

### AR corpus (per-eval inference pass)

| Metric key | Description |
|---|---|
| `corpus/gt_cond_cmp/precision` | Onset precision at 25ms tolerance. |
| `corpus/gt_cond_cmp/recall` | Onset recall at 25ms tolerance. |
| `corpus/gt_cond_cmp/f1` | Onset F1 at 25ms tolerance. |
| `corpus/gt_cond_cmp/binary_f1_at_{N}fps_median` | Median binary F1 across val charts at N FPS. |
| `corpus/gt_cond_cmp/count_mae_at_{N}fps_median` | Median count MAE at N FPS. |
| `corpus/gt_cond_cmp/count_corr_at_{N}fps_median` | Median count correlation at N FPS. |
| `corpus/gt_cond_cmp/density_ratio` | Emitted / GT onset density. |
| `corpus/gt_cond_cmp/dc_human` | TaikoNation direct-compare score (%). |
| `corpus/gt_cond_cmp/oc_human` | TaikoNation overlap-compare score (%). |

### Per-chart comparison (`comparisons/{stem}.json`)

Each generated chart's comparison JSON now contains:

| Field | Description |
|---|---|
| `precision` / `recall` / `f1` | Standard P/R/F1 at 25ms tolerance. |
| `fps_comparisons` | Array of 8 `ResolutionComparison` objects (1, 2, 4, 10, 20, 50, 100, 200 FPS), each with `binary_precision`, `binary_recall`, `binary_f1`, `count_mae`, `count_corr`, `count_accuracy`. |
| `gap_hist_tvd` | Total variation distance between pred and GT gap histograms. |
| `ratio_hist_tvd` | Total variation distance between pred and GT ratio histograms. |
| `density_corr` | Pearson correlation of per-second density timelines. |
| `density_mae` | Mean absolute error of per-second density. |
| `silence_overlap_f1` | F1 of silence region overlap (pred vs GT). |
| `dense_overlap_f1` | F1 of dense region overlap (pred vs GT). |
| `gap_peak_iou` | IoU of gap histogram peak sets. |
| `ioi_mean_ratio` | Ratio of pred/GT mean IOI. |
| `ioi_std_ratio` | Ratio of pred/GT IOI standard deviation. |
| `streak_fraction_delta` | Difference in same-gap streak fractions. |
| `bpm_ratio` | Ratio of pred/GT estimated BPM. |

### Per-chart metrics (`metrics/{stem}.json`)

Each generated chart's self-metrics JSON now contains:

| Field | Description |
|---|---|
| `gap_histogram_dense` | Raw 200-bucket gap histogram (10ms buckets, 0-2000ms). |
| `ratio_histogram_dense` | Raw 200-bucket consecutive-ratio histogram (log2 space). |

---------------------------------------------------------------------
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
---------------------------------------------------------------------

## Results summary

<!-- TODO: fill after run -->

## Visualizations

<!-- TODO: fill after run -->

## Vs prediction

<!-- TODO: fill after run -->

## Takeaways

<!-- TODO: fill after run -->
