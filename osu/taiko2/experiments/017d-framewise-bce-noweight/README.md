# Experiment 017d — Framewise BCE with no pos_weight (symmetric loss)

## Status

`Planned`

## Context

[#017](../017-framewise-bce/) established the framewise framing: a
single forward pass predicts a 500-bin activation map over 2.5 s of
future audio. It used `pos_weight_clamp = [10, 200]`, producing a
typical per-sample weight of 44.5 — each missed onset cost 44x a
false positive. The model responded rationally: it emitted notes on
every audio beat to avoid the asymmetric penalty, reaching
`density_ratio` 1.44 and `hallucination_rate` 0.32 before later
recovering to `dc_human` 91.0 at E4.

[#017b](../017b-framewise-focal/) replaced BCE with focal loss, which
compressed confidence values without fixing selectivity. `density_ratio`
stayed at 2.35 across the run.

[#017c](../017c-framewise-bce-lowweight/) reduced the clamp to [3, 8].
The best eval (E1) reached F1 0.846, precision 0.757, `dc_human` 89.0
— the strongest E1 of any framewise run. But a precision plateau
emerged: across 7 evals, recall improved +0.099 while precision
improved only +0.008, a 12:1 recall-to-precision gradient ratio. Even
8x asymmetry routes all gradient to recall. Once recall saturated near
0.97, there was no gradient pressure to improve precision because
trading a true positive for a false-positive suppression costs 8x.
`density_ratio` never broke below 1.51.

The scalar model [#007](../007-time-stretch/) uses softmax-CE over 501
bins with NO explicit positive weighting — all bins compete for
probability mass in a single distribution. It achieves `density_ratio`
0.87 and `dc_human` 92.0. The competition structure, not any explicit
weight, is what enforces selectivity.

The closest analog for per-bin BCE is `pos_weight = 1`: no asymmetry, a
false positive costs exactly as much as a false negative. This removes
the gradient imbalance mechanism entirely. The risk is the 38:1
neg-to-pos class imbalance — with equal cost, the model could achieve
low loss by predicting all zeros (98% of bins are negative). But #007
proves the task is solvable without explicit weighting, so the
all-zeros collapse is a risk, not a certainty.

Dataset statistics (89,753 sampled windows from taiko2_v1):
- GT=1 bins: 2.54% (38.4:1 neg-to-pos ratio)
- GT onsets per window: median 11, p5=3, p95=27

## Citations

- Direct baseline:
  - [#017 -- framewise BCE](../017-framewise-bce/). `pos_weight_clamp
    = [10, 200]`. Best `density_ratio` 1.44, `dc_human` 91.0,
    `hallucination_rate` 0.32 [exp_017_framewise_bce, step 82,696].
  - [#017b -- framewise focal](../017b-framewise-focal/). Focal loss
    compressed confidences without fixing selectivity. `density_ratio`
    stuck at 2.35 [exp_017b_framewise_focal, step 82,696].
  - [#017c -- framewise BCE low pos_weight](../017c-framewise-bce-lowweight/).
    `pos_weight_clamp = [3, 8]`. Best F1 0.846, precision 0.757,
    `density_ratio` 1.51, `dc_human` 89.0 [exp_017c_framewise_bce_lowweight,
    step 20,674]. Precision plateau: recall +0.099, precision +0.008
    over 7 evals (12:1 recall-to-precision gradient ratio).
  - [#007 -- TimeStretch](../007-time-stretch/). `density_ratio` 0.87,
    `dc_human` 92.0, `hallucination_rate` 0.17
    [exp_007_time_stretch, step 413,480]. Uses softmax-CE with no
    explicit pos_weight.
- Cross-experiment record: [`../README.md`](../README.md).

---
<!--
PRE-RUN. Do not edit after the run.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

If `pos_weight_clamp` is reduced from [3, 8] to [1, 1] (symmetric BCE,
no upweighting), then **recall and precision will develop together**,
because the gradient is no longer biased toward minimizing false
negatives at the expense of false positives — the model must learn to
distinguish onset bins from non-onset bins without an asymmetric cost
signal.

### Mechanism

With `pos_weight = 1`, the BCE loss treats every bit of over-emission
identically to every missed onset. The 12:1 recall-to-precision gradient
ratio observed in #017c (8x loss asymmetry) collapses to 1:1. The model
can no longer improve loss by saturating recall; it must improve both
simultaneously. The 38:1 class imbalance means the negative-class
gradient signal is numerically dominant, but the model also gets a clear
precision signal — predicting a confident positive on a negative bin
carries the same cost as predicting a confident negative on a positive
bin.

The primary risk is all-zeros collapse: 97.5% of bins are negative, so a
model that outputs logit -inf everywhere achieves near-zero BCE loss.
This is a known failure mode for unweighted BCE on heavily imbalanced
data. However, #007 achieves `dc_human` 92.0 without any explicit
positive upweighting in its softmax-CE objective, which demonstrates the
task is solvable without asymmetric loss. The question is whether per-bin
BCE with pos_weight=1 can similarly avoid collapse, or whether the
absence of a competition mechanism (unlike softmax) makes all-zeros the
easy local minimum.

### Predicted numbers

Reference: [#017c](../017c-framewise-bce-lowweight/) best E1 (step
20,674) and [#007](../007-time-stretch/) best (step 413,480).

| Metric | #017c (E1) | #007 | Predicted (#017d) | Notes |
|---|---:|---:|---:|---|
| frame F1 | 0.846 | n/a | **>= 0.80** | may be lower if recall trades for precision |
| frame Precision | 0.757 | n/a | **>= 0.80** | should improve relative to #017c |
| frame Recall | 0.958 | n/a | **>= 0.80** | may drop vs #017c's 0.96 |
| AR `density_ratio` | 1.51 | 0.87 | **0.70-1.10** | primary target |
| AR `dc_human` | 89.0 | 92.0 | **>= 88** | may regress if recall drops too much |
| `pos_rate_pred_50` | 0.067 | n/a | **> 0.005** | collapse sentinel — must not be zero |

## Success criteria

- **Must have:** `pos_rate_pred_50` > 0.005 at every eval after E1 —
  the model is not predicting all-zeros (not collapsed).
- **Must have:** AR `density_ratio` in [0.50, 1.30] at the best eval —
  primary target, substantially better than #017c's 1.51.
- **Must have:** AR `dc_human` >= 85 — pattern quality does not regress
  catastrophically.
- **Nice-to-have:** frame F1 >= 0.80 at any post-warmup eval.
- **Nice-to-have:** AR `density_ratio` in [0.80, 1.10] — near
  #007's 0.87.
- **Fails if:** `pos_rate_pred_50` < 0.001 at every eval — all-zeros
  collapse confirmed, class imbalance dominates.
- **Fails if:** frame Recall < 0.50 at every eval — class imbalance
  killed learning, model cannot detect onsets.
- **Fails if:** AR `dc_human` < 75 — catastrophic pattern regression.

## Changes from baseline

Baseline: [#017c -- framewise BCE low pos_weight](../017c-framewise-bce-lowweight/).

**Single change:** `loss.json` `pos_weight_clamp_min` and
`pos_weight_clamp_max` from 3.0 / 8.0 to 1.0 / 1.0. All other
configs byte-identical to #017c.

No code changes — the existing `FramewiseBCELoss` already supports
arbitrary clamp values.

Config snapshots ([`config/`](./config/)):

- `config/model.json` -- byte-identical to #017c.
- `config/loss.json` -- `pos_weight_clamp_min: 1.0`,
  `pos_weight_clamp_max: 1.0`.
- `config/adapter.json` -- byte-identical to #017c.
- `config/data.json` -- byte-identical to #017c.
- `config/trainer.json` -- byte-identical to #017c.
- `config/infer.json` -- identical decoder to #017c; only checkpoint
  path differs.

## Run config

- Run name: `exp_017d_framewise_bce_noweight`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`.
- Command:
  ```bash
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_017d_framewise_bce_noweight \
      --config-dir osu/taiko2/experiments/017d-framewise-bce-noweight/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/017d-framewise-bce-noweight/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

Future note: a potential architectural approach would make bins compete
directly (e.g. softmax-over-windows, energy-based output, or a learned
NMS layer). With softmax, predicting bin i at high probability forces all
other bins lower — the same competition mechanism that makes #007
selective. A future experiment could add a softmax-over-windows head or
energy-based output that enforces competition without relying on the loss
weight to impose selectivity.

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

### Final vs baseline

| Metric | Baseline (#017c E1) | This run (final) | Delta |
|---|---:|---:|---:|
| AR `density_ratio` | 1.51 | — | — |
| AR `dc_human` | 89.0 | — | — |
| frame F1 | 0.846 | — | — |
| frame Precision | 0.757 | — | — |
| frame Recall | 0.958 | — | — |
| val `loss` | 0.184 | — | — |

### Per-eval progression

| E | Step | Epoch | loss | F1 | Prec | Recall | AR dens | AR dc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Machine-readable copies: [`metrics.json`](./metrics.json).

## Visualizations

## Vs prediction

## Takeaways

## Followup questions
