# Experiment 017b — Framewise focal loss

## Status

`Planned`

## Context

[#017](../017-framewise-bce/) validated the framewise framing: pattern
quality (`dc_human` 91.0 [exp_017_framewise_bce, step 82,696,
val/single/corpus/gt_cond_cmp/dc_human_mean]) matches
[#007](../007-time-stretch/)'s 92.0, with better timing
(`error_median_ms` 6.1 vs 10.2). However, the model over-emits ~44%
extra notes (`density_ratio` 1.44 [exp_017_framewise_bce, step 82,696,
val/single/corpus/gt_cond_cmp/density_ratio_mean],
`hallucination_rate` 0.32 vs #007's 0.17).

The root cause identified in #017: the model confidently predicts
onsets at every rhythmic beat in the audio, not just the ones the
chart author selected. `conf_fp_median` = 0.80
[exp_017_framewise_bce, step 82,696,
val/single/frame/conf_fp_median] is indistinguishable from
`conf_tp_median` = 0.93 — no threshold can separate true from false
positives. The per-window mini-chart data shows `density_ratio` = 4.7
[exp_017_framewise_bce, step 82,696,
val/single/frame/mini/τ50/density_ratio] at the raw model output;
the AR loop's cursor-advance mechanics mask ~70% of the over-emission
down to the 1.44 reported in the AR corpus.

With BCE loss, the gradient is dominated by the ~97% true-negative
bins (confidence near 0, correctly classified). These easy bins
contribute most of the loss magnitude but carry no useful learning
signal for the selectivity problem. The hard cases — metronomic
beats that the model confidently predicts as onsets but that aren't
in the GT chart — are a tiny fraction of the gradient.

This experiment replaces BCE with **focal loss** (Lin et al., 2017),
which down-weights well-classified examples via the `(1 - p_t)^gamma`
modulation. At gamma=2, an easy TN with `p_t = 0.95` gets weight
`0.05^2 = 0.0025` (400x down-weighted), while a hard FP with
`p_t = 0.20` gets weight `0.80^2 = 0.64`. This should redirect
gradient from easy TNs to the hard "correct audio detection, wrong
chart decision" FPs.

## Citations

- Direct baseline:
  - [#017 -- framewise BCE](../017-framewise-bce/). Best `dc_human`
    91.0 [exp_017_framewise_bce, step 82,696,
    val/single/corpus/gt_cond_cmp/dc_human_mean]. Best
    `density_ratio` 1.44, `hallucination_rate` 0.32 at same step.
    `conf_fp_median` = 0.80 across all evals (never separating).
  - [#007 -- TimeStretch](../007-time-stretch/). Best gt
    `matched_rate` 0.7028, `dc_human` 92.0, `density_ratio` 0.87,
    `hallucination_rate` 0.17 [exp_007_time_stretch, step 413,480,
    val/single/corpus/gt_cond_cmp/ fields].
- Loss design:
  - [Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)](https://arxiv.org/abs/1708.02002).
    Source of the `(1 - p_t)^gamma` modulation. gamma=2 is the
    default recommended in the paper for sparse positive classes.
- Cross-experiment record: [`../README.md`](../README.md).

---
<!--
Everything above this divider may be written freely.
Everything between the two dividers is PRE-RUN and must be filled
BEFORE the run. Do not edit it afterwards — use the amendment rule.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

If focal loss (gamma=2) replaces BCE on the otherwise-identical #017
architecture, then **`conf_fp_median` will drop below 0.60**
(separating from `conf_tp_median`), **`density_ratio` will reach
0.85-1.15** (vs #017's 1.44), and **`hallucination_rate` will drop
below 0.20** (vs #017's 0.32), because the focal modulation
redirects gradient from easy true-negatives to the hard false-
positive metronomic beats, teaching the model to suppress them.

### Mechanism

Two effects:

1. **Down-weights easy TNs.** ~97% of bins are negative with
   confidence near 0. Their `p_t ≈ 1.0` gives focal weight
   `(1 - 1.0)^2 ≈ 0` — effectively removed from the gradient.
   In BCE these bins dominate; in focal they contribute almost
   nothing.
2. **Preserves gradient on hard FPs.** The metronomic-beat false
   positives have confidence ~0.80, so `p_t = 1 - 0.80 = 0.20`,
   focal weight `(1 - 0.20)^2 = 0.64` — still a strong gradient
   signal. These are the bins the model needs to learn to suppress.

`pos_weight` (clamp [10, 200]) is kept unchanged to handle the
3%/97% class imbalance. Focal and pos_weight are complementary:
pos_weight addresses count imbalance, focal addresses confidence-
calibration imbalance.

### Predicted numbers

Reference: [#017](../017-framewise-bce/) best (E4, step 82,696) and
[#007](../007-time-stretch/) best (E18, step 413,480). gt_cond.

| Metric | #017 best | #007 best | Predicted (#017b) | Notes |
|---|---:|---:|---:|---|
| AR `density_ratio` | 1.44 | 0.87 | **0.85-1.15** must | primary target — over-emission eliminated |
| AR `hallucination_rate` | 0.32 | 0.17 | **<= 0.20** must | direct consequence of density fix |
| AR `dc_human` | 91.0 | 92.0 | **>= 90** | should hold or improve |
| AR `matched_rate` | 0.90 | 0.70 | **0.70-0.85** | will drop as over-emission drops — this is expected and healthy |
| AR `error_median_ms` | 6.1 | 10.2 | **<= 10** | should hold |
| `conf_fp_median` | 0.80 | n/a | **<= 0.60** | the key diagnostic — FP confidence must separate from TP |
| `conf_tp_median` | 0.93 | n/a | **>= 0.85** | TP confidence should remain high |
| frame F1 (tau=0.5, +/-2) | 0.78 | n/a | **>= 0.80** | precision should improve with fewer FPs |
| `loss/focal_weight_neg` | n/a | n/a | **<= 0.30** | confirms focal is down-weighting easy negatives |
| `loss/focal_weight_pos` | n/a | n/a | **>= 0.10** | confirms positives still get gradient |
| mini tau50 density_ratio | 4.70 | n/a | **<= 2.0** | raw per-window over-emission (before AR masking) |

## Success criteria

- **Must have:** AR `density_ratio` in [0.70, 1.30] at the best eval
  -- the over-emission problem is materially reduced vs #017's 1.44.
- **Must have:** `conf_fp_median` < `conf_tp_median` - 0.15 at any
  post-warmup eval -- the model has learned to assign lower confidence
  to false positives than true positives.
- **Must have:** AR `dc_human` >= 88 -- pattern quality does not
  regress catastrophically.
- **Nice-to-have:** AR `hallucination_rate` <= 0.15 -- below #007's
  0.17.
- **Nice-to-have:** AR `density_ratio` in [0.85, 1.10] -- near-
  perfect density matching.
- **Nice-to-have:** frame F1 >= 0.85.
- **Fails if:** AR `density_ratio` > 1.40 at every eval -- focal did
  not reduce over-emission (the problem is not about easy-vs-hard
  gradient distribution).
- **Fails if:** AR `dc_human` < 80 -- focal loss damaged pattern
  quality (the down-weighting of easy negatives harmed the model's
  ability to learn where NOT to place notes).
- **Fails if:** `loss/focal_weight_pos` < 0.01 at any eval -- focal
  is suppressing TP gradient (gamma too high for this class ratio).

## Changes from baseline

Baseline: [#017 -- framewise BCE](../017-framewise-bce/).

**Single change:** `loss.json` swapped from `FramewiseBCELossConfig`
to `FramewiseFocalLossConfig` with `gamma=2.0`. All other configs
are byte-identical to #017.

Code:

- **`osu/taiko2/training/framewise_focal_loss.py`** (NEW in this
  session, written alongside #017b planning) -- `FramewiseFocalLoss`.
  Applies `(1 - p_t)^gamma` modulation on top of BCE + pos_weight.
  Reports two additional diagnostics vs #017's BCE loss:
  `loss/focal_weight_pos` and `loss/focal_weight_neg` (mean focal
  modulation weight per class, for gamma tuning).
- **`osu/taiko2/cli/train.py`** -- 2-line dispatch addition for
  `FramewiseFocalLossConfig`.
- **Tests**: 4 new tests in `test_framewise_bce.py` (smoke,
  gamma=0 matches BCE, focal reduces easy-neg weight, backward).
  Full suite 625/625 passing.

Config snapshots ([`config/`](./config/)):

- `config/model.json` -- byte-identical to #017.
- `config/loss.json` -- `FramewiseFocalLossConfig`, gamma=2.0,
  pos_weight [10, 200].
- `config/adapter.json` -- byte-identical to #017.
- `config/data.json` -- byte-identical to #017.
- `config/trainer.json` -- byte-identical to #017 (15 epochs).
- `config/infer.json` -- byte-identical to #017 except checkpoint
  path.

## Run config

- Run name: `exp_017b_framewise_focal`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1` (80-row mel; same as #007/#017).
- Command:
  ```bash
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_017b_framewise_focal \
      --config-dir osu/taiko2/experiments/017b-framewise-focal/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/017b-framewise-focal/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

### Final vs baseline

| Metric | Baseline (exp N) | This run (final) | Delta | Direction |
|---|---:|---:|---:|:---:|
| — | — | — | — | — |

### Per-eval progression

{Generated from `runs/exp_017b_framewise_focal/metrics.jsonl`.}

Machine-readable copies (both tables): [`metrics.json`](./metrics.json).

## Visualizations

![Training loss](graphs/01_train_loss.png)
*Training loss over steps (log-y).*

![Validation progression](graphs/02_val_progression.png)
*Watched metric across evals.*

## Vs prediction

## Takeaways

## Followup questions
