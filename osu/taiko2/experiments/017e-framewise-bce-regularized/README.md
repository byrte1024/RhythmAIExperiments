# Experiment 017e — Framewise BCE regularized (label smoothing + head dropout + F1 watch)

## Status

`Planned`

## Context

[#017d](../017d-framewise-bce-noweight/) established symmetric BCE
(pos_weight=1) as the correct loss for framewise onset detection.
Removing positive-class upweighting produced a precision-first
training trajectory — precision 0.92 at E1, then recall climbed
+0.13 over 10 evals while precision gently declined −0.03. At the
optimal operating point (E9, `decode_threshold=0.3`) 017d became the
first framewise model to beat [#007](../007-time-stretch/) on AR
quality: `matched_rate` 0.742 vs #007's 0.703, `hallucination_rate`
0.118 vs #007's 0.172, `density_ratio` 0.925 vs #007's 0.865,
`dc_human` 92.1 vs #007's 92.0
[exp_017d_framewise_bce_noweight, threshold_sweep.json;
exp_007_time_stretch, step 413,480].

Despite those results, 017d has a clear ceiling: **overfitting begins
at E3** and limits further learning. Val loss rises from 0.053 at E2
to 0.076 at E9 while noaug (train-without-augmentation) loss falls
from 0.046 to 0.029 — the gap widens to +0.047 at E9
[exp_017d_framewise_bce_noweight, metrics.jsonl]. AR metrics (matched_rate,
halluc_rate) plateau at E6-E8 and no longer improve despite 7 more
evals of training. The `best.pt` checkpoint saved by the trainer is
the E2 checkpoint (lowest loss) — which has the **worst** AR
metrics of any post-E1 checkpoint. These two observations together
mean the training objective (minimize loss) diverges from the
deployment objective (maximize AR quality) starting at E3.

Three targeted changes address the overfitting wall without altering
the network's fundamental structure:

1. **Label smoothing (eps=0.05):** target values become {0.05, 0.95}
   instead of {0, 1}. Penalizes extreme confidence on any single bin,
   including memorized training examples.
2. **Head dropout 0.2 (was 0.1):** adds regularization between the
   three Conv1D blocks in the detection head, which is the component
   that sees the sharpest train/val divergence.
3. **metric_to_watch = frame/f1:** the trainer now saves `best.pt`
   at the highest frame-level F1 checkpoint rather than the lowest
   loss checkpoint, correcting the E2-best-pt misalignment.

## Citations

- Direct baseline:
  - [#017d -- framewise BCE noweight](../017d-framewise-bce-noweight/).
    `pos_weight_clamp = [1, 1]`. Best AR: E9 (step 186,066),
    `decode_threshold=0.3`, `matched_rate` 0.742, `halluc_rate` 0.118,
    `density_ratio` 0.925, `dc_human` 92.1
    [exp_017d_framewise_bce_noweight, threshold_sweep.json].
    Overfitting gap at E3: +0.012; at E8: +0.041; at E9: +0.047
    [exp_017d_framewise_bce_noweight, metrics.jsonl].
- Related prior:
  - [#017c -- framewise BCE low pos_weight](../017c-framewise-bce-lowweight/).
    `pos_weight_clamp = [3, 8]`. Best F1 0.846 at E1
    [exp_017c_framewise_bce_lowweight, step 20,674].
  - [#007 -- TimeStretch](../007-time-stretch/). `matched_rate` 0.703,
    `halluc_rate` 0.172, `density_ratio` 0.865, `dc_human` 92.0
    [exp_007_time_stretch, step 413,480].

---
<!--
PRE-RUN. Do not edit after the run.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

If label smoothing (eps=0.05), head dropout (0.2), and
`metric_to_watch=frame/f1` are applied together to the #017d
configuration, the overfitting wall shifts from E3 to E6+, giving the
model more epochs of genuine learning before val loss diverges, and the
resulting `best.pt` checkpoint aligns with the AR-quality-optimal
checkpoint rather than the loss-optimal (E2) checkpoint.

### Mechanism

The overfitting gap in #017d (+0.047 at E9
[exp_017d_framewise_bce_noweight, metrics.jsonl]) has two components:
the model memorizes training examples (label smoothing addresses this
by making hard targets impossible to achieve with arbitrary confidence)
and the detection head overfits faster than the trunk (head dropout
0.2 adds gradient noise between the three Conv1D blocks). Together
these should slow the rate at which val loss diverges from noaug loss.

Separately, changing `metric_to_watch` does not directly affect the
training dynamics — it affects only which checkpoint is saved as
`best.pt`. The E2 checkpoint (loss-optimal in 017d) had AR
`matched_rate` 0.554, the worst of any post-E1 checkpoint. Saving the
F1-optimal checkpoint instead means the default inference path uses a
checkpoint that is actually near the AR optimum.

The three changes are independent: label smoothing and head dropout
target regularization; metric_to_watch targets checkpoint selection.
Any two of the three could succeed while the third fails.

### Predicted numbers

Reference: [#017d](../017d-framewise-bce-noweight/) best AR
(E9/tau=0.3) and [#007](../007-time-stretch/) (step 413,480).

| Metric | #017d best | #007 | Predicted (#017e) | Notes |
|---|---:|---:|---:|---|
| AR `matched_rate` | 0.742 | 0.703 | **>= 0.75** | regularization gives more epochs |
| AR `halluc_rate` | 0.118 | 0.172 | **<= 0.12** | precision-first regime maintained |
| AR `density_ratio` | 0.925 | 0.865 | **0.85-1.05** | same selectivity range |
| AR `dc_human` (%) | 92.1 | 92.0 | **>= 92** | pattern quality maintained |
| frame F1 (best eval) | 0.824 | n/a | **>= 0.83** | F1 ceiling rises |
| frame Precision (best) | 0.884 | n/a | **>= 0.85** | maintained |
| frame Recall (best) | 0.772 | n/a | **>= 0.78** | more epochs of recall growth |
| overfitting gap at E8 | +0.041 | n/a | **< +0.041** | must be smaller than 017d |

## Success criteria

- **Must have:** AR `matched_rate` >= 0.72 at best eval checkpoint
  (at least matches #017d's 0.742, allowing for threshold tuning). The
  run does not regress on the key headline metric.
- **Must have:** AR `dc_human` >= 90 at the best eval checkpoint.
  Pattern quality does not collapse under regularization.
- **Must have:** Overfitting gap (val loss - noaug loss) at E8 smaller
  than #017d's E8 gap of +0.041 [exp_017d_framewise_bce_noweight,
  metrics.jsonl]. Regularization measurably delays overfitting.
- **Nice-to-have:** AR `matched_rate` >= 0.75 at optimal threshold —
  beats #017d's best.
- **Nice-to-have:** AR `halluc_rate` <= 0.11 — better selectivity.
- **Nice-to-have:** `best.pt` (F1-optimal) checkpoint has AR
  `matched_rate` >= 0.70 — metric_to_watch change pays off.
- **Fails if:** AR `matched_rate` < 0.60 at every threshold — label
  smoothing or head dropout killed learning.
- **Fails if:** frame Recall < 0.50 at any post-E2 eval — regularization
  over-suppressed positive predictions.

## Changes from baseline

Baseline: [#017d -- framewise BCE noweight](../017d-framewise-bce-noweight/).

Three independent changes:

- `config/loss.json` — `label_smoothing: 0.05` (new field; #017d had
  no label smoothing).
- `config/model.json` — `head_dropout: 0.1 -> 0.2`.
- `config/trainer.json` — `metric_to_watch: "loss" -> "frame/f1_τ_50_tol_2"`,
  `metric_lower_is_better: true -> false`.

All other configs (data.json, adapter.json, infer.json trunk) are
byte-identical to #017d except the checkpoint path in infer.json.

No code changes required — `FramewiseBCELoss` already accepts a
`label_smoothing` field; `FramewiseDetectorConfig` already accepts
`head_dropout`; `TrainerConfig` already accepts arbitrary
`metric_to_watch` strings.

## Run config

- Run name: `exp_017e_framewise_bce_regularized`
- Config snapshots: [`config/`](./config/)
- Dataset: `taiko2_v1`, split `train` / `val`
- Command:
  ```bash
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_017e_framewise_bce_regularized \
      --config-dir osu/taiko2/experiments/017e-framewise-bce-regularized/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/017e-framewise-bce-regularized/config/infer.json \
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

| Metric | Baseline (#017d best AR) | This run (final) | Delta | Direction |
|---|---:|---:|---:|:---:|
| AR `matched_rate` | — | — | — | — |
| AR `hallucination_rate` | — | — | — | — |
| AR `density_ratio` | — | — | — | — |
| AR `dc_human` (%) | — | — | — | — |
| frame F1 | — | — | — | — |
| frame Precision | — | — | — | — |
| frame Recall | — | — | — | — |
| val `loss` | — | — | — | — |

Final eval: eval step `{n}`, wall time `{hh:mm}`, epochs `{k}`.

### Per-eval progression

{One row per eval. Include every metric the trainer reported.
Generated from `runs/exp_017e_framewise_bce_regularized/metrics.jsonl`.}

| Eval | Step | Ep | loss | na_loss | gap | F1 | Prec | Rec | AR match | AR halluc | AR dr | AR dc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | — | — | — | — | — | — | — | — | — | — | — | — |

Machine-readable copies: [`metrics.json`](./metrics.json).

## Visualizations

![Training loss](graphs/01_train_loss.png)
*Training loss (log-y).*

![Val vs noaug loss](graphs/02_val_vs_noaug_loss.png)
*Val vs train_noaug loss, with #017d reference. Overfitting gap comparison.*

![Train recall/precision](graphs/03_train_recall_precision.png)
*Training batch recall, precision, F1 progression.*

![AR corpus all](graphs/04_ar_corpus_all.png)
*AR corpus metrics across all 017 family experiments with #007 reference.*

{Add custom graphs as needed.}

## Vs prediction

- AR `matched_rate` >= 0.75: predicted → actual `{value}` → **{match / beat / miss / wrong direction}**
- AR `halluc_rate` <= 0.12: predicted → actual `{value}` → **{...}**
- AR `density_ratio` 0.85-1.05: predicted → actual `{value}` → **{...}**
- AR `dc_human` >= 92: predicted → actual `{value}` → **{...}**
- overfitting gap at E8 < 0.041: predicted → actual `{value}` → **{...}**

## Takeaways

- {One concrete sentence.}
- {Next.}

## Followup questions

- {Question.} — {suggested next experiment or dataset probe}
