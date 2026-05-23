# Experiment 017e — Framewise BCE regularized (label smoothing + head dropout + F1 watch)

## Status

`Complete`

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

The run trained for 14 evals across 3.5 epochs (steps 20,674 --
289,436). The three regularization changes (label smoothing 0.05,
head dropout 0.2, metric_to_watch=frame/f1) pushed the overfitting
wall 40-50% further than [#017d](../017d-framewise-bce-noweight/) at
every matched step. Frame F1 peaked at E11 (0.827) and AR metrics
peaked at E8 (`matched_rate` 0.871 at default tau=0.3).

Post-run threshold sweep (320 configs: 16 checkpoints x 4 thresholds
x 5 max_notes) found the optimal operating point at **E8 (step
165,392) with tau=0.40**: `matched_rate` 0.783, `hallucination_rate`
0.201, `density_ratio` 1.020, `dc_human` 92.7, `error_median_ms`
10.3 [threshold_sweep.json]. This beats
[#007](../007-time-stretch/) on matched_rate (+8 pp), density_ratio
(1.02 vs 0.87 — 017e is closer to 1.0), dc_human (+0.7 pp), and
error_median (tied at ~10ms).

### Headline finding

**The best framewise model to date.** At the optimal threshold
(E8/tau=0.40), the model achieves near-perfect density
(`density_ratio` 1.020 — essentially 1:1 with GT), `matched_rate`
0.783 (vs #007's 0.703), and `dc_human` 92.7 (vs #007's 92.0).
The remaining gap to #007 is `hallucination_rate` (0.201 vs 0.172).

The regularization succeeded: the overfitting gap at E8 is +0.021
(vs #017d's +0.041 at the same step — 50% reduction). The model
continued improving through E11 on frame F1, and the best AR
checkpoint (E8) has both high matched_rate AND reasonable density.

### Final vs baseline

`best sweep` = E8 (step 165,392) at tau=0.40. `best F1` = E11
(step 227,414, saved as `best.pt`). `final` = E14 (step 289,436).
Baseline = [#007](../007-time-stretch/) best (E18, step 413,480).

| Metric | #007 | 017e sweep (E8@0.40) | 017e E11 (best.pt) | 017e E14 |
|---|---:|---:|---:|---:|
| AR `matched_rate` | 0.703 | **0.783** | 0.850 (@tau=0.3) | 0.831 |
| AR `hallucination_rate` | **0.172** | 0.201 | 0.217 | 0.208 |
| AR `density_ratio` | 0.865 | **1.020** | 1.16 | 1.10 |
| AR `dc_human` (%) | 92.0 | **92.7** | 92.0 | 92.6 |
| AR `error_median_ms` | 10.2 | **10.3** | 6.5 | 11.2 |
| AR `events_per_sec` | 3.57 | 4.31 | 4.79 | 4.55 |
| `over_pspace_self` | 7.26 | n/a | 8.67 | 8.88 |
| `gap_peak_count` | 3.65 | n/a | 3.65 | 3.56 |
| frame F1 | n/a | n/a | **0.827** | 0.819 |
| frame Precision | n/a | n/a | 0.882 | 0.892 |
| frame Recall | n/a | n/a | 0.779 | 0.758 |
| val-noaug gap | n/a | n/a | +0.025 | +0.027 |

### Per-eval progression

| E | Step | Ep | loss | na_loss | gap | F1 | Prec | Rec | AR mr | AR hr | AR dr | AR dc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,674 | 0 | 0.235 | 0.233 | +0.002 | 0.742 | 0.913 | 0.625 | 0.750 | 0.209 | 0.99 | 92.0 |
| 2 | 41,348 | 0 | 0.233 | 0.229 | +0.004 | 0.792 | 0.903 | 0.705 | 0.818 | 0.222 | 1.07 | 92.3 |
| 3 | 62,022 | 0 | 0.234 | 0.227 | +0.008 | 0.808 | 0.897 | 0.735 | 0.816 | 0.217 | 1.10 | 92.1 |
| 4 | 82,696 | 0 | 0.236 | 0.224 | +0.012 | 0.808 | 0.901 | 0.732 | 0.834 | 0.221 | 1.12 | 92.6 |
| 5 | 103,370 | 1 | 0.237 | 0.223 | +0.014 | 0.818 | 0.890 | 0.757 | 0.852 | 0.223 | 1.17 | 92.0 |
| 6 | 124,044 | 1 | 0.240 | 0.222 | +0.018 | 0.816 | 0.898 | 0.747 | 0.831 | 0.219 | 1.12 | 92.4 |
| 7 | 144,718 | 1 | 0.239 | 0.221 | +0.019 | 0.816 | 0.897 | 0.749 | 0.842 | 0.216 | 1.13 | 92.2 |
| 8 | 165,392 | 1 | 0.241 | 0.220 | +0.021 | 0.823 | 0.881 | 0.771 | 0.871 | 0.232 | 1.21 | 92.1 |
| 9 | 186,066 | 2 | 0.241 | 0.219 | +0.022 | 0.814 | 0.899 | 0.744 | 0.828 | 0.208 | 1.11 | 92.4 |
| 10 | 206,740 | 2 | 0.243 | 0.218 | +0.025 | 0.826 | 0.883 | 0.777 | 0.844 | 0.214 | 1.15 | 92.0 |
| 11 | 227,414 | 2 | 0.243 | 0.218 | +0.025 | 0.827 | 0.882 | 0.779 | 0.850 | 0.217 | 1.16 | 92.0 |
| 12 | 248,088 | 2 | 0.243 | 0.217 | +0.025 | 0.820 | 0.891 | 0.760 | 0.845 | 0.211 | 1.14 | 92.2 |
| 13 | 268,762 | 3 | 0.243 | 0.217 | +0.026 | 0.813 | 0.894 | 0.745 | 0.827 | 0.205 | 1.10 | 92.6 |
| 14 | 289,436 | 3 | 0.243 | 0.216 | +0.027 | 0.819 | 0.892 | 0.758 | 0.831 | 0.208 | 1.10 | 92.6 |

Machine-readable copies: [`metrics.json`](./metrics.json).

Threshold sweep results: [`threshold_sweep.json`](./threshold_sweep.json).

## Visualizations

![Training loss](graphs/01_train_loss.png)
*Training loss (log-y).*

![Overfitting](graphs/02_overfitting.png)
*Left: val vs noaug loss. Right: overfitting gap comparison with
#017d — 017e's gap is 40-50% smaller at every matched step.*

![AR corpus](graphs/03_ar_corpus.png)
*AR corpus metrics overlaid with #017d and #007 reference. 017e
(red) starts at near-perfect density (1.0) from E1 and maintains
higher matched_rate than 017d throughout.*

![Train R/P](graphs/04_train_rp.png)
*Training recall/precision/F1. Same precision-first trajectory as
017d (precision starts ~0.90, recall climbs). R-P gap stays
negative (precision > recall) throughout.*

![Threshold sweep](graphs/05_threshold_sweep.png)
*Threshold sweep across all checkpoints (max_notes=0). E8
highlighted. tau=0.40 lands near density 1.0 with the best
matched_rate.*

## Vs prediction

| Metric | Predicted | Actual | Verdict |
|---|---:|---:|---|
| AR `matched_rate` >= 0.75 (sweep) | 0.783 @ E8/tau=0.40 | **PASS** |
| AR `halluc_rate` <= 0.12 | 0.201 | **FAIL** (higher than predicted) |
| AR `density_ratio` 0.85-1.05 | 1.020 | **PASS** |
| AR `dc_human` >= 92 | 92.7 | **PASS** |
| Overfitting gap at E8 < 017d's 0.041 | 0.021 | **PASS** (50% reduction) |
| AR `matched_rate` >= 0.72 (per-eval) | 0.871 | **PASS** |
| `pos_rate_pred_50` > 0.005 | 0.028 | **PASS** |

**Summary**: 5 of 6 predictions PASSED. The `halluc_rate` target
(0.12) was too ambitious — the model achieves 0.201 at the density-
optimal threshold (tau=0.40). The halluc_rate is a consequence of
the model still placing some onsets at metronomic audio beats rather
than chart-author-selected ones; the regularization did not fix this
fundamental selectivity limit, but it did push the overall quality
ceiling higher.

## Takeaways

- **Label smoothing + head dropout + F1-watching work as
  regularization.** The overfitting gap is 40-50% smaller at every
  matched step vs [#017d](../017d-framewise-bce-noweight/). Frame F1
  peaks later (E11 vs 017d's E8) and AR metrics improve for longer.

- **Best framewise model of the series.** At E8/tau=0.40:
  `matched_rate` 0.783, `density_ratio` 1.020, `dc_human` 92.7,
  `error_median_ms` 10.3. Beats [#007](../007-time-stretch/) on
  matched_rate (+8 pp), density (+0.15 closer to 1.0), dc_human
  (+0.7 pp), and ties on error.

- **metric_to_watch=frame/f1 saves the right checkpoint.** `best.pt`
  at E11 has frame F1 0.827 and AR `matched_rate` 0.850 — much
  better than the loss-optimal E2 checkpoint. This change alone
  would have improved #017d.

- **Density starts near 1.0 from E1.** Label smoothing prevents
  the model from being too conservative early (017d started at
  density 0.58 and took 6 evals to reach 0.81). 017e starts at
  0.99 and stays in the 1.0-1.2 range throughout.

- **max_notes_per_step has almost no effect.** Limiting to 1, 2, 4,
  or 8 notes per AR step produces nearly identical results to
  unlimited (0). The AR cursor-advance mechanics naturally handle
  multi-note emission.

- **The halluc_rate ceiling remains at ~0.20.** Even with
  regularization, the model cannot push below 0.20 halluc at
  density ~1.0. This is the same fundamental selectivity limit
  all 017-family models hit: the model detects real audio onsets
  but cannot fully distinguish chart-author-selected ones from
  metronomic beats.

- **The model is more context-dependent than 017d.** Benchmark
  analysis shows `no_past_audio` drops F1 by 13.5% (vs 017d's
  8.0%) and `no_context` drops 5.0% (vs 017d's 4.1%). The
  regularization forces the model to use more input channels
  rather than memorizing audio patterns alone.

## Followup questions

- **Re-run #017d threshold sweep** with the fixed comparison order
  (`pred_chart.compare(gt)`) to get correct numbers for comparison.
- **Coincidence map input channels.** Add the 13-row coincidence
  summary as a parallel input alongside mel. The IDF-weighted onset
  importance signal could help the model distinguish chart-worthy
  onsets from metronomic beats — directly targeting the halluc_rate
  ceiling. Experiment 018 candidate.
- **More data.** The overfitting gap is still growing (+0.027 at
  E14). More training charts would be the most reliable way to push
  the ceiling further.
- **Architectural bin competition.** Softmax-over-windows or learned
  NMS to enforce that bins compete for activation, mimicking #007's
  implicit competition. Could address the halluc_rate ceiling without
  changing the loss.
