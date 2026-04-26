# Experiment 008 — Log-ratio EMD loss (no entropy floor, perception-correct)

## Status

`Planned`

## Context

[#007](../007-time-stretch/) gave us a clean diagnostic: the
ratio-banding ridges at `±log 2` and `±log 3` look identical on val
and on `train_noaug` (graphs 10 and 13). The model fails the same way
on data it has seen many times in training as on held-out val. The
ridges are a **capability** failure, not a generalization failure;
no augmentation strategy will resolve them.

The loss-landscape analysis (`osu/taiko2/analysis/loss_landscape.py`)
identified the structural cause: #002's trapezoid soft CE has an
**entropy floor** — outside its ratio-plateau support, the loss
saturates flat, so the model gets zero gradient signal to move mass
off the octave/triplet bands. Hard CE pulls toward the target peak,
but soft CE just sits at its floor. The user observation during #007
mid-run confirmed this empirically: hard_CE dropped 12 % across the
run while soft_CE only dropped 4.5 %. The trapezoid's partial-credit
machinery never actuated.

The user also identified the perceptual constraint: in human rhythm
perception, octave-up and octave-down errors feel equally wrong —
the right metric is `|log((p+1)/(t+1))|`, symmetric in log-space. A
prediction at `2t` and one at `t/2` should cost the same.

This experiment combines those two insights: **log-ratio Earth-Mover
Distance**, mixed with hard CE to keep the strong "spike at target"
gradient. The EMD term has no entropy floor (its minimum is delta-
at-target, achievable by the model) and uses the perception-correct
log-ratio metric on bin probabilities.

## Citations

- Baseline: [#007 — time-stretch](../007-time-stretch/). Best val
  miss 0.2406 at step 372,132. Same model, schedule, and
  augmentation pipeline as #007 — only the loss differs in #008.
- Diagnostic source: [#007 graph 13](../007-time-stretch/graphs/13_best_train_noaug_ratio_error.png).
  Ridges identical on val and train_noaug — capability-failure
  evidence.
- Loss-landscape analysis: [`analysis/loss_landscape.py`](../../analysis/loss_landscape.py)
  and the rendered surfaces under `analysis/loss_landscapes/`.
  Panels 05 and 06 show the entropy-floor structure of #002's and
  #005's losses; panel 08 shows log-ratio EMD has the right
  combination (perception-correct curved valley + smooth gradient
  outward, no plateau).

---

## Hypothesis

### Claim

If we replace #002's `OnsetLoss` with `LogEmdLoss(hard_alpha=0.5,
exponent=1, stop_weight=1.5)` — keeping #007's full augmentation
pipeline including time-stretch — and run for the same step budget
as #007 (~370k steps to best), the ratio-banding ridges in val's
`ratio_error.png` will visibly compress vs #007's at the same eval,
**and** val miss will improve by at least 0.5 pp at matched step
count vs #007. Equivalently: the entropy-floor analysis predicts
that giving the model gradient signal at the octave bands will
specifically reduce ridge mass; if we see compression, the cause-
chain (entropy floor → ridges → ceiling) is confirmed and #008 is
the next baseline.

### Mechanism

Two effects pulling the same direction:

1. **No entropy floor on the bin term.** `log_emd = Σ Pᵢ |log((i+1)/(t+1))|`
   has minimum 0 iff the predicted bin distribution is delta-at-`t`.
   Mass on `2t` costs `≈ log 2 = 0.69`; mass on a random distant bin
   costs proportionally more. There is no flat region where moving
   mass costs nothing — every bin position has a non-zero gradient
   signal back to the target. Compare to trapezoid soft CE which
   bottoms out at its entropy floor outside the trapezoid support
   and provides no gradient on octave-distance mass.
2. **Bimodal-octave hedging is specifically punished.** A 50/50 split
   between mass at `t/2` and `2t` (the model "hedging the octave")
   scores `≈ log 2 = 0.69` under log-EMD — same as a sharp wrong-
   octave prediction. Compared to a sharp `1.3·t` prediction at
   `≈ log 1.3 = 0.26`, the bimodal hedge costs 2.7× more. The
   gradient pushes the model to commit, not split. This is the
   property the heatmap analysis identified as the specific weapon
   against the ridges.

Hard CE is kept at α=0.5 because the heatmap analysis showed log-EMD
alone has a smooth-but-shallow gradient near the optimum; mixing
with hard CE preserves the strong "sharpen toward target" signal that
we know works (it's been doing all the heavy lifting in #002 / #007).
The two terms agree at the optimum (delta at `t`), so they don't
fight each other the way trapezoid soft CE fought hard CE.

### Predicted numbers

Reference: #007 @ best (E18, step 372,132). Predictions at the same
step count where possible.

| Metric | #007 @ E18 | Predicted (#008, best eval) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.2406 | **≤ 0.235** | must-have, ≥ 0.5 pp improvement |
| val/single/onset/hit  | 0.7512 | ≥ 0.760 | paired |
| val/single/onset/exact | 0.5748 | ≥ 0.57 | should not regress |
| val/single/onset/rhit | 0.6484 | ≥ 0.66 | log-ratio metric directly attacks rhit's failure mode |
| val/single/onset/frame_err_p90 | 30 | ≤ 28 | EMD has gradient at the tail; should pull p90 in |
| `±log 2` ridge intensity (graph 10) | clearly visible | **visibly weaker** | the headline qualitative prediction |

Observational (not gated on numbers):
- **Hard_CE / log_emd component dynamics.** If the entropy-floor
  diagnosis is right, log_emd should drop faster than hard_CE
  (opposite of #007's trapezoid-CE pattern). If it stalls, the
  entropy-floor analysis was wrong.
- **STOP behaviour.** Should match #007's roughly — same softmax
  structure, same `stop_weight=1.5`, no decoupled head.

## Success criteria

- **Must have:** final `val/single/onset/miss` ≤ 0.235 at the best
  eval (≥ 0.5 pp better than #007).
- **Must have:** ridges in val's `ratio_error.png` at best eval are
  visibly weaker than #007's. Subjective check; will be documented
  with a side-by-side comparison.
- **Must have:** `train_noaug/ratio_error.png` ridges shrink in
  parallel with val's. If train ridges shrink but val's don't, we've
  trained for a memorization-only fix; if val ridges shrink, the
  generalization actually moved.
- **Must have:** training runs to completion without NaN / Inf / OOM.
- **Nice-to-have:** miss beats #007 by ≥ 1.0 pp.
- **Nice-to-have:** `frame_err_p90` reaches 28 (it stuck at 30 in #007).
- **Nice-to-have:** `log_emd` term drops faster than `hard_ce` (the
  entropy-floor escape).
- **Fails if:** val miss > 0.2406 at any best eval (loss change made
  things worse).
- **Fails if:** ridges stay visually identical to #007's. If so the
  capability gap is not loss-side and we need an architectural
  intervention (multi-hypothesis head, ratio-aware decoder) instead.

## Changes from baseline

Baseline: [#007](../007-time-stretch/).

- `config/loss.json` — swap `OnsetLossConfig` (`hard_alpha=0.5`,
  `good_pct=0.03`, `fail_pct=0.20`, `frame_tolerance=2`,
  `stop_weight=1.5`) → `LogEmdLossConfig(hard_alpha=0.5, exponent=1,
  stop_weight=1.5)`.
- `training/losses.py` — new `LogEmdLoss` class. `loss = α · hard_CE
  + (1 − α) · log_emd` over (B, n_classes) logits with full softmax
  over 501 classes. log_emd is computed only on bin-target samples
  (zero on STOP-target samples) over the bin part of the softmax;
  STOP samples pay `hard_CE * stop_weight` like #002.
- `cli/train.py` — loss instantiation dispatches on config type
  (`LogEmdLossConfig → LogEmdLoss`).

Nothing else changes from #007: model (`EventEmbeddingDetector`),
adapter, dataset split, optimizer, schedule, seed, cursor-overlap
(0), evals_per_epoch=4, augmentations including
`TimeStretch(p=0.3, max_scale=1.4)`.

## Run config

- Run name: `exp_008_log_emd`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42,
  song-grouped).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_008_log_emd \
      --config-dir osu/taiko2/experiments/008-log-emd/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/008-log-emd/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

---
<!-- Everything below written after the run. Do not pre-populate. -->
---

## Results summary

_(To fill post-run.)_

### Final vs baseline

_(Table.)_

### Per-eval progression

_(Table generated from `runs/exp_008_log_emd/metrics.jsonl`.)_

## Visualizations

_(Graphs post-run.)_

## Vs prediction

_(One line per predicted metric post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
