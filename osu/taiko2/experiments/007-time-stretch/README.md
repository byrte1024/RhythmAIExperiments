# Experiment 007 — Time-stretch augmentation

## Status

`Planned`

## Context

Across [#002](../002-exp45-full/), [#005](../005-gaussian-ce/), and
[#006](../006-cursor-overlap/), two ceilings stayed locked:

1. The `±log 2` / `±log 3` ridges in the `ratio_error` heatmap —
   classic octave / triplet / half-speed confusions the per-sample
   loss cannot discriminate between.
2. Val miss stuck in the 0.26–0.29 band, with the `train_noaug`
   diagnostic added in #005 showing clean overfitting (gap widens
   while val flattens).

#006 tested whether reducing intra-epoch cursor correlation helped;
it made overfitting worse, confirming that removing per-cursor
variety hurts rather than helps. The signal the model WANTS but
doesn't have: tempo-invariance. Two charts where all events have
been uniformly time-scaled by 1.25× are musically the same rhythm
at different speeds — but in the training set, each of those
scalings would be a distinct chart, and the model has no way to
relate them. That mirror's the tempo-CNN literature's finding
(Schreiber 2018) that scale-invariant audio augmentation is the
single strongest intervention for octave errors in onset/tempo
tasks.

This experiment adds a `TimeStretch` post-sample augmentation that
per-sample draws a stretch factor log-uniformly from `[1/1.4, 1.4]`
and rescales both the mel window and past/future event offsets
around the cursor. The rest of the recipe is identical to #002.

## Citations

- Baseline: [#002 — exp 45 full recreation](../002-exp45-full/).
  Best val miss 0.2606 at eval 11 (step 227,414). Same loss, model,
  adapter, dataset that #007 inherits.
- `train_noaug` diagnostic source: [#005 — Gaussian CE](../005-gaussian-ce/).
  First run with the augmentation-off training probe; revealed the
  overfit signature retroactively visible in #002.
- Recent negative result: [#006 — cursor-overlap filtering](../006-cursor-overlap/).
  Confirmed that reducing intra-epoch revisit redundancy hurts;
  overfitting is not a revisit-count problem, it's a data-variety
  problem.
- MIR precedent (unlinked, domain knowledge): mel-frame time-stretch
  at ±40 % is the canonical data-aug trick for tempo-invariance and
  octave-confusion reduction on onset networks.

---

## Hypothesis

### Claim

If we add a per-sample `TimeStretch(prob=0.3, max_scale=1.4)`
augmentation to #002's pipeline and keep everything else identical,
the watched metric `val/single/onset/miss` will reach a best value
at least 1.5 pp BELOW #002's 0.2606 (i.e. ≤ 0.246) **and** the
train_noaug − val miss gap at that best eval will be ≤ 2.0 pp,
because exposing the model to rescaled versions of its training
distribution directly attacks the tempo-invariance gap that is the
most likely cause of both the ratio-banding ridges and the
generalization ceiling.

### Mechanism

Two effects, stacked:

1. **Ratio-banding attack.** The `±log 2` ridge in #002's
   `ratio_error.png` is where the model, given a context with
   dominant gap `g`, predicts the target onset at `g`, `2g`, or
   `g/2` instead of the true value. This is exactly tempo confusion:
   the model has seen versions of the same musical pattern at
   different absolute speeds, but not relative to each other. Time-
   stretching training samples by factor `s` in `[1/1.4, 1.4]`
   forces the model to make consistent predictions across ±40%
   speed variants of the same local context, which should directly
   compress these ridges.
2. **Effective dataset growth.** Each training sample, per pass,
   becomes a family of stretched variants (up to ~40% faster or
   slower). This is pure data diversification — the sample the
   model sees at step N is a different numerical tensor from the
   same sample at step N+epoch, because the stretch factor was
   re-drawn. Memorization cost goes up; generalization should
   improve.

Both effects should reduce the train_noaug / val gap. The first
attacks the specific failure mode visible in artifact graphs; the
second is the generic "more aug = less overfit" story.

### Predicted numbers

Reference: #002 @ best eval (E11, step 227,414):

| Metric | #002 @ E11 | Predicted (#007, best eval) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.2606 | **≤ 0.246** | must-have, ≥ 1.5 pp improvement |
| val/single/onset/hit  | 0.7292 | ≥ 0.744     | paired with miss |
| val/single/onset/exact | 0.5485 | ≥ 0.54 | modest drop acceptable (slight smoothing from interpolation) |
| val/single/onset/frame_err_p90 | 31 | ≤ 28 | tail should shrink if octave errors fall |
| train_noaug − val miss gap (pp) @ best val | unknown for #002 | **≤ −2.0 pp** | hypothesis metric |

Artifact-level predictions (observational, not gated on numbers):

- `ratio_error.png` at best eval should show the `±log 2` ridge
  visibly compressed vs #002's. Same for `±log 3` (triplet ridge),
  though less confidently — triplet rescaling is further outside
  the aug's range.
- `train_noaug/ratio_error.png` (new side-by-side wiring from the
  loop change in this run) should look cleaner than val's, telling
  us whether stretching reduced the ridge on TRAIN (which it
  trivially should if the aug does anything) while we watch whether
  it also reduces it on VAL (the generalization question).

## Success criteria

- **Must have:** final `val/single/onset/miss` ≤ 0.246 (≥ 1.5 pp
  improvement over #002's 0.2606).
- **Must have:** `train_noaug − val miss` gap at the best val eval
  ≤ 2.0 pp.
- **Must have:** training runs to completion without NaN / Inf /
  OOM; all eval + train_noaug artifacts write.
- **Nice-to-have:** visibly narrower `±log 2` ridge in
  `ratio_error.png` vs #002's graph 08.
- **Nice-to-have:** val miss continues improving past the #002 E11
  step count (suggests the augmentation lifted the ceiling, not
  just shifted where training plateaus).
- **Fails if:** final miss > 0.2606 (time-stretch hurt the baseline).
- **Fails if:** `train_noaug − val` gap wider than #002's at a
  comparable step (time-stretch made overfit worse, like #006).

## Changes from baseline

Baseline: [#002](../002-exp45-full/).

- CLI flag: `--time-stretch-prob 0.3 --time-stretch-max-scale 1.4`.
  The `TimeStretch` aug is prepended to the post-augmentation list
  so subsequent event-level augs (EventJitter, dropout, etc.) see
  the stretched sample.
- `training/augmentations.py` — new `TimeStretch` class. Linear
  interpolation on the mel time axis with cursor pinned; event
  `cursor_offset` multiplied by the per-sample scale factor with
  `time_ms` / `bin` recomputed consistently; past events falling
  outside `[-a_bins, 0]` masked as padding; STOP flips propagate
  through the adapter from the scaled future-event offset vs
  `b_pred`.
- `training/loop.py` — eval-artifact save now also runs on the
  `train_noaug` pass (lands under `{run_dir}/eval_{step}/train_noaug/`),
  so we can directly compare val and train-noaug versions of
  `heatmap`, `ratio_error`, `ratio_hit`, `metronome`, etc. to tell
  whether the rays VANISH on train (= generalization problem) or
  PERSIST on both (= capability problem).
- Benchmark `time_shifted` renamed to `context_time_shifted` for
  clarity — it only rescales past-event offsets, not audio. #007's
  `TimeStretch` rescales both, so the distinction matters when
  reading the benchmark table.
- `onset/pred_stop_rate` metric: keeps the fix from #006 (total
  STOP predictions / total samples; the legacy FP-only quantity
  lives under `onset/pred_stop_fp_rate`).

Nothing else changes: model (`EventEmbeddingDetector`), loss
(`OnsetLoss`), adapter, dataset split, optimizer, schedule, seed,
cursor-overlap (back to 0 as in #002/#005), evals_per_epoch=4.

## Run config

- Run name: `exp_007_time_stretch`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42,
  song-grouped).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_007_time_stretch \
      --config-dir osu/taiko2/experiments/007-time-stretch/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/007-time-stretch/config/infer.json \
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

_(Table generated from `runs/exp_007_time_stretch/metrics.jsonl`.)_

## Visualizations

_(Graphs post-run.)_

## Vs prediction

_(One line per predicted metric post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
