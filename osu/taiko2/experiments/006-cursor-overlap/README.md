# Experiment 006 — Cursor-overlap filtering (smaller, less-correlated dataset)

## Status

`Planned`

## Context

[#005](../005-gaussian-ce/) revealed — via its new `train_noaug`
diagnostic pass — that the model overfits cleanly: val miss bottomed
out at step 124k (E6) while `train_noaug` miss kept dropping smoothly
through step 165k (E8), opening a −3.6 pp train/val gap. The flat
val shape #002 showed from its E5 onward retroactively looks like
the same thing; it just lacked the augmentation-off probe to see it.

The core suspicion: with `allowed_overlap_forward =
allowed_overlap_back = 0` (taiko1 parity — every event cursor is a
training sample), consecutive samples share ~95 %+ of their audio
window and 127 / 128 of their event context. The dataset is
nominally in the millions of samples but the **effective sample
count** is closer to hundreds of thousands of mostly-independent
scenes, each visited many times per "epoch" through overlapping
cursor placements. Seen another way: each epoch is essentially 10×
passes over ~10× fewer unique scenes, with augmentations as the only
thing preventing pure memorization.

This experiment tests the dataset-structure hypothesis by enabling
overlap-gap filtering: `allowed_overlap_forward =
allowed_overlap_back = 500` (= `b_bins` / `a_bins`) means consecutive
kept cursors must be at least 500 bins apart — one full prediction
window. Sample count drops ~10×, but each remaining sample is a
genuinely distinct scene. For a comparable total number of gradient
updates, the model sees each scene ~10× more times (across different
epochs, under different augmentation draws and optimizer states)
instead of ~10× adjacent-cursor variants within a single epoch.

## Citations

- Baseline: [#002 — exp 45 full recreation](../002-exp45-full/).
  Final watched metric `val/single/onset/miss = 0.2606` at eval 11
  (step 227,414). Runs with `allowed_overlap = 0`.
- Overfit diagnosis: [#005 — Gaussian CE](../005-gaussian-ce/).
  Best val miss 0.2664 at step 124,044; train/val gap grew from
  −1.0 pp at E1 to −3.6 pp at E8. First run with the `train_noaug`
  diagnostic pass enabled.
- The overlap-gap mechanism exists in `data_samplers/detection.py`
  and has been tested in taiko2 but was never used in a baseline
  recreation — taiko1 kept every cursor.

---

## Hypothesis

### Claim

If we filter training samples with `allowed_overlap_forward =
allowed_overlap_back = 500` (dropping to ~1/10th the sample count,
keeping only cursors at least one prediction window apart from each
other) and keep everything else identical to #002, the model will
reach **a smaller final train-noaug / val miss gap** (≤ 2.0 pp)
while matching #002's val miss within ±1.5 pp at the step where val
peaks. Equivalently: overfitting is primarily driven by redundant-
scene revisits within an epoch, not by training duration per se, so
breaking the intra-epoch revisit pattern should close the gap even
at comparable total step counts.

### Mechanism

Two effects push the same direction:

1. **Decorrelated revisits.** In #002/#005, cursor `i` and cursor
   `i+1` share ~98 % of their audio + 127/128 events — so a gradient
   step on sample `i+1` carries almost no new information relative
   to the step on sample `i`. Effectively the optimizer does a ~10×
   larger effective-batch update on each unique scene. Large-batch
   on correlated data is well-known to worsen generalization.
   Overlap-filtering decorrelates the revisits in time: when the
   model revisits a scene it's in a different epoch with fresh
   augmentation draws and moved-forward optimizer state.
2. **Meaningful epoch semantics.** `evals_per_epoch = 1` becomes
   coherent again — "I've seen every unique scene once" rather than
   "I've seen every scene ~10 times via adjacent cursors". LR
   schedules based on epochs stay sensible.

Counter-effect that might work against us: overlap-filtering removes
the many "same scene with shifted cursor" variants that are
themselves a form of data augmentation (different target bin,
slightly different audio framing). Whether the loss of that signal
outweighs the memorization-reduction benefit is exactly what this
experiment answers.

### Predicted numbers

Reference: #002 @ its best eval (step 227,414). **Comparisons are
by step (or epoch), not by eval index** — #006's
`evals_per_epoch = 1` means each eval sits 4× further in steps than
#002's equivalent-index eval.

| Metric | #002 @ step 227k | Predicted (#006, best eval) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.2606 | 0.246–0.276 | ±1.5 pp, miss criterion |
| val/single/onset/hit  | 0.7292 | 0.714–0.744 | paired |
| val/single/onset/exact | 0.5485 | ≥ 0.53     | should be stable — same loss as #002 |
| train_noaug − val miss gap (pp) @ best val | not measured | **≤ −2.0 pp** | key hypothesis metric |

Observational (not predicted):
- Total steps to best val miss. With ~10× fewer samples per epoch
  and the same optimizer / schedule, wall-time-to-best could go
  either direction.
- Benchmark behaviour — the smaller val split makes low-sample modes
  noisier, so `--benchmark-fraction` is raised to 0.25 (from 0.05).

## Success criteria

- **Must have:** final `val/single/onset/miss` within ±1.5 pp of
  #002's 0.2606 (0.246–0.276).
- **Must have:** `train_noaug − val miss` gap at the best val-miss
  eval ≤ 2.0 pp (vs #005's 2.9 pp at its best eval).
- **Must have:** training runs to completion without NaN / Inf /
  OOM; all eval artifacts write.
- **Nice-to-have:** val miss beats #002.
- **Nice-to-have:** val continues improving past the step at which
  #005 turned over (~124k). If overlap-filtering is the right
  diagnosis, the flat / reversing val shape should not appear here.
- **Fails if:** final miss above 0.28 (overlap-filtering actively
  hurt).
- **Fails if:** gap is *worse* than #005's (overlap-filtering made
  memorization cheaper, not harder).

## Changes from baseline

Baseline: [#002](../002-exp45-full/).

- `config/data.json — allowed_overlap_forward: 0 → 500` and
  `allowed_overlap_back: 0 → 500`. With `a_bins = b_bins = 500`,
  consecutive kept cursors must be ≥ 500 bins (~2.5 s) apart —
  directional audio windows no longer overlap between samples.
- `config/trainer.json — evals_per_epoch: 4 → 1`. Dataset is ~10×
  smaller per epoch; four evals per epoch would give very short
  eval intervals. One eval per epoch gives a meaningful cadence.
- `training/metrics_onset.py` — `onset/pred_stop_rate` fix. Was
  counting FP STOP predictions / non-STOP-target count; now counts
  **total STOP predictions / total samples** (the obvious semantics
  the name implies). The legacy FP-only quantity is preserved as
  `onset/pred_stop_fp_rate`. Retrospective: #002 and #005 reported
  `pred_stop_rate` numbers ~10× too low under the wrong name.
- Run flags: `--benchmark-fraction 0.25` (was 0.05 default) and
  `--train-noaug-fraction 0.25` (was 0.05 in #005) — with ~10×
  fewer val samples per mode and a smaller effective dataset, 5 %
  subsets are too noisy to read; 25 % gives a stable signal.

Nothing else changes: model, loss (`OnsetLoss` from #002), adapter,
augmentation pipeline, optimizer, seed — all identical to #002.

## Run config

- Run name: `exp_006_cursor_overlap`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42,
  song-grouped).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_006_cursor_overlap \
      --config-dir osu/taiko2/experiments/006-cursor-overlap/config \
      --dataset taiko2_v1 --device cuda \
      --benchmarks all --benchmark-fraction 0.25 \
      --train-noaug-fraction 0.25 \
      --infer-corpus-spec osu/taiko2/experiments/006-cursor-overlap/config/infer.json \
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

_(Table generated from `runs/exp_006_cursor_overlap/metrics.jsonl`.)_

## Visualizations

_(Graphs post-run.)_

## Vs prediction

_(One line per predicted metric post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
