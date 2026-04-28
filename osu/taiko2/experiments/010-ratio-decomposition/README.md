# Experiment 010 — Ratio-decomposed onset prediction

## Status

`Running`

## Context

Across experiments #005–#009, we confirmed three things:

1. The `±log 2` / `±log 3` ratio-banding ridges are a **capability**
   problem — present on both val and train_noaug, unchanged by any
   loss-side intervention (Gaussian CE, log-ratio EMD, MDN).
2. The model's top-K contains the right answer 90%+ of the time
   (taiko1 exps 39, 44-C) — it sees multiple plausible onsets but
   can't commit to the right one.
3. The MDN experiments (#009, #009b) showed the model DOES have
   multi-modal internal state (components specialized), but the MDN
   output format fights the bin-precision the task requires.

taiko1 experiment 67 attempted a ratio-decomposed approach that
addresses the root cause: instead of predicting a raw bin offset
(mashing together "what's the tempo?", "where am I in the beat?",
and "what comes next?"), decompose the prediction into three
sequential questions:

- **Divisor**: what's the dominant rhythmic gap? (the "beat")
- **Offset**: how far is the cursor from the last event?
- **Ratio**: what multiple of the divisor is the next onset?

taiko1 exp 67 was abandoned after only 2 evals with promising
signals (divisor 73.9% accurate, AR output "musically structured")
but fixable problems (ratio collapse to ~5 values from 255 bins,
lossy multiplication blur). This experiment reruns the design on
taiko2's backbone with fixes: Conv1d smoothing on the ratio head
(untested in taiko1 67b), cursor-shift augmentation for offset
training, and proper training duration.

## Citations

- Direct precedent: taiko1 exp 67 — ratio-based onset prediction.
  Stopped at eval 2. Divisor 73.9% accurate, ratio collapsed to
  ~5-10 values, AR quality "musically structured" at 27.7% HIT.
- taiko1 exp 67b — proposed Conv1d smoothing fix for ratio collapse.
  Never ran.
- Baseline for metrics: [#007 — time-stretch](../007-time-stretch/).
  Best val miss 0.2406 at step 372,132.
- MDN diagnostic: [#009](../009-mdn/). Confirmed model has
  multi-modal internal state (2 of 3 components specialized).
- Ratio coverage data: taiko1 analysis — 91.6% of targets hit clean
  musical ratios; 82% covered by just 1.0×/0.5×/2.0×.

---

## Hypothesis

### Claim

If we decompose the prediction into divisor + offset + ratio heads
(taiko1 exp 67 design on taiko2's EventEmbeddingDetector backbone),
the ratio head will learn structured predictions that match the
musical-ratio distribution of the training data, and the
per-head heatmaps (divisor target/pred, ratio target/pred) will show
clear diagonal structure by eval 5. Headline derived-bin miss will
be within 5 pp of #007 by eval 10 (≤ 0.29), and the ratio-space
RHIT/RGOOD metrics will show the model learning ratio-level
precision.

### Mechanism

The ratio decomposition attacks the octave confusion at its
structural root:

1. **The divisor head learns tempo.** "What's the beat?" is a
   well-defined question the model can answer from audio + event
   context (taiko1 showed 73.9% accuracy at eval 2). Once the
   model commits to a tempo, the octave ambiguity collapses — if
   divisor = g, then predicting "1.0× ratio" vs "2.0× ratio" is a
   binary decision with concentrated training signal, not a smeared
   decision across 500 bins.

2. **The ratio head operates in log-space.** 255 log-spaced bins
   from 0.125× to 8.0× means each octave spans 42 bins. The
   natural musical ratios (1.0×, 0.5×, 2.0×, 0.33×, 3.0×) are
   well-separated in this space. The softmax over 255 ratio bins
   gives the model a structured vocabulary for rhythmic
   relationships.

3. **Dynamic ratio target from predicted divisor.** The ratio
   target is computed as `(target_bin + offset) / divisor_pred`,
   using the model's OWN divisor prediction. This means the ratio
   head learns to compensate for divisor errors — if the divisor
   head predicts 2× the true beat, the ratio head learns to predict
   0.5× to compensate. The system is self-correcting.

4. **Cursor-shift augmentation.** 30% of training samples shift the
   cursor forward between events, creating non-zero offset targets.
   This trains the offset head for the STOP-hop case at AR
   inference, where the cursor isn't at an event boundary.

### Predicted numbers

Reference: #007 @ best (E18, step 372,132).

| Metric | #007 @ E18 | Predicted (#010, mature eval) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.2406 | ≤ 0.29 | within 5 pp of #007 |
| ratio/div_acc | n/a | ≥ 0.60 | divisor accuracy (taiko1 hit 65.8% at E2) |
| ratio/div_acc_3 | n/a | ≥ 0.70 | divisor within ±3 bins |
| ratio/rgood | n/a | ≥ 0.50 | ratio within ±10% of true ratio |
| ratio/rhit | n/a | ≥ 0.30 | ratio within ±3% of true ratio |

Observational (not gated):
- **Divisor heatmap** should show clear diagonal.
- **Ratio heatmap** should show structured mass at musical ratios
  (1.0×, 0.5×, 2.0×), NOT the horizontal banding taiko1 exp 67 had.
- **Ratio error distribution** should peak at 0 (correct ratio)
  with secondary peaks at ±log(2) (octave errors) — mapping the
  ambiguity structure explicitly.

## Success criteria

- **Must have:** ratio/div_acc ≥ 0.50 by eval 5 (divisor head
  learns something).
- **Must have:** ratio/rgood ≥ 0.30 by eval 5 (ratio head learns
  something).
- **Must have:** divisor heatmap shows visible diagonal structure.
- **Must have:** training runs without NaN.
- **Nice-to-have:** val miss ≤ 0.29 by eval 10.
- **Nice-to-have:** ratio error distribution shows clean peaks at
  musical-ratio positions.
- **Fails if:** ratio head collapses to <10 unique values (same
  pathology as taiko1 exp 67 despite Conv1d smoothing).
- **Fails if:** val miss > 0.40 at eval 5 (fundamental failure to
  learn).

## Changes from baseline

Baseline: [#007](../007-time-stretch/).

- `config/model.json` — swap `EventEmbeddingConfig` →
  `RatioDetectorConfig`. Inherits the full backbone; replaces the
  501-class softmax head with 3 decomposed heads:
  - Divisor: `LN → Linear(384,192) → GELU → Linear(192,500)`.
  - Offset: `LN → Linear(384,192) → GELU → Linear(192,100)`.
  - Ratio: receives cursor_tok + embedded div/off soft expectations
    → `LN → Linear(384,384) → GELU → Linear(384,256)` +
    Conv1d(1→8→1, k=5) smoothing. 255 log-spaced ratio bins + STOP.
  Total: ~16.9M params (+500k over standard head).

- `config/loss.json` — swap `OnsetLossConfig` → `RatioLossConfig`.
  Three loss components:
  - Divisor CE (aux, weight 0.1, fixed GT target from IOI mode).
  - Offset CE (aux, weight 0.1, fixed GT target from cursor−last_event).
  - Ratio CE (primary, dynamic target from predicted div/off).
  Validity masks zero div/off CE on samples with insufficient past
  events. Ratio head frozen for first 20,674 steps (1 eval) while
  div/off warm up.

- `config/infer.json` — swap `ArgmaxDecoder` → `RatioDecoder`.
  Derives bin from `divisor × ratio_value − offset`.

- CLI flags: `--cursor-shift-prob 0.3` (new pre-sample aug shifting
  cursor between events for offset head training).

- Everything else identical to #007: augmentations (TimeStretch
  0.3), dataset, optimizer, schedule, seed.

## Run config

- Run name: `exp_010_ratio`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_010_ratio \
      --config-dir osu/taiko2/experiments/010-ratio-decomposition/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --cursor-shift-prob 0.3 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/010-ratio-decomposition/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

---
<!-- Post-run below -->
---

## Results summary

_(To fill post-run.)_

## Visualizations

_(Post-run.)_

## Vs prediction

_(Post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
