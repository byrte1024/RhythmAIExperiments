# Experiment 024b -- Typing model full run (full data + augs + entropy penalty + AR eval)

## Status

`Planned`

## Context

[#024](../024-typing-baseline/) validated the typing pipeline on
subsample-8: type accuracy 69.8 % (beating the 62 % alternation
baseline by 7.8 pp), strength F1_BIG 0.599 at threshold 0.80, no
crashes, loss dropping. The confidence distribution was the main
concern -- only 4.8 % decisive predictions, 27.4 % conflicted,
entropy at 84 % of maximum. The trajectory stalled at E4 on
subsample-8, suggesting data-limited convergence.

This experiment runs the same 171K-param typing transformer on **full
data** (subsample=1, ~6.3M train / 653K val samples) with three
additions:

1. **Augmentations**: past label dropout (15 %), IOI jitter
   (sigma=0.05), mel noise (sigma=0.1) -- robustness for AR inference
   where past labels come from the model's own predictions and onset
   positions from the (imperfect) onset detector.
2. **Entropy penalty** (weight=0.1 on type head): explicitly penalizes
   uncertain predictions, pushing sigmoid outputs toward 0 and 1.
   Addresses the flat confidence distribution from #024.
3. **AR eval hook**: every 2 evals, runs the typing model
   autoregressively over 100 val charts using
   `inference.typing_pass.type_chart` and computes chart-level
   metrics (symmetry-aware accuracy, pattern match, n-gram TVD,
   alternation rate, strength F1).

## Citations

- Smoke test: [#024](../024-typing-baseline/) -- type_acc 0.698
  [024-typing-baseline/metrics.json, step 18354], strength best_f1
  0.599 at threshold 0.80, 171K params, 6 evals on subsample-8.
- Kind acoustics: [#023](../023-kind-acoustics/) -- Fisher LDA 0.000105
  for D vs K, P(K|D) = 0.623, alternation baseline = 62 %
  [023-kind-acoustics/results/summary.json].
- Inference integration: `inference/typing_pass.py` -- shared
  `type_chart` function used by `cli/infer.py --typing-config`,
  `inference/corpus.py` via `typing_spec`, and the AR hook.

---

## Hypothesis

### Claim

If the typing transformer trains on full data with augmentations and
an entropy penalty (weight=0.1), type accuracy will exceed 73 % and
the decisive mass (predictions with confidence >0.9) will exceed 15 %,
because #024 stalled at E4 due to data exhaustion on subsample-8 and
the entropy penalty directly penalizes the flat confidence distribution.

### Mechanism

Three effects stacked:

1. **8x more data** (6.3M vs 783K train samples). #024 stalled at E4
   (step ~12K) with 783K samples -- the model had seen each chart's
   onsets ~2x by that point. Full data provides ~54K steps/epoch; with
   10 epochs the model sees ~540K steps total, enough to learn
   higher-order pattern structure beyond 4-grams.

2. **Entropy penalty** (`entropy_weight_type=0.1`). H(p) is maximized
   at p=0.5 (0.693 nats). Penalizing it pushes the sigmoid toward 0
   or 1 on every prediction. Combined with BCE (which pulls toward the
   correct class), the total loss rewards confident-and-correct
   predictions more than uncertain-but-correct ones. This should
   produce the bimodal confidence distribution that #024 lacked.

3. **Augmentations** simulate inference-time noise: past label dropout
   (15 %) mimics early-chart AR where few past predictions exist; IOI
   jitter (sigma=0.05) mimics onset detector position noise; mel noise
   (sigma=0.1) mimics audio variation. These should reduce the gap
   between teacher-forced val accuracy and AR chart-level accuracy.

### Predicted numbers

| Metric | #024 (smoke) | Predicted (024b best eval) | Notes |
|---|---:|---:|---|
| type/accuracy | 0.698 | > 0.73 | full data + entropy penalty |
| type/mass_decisive | 0.048 | > 0.15 | entropy penalty effect |
| type/mass_conflicted | 0.274 | < 0.20 | inverse of decisive |
| type/entropy_mean | 0.579 | < 0.50 | penalty pushes down |
| strength/best_f1_BIG | 0.599 | > 0.60 | more data helps |
| combined/accuracy | 0.619 | > 0.66 | both heads improve |
| ar/type_accuracy_sym_mean | n/a | > 0.65 | AR beats alternation baseline |
| ar/type_pattern_match_4_mean | n/a | > 0.15 | above random (6.25 %) |
| ar/strength_f1_BIG_mean | n/a | > 0.30 | AR BIG detection |

## Success criteria

- **Must have:** type/accuracy > 0.72 at best eval (beats #024 by 2+ pp).
- **Must have:** type/mass_decisive > 0.10 (doubles #024's 0.048).
- **Must have:** AR hook fires and produces ar_real_summary.json with
  > 50 charts analyzed.
- **Must have:** ar/type_accuracy_sym_mean > 0.62 (AR beats
  alternation baseline).
- **Nice-to-have:** type/accuracy > 0.75.
- **Nice-to-have:** type/entropy_mean < 0.45 (entropy penalty working
  strongly).
- **Nice-to-have:** ar/type_pattern_match_4_mean > 0.20.
- **Fails if:** type/accuracy below 0.70 (regression from #024's 0.698
  despite 8x data -- would mean augs or entropy penalty hurt).
- **Fails if:** type/mass_decisive below 0.05 (entropy penalty backfired).

## Changes from baseline

Baseline: [#024](../024-typing-baseline/).

- `config/data.json` -- `subsample: 8 -> 1` (full dataset).
- `config/trainer.json` -- `epochs: 3 -> 10`.
- `config/loss.json` -- `entropy_weight_type: 0.0 -> 0.1` (new).
- `config/adapter.json` -- new augmentations:
  - `past_label_dropout_prob: 0.0 -> 0.15`
  - `ioi_jitter_sigma: 0.0 -> 0.05`
  - `mel_noise_sigma: 0.0 -> 0.1`
- AR hook auto-installed (typing mode in `cli/train.py`), fires every
  2 evals on 100 val charts.
- `config/infer.json` -- typing inference spec for downstream use.

## Run config

- Run name: `exp_024b_typing_full`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42),
  **subsample 1** (full data).
- Command:
  ```bash
  PYTHONPATH=. osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_024b_typing_full \
      --config-dir osu/taiko2/experiments/024b-typing-full/config \
      --dataset taiko2_v1 \
      --device cuda \
      --no-augmentation
  ```

---
<!-- Everything below written after the run. Do not pre-populate. -->
---

## Results summary

{Post-run}

## Visualizations

{Post-run}

## Vs prediction

{Post-run}

## Takeaways

{Post-run}

## Followup questions

{Post-run}
