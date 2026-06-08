# Experiment 024 — Typing model smoke test (subsample 8)

## Status

`Planned`

## Context

[#023](../023-kind-acoustics/) established that DON and KA are
acoustically indistinguishable (Fisher LDA 0.000105 on 6.5M onsets)
and that D/K assignment is driven by pattern structure (62 %
alternation rate, 73 % "other" in pattern-repeat analysis). BIG
variants show weak acoustic separation (Fisher 0.0095) but strong IOI
position signal (2x longer preceding gaps).

This experiment is a **smoke test** of the typing model architecture:
a small bidirectional transformer that takes 33 onset tokens (16 past
with known D/K labels + 1 target + 16 future with unknown labels) and
predicts the target onset's type (D/K) and strength (normal/big).
The run uses subsample-8 to minimize GPU time; the goal is to verify
the pipeline works end-to-end (no NaNs, no crashes, loss drops) before
committing to a full run in 024b.

## Citations

- Kind acoustics analysis: [#023](../023-kind-acoustics/) -- Fisher
  LDA 0.000105 for D vs K, 0.0095 for BIG vs NORMAL
  [023-kind-acoustics/results/summary.json:dk_separability.fisher_lda_mean].
- Transition statistics: [#023](../023-kind-acoustics/) -- P(K|D) =
  0.623, P(D|K) = 0.614
  [023-kind-acoustics/results/summary.json:transition_dk].
- Corpus kind distribution: DON 46.8 %, KA 47.5 %, BIG_DON 2.8 %,
  BIG_KA 2.7 %, DRUMROLL 0.1 %, SPINNER 0.2 %
  [023-kind-acoustics/results/summary.json:kind_counts].

---

## Hypothesis

### Claim

If the typing transformer sees 16 past onsets with known D/K labels
and 16 future onset positions, it will exceed the "always alternate"
baseline (62 %) on teacher-forced type accuracy within 3 epochs on
subsample-8 data, because the pattern context (IOIs + past D/K
sequence) carries more information than a simple bigram model.

### Mechanism

The 62 % alternation rate is a first-order Markov baseline. The model
has access to higher-order structure: 4-gram patterns (DKDK = 10.5 %
of all 4-grams from #023), run lengths, IOI context that signals
phrase boundaries, and future onset positions that reveal the
rhythmic structure ahead. A 3-layer transformer with 16 past tokens
can capture at least 4-gram dependencies; the bidirectional attention
over future positions adds phrase-level context.

For the strength head, the IOI signal (BIG onsets have 2x longer
preceding gaps) is directly encoded in the IOI features. Even a
linear model on IOI_before should produce non-trivial BIG recall.

### Predicted numbers

| Metric | Baseline | Predicted (end of smoke) | Notes |
|---|---:|---:|---|
| type/accuracy | 62 % (alternation) | > 62 % | must beat bigram |
| type/accuracy | 50 % (random) | > 55 % | must learn something |
| strength/accuracy | 94.5 % (all normal) | > 94.5 % | must not regress |
| strength/f1_BIG | 0 % (never predict) | > 0.0 | must detect some BIG |
| loss (final eval) | -- | < loss (first eval) | must decrease |
| NaN / crash | -- | none | pipeline integrity |

## Success criteria

- **Must have:** no NaN, no crash, all eval artifacts write.
- **Must have:** loss decreases monotonically across evals (allowing
  small eval-to-eval wobble).
- **Must have:** type/accuracy > 55 % at final eval.
- **Nice-to-have:** type/accuracy > 62 % (beats alternation baseline).
- **Nice-to-have:** strength/f1_BIG > 0.05 (detects some BIG onsets).
- **Fails if:** type/accuracy below 50 % (worse than random).
- **Fails if:** training crashes or produces NaN.

## Changes from baseline

No baseline -- first typing model run.

- New domain types: `domain/typing.py` -- `TypingSample`,
  `TypingInput`, `TypingOutput`, `TypingTarget`, `TypingModelConfig`.
- New sampler: `data_samplers/typing.py` -- `TypingSampler`,
  onset-indexed windows of 33 onsets filtered to hits only.
- New model: `models/typing_model.py` -- `TypingTransformer`, 171K
  params, 3-layer bidirectional transformer with binary sigmoid
  heads for type (D/K) and strength (normal/big).
- New loss: `training/typing_loss.py` -- `TypingLoss`, BCE on type
  + weighted BCE on strength (pos_weight=17).
- New adapter: `training/typing_adapter.py` -- `TypingSampleAdapter`,
  D/K flip augmentation (50 %), future context dropout (10 %).
- New metric: `training/metrics_typing.py` -- `TypingMetric`,
  per-class P/R/F1, confidence stats, entropy, threshold sweeps.
- New artifacts: `training/typing_artifacts.py` --
  `TypingConfusionArtifact`, 14 plots per eval (confusion matrices,
  confidence distributions, calibration curves, entropy histograms,
  threshold sweeps) + 2 npz prediction dumps.
- Integration into `cli/train.py` -- dispatches on `TypingModelConfig`
  for model, loss, sampler, adapter, metrics, and artifacts.
- Fix in `training/loop.py` -- `_infer_b_pred` returns 0 when
  `output.logits` is absent (typing model has no logits field).

## Run config

- Run name: `exp_024_typing_smoke`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42),
  **subsample 8**.
- Command:
  ```bash
  PYTHONPATH=. osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_024_typing_smoke \
      --config-dir osu/taiko2/experiments/024-typing-baseline/config \
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
