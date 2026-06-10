# Experiment 024c -- Typing model with context 32/32

## Status

`Planned`

## Context

[#024b](../024b-typing-full/) plateaued at type accuracy 0.718 with
past/future context of 16 onsets each. The remaining ~28 % error is
consistent with #023's finding that 73 % of pattern-repeat pairs have
"other" D/K assignments (neither same nor flipped), suggesting the
local 16-onset window does not capture enough structure.

This experiment doubles the context to 32 past + 32 future (65 tokens
total) to test whether the 0.718 ceiling is context-limited. If type
accuracy climbs past 0.73 with more context, longer windows are the
right direction. If it stays at ~0.72, the ceiling is elsewhere
(model capacity, or inherent D/K ambiguity).

## Citations

- Baseline: [#024b](../024b-typing-full/) -- type_acc 0.718
  [024b-typing-full/metrics.json, step 489480], AR type_sym 0.542,
  171K params, context=16/16.
- Pattern structure: [#023](../023-kind-acoustics/) -- 73 % "other"
  in pattern repeat analysis, 4-gram DKDK = 10.5 % of corpus
  [023-kind-acoustics/results/summary.json].

---

## Hypothesis

### Claim

If the typing transformer sees 32 past + 32 future onsets instead
of 16/16, teacher-forced type accuracy will exceed 0.73, because the
longer window captures phrase-level D/K structure (8-16 onset phrases)
that the 16-onset window misses.

### Mechanism

At 16 past onsets and typical density ~5 events/sec, the model sees
~3.2 seconds of history. Musical phrases in taiko charts commonly
span 4-8 bars at 170 BPM, which is ~5.6-11.3 seconds or ~28-56
onsets. The 16-onset window catches only half a phrase on each side.
Doubling to 32 gives ~6.4 seconds per side -- enough to capture a
full phrase boundary and its D/K transition pattern.

The model has 172K params (only 1K more than 024b's 171K -- the
difference is the larger position embedding). Transformer
self-attention over 65 tokens is cheap at d_model=64.

### Predicted numbers

| Metric | 024b (ctx 16/16) | Predicted (024c, ctx 32/32) | Notes |
|---|---:|---:|---|
| type/accuracy | 0.718 | > 0.73 | more context lifts ceiling |
| type/entropy_mean | 0.516 | < 0.50 | more context -> more certainty |
| type/mass_decisive | 0.176 | > 0.20 | longer patterns -> clearer signals |
| strength/best_f1_BIG | 0.703 | > 0.70 | more IOI context helps BIG |
| ar/type_accuracy_sym | 0.542 | > 0.55 | slight AR lift from better model |

## Success criteria

- **Must have:** type/accuracy > 0.72 (matches or beats 024b).
- **Must have:** no NaN, no crash, training completes.
- **Nice-to-have:** type/accuracy > 0.73 (ceiling lifted).
- **Nice-to-have:** ar/type_accuracy_sym > 0.55.
- **Fails if:** type/accuracy < 0.71 (longer context hurt).

## Changes from baseline

Baseline: [#024b](../024b-typing-full/).

- `config/model.json` -- `past_context: 16 -> 32`,
  `future_context: 16 -> 32`. Window: 33 -> 65 tokens.
- `config/data.json` -- `past_context: 16 -> 32`,
  `future_context: 16 -> 32`. Sampler extracts wider windows.
- All other configs (loss, trainer, adapter) identical to 024b.
- Position embedding: 33 -> 65 entries (1,056 -> 2,080 params).
  Total model: ~172K params.

## Run config

- Run name: `exp_024c_typing_ctx32`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42),
  subsample 1.
- Command:
  ```bash
  source osu/taiko2/fix.sh
  PYTHONPATH=. osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_024c_typing_ctx32 \
      --config-dir osu/taiko2/experiments/024c-typing-ctx32/config \
      --dataset taiko2_v1 \
      --device cuda
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
