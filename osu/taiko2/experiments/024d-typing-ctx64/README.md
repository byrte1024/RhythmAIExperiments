# Experiment 024d -- Typing model with context 64/64

## Status

`Planned`

## Context

[#024c](../024c-typing-ctx32/) showed that doubling context from
16/16 to 32/32 improved every metric: type accuracy +0.8 pp
(0.718 -> 0.726 [024c metrics.json, step 489480]), AR type sym
+1.3 pp (0.542 -> 0.556), AR ngram TVD -1.9 pp (0.241 -> 0.222).
Crucially, 024c was still climbing at E18 and only flattened in
the last 2 evals -- the ceiling had not been reached.

This experiment doubles context again to 64 past + 64 future (129
tokens). At typical density ~5 events/sec, 64 onsets covers ~12.8
seconds per side -- enough for 2-3 full musical phrases at common
taiko tempos (170 BPM). If the trajectory from 16 -> 32 (+0.8 pp)
continues, 0.73+ type accuracy is reachable.

## Citations

- Baseline: [#024c](../024c-typing-ctx32/) -- type_acc 0.726
  [024c metrics.json, step 489480], AR type_sym 0.556, 172K params,
  context=32/32.
- Context scaling: 024 (ctx16, 0.698) -> 024b (ctx16, 0.718) ->
  024c (ctx32, 0.726). Each step improved.
- Pattern structure: [#023](../023-kind-acoustics/) -- 73 % "other"
  in pattern repeat analysis at 4-onset window
  [023-kind-acoustics/results/summary.json].

---

## Hypothesis

### Claim

If the typing transformer sees 64 past + 64 future onsets, teacher-
forced type accuracy will exceed 0.73, because the 129-token window
captures full musical phrases (2-3 phrases at 170 BPM) whose D/K
structure the 65-token window only partially captured.

### Mechanism

The trajectory from context 16 to 32 showed +0.8 pp on type_acc,
with 024c still climbing at E18. Musical phrases in taiko charts
commonly span 4-8 bars at 170 BPM = 5.6-11.3 seconds = 28-56
onsets. At context=32, the model sees one full phrase per side. At
context=64, it sees two -- enough to capture phrase transitions
and the D/K patterns that span phrase boundaries.

The transformer's self-attention over 129 tokens at d_model=64 is
O(129^2 * 64) = ~1M ops per layer -- still cheap. The model gains
174K params (vs 172K at ctx32 -- only the position embedding grew).

### Predicted numbers

| Metric | 024c (ctx 32/32) | Predicted (024d, ctx 64/64) | Notes |
|---|---:|---:|---|
| type/accuracy | 0.726 | > 0.73 | ceiling lift from more context |
| type/mass_decisive | 0.197 | > 0.20 | more context -> more certainty |
| strength/best_f1_BIG | 0.719 | > 0.72 | more IOI context helps BIG |
| ar/type_accuracy_sym | 0.556 | > 0.56 | slight AR lift |
| ar/type_ngram_tvd_4 | 0.222 | < 0.22 | better distributional match |

## Success criteria

- **Must have:** type/accuracy > 0.726 (improves on 024c).
- **Must have:** no NaN, no crash, training completes.
- **Nice-to-have:** type/accuracy > 0.73 (ceiling lifted past all prior runs).
- **Nice-to-have:** ar/type_accuracy_sym > 0.56.
- **Fails if:** type/accuracy < 0.72 (longer context hurt -- padding noise or attention dilution).

## Changes from baseline

Baseline: [#024c](../024c-typing-ctx32/).

- `config/model.json` -- `past_context: 32 -> 64`,
  `future_context: 32 -> 64`. Window: 65 -> 129 tokens.
- `config/data.json` -- `past_context: 32 -> 64`,
  `future_context: 32 -> 64`. Sampler extracts wider windows.
- All other configs (loss, trainer, adapter) identical to 024c.
- Position embedding: 65 -> 129 entries (2,080 -> 4,128 params).
  Total model: ~174K params.

## Run config

- Run name: `exp_024d_typing_ctx64`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42),
  subsample 1.
- Command:
  ```bash
  source osu/taiko2/fix.sh
  PYTHONPATH=. osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_024d_typing_ctx64 \
      --config-dir osu/taiko2/experiments/024d-typing-ctx64/config \
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
