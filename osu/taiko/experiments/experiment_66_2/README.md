# Experiment 66-2 — Bidirectional Corruption Evaluator

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide.

## Hypothesis

Exp 66-1b showed the unidirectional corruption evaluator has a **regularity bias** — it scores metronomic charts higher because the only "bad" direction it learned was randomness. Real chart quality is an inverted U: too regular (metronomic) AND too random (noise) are both bad.

Fix: train with **bidirectional corruption** — both random corruption (from 66-1) and metronomic corruption (new). The model must learn that CLEAN charts live in the sweet spot between two extremes.

## Architecture

Same model as 66-1 (ChartQualityEvaluator, 5.2M params). Only the corruption pipeline and pair construction change.

## Corruption levels (9 total)

### Random corruption (from 66-1)

| Level | Jitter (per) | Jitter (all) | Insert center | Delete | Insert offset |
|---|---|---|---|---|---|
| LIGHT_RAND | ±10ms | ±10ms | 1% | 1% | 1% |
| MED_RAND | ±30ms | ±30ms | 5% | 5% | 5% |
| HIGH_RAND | ±100ms | ±250ms | 25% | 15% | 10% |
| PURE_RAND | — | — | — | — | — (fully random gaps from global dist) |

### Metronomic corruption (NEW)

| Level | Grid snap | Const-gap fill | Pattern loop | Density flatten | Ratio snap |
|---|---|---|---|---|---|
| LIGHT_METRO | 10% of gaps | 5% of 5s segments | — | — | 10% unusual ratios |
| MED_METRO | 30% of gaps | 15% of segments | 10% of duration (4-8 gap tiles) | 30% toward mean | — |
| HIGH_METRO | 60% of gaps | 40% of segments | 30% of duration (2-4 gap tiles) | 60% toward mean | All ratios to {1:1, 2:1, 1:2, 4:1, 1:4} |
| PURE_METRO | 100% | 100% | — | — | — (constant median gap, or alternating top-2 gaps, or quantize to 1/4 grid) |

## Pair structure

```
CLEAN beats everything (margin scales with severity):
  CLEAN > LIGHT_RAND (margin 1)
  CLEAN > MED_RAND (margin 2)
  CLEAN > HIGH_RAND (margin 3)
  CLEAN > PURE_RAND (margin 4)
  CLEAN > LIGHT_METRO (margin 1)
  CLEAN > MED_METRO (margin 2)
  CLEAN > HIGH_METRO (margin 3)
  CLEAN > PURE_METRO (margin 4)

Within-type ordering:
  LIGHT_RAND > MED_RAND > HIGH_RAND > PURE_RAND (margins 1-3)
  LIGHT_METRO > MED_METRO > HIGH_METRO > PURE_METRO (margins 1-3)

Cross-type ties (same severity):
  LIGHT_RAND ≈ LIGHT_METRO (tie)
  MED_RAND ≈ MED_METRO (tie)
  HIGH_RAND ≈ HIGH_METRO (tie)
  PURE_RAND ≈ PURE_METRO (tie)
```

### Tie loss

For tie pairs: `loss = (score_a - score_b)^2` — pushes scores together. This teaches the model that both corruption directions are equally bad at the same severity.

### Pair sampling per batch

| Source | Proportion |
|---|---|
| CLEAN vs random corruption | 25% |
| CLEAN vs metro corruption | 25% |
| Within-type random pairs | 10% |
| Within-type metro pairs | 10% |
| Cross-type ties | 15% |
| Cross-set rating pairs | 15% |

## Training

| Phase | Epochs | Data | LR |
|---|---|---|---|
| Phase 1: bidirectional corruption | 20 | 85% corruption + ties, 15% rating | 3e-4 |

No separate phase 2 — rating pairs mixed in from the start at low proportion. The 66-1 experiment showed phase 2 fine-tuning didn't help, so we integrate the signal early instead.

### Launch

```bash
python classifier_train_v2.py taiko_v2 --run-name eval_experiment_66_2
```

## Success criteria

- Corruption accuracy: >90% on easy pairs, >70% on hard pairs (same as 66-1)
- **Metro correlations flip:** metro_streak should correlate NEGATIVELY with gen_score
- **Generator ranking correct:** exp 62 > exp 58 > exp 45 > exp 14
- Tie pairs: score difference near zero for same-severity cross-type pairs
- Rating pairwise accuracy: >55% (matching 66-1)

## Result

*(awaiting results)*
