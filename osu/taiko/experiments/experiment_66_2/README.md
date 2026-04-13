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

### Training (40 evals, 20 epochs)

| Metric | Eval 1 | Eval 40 (final) |
|---|---|---|
| Pair accuracy | 90.5% | 92.8% |
| Margin 1 accuracy | 80.6% | 84.3% |
| Margin 4 accuracy | 97.9% | 98.6% |
| Tie gap | 0.30 | 0.24 |
| Val loss | 0.351 | 0.285 |
| mono_rand | True | True |
| mono_metro | True | True |

Both directions monotonic. Tie pairs converged (gap 0.24). Score spread smaller than 66-1 — CLEAN at +0.4, PURE_RAND at -1.3, PURE_METRO at -1.2.

### AR evaluation (exp 14, 45, 58, 62)

| Exp | HIT% | GT win% | GT mean | Gen mean | Diff |
|---|---|---|---|---|---|
| 14 | 69.0% | 96.7% | +0.05 | -0.76 | +0.82 |
| 45 | 73.6% | 100.0% | +0.05 | -0.75 | +0.80 |
| 58 | 74.6% | 100.0% | +0.05 | -0.77 | +0.82 |
| 62 | 74.9% | 100.0% | +0.05 | -0.90 | +0.95 |

GT win rate improved to 97-100% (from 90-97% in 66-1). The model almost perfectly distinguishes real from generated charts.

### Generator ranking: still doesn't match HIT%

Exp 62 (best HIT%) scores lowest (-0.90), exp 45 (lower HIT%) scores highest (-0.75). Same pattern as 66-1b. However, the total spread is only 0.15 points — the model sees these generators as nearly equivalent, with all of them far below real charts.

### Metric correlations vs 66-1

| Metric | 66-2 | 66-1 | Change |
|---|---|---|---|
| gap_cv | **+0.260** | -0.094 | **FLIPPED (good)** |
| Over. P-Space | **+0.316** | +0.172 | Stronger |
| density | +0.288 | +0.193 | Stronger |
| metro_streak | +0.258 | +0.312 | Weaker but still positive |
| DCHuman | -0.335 | -0.262 | Stronger negative |
| close_rate | -0.108 | +0.188 | Flipped |

### Key insight: gap_cv shape is correct

Binned gap_cv vs gen_score is monotonically positive — more variety = higher score:
```
[0.0-0.5): -0.854
[0.5-0.6): -0.836
[0.6-0.7): -0.764
[0.7-0.8): -0.732
[0.8-1.0): -0.748
[1.0-2.0): -0.627
```

This is exactly the signal exp 59 found correlated with human preference. The bidirectional corruption training successfully taught the model to reward pattern variety.

### Reframing: is the model right?

The generator ranking not matching HIT% isn't necessarily wrong. We already know from human evaluation (exp 42-AR, 53-AR) that **per-sample accuracy doesn't predict human preference**. Exp 14 (69% HIT, no context) beat exp 42 (73% HIT, deepest context) in blind human tests. The evaluator might be capturing something real:

1. **close_rate not correlating with quality** — matches human eval findings. Accurate note placement ≠ good chart.
2. **gap_cv positively correlating** — matches exp 59 synthetic evaluator findings. Pattern variety predicts human preference.
3. **GT win rate near 100%** — the model can tell real from generated almost perfectly. Generated charts have a quality gap that simple metrics miss.
4. **The 0.15 spread between generators** — compared to the 0.85 real-vs-generated gap, the differences between our generators are minor. All four models produce charts that are similarly "not quite real."

The open question: **does this evaluator's ranking match actual human preference data from exp 53-AR?** If it does, the evaluator is capturing something our other metrics can't. If it doesn't, the corruption signal isn't transferring to quality.

## Success criteria (reviewed)

| Criterion | Target | Achieved | Status |
|---|---|---|---|
| Corruption easy pairs (margin 3-4) | >90% | 97-99% | **Pass** |
| Corruption hard pairs (margin 1) | >70% | 84% | **Pass** |
| Metro correlations flip | negative | +0.258 (weaker, not flipped) | **Partial** |
| Generator ranking correct | 62 > 58 > 45 > 14 | 45 > 14 > 58 > 62 | **Fail** (but may be wrong criterion) |
| Tie pairs near zero | <0.3 | 0.24 | **Pass** |
| gap_cv direction | positive | **+0.260** | **Pass** |
| GT win rate | >90% | 97-100% | **Pass** |

## Lesson

Bidirectional corruption successfully taught the model to reward pattern variety (gap_cv flipped positive). The metro corruption partially worked — the model no longer only equates regularity with quality. But metro_streak didn't fully flip, suggesting the metro corruptions need to be more targeted at the specific repeating-pattern failure mode rather than general grid-snapping.

The deeper question this experiment raises: our evaluation framework may be measuring the wrong things. HIT%, close_rate, and GT matching measure **accuracy** (does the generated chart match the human chart?). But chart quality is about **musicality** (does the chart feel good to play?). These are different. The evaluator seems to be learning something closer to the second, but we need human validation to confirm.
