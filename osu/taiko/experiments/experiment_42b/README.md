# Experiment 42-B - Pure Hard CE (Frame Tolerance Only)

## Hypothesis

Exp 42's entropy profile is identical to 35-C despite +1.6pp HIT. The proportional soft target (good_pct=3%) creates wider distributions for distant targets — at target=200, 12 bins get full credit. This trains the model to be less confident at distance.

**Test: pure hard CE with ±3 frame tolerance.** No soft trapezoid at all (`hard_alpha=1.0`). The model gets credit only for hitting within 3 bins of the exact target, regardless of target distance. This is maximally sharpening — equal precision demanded at all distances.

Expected: worse HIT rate overall (hard CE is less forgiving), but **equally bad across all distances** — the entropy-distance correlation should flatten. The model will be less accurate but uniformly less accurate, rather than confident at short range and uncertain at long range. If the entropy-distance gradient disappears, the proportional soft targets were the cause.

### Changes from exp 42

- **hard_alpha: 0.5 → 1.0** (pure hard CE, no soft trapezoid)
- **frame_tolerance: 2 → 3** (±3 bins acceptable)
- Short run (1-2 epochs) — diagnostic only

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_42b --model-type event_embed --hard-alpha 1.0 --frame-tolerance 3 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Entropy slashed by 45% — the soft targets WERE the confidence bottleneck. But accuracy drops and skip rate increases.**

### Entropy comparison: 42-B (hard CE) vs 42 (soft targets)

| Metric | Exp 42 (soft) | **42-B (hard CE)** | Change |
|--------|--------------|-------------------|--------|
| Mean entropy | 2.390 | **1.320** | **-45%** |
| Mean confidence | 0.355 | **0.535** | **+51%** |
| target_dist correlation | +0.582 | **+0.363** | **Flatter** |
| density correlation | -0.532 | **-0.286** | Less dependent |
| HIT rate | 73.2% | 68.7% | -4.5pp |
| Skip-1 rate | 11.1% | **13.2%** | +2.1pp worse |

### Entropy by distance

| Range | Exp 42 | **42-B** | Drop |
|-------|--------|---------|------|
| 0-15 | 1.899 | **1.200** | -0.70 |
| 15-30 | 2.017 | **1.157** | -0.86 |
| 30-60 | 2.552 | **1.391** | -1.16 |
| 60-100 | 3.017 | **1.627** | -1.39 |
| 100-200 | 3.377 | **1.787** | -1.59 |
| 200-500 | 3.691 | **1.942** | -1.75 |

Biggest improvements at distant predictions — exactly where it was needed most.

### Skip analysis

| Skip | Exp 42 | **42-B** |
|------|--------|---------|
| 0 (HIT) | 50,337 (67.9%) | 48,197 (65.1%) |
| 1 | 8,242 (11.1%) | **9,779 (13.2%)** |
| Under | 13,781 (18.6%) | 13,876 (18.7%) |

Skip-0 HIT: 94.5% → 93.0%. The model is more confident but less accurate — confidently wrong more often.

## Lesson

- **The proportional soft targets were the primary cause of the entropy-distance correlation.** Hard CE with frame tolerance produces dramatically flatter, more confident predictions.
- **But confidence ≠ accuracy.** Hard CE loses 4.5pp HIT because the soft targets' "nearby credit" helps the model learn the right region even when it can't hit the exact bin. Pure hard CE is too harsh — the model is penalized equally for being 1 bin off and 100 bins off.
- **The ideal is between hard_alpha=0.5 and 1.0.** Soft targets for learning the right region, harder weighting for confidence. Something like hard_alpha=0.7-0.8, or tighter soft targets (good_pct 3% → 1.5%).
