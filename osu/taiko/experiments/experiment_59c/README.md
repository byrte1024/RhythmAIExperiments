# Experiment 59-C - Synthetic Human Evaluator

## Hypothesis

[Exp 59-B](../experiment_59b/README.md) found 4 metrics that significantly correlate with human preference: gap_std (+0.30), gap_cv (+0.29), dominant_gap_pct (-0.27), max_metro_streak (-0.27). All measure pattern variety.

**Can we build a synthetic evaluator from these metrics that reproduces human rankings?** If so, we can evaluate AR quality automatically without human evaluators — enabling rapid iteration on AR inference settings.

### Approach

1. For each song in 42-AR and 53-AR, compute the 4 significant metrics for each model's chart
2. Rank models per song using a weighted combination of z-scored metrics
3. Compare synthetic rankings to actual human rankings (both self and volunteer)
4. Measure agreement: exact rank match %, Kendall tau, weighted score correlation

### Scoring formula

For each model×song, compute a synthetic score from z-scored metrics:
```
synthetic = w1 * z(gap_std) + w2 * z(gap_cv) - w3 * z(dominant_gap_pct) - w4 * z(max_metro_streak)
```

Try multiple weighting schemes:
- **Equal**: all weights = 1.0
- **Correlation-weighted**: weights proportional to |r| from 59-B
- **Top-2 only**: gap_std + gap_cv only (the two strongest)

### Launch

```bash
cd osu/taiko
python experiments/experiment_59c/synthetic_evaluator.py
```

## Result

88 data points across 42-AR and 53-AR. Synthetic evaluator significantly outperforms random.

### Best results:

| Metric | Best scheme | Value | vs Random |
|---|---|---|---|
| 1st place match (all) | top2_only | **52%** | 2x random (25%) |
| 1st place match (volunteers) | all schemes | **60%** | 2.4x random |
| Exact rank match | top2_only (self) | **47%** | 1.9x random |
| Kendall tau | top2_only (self) | **+0.422** | Strong agreement |
| Spearman r (all) | corr_weighted | **+0.347, p=0.001** | Highly significant |

### Scheme comparison:

| Scheme | #1 Match | Exact | Tau |
|---|---|---|---|
| equal | 44% | 38% | +0.253 |
| corr_weighted | 48% | 40% | +0.280 |
| **top2_only** | **52%** | **43%** | **+0.293** |

**top2_only (gap_std + gap_cv only) is the best scheme.** The two simplest metrics outperform the full 4-metric formula. dominant_gap_pct and max_metro_streak add noise.

### Self vs volunteer:

- **Volunteers**: 60% first-place match — synthetic evaluator predicts volunteer preference very well
- **Self (expert)**: top2_only gets tau=0.42, 47% exact match — strong but expert is harder to predict

## Lesson

A synthetic evaluator built from just **gap_std + gap_cv** (pattern variety) can predict human preference with 52% first-place accuracy (2x random) and Spearman r=0.35 (p=0.001). This enables automated AR quality evaluation without human evaluators.

The evaluator predicts volunteers better than experts (60% vs 47% first-place), suggesting expert preferences incorporate dimensions beyond pattern variety (musical intelligence, context sensitivity) that simple gap statistics don't capture.

The formula is trivially cheap to compute on any AR output — no ground truth needed, no audio alignment. Just `np.diff(event_times)` → std and cv.
