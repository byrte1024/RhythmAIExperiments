# Experiment 59-E - Split Synthetic Evaluators

## Hypothesis

[Exp 59-D](../experiment_59d/README.md) found that expert and volunteers correlate with different metrics:
- **Expert**: gap_cv (+0.38), gap_std (+0.28)
- **Volunteers**: dominant_gap_pct (-0.42), gap_entropy (+0.37), max_metro_streak (-0.35)

[Exp 59-C](../experiment_59c/README.md) used the same formula for all evaluator types. Building **separate evaluators** tuned to each group should improve accuracy.

### Formulas

**Expert evaluator** (proportional variety):
```
score = z(gap_cv) * 0.384 + z(gap_std) * 0.281
```

**Volunteer evaluator** (anti-repetition):
```
score = -z(dominant_gap_pct) * 0.420 + z(gap_entropy) * 0.370 - z(max_metro_streak) * 0.351
```

**Combined evaluator** (from 59-B):
```
score = z(gap_std) * 0.299 + z(gap_cv) * 0.289 - z(dominant_gap_pct) * 0.272 - z(max_metro_streak) * 0.269
```

### Launch

```bash
cd osu/taiko
python experiments/experiment_59e/split_evaluator.py
```

## Result

*Pending*

## Lesson

*Pending*
