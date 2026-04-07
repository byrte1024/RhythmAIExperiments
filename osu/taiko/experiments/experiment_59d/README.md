# Experiment 59-D - Self vs Volunteer Metric Correlations

## Hypothesis

[Exp 59-C](../experiment_59c/README.md) showed the synthetic evaluator predicts volunteers better (60% #1 match) than expert self-rankings (47%). This suggests expert and volunteer preferences may correlate with *different* metrics.

**Do self-rankings and volunteer rankings correlate with different chart properties?** Rerun the 59 + 59-B analysis separately for self-only and volunteer-only votes to find metrics that predict each group independently.

### Possible findings

- **Volunteers** may weight pattern variety even more heavily — they respond to "feels good" which is gap variety
- **Expert (self)** may correlate with more nuanced metrics like pattern_change_rate, energy_gap_corr, or gap_autocorr — things that measure musical responsiveness rather than just variety

### Launch

```bash
cd osu/taiko
python experiments/experiment_59d/analyze_split.py
```

## Result

*Pending*

## Lesson

*Pending*
