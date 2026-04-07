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

53 self data points, 35 volunteer data points.

### Expert (self) — significant metrics:

| Metric | r | p | Meaning |
|---|---|---|---|
| **gap_cv** | **+0.384** | **0.005** | Proportional rhythmic variety |
| **gap_std** | **+0.281** | **0.042** | Absolute gap variation |

### Volunteers — significant metrics:

| Metric | r | p | Meaning |
|---|---|---|---|
| **dominant_gap_pct** | **-0.420** | **0.012** | Don't repeat one gap too much |
| **gap_entropy** | **+0.370** | **0.029** | Variety of gap types |
| **max_metro_streak** | **-0.351** | **0.039** | Don't metronome too long |

### Key differences:

| Property | Expert signal | Volunteer signal |
|---|---|---|
| Top metric | gap_cv (+0.38) | dominant_gap_pct (-0.42) |
| What it means | "Are rhythms proportionally interesting?" | "Is this boring/repetitive?" |
| Audio alignment | Slightly negative (dislikes transcription) | Near zero |
| Significant count | 2 metrics | 3 metrics |

Expert values proportional variety (gap_cv normalizes by mean — a chart at any tempo can score well if it varies proportionally). Volunteers value anti-monotony (don't repeat, don't metronome, have entropy).

## Lesson

Expert and volunteer preferences are driven by related but distinct properties:

1. **Expert = proportional variety** (gap_cv). Measures rhythmic creativity relative to the tempo. A good slow chart and a good fast chart both have high gap_cv.

2. **Volunteers = anti-repetition** (dominant_gap_pct, entropy, streak length). Measures "is this boring?" — a simpler, more visceral quality.

3. **Both agree on direction** — more variety is better, more repetition is worse. But the emphasis differs: expert rewards creativity, volunteers punish boredom.

4. **Energy correlation is irrelevant or slightly negative for expert.** Following the audio faithfully (energy_ratio) does not predict preference for either group. Musical intelligence matters more than transcription accuracy.

[Exp 59-E](../experiment_59e/README.md) builds separate synthetic evaluators for self and volunteer preferences.
