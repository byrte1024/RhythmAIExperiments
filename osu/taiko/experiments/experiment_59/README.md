# Experiment 59 - AR Quality Metric Discovery

## Hypothesis

Per-sample metrics (HIT%, metronome benchmark) are poor predictors of human-perceived AR quality. In both 42-AR and 53-AR, models with higher per-sample accuracy didn't win human rankings.

**Goal: discover metrics that actually correlate with human preference.** Analyze all AR-generated charts from 42-AR (3 models × 10 songs) and 53-AR (4 models × 10 songs) to find which computed properties predict human rankings.

### Expected findings

- **Gap histogram spread**: models ranked higher should have more varied gap distributions. Lower-ranked models should have gaps concentrated in a specific area (metronomic).
- **Mel energy correlation**: better-ranked models should place onsets at positions with higher audio energy (transients, drum hits).
- **Pattern diversity**: unique patterns, gap entropy, and pattern change rate should correlate positively with human preference.

### Two analysis levels

1. **Per-model aggregate** (N=7 models across both rounds): total score vs aggregate metrics
2. **Per-song-model** (N=~70 model×song pairs): per-song score vs per-song metrics. More data points, stronger statistical power.

### Candidate metrics

**Gap distribution:**
- Gap entropy (Shannon entropy of discretized gap distribution)
- Dominant gap % (most common gap's share)
- Top-3 gap coverage
- Number of unique gaps (within 5% tolerance)
- Gap std, gap coefficient of variation (std/mean)
- Longest metronome streak

**Pattern dynamics:**
- Pattern change rate (how often the dominant gap changes in sliding windows)
- Gap autocorrelation (lag-1 correlation of consecutive gaps)
- Section-level consistency (within-window variance vs between-window variance)

**Audio alignment:**
- Mel energy at onset positions vs random positions
- Onset-to-transient alignment (distance from each onset to nearest energy peak)
- Energy change correlation (do gap changes happen when audio energy changes?)

**Density:**
- Events per second
- Density stability across windows

### Launch

```bash
cd osu/taiko
python experiments/experiment_59/analyze_metrics.py
```

## Result

88 data points (vote×model×song) across 42-AR and 53-AR. **No metric reaches statistical significance.**

### Top correlations (all non-significant):

| Metric | Spearman r | p-value |
|---|---|---|
| max_metro_streak | -0.154 | 0.15 |
| n_events | -0.145 | 0.18 |
| dominant_gap_pct | -0.136 | 0.21 |
| max_metro_streak_pct | -0.125 | 0.24 |
| density_std | -0.107 | 0.32 |
| gap_autocorr | +0.096 | 0.37 |

Directions are intuitive (metronome streaks bad, dominant gap concentration bad) but signal is buried in noise. Gap entropy, energy ratio, pattern change rate, and all audio alignment metrics showed near-zero correlation.

## Lesson

Raw chart metrics computed across different songs don't predict human preference. The per-song confound dominates — a slow pop song naturally has different gap statistics than a fast J-dance track. Comparing raw metrics across songs adds noise that drowns any real signal.

[Exp 59-B](../experiment_59b/README.md) addresses this by normalizing within each song: computing metric deltas between models on the same song and correlating with score deltas.
