# Experiment 59-B - Within-Song Metric Normalization

## Hypothesis

[Exp 59](../experiment_59/README.md) found zero significant correlations between raw chart metrics and human preference. The problem: different songs have different natural gap distributions, densities, and audio energy — comparing raw metrics across songs adds confounding noise.

**Fix: normalize within each song.** For each song, compute the z-score of each metric across the models that charted it. Then correlate z-scored metrics with the score each model received on that song. This removes the song-level confound and isolates "which model is relatively better on THIS song."

Additionally, compute **pairwise deltas**: for each pair of models on the same song, compute metric difference and score difference. This gives more data points and directly asks: "when model A scores higher than model B on a song, what metrics are higher?"

### Launch

```bash
cd osu/taiko
python experiments/experiment_59b/analyze_normalized.py
```

## Result

88 z-scored data points, 114 pairwise deltas.

### Z-scored correlations (4 significant at p<0.05):

| Metric | Spearman r | p-value | Direction |
|---|---|---|---|
| **gap_std** | **+0.299** | **0.005** | More varied gaps → preferred |
| **gap_cv** | **+0.289** | **0.006** | More relative variation → preferred |
| **dominant_gap_pct** | **-0.272** | **0.010** | Less gap concentration → preferred |
| **max_metro_streak** | **-0.269** | **0.011** | Shorter metronome streaks → preferred |

### Near-significant (p<0.10):

| Metric | r | p |
|---|---|---|
| density_std | -0.198 | 0.065 |
| gap_entropy | +0.194 | 0.071 |
| density | -0.182 | 0.091 |

### Pairwise delta correlations: **no significant results.** All r < 0.11. The z-score normalization is the method that works; raw deltas are too noisy.

### Audio alignment: **no signal.** Energy ratio, onset energy, peak distance, energy-gap correlation all near zero (r < 0.08). Mel correlation does not predict human preference once the song confound is removed.

## Lesson

**Pattern variety is the strongest predictor of human preference.** All 4 significant metrics measure the same underlying property: how spread out the gap distribution is. Models that produce varied rhythmic patterns (high gap_std, high gap_cv, low dominant_gap_pct, short metronome streaks) are ranked higher by humans.

This confirms the hypothesis from exp 59 and explains the 53-AR results: exp44 (lowest metronome score 58.4%) beat exp53 (highest metronome score 64.1%) because exp44 produced more varied patterns, not because of any benchmark metric.

**Practical implications:**
1. gap_std and gap_cv can be computed on AR output without ground truth — usable as a live quality metric during inference development
2. Optimizing for pattern variety (not just per-sample accuracy) should be a training objective
3. Audio alignment metrics are irrelevant to human preference — the model should focus on rhythmic interest, not transient matching

**Normalization is essential.** Raw metrics (exp 59) showed zero correlation. Within-song z-scoring revealed strong signals (r~0.3, p<0.01). Different songs have inherently different gap distributions; only relative comparison within the same song is meaningful.
