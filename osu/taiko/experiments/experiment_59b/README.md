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

*Pending*

## Lesson

*Pending*
