# Experiment 44-B - Metronome Pattern Analysis in Training Data

## Hypothesis

The model's metronome collapse during AR generation may not be purely a failure mode — it could be learning the dominant pattern in the data. Taiko charts are fundamentally rhythmic: evenly-spaced hits are the norm, with brief transitions between patterns.

From dataset-level analysis (analyze_metronome_data.py):
- **90.9% of all events** are part of same-gap streaks (5% tolerance)
- 10,046 / 10,048 charts have metronome streaks
- 1.6M total streaks, mean length 2.9 gaps
- Longest streak: 3,839 consecutive same-gap events (~100ms gap, 600 BPM)
- Gap distribution peaks at ~800 BPM (75ms, 2,842 long streaks) and ~600 BPM (100ms, 1,572 long streaks) — corresponding to 1/4 note streams at 150-200 BPM

**Question:** In the actual training set (subsample=1), what percentage of samples have a target that continues an existing same-gap streak of length >= 8? If the answer is ~50%+, then metronome behavior is the statistically correct response for most samples, and the real challenge is teaching the model *when to break* the pattern, not *when to follow* it.

### Method

Load the full training sample index (same as OnsetDataset with subsample=1). For each sample:
1. Compute past gaps from context events
2. Find the length of the same-gap streak ending at the most recent gap (5% tolerance)
3. Check if the target gap continues the streak

Report: for each streak length K in context, what % of samples continue it?

## Result

*Pending — running analyze_metronome_targets.py*

## Lesson

*Pending*
