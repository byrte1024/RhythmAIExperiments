# Experiment 44-B - Metronome Pattern Analysis in Training Data

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


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

### Dataset-level (analyze_metronome_data.py)

- 90.9% of all events are in same-gap streaks (5% tolerance)
- 10,046 / 10,048 charts have streaks
- 1.6M total streaks, mean length 2.9 gaps, max 3,839
- Gap distribution peaks at ~800 BPM (75ms) and ~600 BPM (100ms) — 1/4 note streams at 150-200 BPM

### Training sample level (analyze_metronome_targets.py)

5.84M samples (subsample=1), 5.82M non-STOP.

| Context streak | % of samples | Target continues |
|---|---|---|
| 1 (any gap) | 100% | **43.9%** |
| 2 | 43.9% | **46.6%** |
| 3 | 20.4% | **55.1%** |
| 5 | 6.6% | **70.8%** |
| 8 | 2.7% | **82.9%** |
| 10 | 1.9% | **86.7%** |
| 16 | 1.0% | **91.5%** |
| 32 | 0.3% | **96.7%** |

**43.9% of all training samples ask the model to repeat the previous gap.** Once context shows 8+ same-gap events, continuing the pattern is correct 83% of the time. At 32+ it's 97%.

## Lesson

- **Metronome behavior is the statistically correct response** for nearly half of all samples. The model isn't broken — it's learning the dominant pattern in the data.
- **The escalation is the key insight**: the longer a streak exists in context, the more likely the correct answer is "continue." This creates a positive feedback loop in AR — once the model starts repeating, the context reinforces it further.
- **The ~17% break points at streak 8** are where chart quality lives. These are the rhythm changes, section transitions, and pattern variations that make a chart interesting. But the model has no special incentive to nail these over the 83% continuation cases.
- **Augmentation alone can't fix this.** The issue is fundamental to the loss function — CE loss treats a wrong continuation the same as a wrong break. The model needs asymmetric incentive to break patterns when appropriate.
- **Possible directions:**
  - Upweight loss on samples where the target BREAKS a streak (transition samples)
  - Adversarial metronome training: inject metronome context, penalize metronome continuation
  - Two-stage prediction: first predict "continue or break", then predict the gap
