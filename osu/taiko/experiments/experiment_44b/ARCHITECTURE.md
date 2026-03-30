# Experiment 44-B — Diagnostic Analysis Specification

## Type

Analysis / diagnostic experiment. No model training. Analyzes metronome pattern prevalence in the training dataset.

## Purpose

Determine whether the model's metronome collapse during AR generation is a learned behavior from data rather than a failure mode. Taiko charts are fundamentally rhythmic — evenly-spaced hits are the norm. This analysis quantifies what percentage of training samples ask the model to continue an existing same-gap streak.

## Data Analyzed

Full training sample index (subsample=1, 5.84M samples, 5.82M non-STOP). Two levels of analysis:

### Dataset-level analysis
- Count all same-gap streaks across all 10,048 charts (5% gap tolerance)
- Report: total streaks, mean length, max length, gap distribution

### Training-sample-level analysis
For each training sample:
1. Compute past gaps from context events
2. Find the length of the same-gap streak ending at the most recent gap (5% tolerance)
3. Check if the target gap continues the streak
4. Report: for each streak length K in context, what percentage of samples continue it

## Scripts

| Script | Location | Purpose |
|---|---|---|
| analyze_metronome_data.py | `osu/taiko/analyze_metronome_data.py` | Dataset-level streak analysis across all charts |
| analyze_metronome_targets.py | `osu/taiko/analyze_metronome_targets.py` | Training-sample-level streak continuation analysis |

## Key Findings

### Dataset-level
- 90.9% of all events are part of same-gap streaks (5% tolerance)
- 10,046 / 10,048 charts have streaks
- 1.6M total streaks, mean length 2.9 gaps, max streak 3,839 consecutive events
- Gap distribution peaks at ~800 BPM (75ms) and ~600 BPM (100ms) — corresponding to 1/4 note streams at 150-200 BPM

### Training-sample-level continuation rates

| Context streak length | % of samples | Target continues streak |
|---|---|---|
| 1 (any gap) | 100% | 43.9% |
| 2 | 43.9% | 46.6% |
| 3 | 20.4% | 55.1% |
| 5 | 6.6% | 70.8% |
| 8 | 2.7% | 82.9% |
| 10 | 1.9% | 86.7% |
| 16 | 1.0% | 91.5% |
| 32 | 0.3% | 96.7% |

- **43.9% of all training samples ask the model to repeat the previous gap** — metronome is the statistically correct response for nearly half of all samples
- Once context shows 8+ same-gap events, continuing the pattern is correct 83% of the time
- At 32+, it is 97%
- The ~17% break points at streak 8 are where chart quality lives — rhythm changes, section transitions, pattern variations
- The positive feedback loop: the longer a streak exists in context, the more likely the correct answer is "continue," which in AR generation reinforces itself further
