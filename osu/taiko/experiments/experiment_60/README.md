# Experiment 60 - DDC Onset Detector Comparison

## Hypothesis

Dance Dance Convolution (DDC, Donahue et al. 2017) is the most cited baseline for rhythm game chart generation. Its onset detection CNN is available as a pip-installable PyTorch package (`ddc_onset`). While DDC targets DDR/StepMania (4-panel), the onset detection task is the same: find musically-relevant onset positions from audio.

**Compare DDC's onset detector against our models on the same val set songs.** DDC is a pure audio onset detector (no context, no density conditioning) — most comparable to our Stage 1 proposer (exp 58) or our no_events benchmark baseline.

### Method

1. Run DDC onset detector on the same 30 val songs used in exp 59-G/H
2. Sweep threshold (0.1-0.5) to find optimal operating point
3. Compare DDC predictions against ground truth osu!taiko events
4. Compare against our models' AR output from exp 59-H (exp44, exp53, exp50b, exp51, exp55, exp58)

### DDC Onset Detector

- **Architecture**: CNN on audio spectrograms, conditioned on difficulty level
- **Output**: Onset salience function at 100fps (10ms resolution) 
- **Our resolution**: ~5ms (200fps equivalent)
- **Framework**: PyTorch (ported from TF)
- **Difficulty**: CHALLENGE (highest, closest to our high-density charts)

### Launch

```bash
cd osu/taiko
python experiments/experiment_60/compare_ddc.py
```

## Result

30 val songs, 9 thresholds swept.

### DDC threshold sweep:

| Threshold | Close% | Hall% | d_ratio | err_med | p/g |
|---|---|---|---|---|---|
| 0.05 | 98.7% | 28.9% | 2.42 | 25ms | 2.55x |
| 0.20 | 88.2% | 22.8% | 1.37 | 26ms | 1.44x |
| 0.40 | 76.9% | 18.4% | 1.02 | 32ms | 1.06x |
| 0.50 | 70.4% | 16.8% | 0.89 | 36ms | 0.93x |

### DDC @ matched density (thr=0.40) vs our best:

| Model | Close% | Hall% | d_ratio | err_med |
|---|---|---|---|---|
| DDC @ 0.40 | 76.9% | 18.4% | 1.02 | 32ms |
| **exp58** | **75.9%** | **15.6%** | **0.92** | **8ms** |

At matched density, close rates are comparable (~76%). But exp58 has lower hallucination (-2.8pp) and **4x better timing precision** (8ms vs 32ms).

### Full comparison (song_density regime):

| Model | Close% | Far% | Hall% | d_ratio | err_med |
|---|---|---|---|---|---|
| DDC @ 0.40 | 76.9% | 15.9% | 18.4% | 1.02 | 32ms |
| exp58 | 75.9% | 16.6% | 15.6% | 0.92 | 8ms |
| exp53 | 73.4% | 19.0% | 17.9% | 0.91 | 8ms |
| exp44 | 71.1% | 20.6% | 14.7% | 0.84 | 9ms |
| exp55 | 69.7% | 21.7% | 15.6% | 0.82 | 14ms |

## Lesson

1. **DDC is a strong onset detector but a weak chart generator.** At low thresholds it catches 98.7% of events — but by predicting 2.5x too many. It finds every audio transient without understanding which ones a chart should use.

2. **At matched density, our models are comparable on catch rate but far superior on timing.** exp58's 8ms median error vs DDC's 32ms reflects both finer resolution (5ms vs 10ms) and our model's ability to predict exact onset positions rather than just flagging peaks.

3. **DDC's hallucination floor (~17%) is structural.** Audio contains many valid onsets that aren't in the chart. DDC detects them all; a good chart generator should select from them. This is exactly what our Stage 1 proposer (exp 58) does — but with context awareness.

4. **DDC validates our Stage 1 concept.** Both are pure audio onset detectors. DDC uses a CNN at 10ms resolution; our proposer uses a transformer at 5ms resolution with the same conv stem as Stage 2. The similar performance confirms audio onset detection is well-solved — the differentiation comes from context-informed selection (Stage 2).

5. **Resolution matters for timing precision.** DDC's 10ms frames impose a 10ms quantization floor. Our 5ms frames allow consistently sub-10ms median errors. For rhythm games where timing is felt at the millisecond level, this matters.

### All-difficulty analysis:

DDC's difficulty parameter acts as density control. Best operating point per difficulty (all at threshold 0.05):

| Difficulty | Close% | Hall% | d_ratio | err_med |
|---|---|---|---|---|
| BEGINNER | 36.3% | 13.5% | 0.43 | 392ms |
| EASY | 67.3% | 18.4% | 0.86 | 38ms |
| **MEDIUM** | **79.0%** | **21.2%** | **1.08** | **29ms** |
| HARD | 94.3% | 27.1% | 1.70 | 25ms |
| CHALLENGE | 98.7% | 28.9% | 2.42 | 25ms |

**MEDIUM is the best DDC setting for osu!taiko** (d_ratio closest to 1.0). It achieves 79% close rate — slightly higher than exp58's 75.9% — but with 21% hallucination (vs 15.6%) and 29ms timing (vs 8ms).

DDC's difficulty conditioning is analogous to our FiLM density conditioning: both control output density. DDC uses discrete levels, we use continuous values. Both work — DDC's MEDIUM produces similar density to osu!taiko charts.

### DDC Oracle (density-matched per song):

Picks the difficulty+threshold whose output density is closest to GT density per song — analogous to perfect density conditioning.

| Model | Close% | Hall% | d_ratio | err_med |
|---|---|---|---|---|
| DDC Oracle | **77.1%** | 19.9% | **1.00** | 27ms |
| exp58 | 75.9% | **15.6%** | 0.92 | **8ms** |

Even with perfect density matching, DDC only edges exp58 by 1.2pp on close rate while losing on hallucination (-4.3pp) and timing (3.4x worse). Our models' advantage is in precision and selectivity, not raw onset detection.
