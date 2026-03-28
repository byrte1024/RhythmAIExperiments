# Experiment 51 - Streak-Ratio Loss Weighting

## Hypothesis

The model metronomically continues patterns because the training data overwhelmingly rewards continuation. From the streak-ratio matrix analysis:

### Training data distribution

![Streak-Ratio Matrix](streak_ratio_matrix.png)

**Continuation dominates at every streak length:**

| Streak | Continue (1/1) | Break (any other) | Continue rate |
|---|---|---|---|
| 1 | 1,697,473 | 1,095,901 | 60.8% |
| 2 | 566,605 | 596,697 | 48.7% |
| 3 | 388,832 | 314,523 | 55.3% |
| 5 | 134,904 | 61,688 | 68.6% |
| 8 | 31,918 | 9,199 | 77.6% |
| 10 | 43,895 | 7,352 | 85.7% |
| 16 | 45,859 | 2,733 | 94.4% |

**The key cells the model almost never sees:**
- Streak 8 + ratio 2/1 (double-time break): 0.10% of data (~5K samples in 5M)
- Streak 8 + ratio 1/2 (half-time break): 0.06%
- Streak 16 + any break: 0.02-0.04%

**The cells that dominate:**
- Streak 1 + ratio 1/1: 34% of all data
- Streak 2 + ratio 1/1: 11.3%

The model is right to continue because continuation IS correct most of the time. But breaking at the right moment is what makes good charts — and the model barely sees those moments.

### Approach

**Per-sample loss weighting based on streak-ratio cell frequency.** Instead of changing the sampler (which changes what the model sees), we change how much the model cares about each error.

For each training sample:
1. Compute the context streak length and target/gap ratio
2. Look up the cell count in the precomputed matrix
3. Weight the loss: `weight = min(cap, (max_count / cell_count) ^ power)`

With power=0.3 and cap=50:

| Cell | Count | Weight |
|---|---|---|
| streak 1, ratio 1/1 | 1,697,473 | 1.0x |
| streak 2, ratio 1/2 | ~217K | ~3x |
| streak 5, ratio 2/1 | ~32K | ~6x |
| streak 8, ratio 2/1 | ~5K | ~18x |
| streak 16, ratio 2/1 | ~1K | ~40x |
| edge cases (<100) | capped | 50x |

Getting a streak-16 break wrong costs 40x more than getting a streak-1 continuation wrong.

### Key principles
- Model still sees the **natural distribution** — it knows continuation is common
- But the **gradient from rare breaks is amplified** to match their importance
- Works alongside existing balanced sampler (they stack)
- Cap at 50x prevents tiny edge bins from exploding gradients

### Predictions
- **Per-sample metrics will likely be worse.** The model is penalized more for rare-cell errors, diverting capacity from the common cells that drive HIT rate.
- **AR generation might be much better.** The model should break patterns more willingly, reducing metronome behavior.
- **Metronome benchmark should improve significantly.** The model learns to distrust long streaks.

### Architecture
Same as exp 45 (EventEmbeddingDetector, gap ratios, tight density jitter). Only change is the per-sample loss multiplier.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_51 --model-type event_embed --streak-loss --streak-power 0.3 --streak-cap 50 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
