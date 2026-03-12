# Experiment 26 - Heavy Audio Augmentation

## Hypothesis

Exp 25 showed the unified fusion architecture matches exp 14 (~68.6% HIT) but overfits from E2 onward (val loss 2.623 → 2.665 while train loss kept falling). Context contribution shrank from 6.8% to 2.3% over 5 epochs. The model memorized audio→onset mappings without learning to leverage gap tokens.

**Audio augmentation as regularizer.** Exp 25 had light audio augmentation (inherited from exp 14) and deliberately reduced context augmentation. The model overfit to audio patterns. Heavier audio augmentation serves two purposes:
1. **Directly fights overfitting** - more variation in audio input makes memorization harder
2. **May indirectly encourage context reliance** - if audio signal is noisier/varied during training, gap tokens become a more stable anchor the model can lean on

### Changes from exp 25

**Architecture: identical.** Same unified fusion model (~19M params). This experiment isolates the effect of augmentation.

**Audio augmentation changes:**

| Augmentation | Exp 25 | Exp 26 | Notes |
|---|---|---|---|
| Gain jitter | ±2dB @ 30% | ±3dB @ 50% | Wider range, more frequent |
| Noise injection | σ=0.1-0.3 @ 15% | σ=0.1-0.4 @ 30% | Stronger noise, 2x frequency |
| Freq jitter | *none* | ±1-5 bins @ 30% | **New** - roll mel bands up/down, zero-fill edges |
| SpecAugment freq mask | 1-8 bands, 1 mask @ 20% | 1-15 bands, 1-2 masks @ 40% | Wider masks, allow multiple |
| SpecAugment time mask | 1-30 frames, 1 mask @ 20% | 1-50 frames, 1-2 masks @ 40% | Wider masks, allow multiple |
| Temporal corruption | *none* | 10-frame chunk shuffle @ 2% | **New** - destroys local time structure, forces context reliance |
| Fade in/out | 10% each | 10% each | Unchanged |

**Context augmentation: unchanged** from exp 25 (event jitter, deletion, insertion, dropout, truncation, cond jitter).

**Training: same hyperparams** - lr=3e-4, batch=64, subsample=4, train from scratch.

### Expected outcomes

1. **Slower convergence** - heavier augmentation means harder training signal, expect 2-3 more epochs to reach exp 25 E2 levels.
2. **Less overfitting** - val loss should not diverge from train loss as quickly. The train/val gap should stay tighter.
3. **Context contribution maintained or improved** - if audio is noisier, the accuracy gap between full model and no_events benchmark should stay above 2.3% (exp 25's final value) and ideally grow.
4. **Higher ceiling** - if overfitting was the bottleneck, we should eventually exceed exp 25's 68.6% HIT / 0.330 score.

### Risk

- Too much augmentation could slow convergence so much that we can't tell if it's working within reasonable epoch count.
- Audio augmentation might be so heavy it degrades audio signal quality, lowering the ceiling rather than raising it.
- Context contribution may still shrink - the model might just overfit more slowly to the same audio-dominant solution.

## Result

*Pending*

## Lesson

*Pending*
