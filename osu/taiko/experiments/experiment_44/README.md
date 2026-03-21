# Experiment 44 - Gentle AR Augmentation

## Hypothesis

Exp 43 showed that aggressive augmentation (~43% context corruption) backfires — the model distrusts context entirely, metronomes from step 0, and performs worse on both per-sample and AR metrics. Meanwhile, exp 43-B proved that context dependency HELPS AR resilience (exp 42 beats exp 14 at every AR step).

**The fix: distort, never destroy.** Every augmented sample should still have recognizable context and audio. No augmentation blanks or fully replaces context. Partial corruption (half of events) teaches robustness while preserving enough real signal for the model to trust context.

### Augmentation changes (from exp 43 → 44)

**Context (~14% total corruption, down from ~43%):**

| Aug | Exp 43 | **Exp 44** |
|-----|--------|-----------|
| Event jitter | ±5, 3x recency | **±3, 2x recency** |
| Event deletion | 15%, 1-4 events | **5%, 1-2 events** |
| Event insertion | 10%, 1-3 events | **3%, 1 event** |
| Metronome (random) | 5%, replaces ALL | **2%, replaces RECENT HALF** |
| Adv metronome (dominant) | 5%, replaces ALL | **2%, replaces OLDEST HALF** |
| Large time shift | 5%, ±100 bins | **2%, ±50 bins** |
| Hallucination burst | 3% | **REMOVED** |
| Context dropout | 5% | **REMOVED** |
| Context truncation | 8% | **5%** |

**Audio (reduced across the board):**

| Aug | Exp 43 | **Exp 44** |
|-----|--------|-----------|
| Mel gain | ±3dB @ 50% | **±2dB @ 30%** |
| Mel noise | σ≤0.4 @ 30% | **σ≤0.3 @ 15%** |
| Freq jitter | ±5 @ 30% | **±3 @ 15%** |
| Temporal shuffle | 2% | **REMOVED** |
| SpecAugment freq | 40%, 1-2 masks, 15 bands | **20%, 1 mask, 10 bands** |
| SpecAugment time | 40%, 1-2 masks, 50 frames | **20%, 1 mask, 30 frames** |

### Key principles
- **Never blank context** — no dropout, no full replacement
- **Partial corruption** — metronome corrupts half of events, not all. Model always has real context to reference.
- **Distort, don't destroy** — jitter and shift events, don't delete or replace them

### Architecture
Same as exp 42 (EventEmbeddingDetector, 16.1M params).

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_44 --model-type event_embed --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
