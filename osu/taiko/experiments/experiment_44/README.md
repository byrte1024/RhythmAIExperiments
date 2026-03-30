# Experiment 44 - Gentle AR Augmentation

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp [43](../experiment_43/README.md) showed that aggressive augmentation (~43% context corruption) backfires — the model distrusts context entirely, metronomes from step 0, and performs worse on both per-sample and AR metrics. Meanwhile, exp [43-B](../experiment_43b/README.md) proved that context dependency HELPS AR resilience (exp [42](../experiment_42/README.md) beats exp [14](../experiment_14/README.md) at every AR step).

**The fix: distort, never destroy.** Every augmented sample should still have recognizable context and audio. No augmentation blanks or fully replaces context. Partial corruption (half of events) teaches robustness while preserving enough real signal for the model to trust context.

### Augmentation changes (from exp 43 → 44)

**Context (~14% total corruption, down from ~43%):**

| Aug | Exp [43](../experiment_43/README.md) | **Exp 44** |
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
Same as exp [42](../experiment_42/README.md) (EventEmbeddingDetector, 16.1M params).

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_44 --model-type event_embed --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**New ATH across the board.** Stopped at eval 20 (epoch 5). Surpasses exp [42](../experiment_42/README.md) on per-sample metrics while adding significant metronome resilience.

### Per-sample metrics (best: eval 19-20)

| Metric | Exp 44 (eval 19) | Exp 44 (eval 20) | Exp [42](../experiment_42/README.md) ATH (eval 9) |
|---|---|---|---|
| HIT | **73.6%** | 73.5% | [73.2%](../experiment_42/README.md) |
| MISS | **25.9%** | 26.0% | [26.4%](../experiment_42/README.md) |
| Accuracy | 54.7% | 54.7% | — |
| Context delta | 5.6pp | 4.9pp | 4.3pp |

### AR resilience

| Metric | Exp 44 (eval 20) | Exp [42](../experiment_42/README.md) (eval 9) |
|---|---|---|
| AR step0 | 74.9% | [74.2%](../experiment_43b/README.md) |
| AR step1 | **48.2%** | [46.3%](../experiment_43b/README.md) |
| AR step3 | 22.2% | [26.4%](../experiment_43b/README.md) |
| AR step5 | 16.8% | [22.4%](../experiment_43b/README.md) |

Step0-1 improved over exp [42](../experiment_42/README.md). Step3+ still lower — deeper AR cascade remains an open problem.

### Metronome resilience

| Metric | Exp 44 (eval 20) | Exp [42](../experiment_42/README.md) (eval 9) |
|---|---|---|
| Metronome benchmark | **45.7%** | [25.4%](../experiment_42/README.md) |
| Advanced metronome | **49.5%** | — |
| Time shifted | **47.3%** | — |

Nearly 2x exp [42](../experiment_42/README.md)'s metronome resilience. The gentle augmentation works — the model maintains accuracy even when context is corrupted to a metronome pattern.

### no_audio behavior

| Metric | Exp 44 (eval 20) | Exp [42](../experiment_42/README.md) (eval 9) |
|---|---|---|
| no_audio stop rate | 96.6% | 3.1% |

The no_audio stop rate is extremely noisy across evals (12% → 68% → 16% → 77% → 96%), so the eval 20 value is not reliable. The model has not consistently learned to stop on silence — this remains an open problem for future work.

### Metronome error analysis (eval 20)

| Peak | Target continues | Pred continues | Pred continues but target breaks |
|---|---|---|---|
| top1 | 47.1% | 48.8% | **11.8%** |
| top2 | 28.3% | 29.4% | 10.2% |
| top3 | 12.4% | 10.3% | 4.4% |

The metronome failure mode (pred continues when target breaks) is 11.8% on top1, down from 13.6% at eval 4. Still the dominant error type.

### Progression

HIT plateaued at ~72.8% from eval 5-15, then broke through to 73.6% at eval 19. The model was not converged at eval 7 as initially suspected — longer training paid off.

## Lesson

- **Gentle augmentation works.** ~14% context corruption rate preserves context trust while building resilience. The "distort, don't destroy" principle is validated.
- **Longer training matters.** HIT appeared plateaued for 10 evals, then broke through. Patience was rewarded.
- **Metronome resilience and per-sample accuracy are not at odds.** Exp 44 achieves both — better HIT than exp [42](../experiment_42/README.md) AND 2x the metronome resilience.
- **no_audio stop rate is unreliable.** Swings wildly between evals (12%-96%). The model has not consistently learned to stop on silence — this needs explicit training or an audio gate mechanism.
- **The 11.8% metronome failure rate remains** the key target for future work. This is where the model continues a pattern when it should break — the dominant error mode in AR generation (see exp [42-AR](../experiment_42ar/README.md) human evaluation).
