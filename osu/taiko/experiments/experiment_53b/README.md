# Experiment 53-B - B_AUDIO/B_PRED Split with A_BINS=500

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 53](../experiment_53/README.md) plateaued at exactly 72.1% HIT — identical to [exp 45](../experiment_45/README.md) which used A_BINS=500 (the default). The B_AUDIO/B_PRED split improved benchmarks but couldn't break past exp 45's per-sample ceiling. One key difference: exp 53 used A_BINS=250 (from [52-D](../experiment_52/README.md)'s finding that 250 is "sufficient"), while exp 45 used A_BINS=500.

**Theory: A_BINS=250 may be the bottleneck.** With 250 past bins (~1.25s, 62 tokens), the model has limited past audio context. Doubling to A_BINS=500 (~2.5s, 125 tokens) restores the baseline past context while keeping the B_AUDIO/B_PRED split benefits. If HIT% improves beyond 72.1%, A_BINS was the limiter. If it stays the same, the ceiling is elsewhere.

Everything else is identical to [exp 53](../experiment_53/README.md):
- B_BINS (B_AUDIO) = 500 (future audio visible)
- B_PRED = 250 (prediction range, N_CLASSES=251)
- Same EventEmbeddingDetector with gap ratios
- Same augmentation, loss, and training params

### Architecture

```
Mel window: 1000 frames (500 past + 500 future)
Conv stem: 1000 → 250 tokens (stride 4)
Cursor: token 125 (500 // 4)
N_CLASSES: 251 (0-249 offsets + STOP)
```

### Key difference from exp 53

| Parameter | Exp 53 | Exp 53-B |
|---|---|---|
| A_BINS | 250 | **500** |
| B_BINS (B_AUDIO) | 500 | 500 |
| B_PRED | 250 | 250 |
| WINDOW | 750 | **1000** |
| Tokens | 187 | **250** |
| Cursor token | 62 | **125** |

More tokens means slightly more compute per step (~33% more tokens through transformer). Same 128 max events but now events can be mapped across 125 past tokens instead of 62.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_53b --model-type event_embed --a-bins 500 --b-bins 500 --b-pred 250 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
