# Experiment 53 - B_AUDIO/B_PRED Split

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 52](../experiment_52/README.md) showed:
- 250 past bins is sufficient ([52-D](../experiment_52/README.md) matches [exp 45](../experiment_45/README.md) at half compute)
- 500 future bins is optimal (1000 breaks STOP, 33 spams transients)
- B=250 gives healthiest density dependence ([52-B](../experiment_52/README.md): 74% zero_density_stop)
- Smaller N_CLASSES converges faster ([52-L](../experiment_52/README.md): 74.2% HIT at eval 2)

**The B_AUDIO/B_PRED split combines these findings:**
- A_BINS = 250 (past audio, proven sufficient)
- B_AUDIO = 500 (future audio for spectral context, full lookahead)
- B_PRED = 250 (prediction range, N_CLASSES=251, easy classification + healthy density)

The model sees 750 mel frames (250 past + 500 future) but only predicts offsets 0-249 (+ STOP at 250). If the next onset is >250 bins away, it predicts STOP and hops. This gives:
- 500-bin future audio for seeing transients ahead (prevents [52-L](../experiment_52/README.md)'s spamming)
- 251-class problem for fast convergence and density dependence (from [52-B](../experiment_52/README.md))
- 250 past for compute savings (from [52-D](../experiment_52/README.md))

### Architecture

```
Mel window: 750 frames (250 past + 500 future)
Conv stem: 750 → 187 tokens (stride 4)
Cursor: token 62 (250 // 4)
N_CLASSES: 251 (0-249 offsets + STOP)
```

Same EventEmbeddingDetector with gap ratios, tight density jitter.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_53 --model-type event_embed --a-bins 250 --b-bins 500 --b-pred 250 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
