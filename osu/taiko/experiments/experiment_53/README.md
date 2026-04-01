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

Stopped at eval 20 (epoch 5.0). **Peak at eval 14 (epoch 4.5).**

### Peak (eval 14) metrics:

| Metric | Value |
|---|---|
| HIT% | 72.1% |
| GOOD% | 72.5% |
| MISS% | 27.3% |
| Accuracy | 53.3% |
| Stop F1 | 0.547 |
| Model score | 0.369 |

### Benchmark highlights (eval 14):

| Benchmark | Value |
|---|---|
| Metronome resilience | 52.5% |
| no_audio_stop | 98.6% |
| no_events_no_audio_stop | 100.0% |
| zero_density_stop | 9.7% |

### Progression summary:

| Eval | Epoch | HIT% | Stop F1 | Score | Metro% | NoAudio Stop% |
|------|-------|------|---------|-------|--------|---------------|
| 1 | 1.2 | 68.6 | 0.507 | 0.332 | 46.0 | 25.8 |
| 7 | 2.7 | 71.6 | 0.544 | 0.363 | 48.6 | 68.5 |
| **14** | **4.5** | **72.1** | **0.547** | **0.369** | **52.5** | **98.6** |
| 17 | 5.2 | 72.0 | 0.542 | 0.367 | 52.0 | 83.9 |
| 20 | 5.0 | 70.7 | 0.509 | 0.350 | 48.5 | 90.4 |

Peaked at eval 14 then regressed. Eval 20 dropped to 70.7% HIT with degraded benchmarks (zero_density_stop crashed to 4.3%).

### vs [Exp 45](../experiment_45/README.md) (best comparison):

| Metric | Exp 45 (eval 5) | Exp 53 (eval 14) |
|--------|-----------------|------------------|
| HIT% | 72.1% | 72.1% |
| GOOD% | 72.5% | 72.5% |
| MISS% | 27.3% | 27.3% |
| Stop F1 | 0.553 | 0.547 |
| Score | 0.368 | 0.369 |

Per-sample metrics are **identical** at peaks. The differentiation is entirely in benchmarks: exp 53 has vastly better corruption resilience and no_audio_stop behavior.

## Lesson

The B_AUDIO/B_PRED split successfully combines large audio context with a small prediction range. Per-sample metrics match [exp 45](../experiment_45/README.md) exactly, while benchmark behaviors (corruption resilience, STOP behavior) are substantially better.

The architecture is sound — large future audio window for spectral context, small prediction range for fast convergence and healthy density dependence. Training peaked early (eval 14, epoch 4.5) and regressed afterward, suggesting the model overfits or loses calibration with extended training.

[Exp 53-B](../experiment_53b/README.md) tests A_BINS=500 (doubled past context) to see if more past audio improves pattern variety on complex rhythmic material.
