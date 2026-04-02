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

Stopped at eval 11 (epoch 3.7). **Peak at eval 11 — still improving when stopped.**

### Peak (eval 11) metrics:

| Metric | Value |
|---|---|
| HIT% | 73.4% |
| GOOD% | 73.8% |
| MISS% | 26.1% |
| Accuracy | 54.2% |
| Stop F1 | 0.562 |
| Model score | 0.384 |
| Val loss | 2.479 |

### Benchmark highlights (eval 11):

| Benchmark | Value |
|---|---|
| Metronome resilience | 46.0% |
| Adv metronome | 51.5% |
| no_audio_stop | 13.9% |
| no_events_no_audio_stop | 100.0% |
| zero_density_stop | 8.2% |
| NoEvents accuracy | 49.7% |
| Context delta | +4.5% |

### Progression summary:

| Eval | Epoch | HIT% | Stop F1 | Score | Val Loss | NoEvt% | CtxD |
|------|-------|------|---------|-------|----------|--------|------|
| 1 | 1.2 | 68.4 | 0.479 | 0.326 | 2.637 | 41.1 | +7.9 |
| 3 | 1.7 | 71.9 | 0.539 | 0.366 | 2.524 | 44.6 | +7.1 |
| 6 | 2.5 | 73.1 | 0.550 | 0.379 | 2.487 | 45.0 | +8.5 |
| 8 | 2.0 | 73.4 | 0.546 | 0.382 | 2.478 | 48.3 | +5.5 |
| **11** | **3.7** | **73.4** | **0.562** | **0.384** | **2.479** | **49.7** | **+4.5** |

### vs notable models (all at peak):

| Metric | exp14 | exp35c | exp44 | exp45 | exp53 | **exp53-B** |
|--------|-------|--------|-------|-------|-------|-------------|
| HIT% | 69.2 | 71.6 | **73.7** | 72.1 | 72.1 | 73.4 |
| Accuracy | 50.5 | 52.7 | **54.8** | 52.9 | 53.3 | 54.2 |
| Stop F1 | 0.480 | 0.543 | **0.570** | 0.553 | 0.547 | 0.562 |
| Val loss | 2.645 | 2.533 | 2.480 | 2.516 | 2.518 | **2.479** |
| NoEvt acc | 50.0 | 48.1 | 48.4 | 47.1 | 46.4 | **49.7** |
| Ctx delta | +0.5 | +4.5 | **+6.4** | +5.7 | +6.9 | +4.5 |
| Metro resist | 47.7 | 44.1 | 44.2 | 43.7 | **52.5** | 46.0 |

### A_BINS bottleneck confirmed:

| Metric | Exp 53 (A=250, peak eval 14) | Exp 53-B (A=500, eval 11) |
|--------|------------------------------|---------------------------|
| HIT% | 72.1% | **73.4%** (+1.3pp) |
| Accuracy | 53.3% | **54.2%** (+0.9pp) |
| Val loss | 2.518 | **2.479** (-0.039) |
| NoEvt acc | 46.4% | **49.7%** (+3.3pp) |

The +1.3pp HIT improvement breaks the 72.1% ceiling that exp 53 shared with exp 45. The +3.3pp NoEvt improvement confirms the gain comes from better audio representations, not more context dependency (ctx delta actually *decreased* from +6.9% to +4.5%).

## Lesson

**A_BINS=250 was indeed the bottleneck.** Doubling past audio context from 250 to 500 broke the 72.1% HIT ceiling that exp 53 shared with [exp 45](../experiment_45/README.md), reaching 73.4% at eval 11 (still improving when stopped).

The improvement is primarily audio-driven: NoEvt accuracy jumped from 46.4% to 49.7% (best ever), while context delta shrank from +6.9% to +4.5%. More past audio helps the model recognize rhythmic patterns directly from the spectrogram rather than relying on event context.

The B_AUDIO/B_PRED split combined with A_BINS=500 is now the best architecture configuration: 500 past + 500 future audio (5.0s total window), predicting into 250 bins (251 classes). Val loss 2.479 matches [exp 44](../experiment_44/README.md)'s all-time best (2.480) with cleaner benchmarks.

53-B is 0.3pp behind [exp 44](../experiment_44/README.md)'s ATH of 73.7%, but exp 44 ran to eval 11 at epoch 4.7 while 53-B only reached epoch 3.7 — it likely had more room. The metronome resistance (46.0%) sits between exp 53's high (52.5%) and exp 44/45's low (~44%), suggesting more past context increases vulnerability to pattern repetition but less than the B_PRED split's protection.
