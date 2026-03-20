# Experiment 43 - AR-Resilient Training with Realistic Augmentation

## Hypothesis

Exp 42-AR blind self-evaluation (volunteer data still pending) revealed a critical finding: **higher per-sample accuracy doesn't mean better AR generation.** In self-ranking, exp 14 (68.9% HIT, no context) scored nearly equal to exp 42 (73.2% HIT, deep context) because exp 42's context dependency causes:
- Metronome regression — model locks into repeating gaps
- AR cascade errors — wrong predictions corrupt context, degrading subsequent predictions
- Inconsistent density — some songs get 2x too many or too few events

Exp 42's metronome benchmark at 25.4% (vs 50.5% for exp 14) confirms the fragility — wrong context is catastrophic.

**Solution: train with realistic AR-failure augmentation** so the model learns to recover from its own mistakes.

### Augmentation changes (from exp 42)

| Augmentation | Exp 42 | **Exp 43** | Simulates |
|---|---|---|---|
| Event jitter | ±2 bins, 2x recency | **±5 bins, 3x recency** | Prediction errors |
| Event deletion | 4%, 1-2 events | **15%, 1-4 events** | Skip errors |
| Event insertion | 4%, 1-2 events | **10%, 1-3 events** | Hallucinations |
| **Metronome corruption** | — | **5%** | Model locks into repeating gap |
| **Advanced metronome** | — | **5%** | Right tempo, no pattern variation |
| **Large time shift** | — | **5%** | AR cursor drift (±100 bins) |
| **Hallucination burst** | — | **3%** | Rapid spam section |
| Context dropout | 2% | **5%** | Total context loss |
| Context truncation | 5% | **8%** | Partial context loss |

### New benchmarks

**autoregress** — 32 consecutive AR predictions per sample:
- Feeds each prediction back as context (like real inference)
- Tracks: survival rate, entropy drift, prediction distribution drift, density comparison
- Graphs: survival curve, entropy over steps, prediction drift, density bar

**lightautoregress** — 32 consecutive predictions compared 1:1 to ground truth:
- pred[i] vs truth[i] — a cascade causes all future notes to misalign
- Tests whether the model can recover from early errors
- Graphs: HIT rate curve over steps, scatter at steps 0/4/8/16/31, frame error curve

### Architecture

Same as exp 42 (EventEmbeddingDetector, 16.1M params). Only augmentation and benchmarks change.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_43 --model-type event_embed --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
