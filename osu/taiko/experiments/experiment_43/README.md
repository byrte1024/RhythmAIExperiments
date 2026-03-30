# Experiment 43 - AR-Resilient Training with Realistic Augmentation

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


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

**Augmentation too aggressive — model distrust context entirely, worse on both per-sample and AR.** Killed after eval 5.

### Per-sample metrics (vs exp 42)

| eval | HIT | Miss | Score | Val loss | Ctx Δ |
|------|-----|------|-------|----------|-------|
| 1 | 64.4% | 34.7% | 0.280 | 2.792 | -1.4% |
| 2 | 67.3% | 31.9% | 0.315 | 2.694 | -1.6% |
| 3 | 68.2% | 31.2% | 0.325 | 2.665 | -0.7% |
| 4 | 68.9% | 30.4% | 0.334 | 2.631 | -0.9% |
| 5 | 68.3% | 31.1% | 0.323 | 2.654 | -0.4% |

Exp 42 at eval 5: 72.0% HIT, 4.3% ctx delta. Exp 43 is **-3.7pp HIT** and **negative context delta**.

### AR resilience (vs exp 42 from 43-B)

| Step | Exp 42 | Exp 43 | Delta |
|------|--------|--------|-------|
| 0 | 74.2% | 66.4% | -7.8pp |
| 1 | 46.3% | 39.2% | -7.1pp |
| 3 | 26.4% | 20.2% | -6.2pp |
| 5 | 22.4% | 15.9% | -6.5pp |
| 8 | 12.8% | 10.8% | -2.0pp |

Worse at EVERY step. The AR augmentation didn't improve AR resilience — it degraded it.

### Metronome collapse

| Step | Exp 42 unique | Exp 43 unique |
|------|--------------|--------------|
| 0 | 36 | **11** |
| 5 | 21 | **7** |
| 10 | 14 | **4** |

Exp 43 starts with only 11 unique predictions (vs 36) and collapses to 3-4 by step 10. The model is metronoming from step 0 — exactly what we were trying to prevent.

### AR set matching

| Metric | Exp 42 (43-B) | Exp 43 |
|--------|--------------|--------|
| Event HIT | 33.9% | **43.2%** |
| Hallucination | 51.5% | 53.5% |
| Density ratio | 1.26x | **1.16x** |

Event HIT is higher (43.2% vs 33.9%) but this is misleading — the model predicts more conservatively (fewer unique values) so the predictions it DOES make are more likely to land on real onsets. It's not more accurate, just less adventurous.

### What went wrong

The augmentation rates were too aggressive:
- **15% deletion** + **10% insertion** + **5% metronome** + **5% adv metronome** + **3% hallucination burst** + **5% large shift** = ~43% of training samples have significantly corrupted context
- The model saw corrupted context nearly half the time → learned to distrust context entirely (ctx delta = -0.4%)
- Without context, it falls back to a narrow "safe" vocabulary of ~11 predictions → metronome behavior from step 0
- The very augmentation designed to prevent metronoming CAUSED it by destroying the model's trust in context

## Lesson

- **Aggressive context augmentation backfires catastrophically.** The model needs to see MOSTLY correct context with OCCASIONAL corruption. ~43% corruption rate taught it to ignore context.
- **Context dependency is fragile.** Exp 42's 5% context delta was built over careful training with mild augmentation. Heavy corruption destroyed it instantly.
- **The augmentation created the exact failure mode it was designed to prevent.** Ironic but informative — the metronome behavior comes from context distrust, and the augmentation maximized distrust.
- **Next: much gentler augmentation rates** — maybe 2-3% for each catastrophic augmentation (metronome, burst, shift) instead of 5%. Or a curriculum: start with clean context, gradually introduce corruption.
