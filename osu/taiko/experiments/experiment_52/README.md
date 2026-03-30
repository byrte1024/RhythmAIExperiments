# Experiment 52 - Audio Window Size Sweep

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

The audio window (A_BINS past + B_BINS future) has been 500/500 (2.5s/2.5s) since experiment 5. This is the most fundamental hyperparameter we've never tuned. It controls:
- How much past audio context the model sees (A_BINS)
- How far ahead the model can predict (B_BINS) — also sets N_CLASSES = B_BINS + 1
- Total attention cost scales with (A_BINS + B_BINS)^2

### Sub-experiments (A/B in mel bins, ~5ms per bin)

| Exp | A_BINS | B_BINS | Past | Future | N_CLASSES | Tokens | Cost vs baseline |
|---|---|---|---|---|---|---|---|
| 52-A | 250 | 250 | 1.25s | 1.25s | 251 | 125 | 0.25x |
| 52-B | 500 | 250 | 2.5s | 1.25s | 251 | 188 | 0.56x |
| 52-C | 1000 | 250 | 5.0s | 1.25s | 251 | 312 | 1.56x |
| 52-D | 250 | 500 | 1.25s | 2.5s | 501 | 188 | 0.56x |
| **52-E** | **500** | **500** | **2.5s** | **2.5s** | **501** | **250** | **1.0x (baseline)** |
| 52-F | 1000 | 500 | 5.0s | 2.5s | 501 | 375 | 2.25x |
| 52-G | 250 | 1000 | 1.25s | 5.0s | 1001 | 312 | 1.56x |
| 52-H | 500 | 1000 | 2.5s | 5.0s | 1001 | 375 | 2.25x |
| 52-I | 1000 | 1000 | 5.0s | 5.0s | 1001 | 500 | 4.0x |
| 52-J | 500 | 125 | 2.5s | 0.625s | 126 | 156 | 0.39x |
| 52-K | 500 | 75 | 2.5s | 0.375s | 76 | 144 | 0.33x |
| 52-L | 500 | 33 | 2.5s | 0.165s | 34 | 133 | 0.28x |

52-E is exp 45 baseline — no need to rerun.

### Predictions

- **More future audio (higher B_BINS) should help most.** The model currently guesses 2.5s ahead. At B=1000 (5s), it can see a full musical phrase and should resolve the 2x/0.5x metric confusion.
- **More past audio (higher A_BINS) helps less.** Past audio is mostly used for spectral context, not timing. Event embeddings already carry rhythm info.
- **B=250 will hurt.** Only 1.25s lookahead means more STOP predictions and worse accuracy on distant onsets.
- **The diagonal (250/250, 500/500, 1000/1000) tests balanced scaling.** Larger windows give more context but cost quadratically.

### Architecture

All use exp 45 settings (EventEmbeddingDetector, gap ratios, tight density jitter). Conv stem stride 4 is unchanged — token count scales with window size.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_52a --model-type event_embed --a-bins 250 --b-bins 250 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52b --model-type event_embed --a-bins 500 --b-bins 250 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52c --model-type event_embed --a-bins 1000 --b-bins 250 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52d --model-type event_embed --a-bins 250 --b-bins 500 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52f --model-type event_embed --a-bins 1000 --b-bins 500 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52g --model-type event_embed --a-bins 250 --b-bins 1000 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiku_v2 --run-name detect_experiment_52h --model-type event_embed --a-bins 500 --b-bins 1000 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52i --model-type event_embed --a-bins 1000 --b-bins 1000 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52j --model-type event_embed --a-bins 500 --b-bins 125 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52k --model-type event_embed --a-bins 500 --b-bins 75 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_52l --model-type event_embed --a-bins 500 --b-bins 33 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

### Recommended run order

Run in this order to maximize information per GPU-hour. Each result informs whether to skip later experiments.

1. [x] **52-A (250/250)** — cheapest grid (0.25x). HIT 70.6%, metronome 50.4%. Smaller window barely hurts HIT but best corruption resilience.
2. [ ] **52-L (500/33)** — cheapest new (0.28x), tests extreme: can the model predict with almost no future? If this works at all, short future is viable.
3. [ ] **52-J (500/125)** — cheap (0.39x), the practical minimum. If 52-L fails but 52-J works, the sweet spot is between 33-125.
4. [ ] **52-B (500/250)** — cheap (0.56x), halved future. Compare with 52-A (250/250) to isolate past vs future contribution.
5. [ ] **52-D (250/500)** — cheap (0.56x), halved past. Same future as baseline — does past audio matter?
6. [ ] **52-F (1000/500)** — moderate (2.25x), doubled past. If 52-D shows past doesn't matter, skip this.
7. [ ] **52-H (500/1000)** — moderate (2.25x), doubled future. The key "does more future help?" test. If 52-B shows halved future barely hurts, skip this.
8. [ ] **52-C (1000/250)** — moderate (1.56x), lots of past + short future. Only if past matters (from 52-D vs 52-E).
9. [ ] **52-G (250/1000)** — moderate (1.56x), little past + lots of future. Only if future matters (from 52-H).
10. [ ] **52-I (1000/1000)** — expensive (4.0x). Only if both past AND future help.
11. [ ] **52-K (500/75)** — cheap (0.33x), between L and J. Only if 52-L and 52-J show interesting tradeoff to narrow down.

**Skip rules:**
- If 52-L (33 future) completely fails → skip 52-K, the minimum viable future is >33
- If 52-B (250 future) matches baseline → skip 52-H/G/I, more future doesn't help
- If 52-D (250 past) matches baseline → skip 52-F/C/I, more past doesn't help
- If both past and future don't matter much → the 500/500 default is already optimal, stop the sweep

## Result

### 52-A: 250/250 (1.25s/1.25s) — stopped at eval 7

| Metric | Eval 3 | Eval 5 | Eval 7 | Exp 45 eval 7 (500/500) |
|---|---|---|---|---|
| HIT | 69.6% | 71.1% | 70.6% | 71.2% |
| Exact | 50.6% | 51.8% | 51.6% | 52.5% |
| Ctx delta | 8.4pp | 11.2pp | 8.0pp | 7.3pp |
| AR step0 | 72.5% | 71.1% | 72.4% | 72.4% |
| AR step1 | 39.7% | 40.9% | 40.0% | 44.1% |
| AR step3 | — | — | 20.5% | 23.1% |
| Metronome | 49.5% | 49.2% | **50.4%** | 41.7% |
| Adv metronome | 48.8% | 48.7% | **49.6%** | 49.9% |
| Time shifted | 48.2% | 48.8% | **49.5%** | 45.1% |
| stop_f1 | 0.495 | 0.525 | 0.520 | 0.508 |
| Unique preds | 246 | — | — | ~446 |

**Observations:**
- HIT capped at ~71%, 1-2pp below 500/500. The smaller lookahead limits prediction range.
- Exact accuracy is higher than expected — 251 classes is an easier classification problem.
- **Corruption resilience is excellent.** Metronome 50.4%, adv metronome 49.6%, time shifted 49.5% — all ATH. The smaller window means less context to corrupt.
- Context delta is very high (8-11pp) — model relies heavily on context with limited audio.
- Unique preds capped at ~246/251 — saturating the class space.
- STOP rate didn't meaningfully increase despite shorter lookahead — still 0.8% of data.
- Same metronome pattern visible in graphs as 500/500 — the fundamental error structure doesn't change with window size.
- Some graph axes still show 500 range (no data there) — cosmetic issue, not functional.

### Other sub-experiments

*Pending — remaining configs not yet run*

## Lesson

*Pending — need more configs to draw conclusions*
