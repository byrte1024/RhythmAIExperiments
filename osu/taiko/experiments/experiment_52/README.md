# Experiment 52 - Audio Window Size Sweep

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

The audio window (A_BINS past + B_BINS future) has been 500/500 (2.5s/2.5s) since experiment [5](../experiment_05/README.md). This is the most fundamental hyperparameter we've never tuned. It controls:
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

52-E is exp [45](../experiment_45/README.md) baseline — no need to rerun.

### Predictions

- **More future audio (higher B_BINS) should help most.** The model currently guesses 2.5s ahead. At B=1000 (5s), it can see a full musical phrase and should resolve the 2x/0.5x metric confusion (exp [48](../experiment_48/README.md)).
- **More past audio (higher A_BINS) helps less.** Past audio is mostly used for spectral context, not timing. Event embeddings already carry rhythm info.
- **B=250 will hurt.** Only 1.25s lookahead means more STOP predictions and worse accuracy on distant onsets.
- **The diagonal (250/250, 500/500, 1000/1000) tests balanced scaling.** Larger windows give more context but cost quadratically.

### Architecture

All use exp [45](../experiment_45/README.md) settings (EventEmbeddingDetector, gap ratios, tight density jitter). Conv stem stride 4 is unchanged — token count scales with window size.

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

1. [x] **52-A (250/250)** — HIT 70.6%, metronome 50.4%. Smaller window barely hurts HIT but best corruption resilience.
2. [x] **52-L (500/33)** — HIT 74.2% (new ATH!) but AR quality poor: spams transients, ignores density. Model NEEDS future audio.
3. [x] ~~**52-J (500/125)**~~ — Skipped. 52-L showed very small B_BINS needs B_AUDIO/B_PRED split to work.
4. [x] ~~**52-K (500/75)**~~ — Skipped. Same reasoning as 52-J.
5. [ ] **52-B (500/250)** — cheap (0.56x), halved future. Compare with 52-A to isolate past vs future contribution.
6. [ ] **52-D (250/500)** — cheap (0.56x), halved past. Same future as baseline — does past audio matter?
7. [ ] **52-F (1000/500)** — moderate (2.25x), doubled past. If 52-D shows past doesn't matter, skip this.
8. [x] **52-H (500/1000)** — Stopped at eval 4. HIT 71.5% (similar to 500/500). STOP broken: no_events_no_audio dropped to 73%. 554/1001 unique preds — model ignores the extra range. More future doesn't help.
9. [x] ~~**52-C (1000/250)**~~ — Skipped. Past doesn't need testing until 52-D runs.
10. [x] ~~**52-G (250/1000)**~~ — Skipped. 52-H showed 1000 future bins breaks STOP and wastes class space.
11. [x] ~~**52-I (1000/1000)**~~ — Skipped. Both 52-H (1000 future) failed. No point in 1000/1000.

**Skip rules:**
- If 52-L (33 future) completely fails → skip 52-K, the minimum viable future is >33
- If 52-B (250 future) matches baseline → skip 52-H/G/I, more future doesn't help
- If 52-D (250 past) matches baseline → skip 52-F/C/I, more past doesn't help
- If both past and future don't matter much → the 500/500 default is already optimal, stop the sweep

## Result

### 52-A: 250/250 (1.25s/1.25s) — stopped at eval 7

| Metric | Eval 3 | Eval 5 | Eval 7 | [Exp 45](../experiment_45/README.md) eval 7 (500/500) |
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

### 52-L: 500/33 (2.5s/0.165s) — stopped at eval 2

| Metric | Eval 1 | Eval 2 | [Exp 45](../experiment_45/README.md) eval 2 (500/500) |
|---|---|---|---|
| HIT | 73.5% | **74.2%** | 70.5% |
| MISS | 25.1% | **23.9%** | 28.2% |
| Exact | 53.2% | 54.8% | 51.7% |
| Accuracy | 65.6% | **67.8%** | — |
| stop_f1 | 0.828 | **0.837** | 0.533 |
| stop_recall | 0.825 | 0.857 | — |
| Ctx delta | -3.5pp | -1.6pp | 7.2pp |
| Metronome | 67.6% | **68.0%** | 44.3% |
| no_audio_stop | 44.0% | 58.7% | — |
| AR step0 | 75.0% | **81.5%** | 71.5% |

**Per-sample metrics are the best we've ever seen.** HIT 74.2% (new ATH, beats exp [44](../experiment_44/README.md)'s [73.6%](../experiment_44/README.md)). MISS 23.9% (new ATH). Stop F1 0.837 (unprecedented). Metronome 68.0% (untouchable). The 34-class problem converges immediately — eval 1 already matched exp [44](../experiment_44/README.md)'s ATH.

**BUT: AR quality is poor despite great metrics.**
- Model always finds the closest possible beat (most beats are in the ~75 bin range, which is ~375ms). With a 33-bin (165ms) window, the optimal strategy is to predict the nearest transient regardless of density conditioning.
- Density has almost no effect — the model spams onsets regardless of what density is requested. Without future audio context, it can't see upcoming silent sections or lower-density passages.
- Errors are "brittle" — with 33 bins, a wrong prediction is either garbage or a false STOP. No graceful degradation to nearby musical positions like with 500 bins.
- The frequent STOP-hopping creates a scanning pattern that produces notes at every detected transient, ignoring chart-level structure.

### 52-H: 500/1000 (2.5s/5.0s) — stopped at eval 4

| Metric | Eval 1 | Eval 4 | [Exp 45](../experiment_45/README.md) eval 4 (500/500) |
|---|---|---|---|
| HIT | 68.6% | 71.5% | 70.2% |
| MISS | 30.6% | 27.9% | 29.1% |
| Ctx delta | 6.9pp | 5.6pp | 7.8pp |
| stop_f1 | 0.530 | 0.547 | 0.515 |
| AR step0 | 67.4% | 73.4% | 72.0% |
| Metronome | 49.2% | 47.0% | 40.5% |
| Unique preds | 610 | 554 | 460 |
| no_audio_stop | 32.1% | **5.9%** | 13.3% |
| no_evt_no_audio_stop | 88.2% | **73.7%** | 100.0% |

**Per-sample HIT is similar to 500/500** — no clear advantage from doubled future audio.

**STOP is broken.** With only 0.2% STOP targets (11,575 out of 5.25M), the model can't learn to stop. no_events_no_audio_stop DECREASED from 88%→74% over training — the model gets worse at stopping even when there's literally nothing. no_audio_stop at 5.9%.

**554/1001 unique predictions** — 45% of the class space is dead. The 200-1000 bin range has only 48K samples across 800 bins. Most bins have single-digit training examples.

**Conclusion:** 1000 future bins is too many. Classification too hard, STOP too rare, sparse long-range bins undertrained.
