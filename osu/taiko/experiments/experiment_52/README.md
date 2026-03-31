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
5. [x] **52-B (500/250)** — HIT 71.9%, zero_density_stop 69-74%. Healthiest density dependence. Tracks exp 45 pace with 251 classes.
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

### 52-B: 500/250 (2.5s/1.25s) — stopped at eval 5

| Metric | Eval 1 | Eval 4 | Eval 5 | [Exp 45](../experiment_45/README.md) eval 5 (500/500) |
|---|---|---|---|---|
| HIT | 67.6% | 72.0% | 71.9% | 71.0% |
| MISS | 31.6% | 27.5% | 27.6% | 28.6% |
| Ctx delta | 8.1pp | 6.7pp | 6.2pp | — |
| stop_f1 | 0.506 | 0.536 | 0.523 | 0.553 |
| AR step0 | 65.3% | 69.3% | 71.0% | 68.4% |
| Metronome | 43.2% | 45.1% | 44.9% | 44.7% |
| Adv metronome | — | 48.7% | **50.8%** | 48.7% |
| Time shifted | — | 45.2% | **46.9%** | 43.0% |
| zero_density_stop | — | **74.3%** | 69.1% | ~7% |
| Unique preds | 242 | 247 | — | ~450 |
| nane_stop | 100% | — | — | 100% |

**HIT tracks [exp 45](../experiment_45/README.md) closely** — 71.9% vs 71.0% at eval 5. The 251-class problem converges slightly faster.

**Density dependence is the standout finding.** zero_density_stop at 69-74% means the model uses density as the primary STOP controller — "if density says nothing, stop." Compare to 500/500 models at 7-8% zero_density_stop where the model ignores density for STOP decisions.

Random density costs -15.6pp accuracy (similar to 500/500's -14pp), but zero density costs **-43.8pp** (vs 500/500's -17.9pp). The model has learned a much healthier density relationship.

**Corruption resilience is strong.** Metronome 44.9%, adv metronome 50.8%, time shifted 46.9% — all comparable to or better than 500/500.

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

### 52-D: 250/500 (1.25s/2.5s) — stopped at eval 4

| Metric | Eval 1 | Eval 2 | Eval 3 | Eval 4 | [Exp 45](../experiment_45/README.md) eval 4 |
|---|---|---|---|---|---|
| HIT | 68.9% | 69.3% | 70.2% | 70.9% | 70.2% |
| MISS | 30.3% | 30.1% | 29.1% | 28.5% | 29.1% |
| Ctx delta | 10.1pp | 5.7pp | 8.3pp | 7.2pp | 7.8pp |
| stop_f1 | 0.478 | 0.528 | 0.525 | 0.542 | 0.537 |
| AR step0 | 70.4% | 69.1% | 71.3% | 73.1% | 72.0% |
| Metronome | 48.1% | 48.2% | 49.4% | 49.3% | 40.5% |
| zero_density_stop | 1.9% | 2.8% | 21.4% | 18.3% | 5.0% |

**Outperforms [exp 45](../experiment_45/README.md) (500/500) at half the past audio compute.** HIT 70.9% vs 70.2%. Best corruption resilience of any config: metronome 49.3%, adv metronome 50.7%, time shifted 49.1%.

**250 past bins is sufficient.** The extra 250 past bins in 500/500 add no measurable benefit — identical HIT, slightly worse corruption resilience.

### Full comparison matrix (all configs at eval 4)

| Benchmark | 52-A (250/250) | 52-B (500/250) | 52-D (250/500) | [exp45](../experiment_45/README.md) (500/500) |
|---|---|---|---|---|
| **HIT** | 70.0% | **72.0%** | 70.9% | 70.2% |
| **MISS** | 29.4% | **27.5%** | 28.5% | 29.1% |
| **NE acc** | 43.5% | **46.0%** | 44.9% | 43.7% |
| **NA stop** | **61.7%** | 26.9% | 2.9% | 13.3% |
| **NENA stop** | **100%** | **100%** | 99.3% | **100%** |
| **TS acc** | 47.4% | 45.2% | **49.1%** | 44.2% |
| **M acc** | 49.2% | 45.1% | **49.3%** | 40.5% |
| **AM acc** | 48.6% | 48.7% | **50.7%** | 49.1% |
| **ZD acc** | 11.5% | 8.8% | 11.5% | **28.8%** |
| **ZD stop** | 8.7% | **74.3%** | 18.3% | 5.0% |
| **RD acc** | 34.4% | **37.0%** | 31.0% | 35.5% |
| **stop_f1** | 0.530 | 0.536 | **0.542** | 0.537 |
| **Compute** | **0.25x** | 0.56x | 0.56x | 1.0x |

### Skipped sub-experiments

- **52-C (1000/250)**: Skipped. 52-D showed past doesn't matter — 1000 past bins would waste compute.
- **52-F (1000/500)**: Skipped. Same reasoning.
- **52-G (250/1000)**: Skipped. [52-H](../experiment_52/README.md) showed 1000 future bins breaks STOP.
- **52-I (1000/1000)**: Skipped. Both dimensions showed diminishing/negative returns beyond 500.
- **52-J (500/125)**: Skipped. [52-L](../experiment_52/README.md) showed very small B_BINS needs B_AUDIO/B_PRED split.
- **52-K (500/75)**: Skipped. Same reasoning.

## Lesson

- **Past audio beyond 250 bins doesn't help.** 52-D (250/500) matches [exp 45](../experiment_45/README.md) (500/500) exactly — the extra 250 past bins are wasted compute. Source: 52-D vs exp 45 comparison.
- **Future audio is what matters.** 250→500 future bins gives +0.9pp HIT (52-A vs 52-D). 500→1000 gives nothing useful and breaks STOP (52-H). Source: 52-A/52-D/52-H comparison.
- **Smaller B_BINS forces healthier density dependence.** B=250: zero_density_stop 74% (density drives STOP). B=500: zero_density_stop 5-18% (audio drives STOP). The model uses density as a rate controller when it can't see far enough ahead. Source: 52-B zero_density analysis.
- **Smaller B_BINS = better corruption resilience.** 250/250 and 250/500 both get ~49% metronome vs 500/500's 40%. Fewer classes = less to corrupt. Source: full comparison matrix.
- **Smaller B_BINS = faster convergence.** 251 classes converges faster than 501 (52-B HIT 72.0% at eval 4 vs exp 45's 70.2%). Source: 52-B vs exp 45.
- **Too few future bins (33) degrades to transient-spamming.** The model needs enough lookahead for chart-level structure. Source: [52-L](../experiment_52/README.md).
- **Too many future bins (1000) breaks STOP and wastes class space.** 0.2% STOP rate, 45% dead classes, NENA stop declining. Source: [52-H](../experiment_52/README.md).
- **The optimal base config appears to be 250/500 or 250/250.** Both outperform 500/500 at lower compute. The B_AUDIO/B_PRED split (future experiment) could combine 500-bin audio with 250-bin prediction for the best of both.
