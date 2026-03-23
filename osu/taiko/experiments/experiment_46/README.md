# Experiment 46 - Hard/Soft Loss Ratio Sweep

## Hypothesis

Exp 44 uses hard_alpha=0.5 (50% hard CE + 50% soft targets). Exp 42-B tested hard_alpha=1.0 but only ran 2 evals before being killed — not enough to draw conclusions. The exact-match vs ±1-frame gap is ~19pp, meaning soft targets provide significant gradient signal for near-miss predictions.

**Question:** What is the optimal hard/soft ratio? Soft targets help learning (gradient for near-misses) but may also contribute to metronome behavior by making continuation predictions "close enough." Hard CE forces precision but may be too harsh early in training.

### Sub-experiments

All identical to exp 44 (EventEmbeddingDetector, gentle augmentation, subsample 1) with two changes adopted from exp 45:
- ±2% density jitter @10% (better AR density adherence)
- Gap ratio features enabled (default on from now)

Only variable is hard_alpha:

| Exp | hard_alpha | Soft weight | Hard weight | Description |
|---|---|---|---|---|
| **46-A** | 0.0 | 100% | 0% | Pure soft targets |
| **46-B** | 0.25 | 75% | 25% | Mostly soft |
| **46-C** | 0.75 | 25% | 75% | Mostly hard |
| **46-D** | 1.0 | 0% | 100% | Pure hard CE |

Exp 44 (hard_alpha=0.5) serves as the baseline — no need to rerun.

All use frame_tolerance=2 (±10ms) and good_pct=0.03 (3%) for the soft target distribution.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_46a --model-type event_embed --hard-alpha 0.0 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_46b --model-type event_embed --hard-alpha 0.25 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_46c --model-type event_embed --hard-alpha 0.75 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
python detection_train.py taiko_v2 --run-name detect_experiment_46d --model-type event_embed --hard-alpha 1.0 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

### Predictions

- **46-A (pure soft):** Higher HIT than exp 44 (soft targets directly optimize tolerance), but lower exact accuracy and possibly worse metronome behavior (more forgiving = easier to continue patterns).
- **46-B (0.25):** Slightly better HIT than exp 44, slight exact accuracy drop. Sweet spot candidate.
- **46-C (0.75):** Slightly lower HIT, better exact accuracy. Could help with metronome if sharper gradients force more decisive predictions at break points.
- **46-D (pure hard):** Similar to exp 42-B — lower HIT, higher precision. May recover if given enough training time (42-B only had 2 evals).

### Key metrics to watch

- Exact match vs HIT gap — does hard CE close the 19pp gap?
- Metronome benchmark — does sharper loss help break patterns?
- pred_continues_target_breaks — the 11.8% metronome failure rate from exp 44
- AR step1+ — does precision help or hurt cascade?

## Result

*Pending*

## Lesson

*Pending*
