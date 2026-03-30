# Experiment 52 - Audio Window Size Sweep

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
```

## Result

*Pending*

## Lesson

*Pending*
