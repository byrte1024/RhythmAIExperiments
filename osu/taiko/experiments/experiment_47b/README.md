# Experiment 47-B - Binary STOP Head with Focal Loss

## Hypothesis

Same as exp 47 — separate binary gate head for STOP prediction. Fixes from 47:
- **Flipped gate targets:** 1=stop, 0=onset (STOP is the positive/rare class)
- **Focal BCE (gamma=2):** Downweights easy negatives (confident onset predictions), focuses on hard positives (STOP boundaries). Designed for extreme class imbalance (0.3% STOP).

### Architecture

Same as exp 47: EventEmbeddingDetector with binary_stop=True.
- Gate head: cursor token → MLP → 1 logit → sigmoid → P(stop)
- Onset head: cursor token → 500-class softmax (onset bins only)
- Gate loss: focal BCE (gamma=2), weight controlled by --gate-weight
- Onset loss: standard CE on non-STOP samples only

Additional logging: stop_pred_rate, stop_target_rate, stop_hallucinations, stop_misses.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_47b --model-type event_embed --binary-stop --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
