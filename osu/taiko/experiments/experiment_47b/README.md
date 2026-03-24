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

**Stopped at eval 1. Stop pred rate 0.000% — gate almost never fires.**

Stop F1=0.066, precision=0.93, recall=0.03. The gate is extremely conservative — when it does predict STOP it's right, but it almost never does.

Root cause: focal BCE with `.mean()` reduction. With 99.7% onset samples, the mean is dominated by onset loss. Focal loss correctly downweights easy onsets, but the total gate loss (~0.007) is tiny compared to onset CE loss (~2.5). With gate_weight=2.0, the gate contributes 0.014 to a total of ~2.5 — the optimizer can't see it.

The math confirms focal is working correctly per-sample (STOP gets 2.75x more gradient than onset), but the 300:1 class ratio means STOP's contribution is still invisible after averaging.

## Lesson

- **Focal loss helps per-sample but doesn't solve the averaging problem.** With 0.3% STOP, even with focal weighting, the mean loss is dominated by the 99.7% onset class.
- **Fix for 47-C:** Average STOP and onset focal losses separately, then combine. This ensures STOP loss has equal standing regardless of class ratio.
