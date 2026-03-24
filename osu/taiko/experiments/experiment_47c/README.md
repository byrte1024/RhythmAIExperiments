# Experiment 47-C - Binary STOP Head with Balanced Focal Loss

## Hypothesis

Same architecture as 47/47-B. Fixes:
- **47 bug:** pos_weight backwards (onset upweighted instead of STOP)
- **47-B bug:** focal loss with `.mean()` — STOP contributes 0.3% of the mean, invisible to optimizer even with focal weighting

**Fix:** Average STOP and onset focal losses separately, then combine with equal weight. This ensures STOP loss has equal standing regardless of class ratio:
```
gate_loss = (mean_focal_onset + mean_focal_stop) / 2
```

With this, a batch of 997 onsets + 3 STOPs produces gate_loss where STOP contributes 50%, not 0.3%.

### Changes from 47-B
- Separate averaging of STOP vs onset focal losses
- Added gate loss (gL) and stop F1 (sF1) to tqdm training bar
- Added stop_pred_rate, stop_target_rate, stop_hallucinations, stop_misses to eval metrics

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_47c --model-type event_embed --binary-stop --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
