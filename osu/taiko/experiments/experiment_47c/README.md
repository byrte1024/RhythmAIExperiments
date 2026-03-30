# Experiment 47-C - Binary STOP Head with Balanced Focal Loss

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Same architecture as [47](../experiment_47/README.md)/[47-B](../experiment_47b/README.md). Fixes:
- **[47](../experiment_47/README.md) bug:** pos_weight backwards (onset upweighted instead of STOP)
- **[47-B](../experiment_47b/README.md) bug:** focal loss with `.mean()` -- STOP contributes 0.3% of the mean, invisible to optimizer even with focal weighting

**Fix:** Average STOP and onset focal losses separately, then combine with equal weight. This ensures STOP loss has equal standing regardless of class ratio:
```
gate_loss = (mean_focal_onset + mean_focal_stop) / 2
```

With this, a batch of 997 onsets + 3 STOPs produces gate_loss where STOP contributes 50%, not 0.3%.

### Changes from [47-B](../experiment_47b/README.md)
- Separate averaging of STOP vs onset focal losses
- Added gate loss (gL) and stop F1 (sF1) to tqdm training bar
- Added stop_pred_rate, stop_target_rate, stop_hallucinations, stop_misses to eval metrics

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_47c --model-type event_embed --binary-stop --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Gate loss stuck at ~0.05, sF1 stuck at 0.15.** The balanced averaging fixed the per-class gradient ratio, but gate_weight=2.0 makes the total gate contribution (~0.1) negligible vs onset loss (~3.8). The optimizer still can't see the gate.

## Lesson

- **Balanced focal averaging works per-class but the absolute scale still matters.** Gate loss ~0.05 x weight 2.0 = 0.1 contribution to total loss ~4.0.
- **Deeper issue:** Both gate and onset head read from the same cursor token (125). The gate is a tiny afterthought on a representation optimized for onset location. The STOP decision ("is there any onset ahead?") is a global question about the forward window, not a point-read at the cursor.
- **Next ([47-D](../experiment_47d/README.md)):** Gate reads from mean-pooled forward tokens (126-249) with its own LayerNorm, not the cursor token.
