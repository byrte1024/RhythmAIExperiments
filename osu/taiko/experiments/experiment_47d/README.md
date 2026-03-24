# Experiment 47-D - Binary STOP Head with Forward-Pool Gate

## Hypothesis

47-C showed the gate can't learn from the cursor token alone. The STOP decision requires looking at the whole forward region.

### Architecture change

Gate reads from mean-pooled forward tokens (126-249) with its own LayerNorm instead of the cursor token.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_47d --model-type event_embed --binary-stop --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Failed immediately.** Gate loss 0.063, sF1=0.09 at 1% through epoch 1. No improvement over time.

The forward-pool approach didn't help. Mean-pooling 124 abstract tokens is too lossy, and gate_weight=2.0 still makes the gradient negligible.

## Lesson

**The binary head approach is fundamentally flawed for this problem.** After 4 iterations (47, 47-B, 47-C, 47-D), every variant failed:

| Exp | Issue | Result |
|---|---|---|
| 47 | pos_weight backwards | STOP rate 0% |
| 47-B | focal loss mean() drowns STOP | F1=0.066, recall=3% |
| 47-C | gate_weight too low, cursor token not right | F1=0.15, stuck |
| 47-D | forward-pool gate, still low weight | F1=0.09, stuck |

The root cause isn't loss weighting or read-out position — it's that **STOP isn't a separate decision from onset prediction.** In the softmax, STOP naturally wins when no onset bin has high confidence — it's the "default when uncertain." A binary gate has to actively learn to fire against a 99.7% onset base rate, and no amount of loss engineering overcame this.

The existing softmax STOP (F1=0.52) works better than every binary head variant because it leverages elimination: "no strong onset candidate = STOP."

**Next direction:** Improve STOP within the existing softmax — STOP augmentation (zero forward audio, force STOP target), higher stop_weight, and/or inference-time audio energy gating.
