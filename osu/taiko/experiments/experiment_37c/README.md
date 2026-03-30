# Experiment 37-C - Focal Dice Multi-Target

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp 37 and 37-B proved that sigmoid BCE with soft targets is fundamentally broken for this task — the model either predicts nothing (with focal) or everything (without). The per-bin BCE loss doesn't incentivize sparsity; it just minimizes per-bin error, which is satisfied by moderate activation everywhere.

**Dice loss** directly optimizes set overlap:
```
dice = 2 * sum(pred * target) / (sum(pred) + sum(target))
loss = 1 - dice
```

Key properties:
- **Predicting everything → low dice** (numerator capped by target sum, denominator explodes)
- **Predicting nothing → zero dice** (zero numerator)
- **Natural balance** — the loss is a ratio, not a sum, so class imbalance doesn't matter
- **No pos_weight or focal tuning needed** — the overlap metric handles sparsity inherently

This is the standard loss for sparse segmentation (medical imaging, where tumors are <1% of pixels). Same structure as our problem: sparse onsets in mostly-empty bins.

A small BCE component (10% weight) is added for stable per-bin gradients — pure dice can have vanishing gradients for bins far from any onset.

### Changes from exp 37-B

- **Loss**: `FocalDiceMultiTargetLoss` (dice_weight=1.0, bce_weight=0.1)
- No pos_weight, no focal_gamma — dice handles balance inherently
- Everything else identical (sigmoid output, mel ramps, multi-target dataset)

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_37c --model-type unified --multi-target --dice-loss --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Dice loss learns slowly — 5.4% HIT at 15% through epoch 1.** Killed early. The architecture is the bottleneck, not the loss.

Train metrics at kill: HIT=5.4%, miss=93.7%, score=-0.801

Dice loss does avoid the over/underprediction extremes of sigmoid BCE, but the fundamental issue remains: extracting a single cursor vector at position 125 and projecting to 501 independent sigmoid outputs is not the right architecture for multi-target detection. The cursor was designed for "predict the next onset" — asking it to simultaneously represent all onsets in the window is asking one feature vector to do too much.

## Lesson

- **Dice loss shows promise** (avoids the extremes of sigmoid BCE) but is slow — the overlap signal is weak when predictions are initially random.
- **The architecture is wrong for multi-target.** Experiments 36-37C all attempted multi-target by changing only the loss/output interpretation. The model (single cursor → 501 logits) was never designed for this. Multi-target needs a fundamentally different output formulation.
- **Three experiments (37, 37-B, 37-C) with three different losses all fail to produce reasonable multi-target predictions from a single cursor token.** The problem is structural.
- **Exp 38 should either**: (A) redesign the architecture for multi-target (framewise detection), or (B) return to single-target (which works at 71.6%) and improve from there.
