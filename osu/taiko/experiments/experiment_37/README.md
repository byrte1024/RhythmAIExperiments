# Experiment 37 - Per-Bin Sigmoid Multi-Target

## Hypothesis

Experiments 36 and 36-B proved that softmax is fundamentally wrong for multi-target: bins compete for probability mass, preventing the model from predicting multiple onsets simultaneously. Per-onset recall loss improved precision but couldn't overcome the softmax bottleneck.

**Per-bin sigmoid** replaces the softmax with 501 independent binary classifiers. Each bin independently predicts P(onset here) with no competition between bins. The model can say "YES at bin 35 AND YES at bin 70" simultaneously.

The loss uses the same log-ratio trapezoid soft labels from OnsetLoss, but as per-bin BCE targets instead of a probability distribution:
- Bins near a real onset get target ≈ 1.0 (within good_pct)
- Bins in the ramp zone get interpolated targets (between good_pct and fail_pct)
- Bins far from any onset get target = 0.0
- Positive bins are upweighted (pos_weight=5.0) since onsets are sparse (~3-5 per 500 bins)

### Architecture

Identical model (OnsetDetector with mel ramps). The output head still produces (B, 501) logits — the interpretation changes from softmax to sigmoid. No architectural changes.

### Changes from exp 36-B

- **Loss**: `SigmoidMultiTargetLoss` replaces `MultiTargetOnsetLoss`. Per-bin BCE instead of softmax CE.
- **Probabilities**: `sigmoid(logits)` instead of `softmax(logits)` — each bin is independent.
- **pos_weight=5.0**: upweights positive bins to handle class imbalance (few onsets per window).
- **focal_gamma=2.0**: focal loss modulation on the per-bin BCE. Downweights easy negatives (~495 bins per sample that are confidently 0), focuses on hard cases (onset bins the model is uncertain about). This is the original RetinaNet use case — sparse detection with massive class imbalance.
- Everything else identical (exponential ramps, amplitude jitter, multi-target dataset).

### Expected outcomes

1. **Higher event recall** — bins don't compete, so the model can fire multiple peaks without suppressing others.
2. **Maintained precision** — the log-ratio trapezoid targets still guide predictions to be near real onsets.
3. **Nearest-target HIT maintained** — the highest-confidence bin should still predict the nearest onset accurately.
4. **Threshold sensitivity may differ** — sigmoid outputs are absolute (not relative like softmax). The 0.05 threshold may need adjustment.

### Risk

- Sigmoid outputs can activate many bins simultaneously → potential hallucination explosion if pos_weight is too high.
- No normalization means the model can predict 0 or 100 onsets per window with no penalty for being "too many." The empty_weight partially addresses this for empty windows.
- The log-ratio trapezoid targets create smooth target distributions — many bins get partial positive labels, which could make the sigmoid overly active.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_37 --model-type unified --multi-target --sigmoid-loss --focal-gamma 2.0 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --pos-weight 5.0 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
