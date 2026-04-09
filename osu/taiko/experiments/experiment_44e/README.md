# Experiment 44e - Distance Ramp Loss

> **[Full Architecture Specification](../experiment_44/ARCHITECTURE.md)** — identical architecture to experiment 44.

## Hypothesis

Our current loss has a flat plateau for far-off predictions: predicting 2x the target and 4x the target produce the exact same loss. This means the model gets zero gradient signal to improve from "very wrong" to "less wrong." The only gradient comes from the trapezoid zone (±20% of target).

Adding a third loss component — a gradual ramp in log-ratio space — gives gradient everywhere. Predictions farther from the target in ratio space get progressively higher loss.

### 3-Component Loss

```
loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE + ramp_alpha * |log(pred/target)|^ramp_exp
```

| Component | Role | Gradient range |
|---|---|---|
| Hard CE (0.5) | Exact bin precision | Only at target bin |
| Soft CE (0.5) | Trapezoid neighborhood | ±20% of target (ratio) |
| Ramp (2.5) | Distance penalty | Everywhere (∞) |

The ramp uses the model's expected prediction (softmax center of mass) in log-ratio space: `|log((E[pred]+1)/(target+1))|`. This is:
- 0 at exact match
- Symmetric in ratio space (2x over = 0.5x under)
- Proportional across all target values
- Differentiable everywhere

### Configuration

| Feature | Value |
|---|---|
| Architecture | EventEmbeddingDetector (same as exp 44) |
| hard_alpha | 0.5 (same as exp 44) |
| ramp_alpha | **2.5** (NEW) |
| ramp_exp | 1.0 (linear in log-ratio) |
| Density jitter | ±10% at 30% (same as exp 44) |
| No gap ratios | Same as exp 44 |

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_44e --model-type event_embed --no-gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --ramp-alpha 2.5 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Expected Behavior

- Per-sample HIT% should be similar or slightly better than exp 44 (73.6%)
- The model should make fewer extreme errors (2x/4x overshoots)
- Overprediction rate (from exp 39: 83% of errors are overshoots) may decrease
- Training loss will be higher due to the ramp component — compare val metrics, not raw loss

## Result

*Pending*

## Lesson

*Pending*
