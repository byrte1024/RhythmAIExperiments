# Experiment 35-D - Exponential Ramps + Focal Loss (gamma=3)

## Hypothesis

Exp 35-C achieved 71.6% HIT with 4.5% sustained context delta — the first breakthrough past 70%. But two issues remain:
1. **2.0x error band** — the model still frequently predicts double the correct value. These are the pattern disambiguation cases where context should help.
2. **High entropy** — predictions are uncertain even when correct.

**Focal loss (gamma=3.0)** should directly target both issues:
- Downweights the easy ~70% of predictions (confident, correct from audio alone), focusing gradient on the hard 30% where the 2.0x confusion happens
- Exp 28 (gamma=2.0) proved focal loss improves entropy calibration (cleaner HIT/MISS separation)
- Gamma=3.0 is more aggressive than exp 28's 2.0 — stronger focus on hard cases
- Combined with mel ramps, the hard cases now have context signal available. Exp 28 failed to improve HIT because context wasn't available; now it is.

### Changes from exp 35-C

**focal_gamma: 0 → 3.0.** Everything else identical (exponential ramps, amplitude jitter 0.25-0.75).

### Expected outcomes

1. **Better HIT on hard cases** — focal loss focuses on the 2.0x confusion cases, and context (via ramps) provides the disambiguation signal.
2. **Lower entropy** — proven from exp 28 that focal loss improves calibration.
3. **Possibly slower convergence** — focal loss makes easy samples contribute less gradient.
4. **Context delta possibly higher** — if hard cases are where context matters most, focusing on them should increase context contribution.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_35d --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --focal-gamma 3.0 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
