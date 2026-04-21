# Experiment 67b — Ratio Prediction with Conv1d Smoothing

## Purpose

Exp 67 showed the ratio head collapses to a few discrete values (horizontal banding in scatter plot), despite 255 available bins. The model snaps to ~5-10 favorite ratios instead of using the full log-spaced range. This creates blurry frame-level predictions due to lossy multiplication.

**Fix:** Add Conv1d smoothing to the ratio head output (same technique used on the onset logits output head since exp42). This forces neighboring ratio bins to have correlated values, preventing isolated spikes.

## Change from Exp 67

Only one change: `ratio_smooth = Conv1d(1→8→1, kernel=5)` added to RatioHeads. Applied as residual: `ratio_logits = ratio_logits + ratio_smooth(ratio_logits)`.

This is the same smoothing that the backbone's onset head uses (`head_smooth` in ProposeSelectDetector). It was shown in exp 31→32 to resolve banding by correlating neighboring output bins.

Everything else identical to exp 67 (same backbone, same loss, same warmup, same augmentation).

## Launch

```bash
cd osu/taiko
python detection_train.py taiko_v2 --run-name detect_experiment_67b \
    --model-type ratio_propose_select \
    --a-bins 500 --b-pred 250 --gap-ratios \
    --density-jitter-rate 0.30 --density-jitter-pct 0.10 \
    --proposer-freeze-evals 0 \
    --ratio-freeze-evals 0 \
    --warm-start runs/detect_experiment_67/checkpoints/eval_001.pt \
    --ramp-alpha 2.5 \
    --epochs 50 --batch-size 48 --evals-per-epoch 4 --workers 3
```

## Expected Behavior

- Ratio scatter should show smoother distribution, fewer horizontal bands
- Frame-level heatmap should be sharper (less blur from quantized ratios)
- Derived HIT should improve (smoother ratios → more precise final positions)
- Divisor accuracy should be similar to exp67 (~65%+ at eval 2)

## Result

*Pending*

## Lesson

*Pending*
