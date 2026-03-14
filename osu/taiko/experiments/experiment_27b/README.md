# Experiment 27-B - Context Pattern Analysis (Diagnostic)

## Hypothesis

Exp 27's best model (eval 8, 69.8% HIT) shows systematic over-prediction: when the correct answer is 75, it predicts 150. Top-3 accuracy is ~90%, meaning the model narrows to a small set of rhythmically valid candidates but picks wrong. Forward error shows predictions cluster at meaningful ratios (1.0x, 0.5x, 2.0x) — the model knows the rhythm vocabulary but can't disambiguate.

**If the model could detect and continue repeating gap patterns from its context, how many misses would it fix?**

For a pattern like `150 150 75 75 150 150 75 75 ?`, the answer `75` is the next element in a repeating `[150 150 75 75]` cycle. The model doesn't need complex reasoning — just pattern detection and continuation.

### Method

Two scripts, run sequentially:

**1. `run_predictions.py`** — general-purpose val inference, saves all predictions + context to `.npz`
```bash
python run_predictions.py taiko_v2 --checkpoint runs/detect_experiment_27/checkpoints/epoch_008.pt --subsample 8
```

**2. `analyze_context.py`** — loads the `.npz` and runs pattern analysis
```bash
python analyze_context.py runs/detect_experiment_27/predictions_epoch_008_sub8.npz
```

Pattern detection logic:
1. For every sample, extract the gap sequence from context (inter-onset intervals)
2. Detect repeating patterns: for each candidate length L ≥ 4, check if the last L gaps repeat at least once earlier in context (≥80% element match to tolerate slight timing variation)
3. If a pattern is found, predict the next element in the cycle
4. Compare: model prediction vs pattern prediction vs ground truth

### What counts as a "pattern"

- Minimum 4 gaps long
- Must repeat at least once in the context (so ≥ 2L gaps needed)
- Elements match within HIT tolerance (≤3% or ±1 frame) to handle slight timing drift
- Aligned to cursor position — pattern[0] is the prediction for the next gap
- When multiple pattern lengths match, prefer the one with more repetitions (more evidence)

### Expected outcome

~90% of misses with sufficient context should be solvable by pattern continuation. Most taiko patterns are repetitive (streams of equal gaps, alternating patterns like `75 75 150 75 75 150`), so the correct answer almost always continues a visible cycle.

## Result

*Pending*

## Lesson

*Pending*
