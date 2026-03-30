# Experiment 41-B — Diagnostic Analysis Specification

## Type

Analysis / diagnostic experiment. No model training. Tracks how skip rate, entropy, and accuracy evolve over training by analyzing multiple checkpoints from a single training run.

## Purpose

Determine whether the skip-1 failure mode (identified in exp 41) improves with more training. If the overprediction rate decreases from early to late checkpoints, the model is gradually learning to prefer the nearest onset and longer training would help. If it is flat, the behavior is structural and architectural changes are needed.

## Model Analyzed

Exp 35-C checkpoints at three stages: eval 1, eval 4, eval 8. OnsetDetector (unified).

## Data Analyzed

Validation set predictions from three exp 35-C checkpoints. For each checkpoint, the same analysis as exp 41:
- Skip count distribution
- Entropy statistics
- Confidence statistics
- HIT rate by skip count
- HIT rate by target distance range
- Unique predictions per step

## Method

1. Run the entropy/skip analysis from exp 41 on three checkpoints: eval 1, eval 4, eval 8
2. Compare all metrics side by side across training stages
3. Track: overall HIT rate, mean entropy, mean confidence, skip-0 HIT, overprediction rate, distance-binned HIT rates

## Scripts

| Script | Location | Purpose |
|---|---|---|
| analyze_entropy.py | `osu/taiko/analyze_entropy.py` | Runs val pass on a checkpoint, computes per-sample entropy, skip count, correlations |
| analyze_entropy_progression.py | `osu/taiko/analyze_entropy_progression.py` | Runs analysis on multiple checkpoints and compares trends |

## Key Findings

- **Entropy and confidence improve with training** — mean entropy drops from 2.690 (eval 1) to 2.391 (eval 8)
- **Skip-0 accuracy improves**: 91.0% to 93.7% — model gets better at cases where it doesn't skip
- **Overprediction rate is flat at ~28%** — the model does not learn to skip less
- **Skip-1 rate is structural at ~11%** throughout training — does not decrease
- **Convergence is slowing**: eval 1-to-4 gained +3.6pp HIT, eval 4-to-8 gained only +1.7pp
- **Distant predictions (100-500)**: volatile, no improvement trend
- Estimated HIT ceiling with current architecture: ~75-78%
