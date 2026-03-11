# Experiment 21 - Relative Quality Selection Loss

## Hypothesis

Exp 20 proved warm-start + freeze are solid infrastructure (69.5% HIT from step 1, 2x speed), and that the gap-based context architecture (exp 19) is correct. But context's overrides are net-harmful (delta -1.18pp) despite best-ever override F1 (22%). The bottleneck is the loss function.

**The problem with hard CE on selection:** Hard cross-entropy rewards only exact correctness — "pick the candidate closest to the true target." This creates three failure modes:

1. **No reward for improvement.** If audio's #1 is 15 frames off and context picks a candidate 3 frames off, hard CE gives the same loss as if context had picked #1. No signal that the override was valuable.
2. **No extra punishment for regression.** If audio's #1 is 1 frame off and context picks something 50 frames off, hard CE gives the same loss as any other wrong answer. No signal that the override was catastrophic.
3. **Missed opportunities are invisible.** When context keeps #1 (no override) but #1 was very wrong and a much better candidate existed, hard CE gives moderate loss. Context learns "keeping #1 is safe" because it avoids the worst-case of picking the wrong override.

The result: "always pick #1" is a stable local minimum under hard CE, since #1 is correct ~70% of the time.

### Changes from exp 20

**1. SelectionLoss (replaces hard CE for K-way selection)**

A relative quality loss that operates in the same trapezoid ratio space as OnsetLoss:

**Step 1 — Quality scoring:** For each of K=20 candidates, compute quality = closeness to true target using the trapezoid (1.0 within 3%, linear ramp to 0 at 20%, frame floor ±1).

**Step 2 — Relative soft targets:** Build a probability distribution over K candidates:
- Candidates at or above audio's #1 quality: weight = their quality score
- Candidates below #1 quality: weight = 0 (suppressed)
- Normalize to sum to 1

This means: "pick the best available, and any candidate as good as or better than #1 gets some credit." If #1 is already the best, soft target peaks at #1 (reward keeping). If k=5 is closer to target than #1, soft target peaks at k=5 (reward overriding).

**Step 3 — Soft CE:** Standard cross-entropy against the soft target distribution. Context is trained to match the quality-weighted distribution, not a hard one-hot.

**Step 4 — Asymmetric miss penalty:** After computing per-sample loss, scale up by `miss_penalty` (2.0x) when:
- Context chose to keep #1 (no override)
- #1 was bad (quality < 0.5)
- A significantly better candidate existed (best quality > #1 quality + 0.1)

This explicitly punishes conservatism when overriding would have helped.

**Step 5 — Skip impossible samples:** If no candidate has any quality (all zero weight), the sample is excluded from the loss. No impossible training signal.

**2. Same infrastructure as exp 20**
- Warm-start from exp 14 best checkpoint
- Freeze all audio components
- Only train 2.5M context params

### Architecture

Identical to exp 20 (gap-based context with own encoders).

| Component | Params | Training |
|-----------|--------|----------|
| AudioEncoder | 8.0M | **Frozen** (from exp 14) |
| EventEncoder | 0.5M | **Frozen** (from exp 14) |
| AudioPath | 5.0M | **Frozen** (from exp 14) |
| cond_mlp | ~8K | **Frozen** (from exp 14) |
| Context gap encoder | 0.9M | Training |
| Context snippet encoder | 0.2M | Training |
| Context selection head | 1.2M | Training |
| Context scoring | 0.025M | Training |
| **Total trainable** | **2.5M** | |

### Loss comparison

| Aspect | Exp 20 (hard CE) | Exp 21 (SelectionLoss) |
|--------|-------------------|------------------------|
| Target | One-hot at closest candidate | Soft distribution peaked at best, suppressing below-baseline |
| "Better than #1" | Same loss as any correct pick | High weight in soft target (rewarded) |
| "Worse than #1" | Same loss as any wrong pick | Zero weight (suppressed) |
| "Kept #1 when wrong" | Moderate loss | 2x loss (miss_penalty) |
| No good candidate | Trains on least-wrong | Skipped (no signal) |

### Expected outcomes

1. **Audio HIT = 69.5%** — frozen, identical to exp 20.
2. **Context delta > 0** — the relative loss directly rewards improvement over #1. Even small positive delta would be a breakthrough.
3. **Override accuracy > 50%** — context should learn to override only when it has a better candidate, not randomly.
4. **Fewer false_topK** — suppressing below-baseline candidates should reduce bad overrides.
5. **More true_topK** — miss_penalty should push context to override when #1 is wrong and a better option exists.
6. **Override F1 increasing over epochs** — unlike previous experiments where F1 declined or plateaued.

### Risk

- The soft target distribution may be too flat when multiple candidates are near-equal quality, giving weak gradient signal.
- miss_penalty=2.0 may be too aggressive, causing context to override too often (high false_topK). Or too mild to break the "keep #1" habit.
- Asymmetric scaling based on argmax (what context "chose") creates a non-differentiable dependency — the scale factor is a step function of the logits. This could cause instability if context is near the decision boundary.
- The quality threshold for "missed opportunity" (baseline < 0.5, best > baseline + 0.1) may not match the actual distribution of override opportunities well.

### Command

```bash
python detection_train.py \
  --name experiment_21 \
  --warm-start runs/detect_experiment_14/checkpoints/best.pt \
  --freeze-audio \
  --miss-penalty 2.0 \
  --epochs 50
```

## Result

*Pending*
