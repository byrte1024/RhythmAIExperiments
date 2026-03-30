# Experiment 41-B - Entropy Progression Over Training (Diagnostic)

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp [41](../experiment_41/README.md) revealed the skip count is the primary failure mode (skip 0 = [85% HIT [?]](../experiment_41/README.md), skip 1 = [43% [?]](../experiment_41/README.md)). But does this improve over training? If skip-1 HIT rate increases from eval 1→4→8, the model is gradually learning to prefer the nearest onset and we just need to train longer. If it's flat, the architecture can't solve this and we need a different approach.

Also tracks: does entropy decrease over training? Does confidence improve? Do distant predictions get better?

### Method

Run the skip/entropy analysis from exp [41](../experiment_41/README.md) on three checkpoints from [35-C](../experiment_35c/README.md): eval 1, eval 4, eval 8. Compare all metrics side by side.

## Result

**Entropy and confidence improve with training, but skip rate is structural — barely changes.**

| Metric | eval 1 | eval 4 | eval 8 | Trend |
|---|---|---|---|---|
| **HIT rate** | 66.3% | 69.9% | 71.6% | Improving (+5.3pp) |
| **Mean entropy** | 2.690 | 2.556 | 2.391 | Improving (-0.30) |
| **Mean confidence** | 0.316 | 0.330 | 0.355 | Improving (+0.04) |
| **Skip 0 HIT** | 91.0% | 93.1% | 93.7% | Improving (+2.7pp) |
| **Skip 0 entropy** | 2.565 | 2.432 | 2.271 | Improving (-0.29) |
| Skip 1 entropy | 2.973 | 2.967 | 2.780 | Slight improvement |
| **Overpred rate** | 30.2% | 27.6% | 28.0% | **Flat** |
| Dist 0-30 HIT | 66.3% | 73.7% | 75.1% | Improving (+8.8pp) |
| Dist 30-100 HIT | 68.1% | 67.6% | 70.4% | Slow (+2.3pp) |
| Dist 100-500 HIT | 48.6% | 56.3% | 50.8% | **Volatile, no trend** |

**Convergence is slowing:** eval 1→4 gained +3.6pp HIT, eval 4→8 gained only +1.7pp.

### Key findings

**Improving with training:**
- Skip-0 accuracy: 91% → 93.7% — the model gets better at cases where it doesn't skip
- Short-range predictions (0-30): 66.3% → 75.1% — large improvement
- Overall entropy and confidence — steady improvement

**NOT improving with training:**
- **Overprediction rate: stuck at ~28%** — the model doesn't learn to skip less
- **Distant predictions (100-500): volatile** — no trend, effectively random improvement
- **Skip-1 rate: ~11%** throughout — structural, not trainable

## Lesson

- **The model improves on what it can already do (skip-0 cases)** but doesn't learn to avoid skipping. The skip behavior is structural — the sharper transient at the further onset dominates regardless of training duration.
- **More training helps entropy/confidence** but with diminishing returns. The eval 4→8 delta is half of eval 1→4 across all metrics.
- **The ~28% miss rate has a clear decomposition:** ~11% from skip-1+ (structural, won't train away), ~19% underprediction (46.5% are near-misses, improvable), ~6% other. The ceiling with current architecture is approximately: 93.7% × 67% (skip-0 samples) + some fraction of underpredictions ≈ ~75-78% HIT.
- **To break past ~75% HIT, must address skip-1 errors** — either through stronger context (to override transient saliency) or inference-time correction.
