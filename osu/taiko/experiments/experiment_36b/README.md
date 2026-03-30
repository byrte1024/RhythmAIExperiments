# Experiment 36-B - Multi-Target with Per-Onset Recall Loss

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp 36 showed multi-target training doesn't hurt nearest-target HIT (66.2% = 35-C) but event recall was only 8.2%. Root cause: the normalized soft targets dilute per-onset gradient — missing one of 5 onsets costs only 1/5 of the soft loss. The model rationally underpredicts.

**Per-onset recall loss** directly penalizes `-log(prob[target_bin])` for EACH real onset independently. Every missed onset costs the same regardless of how many others exist in the window. Combined with the existing soft CE and hard CE:

```
loss = hard_alpha * hard_CE(nearest)
     + (1 - hard_alpha) * soft_CE(multi_target_distribution)
     + recall_weight * mean(-log(prob[onset_i]) for each real onset)
```

The recall term creates direct gradient pressure at each real onset bin — the model must put probability mass there or pay.

### Changes from exp 36

- **recall_weight=1.0** (new) — per-onset `-log(prob)` loss, equal weight to the CE terms
- **Threshold sweep optimized** — 11 thresholds (was 42) + 4x subsampling + tqdm progress bar
- Everything else identical (multi-target, exponential ramps, amplitude jitter)

## Result

**Recall loss improves precision but doesn't fix conservatism. Softmax is the wrong output for multi-target.** Killed after eval 1.

| Metric | 36-B eval1 | 36 eval1 |
|--------|------------|----------|
| Nearest HIT | 64.7% (-1.5pp) | 66.2% |
| Event recall | 9.5% (+1.3pp) | 8.2% |
| Pred precision | 63.2% (+10.8pp) | 52.4% |
| F1 | 0.165 (+0.023) | 0.142 |
| Hallucination | 3.6% | 1.0% |
| Preds/win | 2.4 | 2.6 |
| Ctx Δ | 0.9% | ~4.1% |

Precision improved significantly but recall barely moved. The model makes fewer, better predictions but still misses 90% of onsets.

**Root cause: softmax can't represent multiple targets.** Pushing probability up at bin 35 pushes it down at bin 70 — both real targets compete for the same probability mass. No loss function can fix this; the output formulation is fundamentally wrong for multi-target prediction.

## Lesson

- **Per-onset recall loss works for precision** (+10.8pp) but can't overcome the softmax bottleneck for recall.
- **Softmax is single-target by design.** Multi-target requires per-bin independent outputs (sigmoid) or a fundamentally different output architecture.
- **The loss-only approach (36, 36-B) proves that multi-target needs an architecture change**, not just a loss change.
