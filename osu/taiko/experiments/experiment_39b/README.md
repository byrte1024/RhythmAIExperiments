# Experiment 39-B - Top-K Reranking by Confidence × Proximity

## Hypothesis

Exp 39 proved 83.2% of overpredictions match real future onsets. The model sees valid onsets but picks the wrong one (further instead of nearest). The theoretical HIT if we count these is 86.5%.

**Rerank the top-K candidates** by a weighted score combining confidence rank and position proximity:
```
score = conf_weight * normalized_confidence + pos_weight * (1 - bin/500)
```

Higher position weight = prefer closer predictions. Higher confidence weight = trust the model's ranking. Sweep different weight combinations to find the sweet spot that maximizes HIT rate.

Also track regression: how many current HITs become misses from reranking (we don't want to break what works).

### Method

1. Run 35-C eval 8 on val set, collect top-10 predictions with probabilities
2. For each weight combination, rerank top-10 and pick the highest-scoring candidate
3. Measure: HIT rate, improvements (miss→hit), regressions (hit→miss), net gain

## Result

**Only +0.9pp from reranking — blunt proximity bias causes too many regressions.**

```
Baseline (argmax): 71.6%
Best reranked: 72.5% (conf_w=0.5, pos_w=2.0)
Improvement: +0.9%
  Improved: 1,994 (miss→hit)
  Regressed: 1,299 (hit→miss)
  Net: +695
```

The theoretical ceiling from exp 39 is +14.9pp, but reranking only recovers +0.9pp. The problem: a global position weight helps overpredictions but hurts correct distant predictions. When the nearest onset IS far, the proximity bias pushes toward a spurious close candidate.

**conf_w=0 is catastrophic** (-59.3%) — pure proximity picks the smallest bin regardless of confidence, destroying everything.

## Lesson

- **Global reranking is too blunt** — 1,299 regressions per 1,994 improvements. The proximity bias can't distinguish "model is wrong, pick closer" from "model is right about a distant onset."
- **Need confidence-aware reranking** — only apply proximity bias when the model is uncertain (top candidates are close in confidence). When confident, trust the model.
- **Next: add entropy/uncertainty weighting** — downweight candidates farther from #1 in confidence, so proximity only matters when the model is hedging.
