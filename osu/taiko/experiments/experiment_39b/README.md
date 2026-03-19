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

*Pending*

## Lesson

*Pending*
