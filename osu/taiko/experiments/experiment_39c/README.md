# Experiment 39-C - Entropy-Weighted Reranking

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp 39-B showed global proximity reranking only gains +0.9pp because it regresses 1,299 correct predictions. The fix: only apply proximity bias when the model is uncertain.

**Entropy weight** downweights candidates further from #1 in confidence. When the model is confident (top-1 >> top-2), the entropy weight keeps the original pick. When hedging (top-1 ≈ top-2), it lets proximity tip the balance.

```
score = conf_w * confidence + pos_w * (1 - bin/500) + ent_w * (confidence / top1_confidence)
```

The third term `ent_w * (confidence / top1_confidence)` acts as a confidence-relative bonus. High ent_w = strongly prefer candidates close to #1 in confidence (conservative). Low ent_w = let position override even when #1 is more confident (aggressive).

## Result

**Barely better than 39-B — entropy weight doesn't help.**

```
Baseline: 71.6%
Best: 72.6% (conf=0.7, pos=3.0, ent=0.5)
Improvement: +1.0%
  Improved: 2,269
  Regressed: 1,559
  Net: +710
```

The entropy weight adds only +0.1pp over 39-B's +0.9pp. The regression ratio (~2:3) is unchanged. All weight combinations cluster around the same +0.9-1.0pp ceiling.

## Lesson

- **Post-hoc reranking of the model's own top-K is fundamentally limited to ~+1pp.** The model's confidence ranking is already strongly correlated with its predictions — any scoring function based on these confidences can only marginally improve selection.
- **The 14.9pp theoretical ceiling requires oracle knowledge** (which candidate is nearest to cursor), not available from the model's output alone.
- **The path forward is either**: train the model differently (to prefer nearest), or use beam search at inference (explore multiple paths).
