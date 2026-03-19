# Experiment 39-C - Entropy-Weighted Reranking

## Hypothesis

Exp 39-B showed global proximity reranking only gains +0.9pp because it regresses 1,299 correct predictions. The fix: only apply proximity bias when the model is uncertain.

**Entropy weight** downweights candidates further from #1 in confidence. When the model is confident (top-1 >> top-2), the entropy weight keeps the original pick. When hedging (top-1 ≈ top-2), it lets proximity tip the balance.

```
score = conf_w * confidence + pos_w * (1 - bin/500) + ent_w * (confidence / top1_confidence)
```

The third term `ent_w * (confidence / top1_confidence)` acts as a confidence-relative bonus. High ent_w = strongly prefer candidates close to #1 in confidence (conservative). Low ent_w = let position override even when #1 is more confident (aggressive).

## Result

*Pending*

## Lesson

*Pending*
