# Experiment 50-B - Anti-Entropy Loss (Weight 0.5)

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp 50 (weight 0.1) produced a sidegrade: better resilience, slightly lower HIT. The pressure was enough to improve robustness but too gentle to force better selection. At 0.1, entropy of ~2.3 nats adds ~0.23 to loss (~8%).

At weight 0.5, entropy adds ~1.15 to loss (~40% of onset loss). This is aggressive — the model will be strongly incentivized to produce peaked distributions. The question is whether this forces better discrimination between candidates or causes mode collapse.

### Changes from exp 50

- `--entropy-weight 0.5` (was 0.1)

### Risks

- At 40% of onset loss, the entropy penalty dominates training dynamics. The model may optimize for low entropy over correct predictions.
- Unique predictions could collapse — monitor closely at eval 1-2.
- Soft targets (which spread probability by design) fight directly against anti-entropy. This tension could destabilize training.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_50b --model-type event_embed --entropy-weight 0.5 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Converged at eval 10. Same HIT ceiling as w=0.1, no additional benefit from stronger pressure.**

### Progression

| Metric | Eval 3 | Eval 8 | Eval 10 | 50 (w=0.1) e10 | Exp 44 ATH |
|---|---|---|---|---|---|
| HIT | 70.9% | 72.1% | 72.5% | 72.3% | **73.6%** |
| Exact | 52.4% | 53.3% | 53.7% | 53.6% | 54.7% |
| AR step0 | 70.9% | 71.7% | 72.8% | 73.2% | 76.7% |
| AR step1 | 40.3% | 41.7% | 41.9% | **45.0%** | 48.2% |
| Metronome | 43.5% | 45.2% | 46.5% | 45.4% | 42.9% |
| Adv metronome | 51.3% | 51.0% | 51.7% | — | 50.1% |
| Unique preds | 448 | 454 | 452 | 434 | ~450 |

### Key observation: bimodal entropy

At w=0.5, the classic trimodal entropy distribution (peaks at ~1.4, ~2.3, ~3.0 nats) collapsed into a bimodal distribution (peaks at ~0.8 and ~1.7 nats). The "disambiguation zone" at 2.3 nats — where the model hedges between options — was eliminated entirely. The model either commits confidently (→ usually correct) or commits with moderate confidence (→ often wrong). No more hedging.

Despite this cleaner entropy structure, HIT didn't improve over w=0.1.

### 50 vs 50-B comparison

| | w=0.1 (exp 50) | w=0.5 (exp 50-B) |
|---|---|---|
| Entropy contribution to loss | ~8% | ~40% |
| HIT at convergence | 72.9-73.2% | 72.5% |
| AR step1 | **45.5%** | 41.9% |
| Metronome | 45.5% | 46.5% |
| Unique preds | 430-442 | 448-454 |
| Entropy structure | Compressed trimodal | Bimodal |

AR step1 is notably worse at w=0.5 — the model commits too hard to wrong predictions, leaving less uncertainty for the next step to recover from.

## Lesson

- **Stronger anti-entropy doesn't help.** w=0.5 produces a cleaner entropy structure but the same HIT ceiling and worse AR cascade than w=0.1.
- **The disambiguation zone isn't the problem.** Eliminating it (bimodal entropy) didn't improve accuracy. The model's hedging was informative, not harmful.
- **Anti-entropy hurts AR at high weight.** Over-committed wrong predictions cascade worse than uncertain ones. w=0.1 is the sweet spot if using anti-entropy at all.
- **The HIT ceiling at ~73% is not an entropy problem.** Both weights converge to the same place. The remaining errors are structural (exp 48: 2x/0.5x metric confusion), not a confidence issue.
