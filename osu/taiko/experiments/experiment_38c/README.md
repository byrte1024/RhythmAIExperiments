# Experiment 38-C - Framewise with Unweighted BCE

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp [38-B](../experiment_38b/README.md) proved the framewise architecture works (24% event recall, well above random) but pos_weight=7 causes 3x overprediction (46.7 preds/win vs 16.2 real). Every positive weighting attempt (exp [37](../experiment_37/README.md), [37-B](../experiment_37b/README.md), [38-B](../experiment_38b/README.md)) causes overprediction.

**Remove positive weighting entirely.** The 13% natural positive ratio is learnable without upweighting — the model just needs to discover the right activation threshold on its own. BCE without weighting naturally balances: missing a positive costs ~2x more gradient than a false positive (since -log(p) at p=0.1 is steeper than at p=0.9).

### Changes from exp [38-B](../experiment_38b/README.md)

- **pos_weight removed** — plain `F.binary_cross_entropy(probs, target)`, no class weighting
- Everything else identical (framewise model, fixed causal mask, mel ramps, no teacher forcing)

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_38c --model-type framewise --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Slight improvement over 38-B but F1 still too low.** Killed after eval 1.

| Metric | 38-C (pw=1) | [38-B](../experiment_38b/README.md) (pw=7) |
|--------|-------------|-------------|
| Event recall | 23.4% | 24.1% |
| Pred precision | 11.7% | 8.4% |
| F1 | 0.156 | 0.124 |
| Preds/window | 32.5 | 46.7 |
| Hallucination | 50.1% | 65.2% |

Removing pos_weight reduced overprediction (46→32 preds/win) and improved precision (8→12%), but F1 at 0.156 is far too low for practical use. The framewise architecture learns to find onsets (~24% recall) but can't achieve the precision needed.

## Lesson

- **Unweighted BCE is better** — reduces hallucination without losing recall.
- **Framewise from a single model is hard** — the model needs to simultaneously learn audio features AND spatial onset detection from scratch. The single-target model (exp [35-C](../experiment_35c/README.md)) took 20+ experiments of iteration to reach 71.6% HIT; framewise needs similar refinement but we've exhausted the quick iterations.
- **Return to exp [35-C](../experiment_35c/README.md)** — the single-target approach works at 71.6% HIT with 5% sustained context delta. Better to push that further (longer training, beam search, ramp refinements) than to keep iterating on framewise which is at F1=0.156.
