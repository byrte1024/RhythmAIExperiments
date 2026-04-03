# Experiment 55 - Auxiliary Ratio Head

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

The soft trapezoid loss is asymmetric in bin space: target 20 has ~3 valid bins within tolerance, target 200 has ~30. This confuses the model — it gets different gradient landscapes depending on where in the prediction range the target falls. Log-ratio space is symmetric by construction (`log10(2) = -log10(0.5)`), making the soft trapezoid naturally balanced.

However, predicting in ratio space during AR inference causes error accumulation — ratio errors compound multiplicatively across steps. The solution: **a training-only auxiliary head** that predicts in log10-ratio space.

Two heads share the same backbone:
- **Bin head** (existing): Hard+soft CE in bin space, 251 classes. Used at inference.
- **Ratio head** (new): Soft CE in log10-ratio space, 201 classes. Training only.

The ratio head teaches the backbone to think proportionally. The bin head learns concrete predictions for AR. The shared backbone gets gradients from both, learning representations that satisfy both objectives.

### Key design decisions

- **Class-based (201 bins), not scalar regression**: Supports multimodality. The bin head can express "probably bin 30 or bin 60" — a scalar ratio head would average to bin 45 (wrong for both). 201 bins in log10 space gives ~3% ratio resolution.
- **Pure soft CE (hard_alpha=0.0)**: The entire point is symmetric soft targets. Hard CE would reintroduce the asymmetry.
- **No agreement loss**: The shared backbone already couples the heads through gradients. Monitor head agreement as a metric first.
- **ratio_weight=0.3**: Auxiliary, shouldn't dominate onset learning.

### Architecture

```
Mel window: 1000 frames (500 past + 500 future)
Conv stem: 1000 → 250 tokens (stride 4)
Cursor: token 125 (500 // 4)
Onset head: cursor → LayerNorm → Linear(384, 251) → smooth conv → 251 logits
Ratio head: cursor → LayerNorm → Linear(384, 201) → smooth conv → 201 logits
```

Same backbone as [exp 53-B](../experiment_53b/README.md). Ratio head adds ~77K parameters (~0.5% of total).

### Ratio target computation

- `prev_gap = |past_bins[-1] - past_bins[-2]|` (gap between last two context events)
- `ratio = target_bin / prev_gap`, clamped to [0.05, 20.0]
- `log10_ratio = log10(ratio)`, discretized into 201 bins covering [-1.301, 1.301]
- Bin 100 = ratio 1.0 (repeat same gap). Bin 77 = ratio 0.5 (halved). Bin 123 = ratio 2.0 (doubled).
- Masked (-1) for STOP targets, <2 context events, or prev_gap/target = 0.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_55 --model-type event_embed --a-bins 500 --b-bins 500 --b-pred 250 --ratio-head --ratio-weight 0.3 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

Stopped at eval 12 (epoch 3.0). **Peak at eval 7 (epoch 2.7), tied at eval 12.**

### Peak (eval 7) metrics:

| Metric | Value |
|---|---|
| HIT% | 73.6% |
| GOOD% | 74.0% |
| MISS% | 25.8% |
| Accuracy | 54.2% |
| Stop F1 | 0.570 |
| Model score | 0.386 |
| Val loss | 2.463 |

### Ratio head metrics at peak:

| Metric | Value |
|---|---|
| Ratio HIT% | 34.2% |
| Ratio MISS% | 26.1% |
| Ratio accuracy | 34.2% |
| Ratio unique preds | 192 / 201 |

### Progression summary:

| Eval | Epoch | HIT% | Score | Val Loss | NE% | CtxD | NA_stop | rHIT% |
|------|-------|------|-------|----------|-----|------|---------|-------|
| 1 | 1.2 | 69.6 | 0.340 | 2.608 | 37.3 | +11.8 | 11.8% | 27.4% |
| 5 | 2.2 | 73.2 | 0.380 | 2.475 | 47.0 | +6.9 | 36.3% | 34.6% |
| **7** | **2.7** | **73.6** | **0.386** | **2.463** | **48.4** | **+5.8** | **16.3%** | **34.2%** |
| 10 | 3.5 | 73.5 | 0.384 | 2.473 | 50.0 | +4.4 | 62.6% | 36.3% |
| 12 | 3.0 | 73.6 | 0.384 | 2.461 | 47.7 | +6.9 | 47.8% | 32.0% |

### vs [exp 53-B](../experiment_53b/README.md) (same architecture, no ratio head):

| Metric | 55 (eval 7) | 53-B (eval 11) |
|--------|-------------|----------------|
| HIT% | **73.6%** | 73.4% |
| Accuracy | 54.2% | 54.2% |
| Stop F1 | **0.570** | 0.562 |
| Val loss | **2.463** | 2.479 |
| NoEvt acc | 48.4% | **49.7%** |
| Peak epoch | **2.7** | 3.7 |

The ratio head accelerated convergence (~1 epoch earlier to peak) and improved HIT by +0.2pp, val loss by -0.016, and stop F1 by +0.008. Modest but consistent improvements.

### vs all-time bests:

| Metric | Exp 55 peak | Exp 44 peak | Exp 53-B peak |
|--------|-------------|-------------|---------------|
| HIT% | 73.6% | **73.7%** | 73.4% |
| Val loss | **2.463** | 2.480 | 2.479 |
| NoEvt acc | 48.4% | 48.4% | **49.7%** |

Tied exp 44's ATH within 0.1pp. Best val loss ever.

## Lesson

The auxiliary ratio head provides a modest but real improvement. The symmetric soft trapezoid in log10 space gives the backbone proportional reasoning that the asymmetric bin-space loss doesn't. Key findings:

1. **Faster convergence**: Peaked at epoch 2.7 vs 53-B's 3.7 and exp 44's 4.7. The ratio head's smooth symmetric gradients help early training.
2. **Better val loss**: 2.461 (eval 12) is the best ever observed — the model is better calibrated even when HIT% is similar.
3. **Stop F1 improvement**: 0.570 ties exp 44's best despite no explicit STOP modifications. The ratio head teaches uncertainty awareness as a side effect.
4. **NoEvt accuracy hit 50.0%** (eval 10) — first model to reach 50% audio-only accuracy.
5. **Didn't break through 73.7%**: The ratio head is a quality-of-training improvement, not an architectural breakthrough. The HIT% ceiling appears structural.

The ratio head infrastructure is proven and can be combined with other changes going forward.
