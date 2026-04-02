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

*Pending*

## Lesson

*Pending*
