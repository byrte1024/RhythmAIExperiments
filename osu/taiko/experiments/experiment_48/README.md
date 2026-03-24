# Experiment 48 - Cross-Model Failure Analysis

## Hypothesis

All our models (exp 14, 35-C, 44, 45) achieve 68-74% HIT despite very different architectures (no context, mel ramps, event embeddings, gap ratios). If they all fail on the **same** validation samples, those failures are structural — inherent to the data, audio, or task — and no architecture change will fix them. If they fail on **different** samples, each architecture has unique blind spots that could be addressed.

### Method

Run each model on the full validation set (subsample 1). For each sample, record:
- Per-sample score [-1, +1]
- Predicted bin
- Whether it was a HIT/MISS

Then analyze:
1. **Success heatmaps** — 512x512 image per model, same sample ordering, visual comparison
2. **Overlap rates** — what % of failures are shared across all models vs model-specific?
3. **Agreement table** — for each pair of models, how often do they agree on HIT/MISS?
4. **Failure mode comparison** — when models fail on the same sample, do they predict the same wrong bin or different ones?
5. **Model-specific failures** — samples where one model fails but all others succeed. What characterizes these?

### Models

| Label | Experiment | HIT | Architecture |
|---|---|---|---|
| exp14 | Exp 14 | 68.9% | No context, audio-only |
| exp35c | Exp 35-C | 71.6% | Mel-embedded exponential ramps |
| exp44 | Exp 44 | 73.6% | Event embeddings, gentle augmentation |
| exp45 | Exp 45 | 71.9% | Event embeddings + gap ratios + tight density |

### Predictions

- **Most failures will be shared.** The same audio sections (ambiguous transients, polyrhythmic sections, quiet passages) will trip up every model.
- **Context models (44, 45) will have fewer unique failures** than no-context (14), since context resolves some ambiguity.
- **When models fail on the same sample, they'll predict similar wrong bins** — the "sharper transient at wrong position" pattern from exp 39-E.

### Scripts

- `analyze_val_heatmap.py` — runs a model on val set, produces 512x512 heatmap + raw scores .npy
- `analyze_cross_model.py` — loads multiple .npy score files, produces overlap/agreement/failure analysis

### Launch

```bash
python analyze_val_heatmap.py --checkpoint runs/detect_experiment_14/checkpoints/best.pt --label exp14
python analyze_val_heatmap.py --checkpoint runs/detect_experiment_35c/checkpoints/eval_008.pt --label exp35c
python analyze_val_heatmap.py --checkpoint runs/detect_experiment_44/checkpoints/eval_019.pt --label exp44
python analyze_val_heatmap.py --checkpoint runs/detect_experiment_45/checkpoints/best.pt --label exp45
python analyze_cross_model.py
```

## Result

*Pending*

## Lesson

*Pending*
