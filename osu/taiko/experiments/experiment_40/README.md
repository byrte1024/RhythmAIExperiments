# Experiment 40 - Stronger Balanced Sampling (power=0.7)

## Hypothesis

Exp 39-E revealed the model picks the sharpest transient, not the nearest onset. But exp 39-D showed the model is systematically underconfident for distant predictions — entropy rises sharply with target distance from cursor.

The class distribution is severely skewed: bins 10-50 have ~77% of all samples, bins 100-500 have <5%. The current balanced sampling (`1/sqrt(count)`, power=0.5) gives rare bins ~57x weight, but they're still seen far less often than common bins in absolute terms. The model builds strong, confident features for short-range predictions but remains uncertain about long-range ones — exactly where the overprediction errors cluster.

**Stronger balanced sampling (power=0.7)** gives distant bins more representation so the model can build confident features there. The model is currently underconfident for distant predictions because it rarely sees them during training.

Effective gradient distribution across bin ranges:

| Bin range | Natural | power=0.5 (current) | power=0.7 (new) |
|---|---|---|---|
| 10-25 (common) | 41.8% | 21.3% | 12.2% |
| 50-100 | 16.6% | 23.8% | 20.9% |
| 100-200 (rare) | 3.9% | 16.1% | 21.7% |
| 200-500 (very rare) | 0.8% | 11.5% | 26.5% |

Power=0.7 gives distant bins (200-500) 26.5% of training exposure (up from 11.5%) while common bins (10-25) still get 12.2% — enough to maintain confident short-range predictions. Weights capped at 1.0 to prevent empty classes from dominating.

### Changes from exp 35-C

- **balance_power: 0.5 → 0.7**
- Everything else identical (exponential ramps, amplitude jitter, unified fusion, full dataset)

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_40 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --balance-power 0.7 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
