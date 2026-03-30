# Experiment 39-D - Top-K Depth Analysis

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

We know top-10 contains the answer [~96%](../experiment_27/README.md) of the time, but two unknowns:

1. **How many HITs per sample?** If top-10 has 5 candidates that all match the target (near-duplicate bins like 74, 75, 76), the model is concentrated but imprecise. If it has exactly 1 match, the model knows the answer but ranks it wrong.

2. **When wrong, how close in confidence is the correct answer?** If the correct candidate is at rank 2 with 0.95x the confidence of rank 1, a tiny nudge could fix it. If it's at rank 8 with 0.1x confidence, the model genuinely doesn't know.

## Result

**The model clusters 3 HITs per sample in top-10 (74.5%), and when wrong, the correct answer is rank 1-2 with ~40% the confidence of the wrong pick.**

### Analysis 1: HIT count per sample

| HITs in top-10 | Samples | % |
|---|---|---|
| 0 | 3,048 | 4.1% |
| 1 | 3,147 | 4.2% |
| 2 | 7,041 | 9.5% |
| **3** | **55,212** | **74.5%** |
| 4+ | 5,627 | 7.6% |

Mean: 2.86, Median: 3. The model produces a tight cluster of ~3 adjacent correct bins, not scattered candidates.

### Analysis 2: When wrong, where is the correct answer?

- Wrong predictions: 21,037 (28.4%)
- Correct in top-10: **17,989 (85.5%)**
- Correct NOT in top-10: 3,048 (14.5%)

Of the 17,989 fixable errors:

| Rank | Count | % |
|---|---|---|
| 1 (2nd choice) | 7,910 | 44.0% |
| 2 (3rd choice) | 5,352 | 29.8% |
| 3-4 | 2,529 | 14.1% |
| 5-9 | 2,198 | 12.2% |

**Confidence analysis:**
- Correct answer confidence: mean 0.108 (10.8%)
- Wrong chosen confidence: mean 0.284 (28.4%)
- **Ratio: 0.43** — the model is 2.3x more confident in the wrong answer
- Nearly equal (>0.9 ratio): only 6.5% — rare to have a close call
- Model very sure of wrong (<0.1 ratio): 9.9%

## Lesson

- **The model concentrates probability correctly** — 3 adjacent correct bins in top-10 (74.5% of samples). The predicted region is right, just not always the peak.
- **When wrong, the answer is usually the 2nd or 3rd choice** — 73.8% at rank 1-2. But the confidence gap is 2.3x, too large for post-hoc reranking to overcome.
- **This is a confidence/calibration problem, not a detection problem.** The model sees the right region but can't confidently commit to the nearest onset when multiple valid onsets exist in the window.
- **The 2.5x confidence gap explains why [reranking only gets +1pp](../experiment_39b/README.md)** — no weighting scheme can reliably flip a 28% vs 11% confidence decision without also breaking correct predictions.
- **Fix must happen at training time** — the model needs to learn to be more confident about nearest onsets specifically.
