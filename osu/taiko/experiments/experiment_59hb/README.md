# Experiment 59-HB - Ground Truth Comparison for 59-H Models

## Hypothesis

[Exp 59-H](../experiment_59h/README.md) showed exp51 dominating synthetic evaluation despite 67.5% HIT. Before investing in human evaluation, check the actual AR quality vs ground truth — matched/close/far rates, hallucination, density adherence — using the CSVs already generated in 59-H.

If exp51 has extremely high hallucination or very low match rates, its high gap variety is likely noise. If it maintains reasonable match rates while having varied gaps, it's genuinely interesting.

### Launch

```bash
cd osu/taiko
python experiments/experiment_59hb/gt_comparison.py
```

## Result

### Song density regime:

| Model | Close% | Far% | Hall% | d_ratio | err_med | p/g |
|---|---|---|---|---|---|---|
| **exp58** | **75.9%** | **16.6%** | 15.6% | 0.92 | 8ms | 0.96 |
| exp53 | 73.4% | 19.0% | 17.9% | 0.91 | 8ms | 0.96 |
| exp44 | 71.1% | 20.6% | 14.7% | 0.84 | 9ms | 0.88 |
| exp55 | 69.7% | 21.7% | 15.6% | 0.82 | 14ms | 0.86 |
| exp50b | 66.5% | 25.2% | 14.8% | 0.78 | 19ms | 0.82 |
| **exp51** | **56.8%** | **36.0%** | 14.9% | **0.64** | **40ms** | **0.70** |

### Key finding on exp51:

exp51's hallucination rate is **normal** (14.9%) — it's not spamming random notes. But it **under-predicts severely** (only 64% of expected density, 70% event ratio). Its high synthetic gap variety comes from **omission, not creativity** — fewer events placed further apart naturally produces high gap_std/cv.

### exp58 is the actual best:

exp58 (propose-select ATH) has the best GT matching across all metrics: highest close rate (75.9%), lowest far rate (16.6%), best density adherence (0.92), lowest median error (8ms).

### Fixed density regime confirms the pattern:

Under fixed_53ar density, all models improve (higher density conditioning → more events) but the ranking holds: exp58 best, exp51 worst on GT matching.

## Lesson

1. **The synthetic evaluator's exp51 preference is a false positive.** High gap variety from under-prediction, not musical intelligence. The evaluator rewards omission because fewer events = more varied gaps.

2. **exp58 is the true best model** — highest per-sample HIT (74.6%), best AR GT matching (75.9% close), and competitive synthetic scores. The propose-select architecture translates to real AR quality.

3. **Synthetic evaluator needs a minimum density/match filter.** A model with <70% close rate or <0.75 density ratio should be penalized regardless of gap variety. This would correctly rank exp51 last.

4. **exp51 should still be included in human eval** — its unusual under-prediction style might produce a distinctive "less is more" feel that some players prefer, even if GT matching is poor.

5. **The quality relationship is non-linear (inverted U-shape).** Too little gap variety = metronomic (bad). Too much gap variety = under-prediction/noise (also bad). Linear correlations from 59-B captured the upward slope but missed the peak and downturn. A proper quality algorithm would need to model this non-linearity — potentially a small neural network trained on human preference data. The synthetic evaluator from 59-C through 59-F is a useful screening tool when combined with GT matching and density checks, but should not be used as a standalone judge.
