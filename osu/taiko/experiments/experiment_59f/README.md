# Experiment 59-F - Evaluator Weight Sweep

## Hypothesis

[Exp 59-E](../experiment_59e/README.md) tested 4 hand-picked evaluator formulas. combined_top2 (gap_std + gap_cv, equal weights) won overall, but maybe a better combination exists. The weight space hasn't been explored.

**Sweep through combinations**: top-N metrics (N=1..6) with varying temperature (how sharply to weight by correlation strength). Find the optimal formula for each audience.

### Launch

```bash
cd osu/taiko
python experiments/experiment_59f/sweep_evaluator.py
```

## Result

144 combinations tested (8 top-N × 6 temperatures × 3 audiences).

### Best formula per audience:

| Audience | Top-N | Temp | #1 Match | Tau | r | p |
|---|---|---|---|---|---|---|
| All | 6 | 0.0 | **52%** | +0.333 | +0.425 | <0.001 |
| Self | 2 | any | **47%** | +0.422 | +0.404 | 0.003 |
| Volunteer | 7 | 0.5 | **70%** | +0.400 | +0.484 | 0.003 |

### Heatmap observations:

- **Self**: uniformly green at top-2, uniformly yellow-red at top-5+. Temperature doesn't matter. Adding metrics always hurts.
- **Volunteer**: broadly green across most cells. top-7 is best. Mild temperature helps (0.5-1.0). Nearly all metrics contribute.
- **All**: sweet spot at top-6, equal weights. A compromise between the narrow expert formula and the broad volunteer formula.

## Lesson

1. **Expert preference is captured by exactly 2 metrics** (gap_std + gap_cv). Temperature, weighting, and additional metrics are all noise. The expert evaluator is already optimal at its simplest form.

2. **Volunteer preference is multi-dimensional**. 7 metrics at mild temperature achieves 70% first-place accuracy — correctly picks the human-preferred chart 7/10 times. Volunteers respond to variety, anti-repetition, entropy, AND density stability.

3. **Temperature is largely irrelevant.** The metrics from 59-B have similar correlation strengths (~0.27-0.30), so sharpening weights doesn't differentiate them meaningfully.

4. **Recommended formulas:**
   - Expert: `z(gap_std) + z(gap_cv)` (2 metrics, equal weights)
   - Volunteer: top-7 metrics at temp=0.5 (all metrics from 59-B weighted by |r|^0.5)
   - General: top-6 metrics at equal weights

### Leaderboard validation:

53-AR: All 3 optimized evaluators correctly predict #1 (exp45) and #2 (exp44). Only swap 3rd/4th. 42-AR: All miss — exp14 won by avoiding failure modes, not positive quality traits. Evaluators can't detect absence of defects.
