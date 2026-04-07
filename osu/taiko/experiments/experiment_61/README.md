# Experiment 61 - TaikoNation Evaluation Metrics

## Hypothesis

TaikoNation (Halina & Guzdial, 2021) defined 5 patterning-focused evaluation metrics for chart generation. Their key insight: pattern diversity and pattern overlap with human charts matter more than raw onset accuracy. Their metrics operate on binary arrays at 23ms resolution.

**Compute TaikoNation's exact evaluation metrics on our AR-generated charts** from exp 59-H. Compare to their published scores.

### TaikoNation's published results (from paper):

| Model | Over. P-Space | HI P-Space (Taiko) | DCHuman (Taiko) | DCRand |
|---|---|---|---|---|
| TaikoNation | 21.3% | 94.1% | 75.0% | 50.4% |
| DDC | 15.9% | 83.2% | 77.9% | 49.9% |
| Human Taiko | 14.5% | — | — | 50.2% |

### Metrics

1. **DCRand**: % similarity to random noise. ~50% = structured (not random).
2. **DCHuman**: Direct binary comparison at each 23ms step.
3. **OCHuman**: DCHuman with ±1 step tolerance buffer for hits.
4. **Over. P-Space**: Unique 8-step (184ms) patterns as % of all 256 possible patterns. Measures pattern diversity.
5. **HI P-Space**: % of human chart patterns also found in the AI chart. Pattern overlap.

### Launch

```bash
cd osu/taiko
python experiments/experiment_61/taikonation_eval.py
```

## Result

30 val songs, 6 models + Human GT.

### Full results:

| Model | Over. P-Space | HI P-Space | DCHuman | OCHuman | DCRand |
|---|---|---|---|---|---|
| **exp58** | 10.1% | **81.1%** | **90.8%** | **93.0%** | 50.0% |
| exp44 | 10.0% | 79.0% | 90.6% | 92.9% | 50.0% |
| exp53 | 10.2% | 78.3% | 90.5% | 92.7% | 50.0% |
| exp55 | 9.4% | 73.6% | 90.9% | 93.0% | 50.0% |
| exp50b | 9.2% | 75.3% | 90.2% | 92.5% | 50.1% |
| exp51 | 7.6% | 60.7% | 89.9% | 92.0% | 49.9% |
| Human GT | 11.7% | — | — | — | 50.0% |
| TaikoNation* | 21.3% | 94.1% | 75.0% | — | 50.4% |
| DDC* | 15.9% | 83.2% | 77.9% | — | 49.9% |

(*) Published results on different songs — not directly comparable.

### Key findings:

1. **We dominate on placement accuracy.** DCHuman 90.8% vs TaikoNation's 75.0% (+15.8pp). Our notes land in the right positions. OCHuman 93.0% shows even with tolerance, we're far ahead.

2. **TaikoNation has more pattern diversity.** Over. P-Space 21.3% vs our 10.1%. But Human GT is 11.7% — our models are closer to human diversity (14% below) while TaikoNation overshoots (47% above). Higher diversity isn't necessarily better.

3. **HI P-Space: we cover 81% of human patterns.** TaikoNation covers 94%. Their higher diversity means they naturally hit more human patterns, but also produce many non-human patterns (Over. P-Space 21% vs Human 14.5%).

4. **exp51 is worst on every metric.** Confirms 59-HB finding: its "variety" is under-prediction, not creativity.

5. **DCRand ~50% for everything.** All models (ours, TaikoNation, human) are equally structured — none are close to random noise.

## Lesson

The tradeoff between our approach and TaikoNation is **precision vs diversity**:

- **Ours**: Very accurate placement (90.8% DCHuman), slightly below human pattern diversity (10.1% vs 11.7%)
- **TaikoNation**: More diverse patterns (21.3% Over. P-Space), less accurate placement (75.0% DCHuman)

This maps directly to the 59-series finding: our models are metronomic but precise. We correctly place notes but repeat patterns too much. TaikoNation places notes less accurately but with more varied rhythmic patterns.

**The ideal model would combine both**: our placement accuracy with TaikoNation's pattern diversity. This may be achievable by:
1. Adding pattern diversity as an explicit training objective
2. Post-processing to vary note type assignments (TaikoNation's strength)
3. Temperature/sampling strategies that increase gap variety without sacrificing accuracy

Adding these results to [PERFORMANCE.md](../../PERFORMANCE.md).
