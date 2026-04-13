# Experiment 66-2c — Bidirectional Evaluator vs osu! Ratings

## Hypothesis

Exp 66-2b showed the bidirectional evaluator perfectly matches human preference on 42-AR (large quality gaps) but fails on 53-AR (subtle differences). How does it correlate with osu! user ratings across the full 10k chart dataset?

66-1 achieved Spearman 0.091 and 55.9% pairwise accuracy on ratings. The bidirectional model may do better or worse — it was trained with only 15% rating pairs (vs 40% in 66-1 P2), but its corruption understanding is more nuanced.

## Method

Score all 10,027 rated charts in taiko_v2 with the 66-2 checkpoint (8 windows each). Compute:
- Overall Spearman/Pearson with osu! rating
- Per-beatmapset Spearman (deduplicates same-set charts)
- Per-star-rating tier Spearman (controls for difficulty)
- Pairwise accuracy on rating pairs (star_rating ±0.5, gap ≥1.0)
- Top/bottom charts by model score

## Launch

```bash
python classifier_eval_ratings.py --checkpoint runs/eval_experiment_66_2/checkpoints/best.pt --dataset taiko_v2
```

## Result

### Overall correlation

| Metric | 66-2 (bidirectional) | 66-1 (unidirectional) |
|---|---|---|
| Spearman (all charts) | -0.015 (p=0.13, **not significant**) | +0.091 (p=1e-19) |
| Spearman (per-beatmapset) | -0.004 (p=0.83) | +0.107 (p=8e-8) |
| Pairwise accuracy | **58.3%** | 55.9% |

Overall Spearman collapsed to near zero — the bidirectional model's scores don't linearly track ratings. But pairwise accuracy improved to 58.3% (best yet), meaning it's better at relative ordering even if absolute correlation is gone.

### Per star-rating tier

| Tier | n | Spearman | p-value |
|---|---|---|---|
| <2* | 2198 | -0.021 | 0.33 |
| 2-3* | 2125 | +0.016 | 0.47 |
| 3-4* | 2182 | +0.021 | 0.33 |
| 4-5* | 1574 | **+0.100** | 6.8e-5 |
| 5-6* | 974 | +0.029 | 0.37 |
| 6+* | 974 | **+0.191** | 1.8e-9 |

The model only correlates with ratings for harder charts (4+*). Hard charts have more room for quality variation — more complex patterns, more ways to be creative or metronomic. The evaluator's sensitivity to pattern variety has more signal to work with at higher difficulties.

### Score distribution

Scores center around 0 with range -1.5 to +1.5 (much less extreme than 66-1's -10 to +30). The bidirectional tie loss compressed the score range.

## Lesson

The bidirectional model trades overall Spearman for better pairwise accuracy. It's better at relative comparisons (58.3% vs 55.9%) but worse at absolute scoring. The non-linear relationship with ratings and the tier-dependent correlation suggest the evaluator captures something real about chart quality — but only for charts complex enough to have meaningful quality variation.
