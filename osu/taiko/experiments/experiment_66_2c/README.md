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

*(awaiting results)*
