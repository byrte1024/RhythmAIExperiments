# Experiment 44-D - Temperature Sampling on Top-K and Top-U

## Hypothesis

Exp 44-C showed the model knows the right answer 91.8% of the time (Top-U 3 oracle) but only picks it 73.6% of the time (argmax). The gap between "considered" and "selected" is 18pp.

Currently we use argmax (greedy) decoding — always pick the highest confidence prediction. But from the threshold analysis, the model typically has 2-4 meaningful options at 5% confidence. Temperature sampling could help by:
- **Low temperature (T < 1):** Sharpens the distribution, making the top choice more dominant. Likely hurts — we already argmax, sharpening doesn't help.
- **High temperature (T > 1):** Softens the distribution, giving lower-ranked options a chance. Could help break metronome patterns by occasionally sampling a non-dominant prediction.
- **Very high temperature (T >> 1):** Approaches uniform sampling over the top-K/U, essentially random selection from the candidate set.

### Predictions

- **Temperature sampling will likely be WORSE on per-sample metrics.** Argmax is the optimal single-sample strategy by definition — any deviation from the highest-confidence prediction lowers expected accuracy. HIT rate should monotonically decrease as temperature increases.
- **But it might produce better charts.** The metronome problem (44-B) is caused by the model always picking the statistically dominant option. Temperature introduces variety — occasionally sampling a non-dominant prediction could break metronome lock-in during AR generation, even if individual predictions are less accurate.
- **This experiment measures the per-sample cost of temperature.** If Top-U 3 at T=2.0 only loses 2-3pp HIT vs argmax, that's a cheap price for potentially much better AR behavior. If it loses 15pp, it's too expensive.
- **Top-U sampling should degrade more gracefully** than Top-K at the same temperature, because all Top-U candidates are genuinely different options rather than near-duplicates clustered around the same onset.

### Method

Reuse the softmax probabilities from exp 44 (eval 20 checkpoint, subsample 8 val set). For each sample:

1. Compute Top-K (3, 5, 10, 20) and Top-U (3, 5, 10, 20) candidate sets with their confidences
2. For 50 temperature values from 0.01 to 100 (log-spaced):
   - Apply temperature to the candidate confidences: `p_i^(1/T) / sum(p_j^(1/T))`
   - Sample from the reweighted distribution (deterministic: use the expected value by weighting candidates by their tempered probabilities, picking the one with highest tempered prob — i.e. argmax after temperature)
3. Measure HIT/GOOD/MISS at each temperature

Note: at T→0 sampling collapses to argmax (always picks highest confidence). At T→∞ it's uniform random across candidates. This uses actual random sampling (not oracle, not argmax) — it's what the model would produce if we used temperature sampling at inference time. Results are averaged over 5 random trials per temperature for stability.

### Scripts

- `analyze_temperature.py` — runs temperature sweep on cached probabilities

## Result

74,075 non-STOP val samples, 5 trials averaged per temperature.

### Temperature sampling is strictly worse on per-sample metrics

Best HIT is always at T=0.01 (argmax) for every K and U. No sweet spot exists.

| Mode | T=0.01 | T=1.0 | T=100 |
|---|---|---|---|
| Top-K 3 | 73.6% | 69.5% (-4.1pp) | 63.4% |
| Top-K 5 | 73.6% | 64.9% (-8.7pp) | 49.2% |
| Top-K 10 | 73.6% | 58.8% (-14.8pp) | 29.0% |
| Top-K 20 | 73.6% | 55.1% (-18.5pp) | 15.9% |
| Top-U 3 | 73.0% | 64.2% (-8.8pp) | 45.4% |
| Top-U 5 | 73.0% | 59.4% (-13.6pp) | 31.2% |
| Top-U 10 | 73.0% | 56.3% (-16.7pp) | 17.2% |
| Top-U 20 | 73.0% | 55.4% (-17.6pp) | 9.0% |

### Top-K 3 is most resilient to temperature

Only loses 4.1pp at T=1.0 because the 3 candidates are near-duplicates — sampling between them doesn't change the prediction much. Top-U degrades faster because its candidates are genuinely different.

### MISS rate at T=1.0

| Mode | T=0.01 | T=1.0 |
|---|---|---|
| Top-K 3 | 25.8% | 28.9% (+3.1pp) |
| Top-U 3 | 26.5% | 29.8% (+3.3pp) |
| Top-U 5 | 26.5% | 30.5% (+4.0pp) |

## Lesson

- **Temperature sampling cannot improve per-sample metrics.** Argmax is provably optimal for single-sample accuracy. Any randomness only adds noise.
- **Top-K 3 is the cheapest temperature option** (-4.1pp at T=1.0) because candidates are clustered. But this also means it provides the least diversity.
- **Top-U degrades faster but provides real diversity.** The per-sample cost of Top-U 5 at T=1.0 is 13.6pp — steep, but each sampled prediction is a genuinely different onset.
- **The value of temperature is in AR generation, not per-sample metrics.** Whether the per-sample cost buys better charts through metronome breaking is a listening test question. Added `--random-seed` mode to detection_inference.py for AR testing.
