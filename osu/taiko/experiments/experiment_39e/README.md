# Experiment 39-E - Audio Analysis of Failure Cases

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp [39-D](../experiment_39d/README.md) showed that when the model is wrong, the correct answer is at rank 1-2 with only 43% the confidence of the wrong pick. The model genuinely believes the wrong answer is correct despite heavy training loss penalizing these errors.

**Theory: the audio at the target and predicted positions will be nearly identical.** Both positions are real onsets (exp [39](../experiment_39/README.md) showed 83% of overpredictions match future onsets), so both have real transients. The model isn't picking a louder transient over a quieter one — it's choosing between two equally valid audio events and can't determine which is nearer.

If the audio energy/flux/onset strength is similar at both positions, the failure isn't an audio saliency problem — it's a proximity/ordering problem. The model detects onsets correctly but lacks the mechanism to rank them by distance from cursor.

### Method

For each failure case where the correct answer is in top-K:
1. Compare mel energy (mean across bands, ±5 frame window) at target vs predicted position
2. Compare spectral flux (frame-to-frame change) at both positions
3. Compare onset strength (max flux in ±3 window) at both positions
4. Export 10 visual mel windows with target and predicted positions marked

### Expected: audio features will be nearly identical at target and predicted positions

## Result

**Mel energy is identical (50/50) but spectral flux is biased toward the predicted position — for overpredictions, 78-81% have sharper transients at the wrong pick.**

| Metric | At target | At predicted | Pred higher |
|--------|-----------|-------------|-------------|
| Mel energy | 22.21 | 22.17 | 50.0% (equal) |
| Spectral flux | 35.16 | 45.24 | 60.6% |
| Onset strength | 49.97 | 62.32 | 61.2% |

**Overpredictions specifically (63.3% of failures):**
- Pred energy > target: 47.8% (equal)
- Pred flux > target: **77.8%**
- Pred onset strength > target: **81.0%**

**Underpredictions (36.7% of failures):**
- Pred energy > target: 53.9% (equal)
- Pred flux > target: 31.0% (target is sharper)

### Interpretation

- **Raw energy is identical** — the model isn't picking louder sounds. Both positions are real onsets with similar loudness.
- **But the further onset often has sharper transients** — 78-81% of overpredictions land on a position with stronger spectral flux. The model picks the most "onset-like" audio event, not the nearest one.
- **This is NOT universal** — ~20% of overpredictions have WEAKER transients at the predicted position, and underpredictions show the opposite pattern (target is sharper). The transient-saliency bias explains a significant portion of failures but not all.
- **The further onset is often the start of a new rhythmic group** — in patterns like `75 75 150`, the onset at 150 (new group) tends to have a harder attack than the continuation at 75.

## Lesson

- **The model confuses audio saliency with temporal order** — it picks the sharpest transient, not the nearest onset. This is rational for a model that learned "sharp transient = onset" but wasn't taught "prefer nearer."
- **~80% of overpredictions are saliency-driven** — the model isn't randomly wrong, it's systematically picking the more prominent audio event.
- **~20% of failures have other causes** — not all errors are from transient saliency. Some may be context/pattern errors, noise, or genuinely ambiguous cases.
- **Context ramps partially address this** ([5% delta](../experiment_35c/README.md)) — they encode "there's an onset nearer" — but the saliency signal overwhelms the ramp signal for ~80% of failures.
- **A training fix needs to either**: (1) reduce the model's reliance on transient sharpness for ordering, or (2) amplify the proximity signal to compete with saliency.
