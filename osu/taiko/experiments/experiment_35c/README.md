# Experiment 35-C - Exponential Decay Ramps + Amplitude Jitter

## Hypothesis

Exp 35-B showed full-band mel ramps provide the best sustained context delta (3.5-5%) but two issues remain:
1. **Linear ramps are ambiguous** — the model sees a gradual slope but can't pinpoint exact beat positions. This contributes to high entropy across all predictions.
2. **Fixed 0.5x audio scaling reduces confidence** — weakened audio signal means the model is uncertain even when correct.

**Exponential decay ramps** make beat positions sharp and unambiguous. At each event, the signal spikes to 1.0 and decays exponentially with a half-life of 3% of the gap. By the time the next event arrives, the signal is near zero. The model sees clear "beat markers" — a spike pattern whose frequency IS the rhythm.

**Amplitude jitter** (0.25-0.75x per sample during training) prevents the model from learning a fixed ramp-to-audio ratio. It must be robust to varying audio strengths, encouraging confidence calibration.

### Changes from exp 35-B

- **Ramp shape**: linear (1→0 over gap) → exponential decay (spike + fast falloff, half-life = 3% of gap)
- **Audio scaling**: fixed 0.5 → random 0.25-0.75 per sample during training (0.5 at eval)

### Expected outcomes

1. **Lower entropy** — sharper beat markers give the model more confident features to attend to.
2. **Context delta maintained or improved** — exponential decay makes event positions more distinct, potentially more useful.
3. **Better calibration** — amplitude jitter trains robustness, reducing overconfidence/underconfidence.

### Risk

- New exponential may be way too weak for the model to detect, or become easy to ignore.

### Visualizations

![Ramp Comparison](ramp_visualization.png)
![Real Sample](real_sample_visualization.png)

## Result

*Pending*

## Lesson

*Pending*
