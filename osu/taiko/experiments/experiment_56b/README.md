# Experiment 56-B - Density Sensitivity Test

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 56](../experiment_56/README.md) showed the model under-predicts density (ratio 0.83 avg) and that density conditioning has limited control over output. But does varying the conditioned density actually change the model's output at all?

**Test: run each song 3 times** with different density conditioning:
- **0.8x**: 80% of the chart's actual density
- **1.0x**: actual chart density (baseline, same as exp 56)
- **1.2x**: 120% of the chart's actual density

If the model is sensitive to density, we should see proportional changes in predicted event count. If it's largely ignoring density, all three runs will produce similar output.

### Launch

Same model as [exp 56](../experiment_56/README.md): [exp 45](../experiment_45/README.md).

```bash
cd osu/taiko
python experiments/experiment_56b/run_density_sweep.py --checkpoint runs/detect_experiment_45/checkpoints/best.pt
```

## Result

Ran on 50 val songs, 3 scales each (150 inference runs).

### Summary:

| Scale | Avg events | Close rate (<50ms) | Hallucination (>100ms) | Density ratio |
|-------|-----------|-------------------|----------------------|---------------|
| 0.8x | 652 | 59.5% | 13.3% | 0.68 |
| 1.0x | 788 | 70.0% | 15.3% | 0.84 |
| 1.2x | 982 | **77.5%** | 17.5% | **1.01** |

### Key findings:

1. **The model IS density-sensitive.** Avg sensitivity (1.2x/0.8x event ratio) = **1.53**, almost exactly proportional to the 1.5x density change. Density conditioning is not decorative.

2. **The model needs ~1.2x density to match reality.** At 1.0x (correct density), the model only reaches 0.84 density ratio. At 1.2x, it reaches 1.01 — nearly perfect adherence.

3. **Higher density improves catch rate with minimal hallucination cost.** Going from 0.8x to 1.2x gains +18pp catch rate (59.5% → 77.5%) for only +4.2pp hallucination (13.3% → 17.5%).

4. **The systematic under-prediction from [exp 56](../experiment_56/README.md) is calibration, not deafness.** The model hears the density signal and responds proportionally — it's just calibrated 20% low.

## Lesson

Density conditioning works as intended — the model responds proportionally to density changes. The ~20% under-prediction is a consistent calibration offset, not a failure of the conditioning mechanism.

**Practical implication**: For inference, condition at ~1.2x the desired density to get accurate output density. This is simpler than any architectural fix — just inflate the input.

The hallucination tradeoff is favorable: the extra predictions from higher density land near real events, not in silence. The model knows where the onsets are — it just needs permission (via density) to place them.
