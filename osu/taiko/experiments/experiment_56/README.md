# Experiment 56 - Density Conditioning AR Analysis

## Hypothesis

In [53-AR](../experiment_53ar/README.md) human evaluation, [exp 45](../experiment_45/README.md) outperformed [exp 44](../experiment_44/README.md) on expert rankings but volunteers preferred exp 44. The key difference between them was density conditioning: exp 45 tightened density jitter from ±10%/30% to ±2%/10%, making the model more density-adherent.

**Theory: density conditioning may be giving the model too much information.** Each song has an "ideal" density, but the model may over-rely on density rather than audio to determine note placement. This could explain high hallucination rates — the model places notes to hit the target density rather than because it hears something.

This experiment runs full AR inference on val songs using each song's actual chart density, then analyzes the relationship between density conditioning and AR quality.

### Approach

1. Select 50 diverse val songs (varying density, genre, duration)
2. For each song, pick one chart and run AR inference with that chart's density_mean/peak/std
3. Compare predicted onsets to ground truth onsets
4. Compute: event hit/good/miss rates, hallucination rate, density ratio (predicted vs actual)
5. Analyze whether density-adherent predictions are better or worse

### Launch

Starting with [exp 45](../experiment_45/README.md) as the test model — it has the tightest density jitter (±2%/10%) and ranked highest on expert self-evaluation in [53-AR](../experiment_53ar/README.md), though volunteers preferred [exp 44](../experiment_44/README.md). The tight density adherence makes it the best candidate to investigate density over-reliance.

```bash
cd osu/taiko
python experiments/experiment_56/run_ar_analysis.py --checkpoint runs/detect_experiment_45/checkpoints/best.pt
```

## Result

*Pending*

## Lesson

*Pending*
