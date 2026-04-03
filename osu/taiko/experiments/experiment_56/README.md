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

Ran on 48 val songs (2 skipped — no audio found).

### Summary:

| Metric | Value |
|---|---|
| Avg density ratio (pred/gt) | 0.83 (under-predicts by 17%) |
| Avg matched (<25ms) | 67.1% |
| Avg close (<50ms) | 70.0% |
| Avg far (>100ms) | 19.8% |
| Avg hallucination (>100ms from any GT) | 14.9% |
| Avg GT error median | 16ms |

### Key findings:

1. **Model consistently under-predicts density** (ratio 0.83 avg). Not a single song is significantly over-predicted. Worst: monoqlom (d=5.7, ratio 0.41), Mrs GREEN APPLE (d=6.4, ratio 0.54).

2. **Hallucination inversely correlates with density**: Low density songs (d=1-3) have 25-50% hallucination — the model fills silence with notes. High density songs (d=7+) have 1-7% hallucination — the model is conservative.

3. **Catch rate has no clear density correlation**: Songs across all densities achieve 60-85% close rate. The worst performers are sparse songs with long silences, not dense songs.

4. **Density conditioning is asymmetric**: The model can't be pushed to produce enough notes at high density, but also can't be restrained at low density. It defaults to a conservative ~3-5 events/sec regardless of conditioning.

### Density adherence by range:

| Density range | Avg ratio | Avg hallucination |
|---|---|---|
| d < 3.5 | 0.88 | 28.3% |
| 3.5 < d < 5.5 | 0.84 | 14.0% |
| d > 5.5 | 0.77 | 8.7% |

Higher conditioned density → worse adherence but lower hallucination.

## Lesson

The model under-predicts density across the board. Density conditioning has limited control — the model's output density is driven more by audio content than by the conditioning signal. At low density the model hallucinates to fill silence; at high density it can't produce enough notes even when told to.

[Exp 56-B](../experiment_56b/README.md) tests whether varying density (0.8x, 1.0x, 1.2x) actually changes model output, or if the conditioning is largely ignored.
