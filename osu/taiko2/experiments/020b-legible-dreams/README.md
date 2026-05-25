# Experiment 020b — Legible audio dreams

## Status

`Planned`

## Context

[#020](../020-activation-maximization/) established activation
maximization on the 017e checkpoint. The key finding was that dreamed
mels had near-zero energy everywhere -- the optimizer converged by
making tiny perturbations to noise rather than building structured
spectrograms. This limited interpretability: band-energy ratios and
mel-confidence correlations were near zero because both numerator and
denominator were near zero
[020-activation-maximization/custom/dream_gt_s0/chart_00_*/dream_empty_analysis.npz,
onset_mean_energy=-0.004, notonset_mean_energy=0.002].

The saliency and confidence findings from #020 were robust (high-band
dominance, bin-exact confidence, dense pattern capacity wall), but
the dreamed spectrograms themselves were not human-interpretable.
The Griffin-Lim audio sounded like noise.

This experiment reruns the same dreams with three changes designed to
produce legible, structured spectrograms:

1. **Moderately lower regularization** (lambda_tv 0.01 -> 0.003,
   lambda_l2 0.001 -> 0.0003, both 3x lower). Allows the optimizer
   to build larger energy structures while still smoothing noise.
2. **Realistic initialization** from per-band dataset statistics
   (mean 10-26 dB across bands, std 11-17 dB) instead of flat
   Gaussian noise at mean=15. The optimizer perturbs a realistic
   spectral baseline.
3. **Realism penalty** (lambda_realism=0.01) that penalizes per-band
   mean deviation from the dataset distribution. This is the
   dominant regularizer -- keeps the dream mel-like rather than
   drifting into adversarial noise. Stronger than the TV/L2 terms.
4. **Adam optimizer** retained (not L-BFGS). Adam's noisy gradients
   help avoid sharp adversarial patterns. 2000 iterations (fewer
   than #020's 3000 since realistic init gives a head start).

## Citations

- Direct baseline:
  - [#020 -- activation maximization](../020-activation-maximization/).
    Dreamed mels had near-zero energy, confidence bin-exact, high-band
    saliency dominance
    [020-activation-maximization/custom/dream_gt_s0/chart_00_*/dream_empty_analysis.npz].
- Model checkpoint:
  - [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).
    `matched_rate` 0.783, `dc_human` 92.7
    [exp_017e_framewise_bce_regularized, threshold_sweep.json].
- Technique:
  - Ardila 2016: L-BFGS recommended for audio DeepDream.
- Implementation: `cli/dream.py --preset legible`.

---
<!--
PRE-RUN. Do not edit after the run.
-->
---------------------------------------------------------------------

## Hypothesis

### Claim

The `legible` preset (3x lower TV/L2, strong realism penalty,
realistic init) will produce dreamed spectrograms with visible
transient structures at onset positions, realistic per-band energy
profiles, and Griffin-Lim audio that sounds percussive rather than
noise-like. The high-band saliency finding from #020 should manifest
as visible high-band energy at onset frames in the dreamed mel.

### Mechanism

#020's near-zero energy was caused by the optimizer finding a local
minimum where tiny perturbations suffice. Two factors conspired:
(a) TV loss smoothed away emerging structure, (b) flat noise
initialization at mean=15 meant the optimizer had no realistic
spectral baseline to perturb from. The legible preset addresses both:
moderately lower TV/L2 (3x, not 10x) allows structure to form without
producing adversarial noise, and realistic init provides a meaningful
starting point. The realism penalty (the strongest regularizer) keeps
per-band means near the dataset distribution so energy ratios are
interpretable and the dreamed mel doesn't drift into random noise.

### Predicted numbers

| Observable | #020 result | Predicted (#020b) | Notes |
|---|---|---|---|
| Dreamed mel energy range | -0.06 to +0.07 dB | **10-30 dB** | Realistic range |
| Onset vs non-onset energy delta | ~0 | **>= 3 dB** | Visible difference |
| Band energy at onset: low vs high | unmeasurable | **High bands brighter** | Matches saliency finding |
| Confidence at target bins | 0.96 | **>= 0.90** | May be lower with less regularization |
| Griffin-Lim audio | noise | **Percussive clicks** | Realistic init helps inversion |
| Dreamed-real per-band correlation | ~0 | **>= 0.3** | Realistic init anchors the profile |

## Success criteria

- **Must:** dreamed mel energy in the 5-40 dB range (not near zero).
- **Must:** visible onset-aligned vertical structures in the dreamed
  mel spectrogram.
- **Must:** confidence at target bins >= 0.80.
- **Fails if:** dreamed mels are pure random noise (realism penalty
  too weak) or worse than #020 (regularization too aggressive).
- **Nice-to-have:** Griffin-Lim audio has audible transients at onset
  positions.
- **Nice-to-have:** band-energy analysis shows high-band selectivity
  at onset frames, consistent with #020's saliency finding.

## Changes from baseline

Baseline: [#020 -- activation maximization](../020-activation-maximization/).

One code change:
- `cli/dream.py` -- added `--preset legible` flag. Legible preset:
  Adam optimizer (2000 iterations), lambda_tv=0.003, lambda_l2=0.0003,
  lambda_realism=0.01, realistic_init=True (per-band dataset mean/std).
  All analysis outputs (temporal, band, past events) are identical.

## Run config

- Checkpoint: `osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt`
- Dataset: `taiko2_v1`, split `val`
- No training -- analysis only.

### Run 1: GT dream with legible preset

```bash
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --dataset taiko2_v1 --n-charts 5 \
    --mode dream --target gt \
    --preset legible \
    --cond-sweep \
    --out-dir osu/taiko2/experiments/020b-legible-dreams/custom/dream_gt \
    --device cuda
```

### Run 2: Single onset + metronome

```bash
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --mode dream --target single --target-bin 250 \
    --preset legible \
    --out-dir osu/taiko2/experiments/020b-legible-dreams/custom/dream_single \
    --device cuda

osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --mode dream --target metro --metro-gap 40 \
    --preset legible \
    --out-dir osu/taiko2/experiments/020b-legible-dreams/custom/dream_metro_sparse \
    --device cuda
```

---------------------------------------------------------------------
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
---------------------------------------------------------------------

## Results summary

<!-- TODO: fill after runs -->

## Visualizations

<!-- TODO: fill after runs -->

## Vs prediction

<!-- TODO: fill after runs -->

## Takeaways

<!-- TODO: fill after runs -->
