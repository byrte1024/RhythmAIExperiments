# Experiment 020 — Activation maximization (input dreaming)

## Status

`Planned`

## Context

[#017e](../017e-framewise-bce-regularized/) established the strongest
framewise onset detector: `matched_rate` 0.783, `density_ratio` 1.020,
`dc_human` 92.7 at the optimal sweep point E8/tau=0.40
[exp_017e_framewise_bce_regularized, threshold_sweep.json].
The 017e no_context benchmark showed the model is 95% audio-driven
(F1 drops only ~5% with events zeroed
[exp_017e_framewise_bce_regularized, benchmarks]). But what the model
actually sees in the audio -- which frequency bands, what temporal
patterns, how conditioning modulates the response -- remains opaque.

This experiment applies activation maximization (feature visualization)
to the 017e checkpoint. The model weights are frozen; the input mel
spectrogram `(80, 1000)` becomes a learnable parameter optimized via
gradient descent to produce a target activation map `(500,)`. The
resulting "dreamed" mel reveals what spectral patterns the model
associates with onsets. Griffin-Lim inversion produces listenable
audio from the dreamed spectrograms.

This is an **analysis experiment** -- no model is trained. The
checkpoint is 017e's `best.pt`.

## Citations

- Model checkpoint:
  - [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).
    Best sweep: E8 (step 165,392), tau=0.40, `matched_rate` 0.783,
    `density_ratio` 1.020, `dc_human` 92.7
    [exp_017e_framewise_bce_regularized, threshold_sweep.json].
- Context dependency:
  - [#017e benchmarks](../017e-framewise-bce-regularized/): no_context
    F1 drop ~5%, no_future_audio F1 = 0.000
    [exp_017e_framewise_bce_regularized, benchmarks].
- Technique:
  - Ardila, "Audio DeepDream: Optimizing Raw Audio with Convolutional
    Networks" (2016).
  - Erhan et al., "Visualizing Higher-Layer Features of a Deep Network"
    (2009).
- Implementation: `cli/dream.py`.

---
<!--
PRE-RUN. Do not edit after the run.
-->
---------------------------------------------------------------------

## Hypothesis

### Claim

Activation maximization on the 017e checkpoint will produce dreamed
spectrograms with structured energy concentrated in mel bands 0-30
(low frequency, where taiko drum hits live) and unstructured noise in
bands 40-79 (mid-high frequency, which taiko onsets do not occupy).
Conditioning density will strongly modulate the dream -- higher
`density_mean` will produce denser onset patterns and more sustained
low-band energy -- because the model's `density_ratio` tracks near 1.0
at the optimal threshold [exp_017e, threshold_sweep.json, density_ratio 1.020].
Event context will produce only minor changes to the dreamed mel
because the no_context benchmark showed only ~5% F1 drop
[exp_017e, benchmarks].

### Mechanism

Taiko drums (DON = center hit, KA = rim hit) produce energy
predominantly below 2 kHz. The mel spectrogram's 80 bands span 20 Hz
to 8 kHz; bands 0-30 cover roughly 20-1000 Hz (the taiko fundamental
range), while bands 40-79 cover 1-8 kHz. If the model learned to
detect taiko onsets rather than generic audio transients, optimizing
toward an onset activation should place energy where taiko hits
naturally occur. The upper bands carry no discriminative signal for
taiko onset detection and should receive only regularization-driven
noise.

The density conditioning MLP (3 -> 64 -> 64, FiLM) directly modulates
the transformer via gamma/beta scaling at every layer. A model that
correctly uses density conditioning should produce different dreamed
audio when density_mean changes -- specifically, the dreamed mel at
high density should show more frequent transient-like patterns than at
low density, because the model needs to emit more onsets per second.

Event context enters via scatter-added embeddings on audio tokens at
past onset positions. Since no_context drops F1 by only ~5%, the
dreamed mel should look similar with and without events -- the model
routes primarily through the audio pathway.

### Predicted numbers

| Observable | Predicted | Notes |
|---|---|---|
| Energy in bands 0-30 vs 40-79 | **>= 3:1 ratio** in dreamed mels | Taiko is low-frequency |
| Dreamed mel at onset bins | **Sharp transients** (2-5 frame peaks) | Not diffuse blobs |
| Dreamed mel at non-onset bins | **Low / flat energy** | Model should suppress |
| Cond sweep: density_mean effect | **Strong** -- onset density in dream scales with density_mean | density_ratio ~1.0 proves conditioning works |
| Event sweep: empty vs real delta | **Small** -- dreamed mels visually similar | no_context F1 drop only ~5% |
| Counterfactual: hallucination suppression | **Localized** mel change at the hallucinated bin's time position | Not global |
| Griffin-Lim audio at onset positions | **Audible percussive transients** | Not white noise |

## Success criteria

- **Must:** dreamed mels show structured patterns (not random noise)
  that correlate with target onset positions. Confidence at target bins
  must reach >= 0.9 by iteration 1000.
- **Must:** conditioning sweep produces visibly different dreamed mels
  across density_mean values 0.5 to 12.0.
- **Fails if:** optimization does not converge (loss does not decrease)
  or dreamed mels are indistinguishable from random noise.
- **Nice-to-have:** low-band energy dominance (>= 3:1 ratio bands
  0-30 vs 40-79).
- **Nice-to-have:** Griffin-Lim audio sounds percussive at onset
  positions.

## Changes from baseline

Baseline: [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).

No model changes. This is an analysis experiment using the 017e
`best.pt` checkpoint.

New code:
- `cli/dream.py` -- activation maximization script. Modes: dream,
  counterfactual, saliency. Axes: event sweep (empty vs real),
  conditioning sweep (density_mean 0.5 to 12.0). Outputs: PNG
  visualizations, WAV audio (Griffin-Lim), NPZ data.

## Run config

- Checkpoint: `osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt`
- Dataset: `taiko2_v1`, split `val`
- No training -- analysis only.

### Run 1: GT target dream with sweeps

```bash
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --dataset taiko2_v1 --sample-idx 0 \
    --mode dream --target gt \
    --cond-sweep \
    --out-dir osu/taiko2/experiments/020-activation-maximization/custom/dream_gt_s0 \
    --device cuda
```

### Run 2: Single-onset dream

```bash
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --mode dream --target single --target-bin 250 \
    --cond-sweep \
    --out-dir osu/taiko2/experiments/020-activation-maximization/custom/dream_single \
    --device cuda
```

### Run 3: Metronome dream (sparse vs dense)

```bash
# Sparse: 5 onsets/s (gap=40 bins)
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --mode dream --target metro --metro-gap 40 \
    --out-dir osu/taiko2/experiments/020-activation-maximization/custom/dream_metro_sparse \
    --device cuda

# Dense: 20 onsets/s (gap=10 bins)
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --mode dream --target metro --metro-gap 10 \
    --out-dir osu/taiko2/experiments/020-activation-maximization/custom/dream_metro_dense \
    --device cuda
```

### Run 4: Saliency on real samples

```bash
for idx in 0 1 2 3 4; do
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --dataset taiko2_v1 --sample-idx $idx \
    --mode saliency --target gt \
    --out-dir osu/taiko2/experiments/020-activation-maximization/custom/saliency_s${idx} \
    --device cuda
done
```

### Run 5: Counterfactual

```bash
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.dream \
    --checkpoint osu/taiko2/runs/exp_017e_framewise_bce_regularized/checkpoints/best.pt \
    --dataset taiko2_v1 --sample-idx 0 \
    --mode counterfactual --target gt \
    --out-dir osu/taiko2/experiments/020-activation-maximization/custom/counterfactual_s0 \
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
