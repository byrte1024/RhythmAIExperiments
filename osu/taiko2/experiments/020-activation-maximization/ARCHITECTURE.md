# Experiment 020 — Activation maximization · Architecture

> **This document is self-contained.** It must describe everything needed to
> reproduce the experiment from scratch: data pipeline, model, loss,
> training schedule, inference procedure, environment. No cross-references
> to other experiments, other documents, or external URLs. Links and
> citations belong in `README.md`.

---

## Task

This is an analysis experiment, not a training experiment. No model is
trained. A frozen FramewiseDetector checkpoint is probed by optimizing
the input mel spectrogram to produce a target activation map. The
resulting "dreamed" mel reveals what spectral patterns the model
associates with onset detection.

## Frozen model

The model whose weights are frozen is a FramewiseDetector (~21.89 M
params) trained with symmetric BCE + label smoothing 0.05 + head
dropout 0.2. Checkpoint: `best.pt` selected by frame/f1_tau_50_tol_2.

### Model inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| mel | (1, 80, 1000) | float32 | Log-mel spectrogram; 80 bands, 1000 time frames at 5 ms/frame (500 past + 500 future). |
| event_offsets | (1, 128) | int64 | Last 128 onset bin positions relative to cursor. |
| event_mask | (1, 128) | bool | True = padding, False = real event. |
| conditioning | (1, 3) | float32 | [density_mean, density_peak, density_std]. |

### Model output

| Name | Shape | Dtype | Description |
|---|---|---|---|
| logits | (1, 500) | float32 | Raw pre-sigmoid logits per future-time bin. |
| confidence_map | (1, 500) | float32 | sigmoid(logits), detached. |

---

## Optimization procedure

### Learnable variable

The mel spectrogram `(1, 80, 1000)` is the only optimized tensor. All
other inputs (events, conditioning) and all model weights are frozen.

### Initialization

- **Dream mode**: Gaussian noise, mean=15.0, std=5.0 (approximating
  the training distribution mean ~18, range -31 to +53 dB).
- **Counterfactual mode**: Real mel from a dataset sample.
- **Saliency mode**: No optimization; single backward pass.

### Target map

A binary vector `(500,)` specifying which future-time bins should have
high confidence. Built from one of:

| Target | Description |
|---|---|
| gt | Ground truth onset positions from a dataset sample. |
| single | One active bin at a specified position (default: bin 250). |
| metro | Evenly spaced onsets at a given gap (e.g., every 40 bins = 5/s). |
| model | Threshold the model's own prediction on a real sample. |

### Loss function

```
loss = BCE(logits, target_map)
     + lambda_tv * TV(mel)
     + lambda_l2 * ||mel||_2
```

where:
- `BCE` = `F.binary_cross_entropy_with_logits`, mean reduction.
- `TV(mel)` = total variation: mean absolute differences along both
  the frequency axis (band-to-band) and time axis (frame-to-frame).
  Encourages spatial smoothness, suppresses adversarial noise.
- `||mel||_2` = L2 norm of the full mel tensor. Prevents values from
  drifting to extreme magnitudes.

For counterfactual mode, an additional perturbation budget term:

```
loss += 0.1 * relu(||mel - mel_anchor||_2 - budget)
```

This penalizes deviations beyond the L2 budget from the original mel,
keeping changes localized.

### Clamping

After each optimizer step, mel values are hard-clamped to
`[mel_min, mel_max]` = `[-35, 55]` dB. This range covers the observed
training distribution (min ~-31, max ~+53 across the dataset).

### Jitter

On even-numbered iterations, the mel is randomly shifted by up to
`jitter_px=2` pixels along both axes before the forward pass. The
shift is not applied to the stored mel parameter, only to the forward
pass input. This reduces grid-aligned artifacts (per Mordvintsev et al.
2015, adapted from image DeepDream).

### Optimizer and schedule

| Param | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.03 |
| LR schedule | CosineAnnealingLR, eta_min=0.001 |
| Iterations | 3000 |
| lambda_tv | 0.01 |
| lambda_l2 | 0.001 |
| mel_min | -35.0 |
| mel_max | 55.0 |
| jitter_px | 2 |
| perturbation_budget (counterfactual) | 50.0 |
| seed | 42 |

---

## Axes of variation

Each dream is run across two axes, producing multiple outputs per
target.

### 1. Event context (automatic)

Every dream runs twice when a dataset sample is available:

| Mode | event_offsets | event_mask | Purpose |
|---|---|---|---|
| empty | zeros | all True (all padding) | Audio-only: what does the model see without event history? |
| real | from dataset sample | from dataset sample | Audio + context: how does event history change the dream? |

### 2. Conditioning sweep (--cond-sweep flag)

Runs the same dream at 7 density_mean values with empty events:

| density_mean | Interpretation |
|---|---|
| 0.5 | Very sparse chart (~0.5 onsets/sec) |
| 1.0 | Sparse |
| 2.0 | Below average |
| 3.0 | Average (dataset mean ~2.75) |
| 5.0 | Above average |
| 8.0 | Dense |
| 12.0 | Very dense |

density_peak and density_std are held at the sample's values (or
defaults: peak=8.0, std=2.0).

---

## Saliency mode

No optimization. For a real mel input:

1. Enable gradients on the mel tensor.
2. Forward pass through the frozen model.
3. Compute `sum(sigmoid(logits))` and backpropagate.
4. The gradient `d(conf_sum) / d(mel)` is the saliency map `(80, 1000)`.

Positive saliency at `(band, frame)` means increasing energy there
increases total onset confidence. Negative means it suppresses onsets.

---

## Mel-to-audio inversion

Dreamed mels are converted to listenable audio via Griffin-Lim:

1. Reconstruct the mel filterbank matrix from the training config
   (sr=22000, n_fft=2048, n_mels=80, fmin=20, fmax=8000).
2. Invert the mel filterbank via pseudoinverse to get an approximate
   STFT magnitude.
3. Convert from dB to power, then power to magnitude
   (sqrt for power=2.0).
4. Run Griffin-Lim phase estimation (64 iterations).
5. Normalize peak to 0.9.
6. Save as 22 kHz mono WAV.

Griffin-Lim produces approximate audio with phase artifacts. The audio
is for qualitative assessment (does it sound percussive? tonal?
noise-like?) not quantitative analysis.

---

## Output artifacts

All mel visualizations share a common layout:
- White dashed vertical line at frame 500 = cursor position.
- "PAST" / "FUTURE" labels on each half.
- Cyan vertical lines = target onset positions (future half only).
- Yellow vertical lines = past event positions (when event context is
  "real").
- Confidence plots use x-axis "Future bin (0-499, each = 5ms)".

### Per dream run (one target + one event mode)

| Artifact | Description |
|---|---|
| `{slug}_mel.png` | Real mel vs dreamed mel with cursor/onset/event annotations, plus target vs output confidence maps. |
| `{slug}_trajectory.png` | Optimization loss curve (log-y) + confidence at target/non-target bins over 3000 iterations. |
| `{slug}_dreamed.wav` | Griffin-Lim audio from dreamed mel. |
| `{slug}_data.npz` | Raw data: dreamed_mel, confidence_map, target_map, trajectory arrays. |
| `{slug}_analysis.png` | 2x2 grid: (1) per-band energy at onset vs non-onset frames, (2) per-band onset selectivity delta, (3) band-group summary bar chart, (4) mel energy vs confidence scatter with Pearson r for all/low/high bands. |
| `{slug}_vs_real.png` | 2x2 grid: (1) onset frame per-band profile dreamed vs real, (2) non-onset frame per-band profile dreamed vs real, (3) per-band dreamed-real Pearson r, (4) mel value distribution histogram dreamed vs real. |
| `{slug}_analysis.npz` | Numeric data: onset_band_mean, notonset_band_mean, band_delta, per_frame_energy, low/high_band_energy, confidence_map, corr_all/low/high_bands, low_high_energy_ratio, onset/notonset_mean_energy. |

### Per event sweep

| Artifact | Description |
|---|---|
| `events_sweep.png` | Annotated dreamed mels for empty vs real events (with cursor/onset/event markers), plus overlaid confidence maps. |

### Per conditioning sweep

| Artifact | Description |
|---|---|
| `cond_sweep.png` | Annotated dreamed mels at all 7 density_mean values (with cursor/onset markers), plus overlaid confidence maps. |
| `cond_dm{N}_dreamed.wav` | Griffin-Lim audio per density value. |
| `cond_dm{N}_analysis.png` | Per-band analysis for each density value. |
| `cond_dm{N}_vs_real.png` | Dream-vs-real comparison for each density value. |
| `cond_sweep_data.npz` | All dreamed mels and confidence maps. |

### Per saliency run

| Artifact | Description |
|---|---|
| `saliency.png` | Input mel with cursor line, saliency map (RdBu diverging colormap) with cursor line, confidence map (future bins only). |
| `saliency.npz` | Raw saliency, confidence_map, mel arrays. |

### Per experiment run

| Artifact | Description |
|---|---|
| `manifest.json` | List of all charts processed with chart_id, density_mean, n_onsets, mode. |
| `real.wav` | Griffin-Lim audio from real mel (one per chart). |

### Chart selection

Charts are selected by sorting all val-split charts with
density_mean in [2.0, 7.0] by density, then picking N evenly spaced
across the sorted list. This ensures coverage from low-density
(~2 onsets/s) to high-density (~7 onsets/s) charts while excluding
extreme outliers.

---

## Dataset

Real samples are drawn from:

- Name: taiko2_v1
- Split: val
- Audio sampler: MelSampler (80 bands, 22 kHz, hop 110)
- Mel value range: approximately -31 to +53 dB
- Mean mel value: approximately 15-19 dB

---

## Environment

| Component | Version |
|---|---|
| Python | 3.13.13 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070, 11.5 GB VRAM |
| OS | CachyOS Linux (Arch-based), kernel 7.0.9-1-cachyos |
| numpy | 2.4.2 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |
| soundfile | (system) |

---

## Addenda

(None yet.)
