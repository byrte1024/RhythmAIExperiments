# Experiment 019 — Coincidence input · Architecture

> **This document is self-contained.** It must describe everything needed to
> reproduce the experiment from scratch: data pipeline, model, loss,
> training schedule, inference procedure, environment. No cross-references
> to other experiments, other documents, or external URLs. Links and
> citations belong in `README.md`.

---

## Task

Predict a per-bin activation map over 500 future-time bins (2.5 s at
5 ms/bin) given the current cursor position, 500 bins of past audio,
500 bins of future audio, and up to 128 past onset events. Each bin is
independently classified as onset-present (1) or onset-absent (0) via
a single forward pass. No STOP class.

## Inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| mel | (B, 93, 1000) | float32 | Concatenated input: 80 log-mel bands + 13 coincidence summary rows. 1000 time frames at 5 ms/frame (500 past + 500 future). See "Input channel layout" below. |
| event_offsets | (B, 128) | int64 | Last 128 onset bin positions relative to the cursor (negative = past). |
| event_mask | (B, 128) | bool | True = padding, False = real event. |
| conditioning | (B, 3) | float32 | [density_mean, density_peak, density_std] from chart metadata. |

### Input channel layout

The 93-channel input tensor is formed by concatenating the mel and
coincidence arrays along the frequency axis at sample-construction time:

```
channels 0-79  : log-mel spectrogram (80 bands, 20 Hz - 8000 Hz)
channels 80-92 : coincidence summary (13 rows)
```

**Coincidence summary row definitions (channels 80-92):**

| Row index | Channel | Contents |
|---|---|---|
| 0 | 80 | LSH onset-type color — R component. Locality-sensitive hash of the onset's spectral fingerprint, red channel. Metronomic beats cluster to similar R values; unusual onsets scatter. |
| 1 | 81 | LSH onset-type color — G component. Green channel of the same LSH hash. |
| 2 | 82 | LSH onset-type color — B component. Blue channel of the same LSH hash. |
| 3 | 83 | IDF population / importance weight. Inverse-document-frequency score across the full track corpus. Low for common/expected onsets (metronomic beats), high for rare/important ones (chart highlights). Values in [0, 1] after normalization. |
| 4 | 84 | Spike energy. Magnitude of the onset spike in the flux signal, independent of IDF. Captures loudness of the onset event. |
| 5 | 85 | Band-group average 0 (lowest frequency band group). |
| 6 | 86 | Band-group average 1. |
| 7 | 87 | Band-group average 2. |
| 8 | 88 | Band-group average 3. |
| 9 | 89 | Band-group average 4. |
| 10 | 90 | Band-group average 5. |
| 11 | 91 | Band-group average 6. |
| 12 | 92 | Band-group average 7 (highest frequency band group). |

Rows 5-12 partition the spectrum into 8 equal-width frequency-band groups
and store the average spectral energy per group at each frame. Together
they encode the spectral shape of each onset (bass-heavy, treble-heavy,
or broadband), complementing the IDF importance signal with timbre information.

## Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| logits | (B, 500) | float32 | Raw pre-sigmoid logits; one per future-time bin. |
| confidence_map | (B, 500) | float32 | sigmoid(logits), detached. Values in [0, 1]. |

---

## Data pipeline

### Audio preprocessing (mel)

| Param | Value |
|---|---|
| Sample rate | 22 000 Hz |
| FFT size | 2048 samples |
| Hop length | 110 samples = 5.000 ms/frame |
| Mel bands | 80 |
| Frequency range | 20 Hz - 8 000 Hz |
| Power spectrum | power=2.0 |
| Amplitude to dB | top_db=80 |
| On-disk dtype | float16 |
| Served dtype | float32 |

### Coincidence preprocessing

The coincidence summary is computed by the ``CoincidenceMelSampler``
(``samplers/coincidence_mel.py``) at dataset build time. The sampler
runs the mel pipeline and the coincidence pipeline in sequence,
concatenates the outputs, and writes a single ``(93, T)`` float16
feature file to disk. The same sampler is used at inference time to
produce ``(93, T)`` features from raw audio.

**Pipeline (executed inside ``CoincidenceMelSampler._transform``):**

1. **Mel spectrogram** — same parameters as above (22 000 Hz, FFT 2048,
   hop 110, 80 bands, 20-8000 Hz, power=2.0, top_db=80). Produces
   rows 0-79.
2. **Spectral flux** — half-wave rectified first-order difference along
   the time axis of the mel power spectrogram.
3. **Spike detection** — local maxima of the flux signal above an
   adaptive threshold (median + k * MAD, per-band).
4. **IDF weighting** — per-spike inverse-document-frequency score based
   on how many bands fire simultaneously. Common patterns receive low
   IDF; rare patterns receive high IDF. Normalized to [0, 1] per track.
5. **LSH coloring** — locality-sensitive hashing maps each onset's
   band co-activation pattern to an RGB color triplet via stable random
   projections.
6. **13-row summary** — rows 80-92 of the output:
   - Rows 80-82: LSH R, G, B (onset type identity)
   - Row 83: IDF population (onset importance)
   - Row 84: total spike energy
   - Rows 85-92: 8 frequency-band-group averages of spike confidence

**On-disk format:** single ``features/{stem}.npy`` file per chart,
shape ``(93, T)`` float16. The first 80 rows are mel; the last 13
are coincidence summary. No separate coincidence files — everything
is in one array.

**Dataset preparation:**

```bash
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.prepare_dataset \
    --name taiko2_v1_coin \
    --charts-dir /path/to/osz/packs/ \
    --audio-sampler coincidence_mel
```

**Temporal alignment:** hop_length=110 is shared between mel and
coincidence (both computed from the same waveform in the same sampler
call). The T dimension is identical by construction.

### Event encoding

| Param | Value |
|---|---|
| Bin duration | 5.000 ms |
| Grid rate | 200 bins/second |
| Onset kinds retained | DON, KA, BIG_DON, BIG_KA, DRUMROLL, SPINNER |
| Bin index formula | floor(time_ms / bin_ms) |

Kind codes:

| Code | Kind |
|---|---|
| 0 | DON |
| 1 | KA |
| 2 | BIG_DON |
| 3 | BIG_KA |
| 4 | DRUMROLL |
| 5 | SPINNER |
| 6 | UNKNOWN |

### Sample construction

| Param | Value |
|---|---|
| Past audio bins (A_BINS) | 500 |
| Future audio bins (B_BINS) | 500 |
| Past events (C_EVENTS) | 128 |
| Future events (D_EVENTS) | 100 |
| Min cursor bin | 6000 |
| Allowed overlap forward | 0 |
| Allowed overlap backward | 0 |

### Target construction

Binary activation map (B, 500) with value 1.0 at each bin where a GT
onset falls, 0.0 elsewhere. No Gaussian smoothing (sigma = None). The
label smoothing applied at loss time transforms these 0/1 targets into
0.05/0.95 before computing BCE; the raw targets stored in the batch
remain {0, 1}.

### Train/val split

- Song-level grouping by beatmapset_id.
- Split seed: 42.
- Ratios: train = 0.9, val = 0.1.

### Augmentations

| Augmentation | Probability | Parameters |
|---|---|---|
| Time stretch | 30% | log-uniform in [1/1.4, 1.4] |
| Mel gain | 30% | +/-2 dB uniform — applied to mel channels 0-79 only |
| Mel noise | 15% | Gaussian sigma uniform in [0.1, 0.3] — applied to mel channels 0-79 only |
| Frequency roll | 15% | shift in {-3, ..., +3} — mel rows (0-79) only; coincidence rows (80-92) are not rolled |
| SpecAug freq mask | 20% | 1 mask, width <= 10 bands — mel channels 0-79 only |
| SpecAug time mask | 20% | 1 mask, width <= 30 frames — applied to all 93 channels |
| Event jitter | 100% | global +/-3 bins + per-event +/-3 * recency-scaled |
| Event deletion | 5% | drop 1-2 random past events |
| Event insertion | 3% | add 1 synthetic event between two reals |
| Partial metronome (recent half) | 2% | replace with evenly-spaced events |
| Partial adv metronome (older half) | 2% | replace with dominant-gap-spaced events |
| Large time shift | 2% | +/-50 bin shift on 2-4 recent events |
| Context truncation | 5% | keep only most recent 8-32 events |
| Conditioning jitter | 10% | +/-2% on density_mean / peak / std |

**Frequency-axis augmentations apply to mel rows only (0-79).** The
coincidence rows (80-92) represent onset-detection features indexed by
time, not by frequency band. Rolling them along the frequency axis would
corrupt their meaning. Mel gain, mel noise, freq roll, and SpecAug freq
mask therefore operate on a slice of the input tensor (channels 0-79).

Order: time stretch first, then audio augs, then event augs, then
conditioning jitter.

---

## Benchmarks

Two new benchmarks in addition to the standard 017e set:

| Benchmark | Description |
|---|---|
| `no_coincidence` | Zero out channels 80-92 at eval time. Measures model reliance on coincidence channels. Expected: >= 3 % F1 drop if hypothesis holds. |
| `no_mel` | Zero out channels 0-79 at eval time. Measures whether coincidence alone is informative. Expected: large F1 drop (model cannot localise onsets precisely from coincidence alone). |

All other benchmarks from the 017e series (e.g., `no_past_audio`,
`no_context`, `no_conditioning`) are also run.

---

## Model architecture

Total parameters: ~21.92 M (vs 017e's ~21.89 M — difference from wider
conv stem input layer only).

### 1. Conditioning MLP

```
conditioning (B, 3) -> Linear(3, 64) -> GELU -> Linear(64, 64) -> cond (B, 64)
```

### 2. Conv stem

4x downsample; 1000 frames -> 250 tokens. Input is now (B, 93, 1000)
(80 mel channels + 13 coincidence channels).

```
mel (B, 93, 1000)
  -> Conv1d(in=93, out=192, kernel=7, stride=2, padding=3)
  -> GELU -> GroupNorm(1, 192)
  -> Conv1d(in=192, out=384, kernel=7, stride=2, padding=3)
  -> GELU -> transpose -> (B, 250, 384)
  -> LayerNorm(384) + SinusoidalPosEmb(0..249) + FiLM(cond)
```

Cursor at token index 125.

The first Conv1d layer is the only weight-incompatible change from 017e:
Conv1d(80, 192, ...) becomes Conv1d(93, 192, ...). All downstream layers
are identical.

### 3. Event embeddings

5-component feature vectors per event:

| Component | Dim | Description |
|---|---|---|
| presence | 384 | Learned parameter broadcast to every slot |
| gap_before | 384 | SinusoidalPosEmb of inter-onset gap, clamped >= 1 |
| gap_after | 384 | SinusoidalPosEmb of next gap; last valid uses gap_before as proxy |
| gap_ratio_before | 384 | SinusoidalPosEmb of ratio * 50, clamped [0.1, 10] |
| gap_ratio_after | 384 | SinusoidalPosEmb of ratio * 50, clamped [0.1, 10] |

Concatenated: (B, 128, 1920) -> Linear(1920, 384) -> GELU ->
Linear(384, 384). Scatter-added onto audio tokens at past positions.

### 4. Transformer trunk

8 layers of pre-norm encoder blocks:

```
for each of 8 layers:
    x = TransformerEncoderLayer(
            d_model=384, nhead=8, dim_feedforward=1536,
            dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )(x)
    x = FiLM(cond)(x)
```

### 5. Output head — Conv1D-on-bin-axis

head_dropout = 0.2. Each Dropout layer between Conv1D blocks uses p=0.2.

```
cursor_token = x[:, 125, :]           (B, 384)
audio_features = x[:, 125:, :]        (B, 125, 384)

Per-bin channels:
  pos_embed:       (B, 32, 500)   sinusoidal over bin positions
  audio_features:  (B, 384, 500)  linearly interpolated from 125 tokens
  cursor_broadcast:(B, 32, 500)   Linear(384, 32) broadcast

h: (B, 448, 500)
  -> Conv1d(448, 256, k=31, p=15) -> GroupNorm(8) -> SiLU -> Dropout(0.2)
  -> Conv1d(256, 256, k=15, p=7)  -> GroupNorm(8) -> SiLU -> Dropout(0.2)
  -> Conv1d(256, 256, k=15, p=7)  -> GroupNorm(8) -> SiLU -> Dropout(0.2)
  -> Conv1d(256, 1, k=1)
  -> logits: (B, 500)
```

confidence_map = sigmoid(logits).detach()

### FiLM module

```
cond (B, 64) -> Linear(64, 768) -> split (gamma, beta) each (B, 384)
output = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
```

Weight + bias initialized to zero.

---

## Loss

**Binary cross-entropy with logits**, per-bin, with **no**
positive-class upweighting and **label smoothing (eps=0.05)**.

### Target after smoothing

Raw binary targets {0, 1} are smoothed before BCE:

```
smoothed_target = target * (1 - eps) + (1 - target) * eps
                = target * 0.95 + (1 - target) * 0.05
```

GT-positive bins: target becomes 0.95 (not 1.0).
GT-negative bins: target becomes 0.05 (not 0.0).

This prevents the model from being rewarded for pushing logits to
+inf on GT-positive bins or -inf on GT-negative bins. When the model
already predicts sigmoid(logit) near 0.95/0.05, the gradient is
near-zero regardless of how confident the raw logit is — memorized
training examples stop receiving gradient updates. The BCE loss minimum
under label smoothing is achieved at exactly the smoothed target
values, not at {0, 1}.

### Positive-class weight

Per-sample: `pos_weight = clamp(n_neg / max(n_gt, 1), min=1, max=1)`.

The clamp always returns 1.0, regardless of window density.

Dataset statistics (89,753 sampled windows):
- GT=1 bins: 2.54% (38.4:1 neg-to-pos)
- GT onsets per window: median 11, p5=3, p95=27

Effect of the clamp:

| Window type | n_gt | raw weight | clamped [1, 1] |
|---|---:|---:|---:|
| Sparse (p5) | 3 | 165.7 | 1.0 |
| Below median | 7 | 70.4 | 1.0 |
| Median | 11 | 44.5 | 1.0 |
| Above median | 17 | 28.4 | 1.0 |
| Dense (p95) | 27 | 17.5 | 1.0 |
| Very dense | 52 | 8.6 | 1.0 |

All window types receive weight 1.0. A missed onset costs exactly as
much as a false positive. The gradient is symmetric between recall and
precision.

### Loss formula

```
smoothed = target * 0.95 + (1 - target) * 0.05
per_bin = F.binary_cross_entropy_with_logits(
    logits, smoothed, weight=weight_map, reduction="none",
)
loss = per_bin.mean()
```

where `weight_map` is all-ones (every bin, every sample, weight=1.0).

### Hyperparameters

| Param | Value |
|---|---|
| pos_weight_clamp_min | 1.0 |
| pos_weight_clamp_max | 1.0 |
| label_smoothing | 0.05 |
| canonical_threshold | 0.5 |
| canonical_tolerance_frames | 2 |

---

## Training

metric_to_watch = `frame/mini/tau50/fps_50/binary_f1` (higher is better).
The trainer saves `best.pt` at the eval with the highest mini-chart
binary F1 at 50 FPS (20 ms frames). This metric was found in #017f to
track AR chart quality better than frame-level F1, peaking later in
training (E15 vs E11).

| Param | Value |
|---|---|
| Optimizer | AdamW, lr=3e-4, wd=0.01 |
| Gradient clip | 1.0 |
| Batch size | 64 |
| Epochs | 15 |
| Scheduler | CosineAnnealingLR |
| torch.compile | on (triton backend) |
| Balanced sampling | off |
| Evals per epoch | 4 |
| Watched metric | frame/mini/tau50/fps_50/binary_f1 (higher is better) |
| metric_lower_is_better | false |
| Train-noaug fraction | 0.05 |
| Benchmarks | all (including no_coincidence, no_mel) |
| Seed | 42 |

---

## Inference (autoregressive)

1. Forward pass -> sigmoid -> confidence_map.
2. NMS via max_pool1d(kernel=3).
3. Threshold at 0.4.
4. Empty positive set -> cursor += 20 bins.
5. Else: emit all passing bins, cursor advances to last emitted.

The decode_threshold of 0.4 is the optimal threshold identified via the
017e post-run threshold sweep (320 configs: 16 checkpoints x 4 thresholds
x 5 max_notes). It is used here as the starting inference configuration;
a new threshold sweep after this run may identify a different optimal.

| Param | Value |
|---|---|
| decode_threshold | 0.4 |
| nms_kernel | 3 |
| stop_hop_bins | 20 |
| min_emit_gap_bins | 1 |
| max_notes_per_step | 0 |

---

## Metrics

The following per-eval metrics are logged and tracked:

### Frame-level (validation set, tau=50 threshold, tol=2 frames)

| Metric | Description |
|---|---|
| val/single/frame/precision_tau_50_tol_2 | Precision at threshold 0.50, 2-frame tolerance |
| val/single/frame/recall_tau_50_tol_2 | Recall at threshold 0.50, 2-frame tolerance |
| val/single/frame/f1_tau_50_tol_2 | F1 at threshold 0.50, 2-frame tolerance |
| val/single/frame/auc_pr | Area under precision-recall curve |
| val/single/frame/auc_roc | Area under ROC curve |
| val/single/frame/pos_rate_pred_50 | Fraction of predicted-positive bins at tau=0.50 |
| val/single/frame/pos_rate_target | Fraction of GT-positive bins |
| val/single/frame/mean_act_pos | Mean confidence on GT-positive bins |
| val/single/frame/mean_act_neg | Mean confidence on GT-negative bins |
| val/single/frame/separation | mean_act_pos - mean_act_neg |
| val/single/frame/brier | Brier score (mean squared error of probabilities) |
| val/single/frame/conf_tp_median | Median confidence on true positives |
| val/single/frame/conf_fn_median | Median confidence on false negatives |
| val/single/frame/conf_fp_median | Median confidence on false positives |
| val/single/frame/conf_tn_median | Median confidence on true negatives |

### Loss components

| Metric | Description |
|---|---|
| val/single/loss | Total BCE loss (computed on smoothed targets) |
| val/single/loss/pos_only | Loss on GT-positive bins only |
| val/single/loss/neg_only | Loss on GT-negative bins only |
| val/single/loss/pos_neg_ratio | Ratio pos_only / neg_only |

### Frame-level FPS resolution (validation set)

Per threshold, each val window is compared at 8 temporal resolutions
(1, 2, 4, 10, 20, 50, 100, 200 FPS).

| Metric | Description |
|---|---|
| frame/mini/tau{T}/fps_{F}/binary_f1 | Binary F1 at F FPS, threshold T/100 |
| frame/mini/tau{T}/fps_{F}/binary_precision | Binary precision at F FPS |
| frame/mini/tau{T}/fps_{F}/binary_recall | Binary recall at F FPS |
| frame/mini/tau{T}/fps_{F}/count_mae | Count MAE per frame at F FPS |
| frame/mini/tau{T}/fps_{F}/count_corr | Count correlation at F FPS |

The watched metric is frame/mini/tau50/fps_50/binary_f1.

### AR corpus metrics (inference pass per eval)

| Metric | Description |
|---|---|
| corpus/precision | Onset precision at 25ms tolerance |
| corpus/recall | Onset recall at 25ms tolerance |
| corpus/f1 | Onset F1 at 25ms tolerance |
| corpus/density_ratio | Emitted notes/s / GT notes/s |
| corpus/dc_human | Pattern quality score [0, 100] |
| corpus/oc_human | Overlap correctness [0, 100] |
| corpus/error_median_ms | Median timing error on matched notes |
| corpus/binary_f1_at_{F}fps_median | Median binary F1 at F FPS |
| corpus/gap_hist_tvd | Gap distribution total variation distance |
| corpus/ratio_hist_tvd | Ratio distribution total variation distance |
| corpus/density_corr | Per-second density Pearson correlation |
| corpus/gap_peak_iou | Gap histogram peak IoU |
| corpus/silence_overlap_f1 | Silence region overlap F1 |
| corpus/dense_overlap_f1 | Dense region overlap F1 |
| corpus/bpm_ratio | Pred/GT estimated BPM ratio |

### Benchmark metrics (at each eval)

| Benchmark | Metric prefix | Notes |
|---|---|---|
| `no_coincidence` | val/no_coincidence/frame/ | Rows 80-92 zeroed. Primary ablation for this experiment. |
| `no_mel` | val/no_mel/frame/ | Rows 0-79 zeroed. Tests coincidence-only informativeness. |
| Standard benchmarks | val/{mode}/frame/ | Same set as 017e series (no_past_audio, no_context, etc.) |

### Collapse sentinels

| Metric | Collapse threshold | Description |
|---|---|---|
| val/single/frame/pos_rate_pred_50 | < 0.001 | All-zeros if below this at every eval |
| val/single/frame/recall_tau_50_tol_2 | < 0.50 | Class imbalance killed learning |

---

## Per-eval artifacts

Saved under `runs/exp_019_coincidence_input/eval_{step}/`:

| Artifact | Description |
|---|---|
| framewise_heatmap.png/.npz | Predicted confidence map vs GT for 64 representative windows |
| framewise_distribution.png/.npz | Histogram of predicted confidences at GT-positive vs GT-negative bins |
| per_bin_rate.png/.npz | Per-bin P(target=1), recall@bin, FPR@bin |
| value_hist_target.png/.npz | Histogram of target values, linear + log y-axis |
| value_hist_pred.png/.npz | Histogram of predicted confidences, linear + log y-axis |
| value_hist_combined.png | Side-by-side 2x2 grid: target vs prediction, linear vs log |
| confidence_by_outcome.png/.npz | Overlaid histograms of confidence for TP/FN/FP/TN |
| calibration.png/.npz | Calibration curve (predicted conf vs empirical positive rate) + bucket populations |

---

## Dataset

- Name: taiko2_v1_coin
- Source: osu!taiko .osz packs
- Mel bands: 80 + 13 coincidence rows = 93 input channels
- GT positive rate: 2.54% of bins
- Median GT onsets per window: 11
- Split seed: 42
- Train/val: 0.9 / 0.1

---

## Environment

| Component | Version |
|---|---|
| Python | 3.13.13 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| torchaudio | 2.11.0.dev20260227+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070, 11.5 GB VRAM |
| OS | CachyOS Linux (Arch-based), kernel 7.0.6-1-cachyos |
| numpy | 2.4.2 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |

---

## Addenda

(None yet.)
