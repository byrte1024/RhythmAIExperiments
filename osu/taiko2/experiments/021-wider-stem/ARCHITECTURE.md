# Experiment 021 — Wider conv stem · Architecture

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
| mel | (B, 80, 1000) | float32 | Log-mel spectrogram; 80 bands, 1000 time frames at 5 ms/frame (500 past + 500 future). |
| event_offsets | (B, 128) | int64 | Last 128 onset bin positions relative to the cursor (negative = past). |
| event_mask | (B, 128) | bool | True = padding, False = real event. |
| conditioning | (B, 3) | float32 | [density_mean, density_peak, density_std] from chart metadata. |

## Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| logits | (B, 500) | float32 | Raw pre-sigmoid logits; one per future-time bin. |
| confidence_map | (B, 500) | float32 | sigmoid(logits), detached. Values in [0, 1]. |

---

## Data pipeline

### Audio preprocessing

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
| Mel gain | 30% | +/-2 dB uniform |
| Mel noise | 15% | Gaussian sigma uniform in [0.1, 0.3] |
| Frequency roll | 15% | shift in {-3, ..., +3} |
| SpecAug freq mask | 20% | 1 mask, width <= 10 bands |
| SpecAug time mask | 20% | 1 mask, width <= 30 frames |
| Event jitter | 100% | global +/-3 bins + per-event +/-3 * recency-scaled |
| Event deletion | 5% | drop 1-2 random past events |
| Event insertion | 3% | add 1 synthetic event between two reals |
| Partial metronome (recent half) | 2% | replace with evenly-spaced events |
| Partial adv metronome (older half) | 2% | replace with dominant-gap-spaced events |
| Large time shift | 2% | +/-50 bin shift on 2-4 recent events |
| Context truncation | 5% | keep only most recent 8-32 events |
| Conditioning jitter | 10% | +/-2% on density_mean / peak / std |

Order: time stretch first, then audio augs, then event augs, then
conditioning jitter.

---

## Model architecture

Total parameters: ~22.10 M (+0.9% vs 017f's 21.89 M).

### 1. Conditioning MLP

```
conditioning (B, 3) -> Linear(3, 64) -> GELU -> Linear(64, 64) -> cond (B, 64)
```

### 2. Conv stem

4x downsample; 1000 mel frames -> 250 tokens. stem_width=256 (was
192 in 017f).

```
mel (B, 80, 1000)
  -> Conv1d(in=80, out=256, kernel=7, stride=2, padding=3)
  -> GELU -> GroupNorm(1, 256)
  -> Conv1d(in=256, out=384, kernel=7, stride=2, padding=3)
  -> GELU -> transpose -> (B, 250, 384)
  -> LayerNorm(384) + SinusoidalPosEmb(0..249) + FiLM(cond)
```

Cursor at token index 125.

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

### 5. Output head -- Conv1D-on-bin-axis

head_dropout = 0.2. Each Dropout layer between Conv1D blocks uses
p=0.2.

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

### Positive-class weight

Per-sample: `pos_weight = clamp(n_neg / max(n_gt, 1), min=1, max=1)`.

The clamp always returns 1.0, regardless of window density.

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
| Benchmarks | all (10 modes) |
| Seed | 42 |

---

## Inference (autoregressive)

1. Forward pass -> sigmoid -> confidence_map.
2. NMS via max_pool1d(kernel=3).
3. Threshold at 0.3.
4. Empty positive set -> cursor += 20 bins.
5. Else: emit all passing bins, cursor advances to last emitted.

| Param | Value |
|---|---|
| decode_threshold | 0.4 |
| nms_kernel | 3 |
| stop_hop_bins | 20 |
| min_emit_gap_bins | 1 |
| max_notes_per_step | 0 |

---

## Metrics

This run collects a significantly expanded set of metrics compared to
prior framewise experiments. The metrics below are logged per-eval.

### Frame-level (validation set)

Computed by `FramewiseMetric` at 5 thresholds (10, 30, 50, 70, 90)
and 5 tolerances (0, 1, 2, 3, 5 frames).

| Metric | Description |
|---|---|
| frame/precision_tau_{T}_tol_{N} | Precision at threshold T/100, N-frame tolerance |
| frame/recall_tau_{T}_tol_{N} | Recall at threshold T/100, N-frame tolerance |
| frame/f1_tau_{T}_tol_{N} | F1 at threshold T/100, N-frame tolerance |
| frame/auc_pr | Area under precision-recall curve |
| frame/auc_roc | Area under ROC curve |
| frame/mean_act_pos | Mean confidence on GT-positive bins |
| frame/mean_act_neg | Mean confidence on GT-negative bins |
| frame/separation | mean_act_pos - mean_act_neg |
| frame/brier | Brier score |

### Frame-level mini-chart (validation set)

Per threshold, each val window is decoded into a mini-chart and
compared against GT using `gt_match_metrics`.

| Metric | Description |
|---|---|
| frame/mini/tau{T}/matched_rate | Fraction of GT onsets matched within 25ms |
| frame/mini/tau{T}/hallucination_rate | Fraction of pred onsets with no GT match |
| frame/mini/tau{T}/density_ratio | Pred / GT onset density |
| frame/mini/tau{T}/error_median_ms | Median timing error on matched onsets |

### Frame-level FPS resolution (validation set)

Per threshold, each val window is compared at 8 temporal resolutions
(1, 2, 4, 10, 20, 50, 100, 200 FPS). Each FPS level bins both pred
and GT onset lists into frames of width 1000/fps ms.

| Metric | Description |
|---|---|
| frame/mini/tau{T}/fps_{F}/binary_precision | At F FPS: fraction of pred-active frames that are GT-active |
| frame/mini/tau{T}/fps_{F}/binary_recall | At F FPS: fraction of GT-active frames that are pred-active |
| frame/mini/tau{T}/fps_{F}/binary_f1 | At F FPS: harmonic mean of binary precision and recall |
| frame/mini/tau{T}/fps_{F}/count_mae | At F FPS: mean absolute error of onset count per frame |
| frame/mini/tau{T}/fps_{F}/count_corr | At F FPS: Pearson correlation of onset counts per frame |
| frame/mini/tau{T}/fps_{F}/count_accuracy | At F FPS: fraction of frames with exact count match |

The watched metric is `frame/mini/tau50/fps_50/binary_f1` -- binary F1
at 50 FPS (20 ms frames) using threshold 0.50. This measures whether
the model agrees with GT on onset presence in each 20 ms window.

### AR corpus metrics (inference pass per eval)

Computed by `corpus.py` running full AR inference on the val split.

| Metric | Description |
|---|---|
| precision | Onset precision at 25ms tolerance |
| recall | Onset recall at 25ms tolerance (= matched_rate) |
| f1 | Harmonic mean of precision and recall at 25ms |
| density_ratio | Emitted notes/s / GT notes/s |
| dc_human | TaikoNation direct-compare score (%) |
| oc_human | TaikoNation overlap-compare score (%) |
| error_median_ms | Median timing error on matched notes |

### AR corpus FPS resolution (per eval)

Aggregated across val charts into `fps_summary.json` with percentile
stats (min, p25, median, p75, p95, max, mean) per (fps, metric) pair.

| Metric | Description |
|---|---|
| binary_f1_at_{F}fps | Median binary F1 across val charts at F FPS |
| count_mae_at_{F}fps | Median count MAE at F FPS |
| count_corr_at_{F}fps | Median count correlation at F FPS |

### AR corpus distributional comparison (per chart)

Each per-chart comparison JSON contains distributional metrics
comparing the generated chart's rhythmic structure against GT.

| Metric | Description |
|---|---|
| gap_hist_tvd | Total variation distance between pred and GT gap histograms |
| ratio_hist_tvd | Total variation distance between pred and GT ratio histograms |
| density_corr | Pearson correlation of per-second density timelines |
| density_mae | Mean absolute error of per-second density |
| silence_overlap_f1 | F1 of silence region overlap |
| dense_overlap_f1 | F1 of dense region overlap |
| gap_peak_iou | IoU of gap histogram peak sets |
| ioi_mean_ratio | Ratio of pred/GT mean IOI |
| ioi_std_ratio | Ratio of pred/GT IOI standard deviation |
| streak_fraction_delta | Difference in same-gap streak fractions |
| bpm_ratio | Ratio of pred/GT estimated BPM |

### Per-chart self-metrics

Each generated chart's self-metrics JSON includes:

| Metric | Description |
|---|---|
| gap_histogram_dense | Raw 200-bucket gap histogram (10ms buckets, 0-2000ms) |
| ratio_histogram_dense | Raw 200-bucket ratio histogram (log2 space) |

### Collapse sentinels

| Metric | Collapse threshold | Description |
|---|---|---|
| frame/pos_rate_pred_50 | < 0.001 | All-zeros if below this at every eval |
| frame/recall_tau_50_tol_2 | < 0.50 | Class imbalance killed learning |

---

## Per-eval artifacts

Saved under `runs/exp_021_wider_stem/eval_{step}/`:

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

AR corpus artifacts per eval:

| Artifact | Description |
|---|---|
| generated/{stem}.zip | Full generated chart bundle |
| metrics/{stem}.json | Per-chart self-metrics (includes dense histograms) |
| comparisons/{stem}.json | Per-chart comparison (includes FPS comparisons + distributional metrics) |
| steps/{stem}.jsonl | Per-step AR decode log |
| fps_summary.json | Aggregated FPS resolution stats across val charts |

---

## Dataset

- Name: taiko2_v1
- Source: osu!taiko .osz packs
- Mel bands: 80
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
| OS | CachyOS Linux (Arch-based), kernel 7.0.9-1-cachyos |
| numpy | 2.4.2 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |

---

## Addenda

(None yet.)
