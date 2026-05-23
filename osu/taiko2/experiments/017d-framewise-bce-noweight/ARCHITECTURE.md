# Experiment 017d — Framewise BCE with no pos_weight · Architecture

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
onset falls, 0.0 elsewhere. No Gaussian smoothing (sigma = None).

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

Total parameters: ~21.89 M.

### 1. Conditioning MLP

```
conditioning (B, 3) -> Linear(3, 64) -> GELU -> Linear(64, 64) -> cond (B, 64)
```

### 2. Conv stem

4x downsample; 1000 mel frames -> 250 tokens.

```
mel (B, 80, 1000)
  -> Conv1d(in=80, out=192, kernel=7, stride=2, padding=3)
  -> GELU -> GroupNorm(1, 192)
  -> Conv1d(in=192, out=384, kernel=7, stride=2, padding=3)
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

```
cursor_token = x[:, 125, :]           (B, 384)
audio_features = x[:, 125:, :]        (B, 125, 384)

Per-bin channels:
  pos_embed:       (B, 32, 500)   sinusoidal over bin positions
  audio_features:  (B, 384, 500)  linearly interpolated from 125 tokens
  cursor_broadcast:(B, 32, 500)   Linear(384, 32) broadcast

h: (B, 448, 500)
  -> Conv1d(448, 256, k=31, p=15) -> GroupNorm(8) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 256, k=15, p=7)  -> GroupNorm(8) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 256, k=15, p=7)  -> GroupNorm(8) -> SiLU -> Dropout(0.1)
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
positive-class upweighting. A false positive costs identically to a
false negative in the loss function.

### Target

Binary activation map (B, 500). Value 1.0 at GT onset bins, 0.0
elsewhere. No Gaussian smoothing.

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
precision. The 38:1 class imbalance means the sum of false-positive
gradient signals is numerically 38x larger than the sum of
false-negative gradient signals, but each individual bin is weighted
equally.

### Loss formula

```
per_bin = F.binary_cross_entropy_with_logits(
    logits, target_map_binary, weight=weight_map, reduction="none",
)
loss = per_bin.mean()
```

where `weight_map` is all-ones (every bin, every sample, weight=1.0).

### Hyperparameters

| Param | Value |
|---|---|
| pos_weight_clamp_min | 1.0 |
| pos_weight_clamp_max | 1.0 |
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
| Watched metric | loss (lower is better) |
| Train-noaug fraction | 0.05 |
| Benchmarks | all (10 modes) |
| Seed | 42 |

---

## Inference (autoregressive)

1. Forward pass -> sigmoid -> confidence_map.
2. NMS via max_pool1d(kernel=3).
3. Threshold at 0.5.
4. Empty positive set -> cursor += 20 bins.
5. Else: emit all passing bins, cursor advances to last emitted.

| Param | Value |
|---|---|
| decode_threshold | 0.5 |
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
| val/single/loss | Total BCE loss |
| val/single/loss/pos_only | Loss on GT-positive bins only |
| val/single/loss/neg_only | Loss on GT-negative bins only |
| val/single/loss/pos_neg_ratio | Ratio pos_only / neg_only |

### AR corpus metrics (inference pass per eval)

| Metric | Description |
|---|---|
| infer_corpus/density_ratio | Emitted notes / GT notes |
| infer_corpus/dc_human | Pattern quality score [0, 100] |
| infer_corpus/oc_human | Overlap correctness [0, 100] |
| infer_corpus/hallucination_rate | Fraction of emitted notes with no GT match |
| infer_corpus/matched_rate | Fraction of GT notes matched |
| infer_corpus/error_median_ms | Median timing error on matched notes |
| infer_corpus/events_per_sec | Emitted notes per second |

### Collapse sentinels

| Metric | Collapse threshold | Description |
|---|---|---|
| val/single/frame/pos_rate_pred_50 | < 0.001 | All-zeros if below this at every eval |
| val/single/frame/recall_tau_50_tol_2 | < 0.50 | Class imbalance killed learning |

---

## Per-eval artifacts

Saved under `runs/exp_017d_framewise_bce_noweight/eval_{step}/`:

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
| OS | CachyOS Linux (Arch-based), kernel 7.0.6-1-cachyos |
| numpy | 2.4.2 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |

---

## Addenda

> *2026-05-23: ``threshold_sweep.json`` and all threshold sweep numbers
> in ``README.md`` are **INVALID**. The sweep script
> (``cli/threshold_sweep.py``) called ``gt.compare(pred_chart)``
> instead of ``pred_chart.compare(gt)``, flipping self/other in
> ``_gt_match_metrics``. This reversed matched_rate (became precision-
> like), hallucination_rate (became miss-like), and density_ratio
> (became GT/pred instead of pred/GT). All claims about "best
> threshold at E9/tau=0.3" are based on inverted metrics. The bug
> was fixed in ``threshold_sweep.py``; the sweep must be re-run to
> obtain correct numbers. The per-eval AR corpus metrics (from
> ``InferCorpusHook``) are NOT affected — ``corpus.py`` correctly
> calls ``generated.compare(gt_chart)``.*
