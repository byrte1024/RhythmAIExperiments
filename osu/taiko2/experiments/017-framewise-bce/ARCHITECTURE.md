# Experiment 017 — Framewise BCE (non-diffusion control) -- Architecture

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
a single forward pass. No STOP class -- the AR loop infers "no more
onsets in this window" when no bin exceeds the decode threshold.

## Inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| mel | (B, 80, 1000) | float32 | Log-mel spectrogram; 80 bands, 1000 time frames at 5 ms/frame (500 past + 500 future). |
| event_offsets | (B, 128) | int64 | Last 128 onset bin positions relative to the cursor (negative = past). Padded at the start with zeros where fewer than 128 real events exist. |
| event_mask | (B, 128) | bool | True = padding, False = real event. |
| conditioning | (B, 3) | float32 | [density_mean, density_peak, density_std] from chart metadata. |

## Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| logits | (B, 500) | float32 | Raw pre-sigmoid logits; one per future-time bin. |
| confidence_map | (B, 500) | float32 | sigmoid(logits), detached. Values in [0, 1]. Consumed by decoder and diagnostics. |

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
| Grid rate | 200 bins/second (exact integer) |
| Onset kinds retained | DON, KA, BIG_DON, BIG_KA, DRUMROLL, SPINNER |
| Bin index formula | floor(time_ms / bin_ms) |

Kind codes in the stored .npz arrays:

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
| Min cursor bin (skip warmup) | 6000 |
| Allowed overlap forward | 0 bins |
| Allowed overlap backward | 0 bins |
| Past-event padding | start (oldest-first; front-padded) |
| Future-event padding | end (nearest-first; back-padded) |

### Target construction

Binary activation map (B, 500) with value 1.0 at each bin where a GT
onset falls (within [0, b_pred)), 0.0 elsewhere. No Gaussian smoothing
(sigma = None). The smoothed map field is set equal to the binary map.

`gt_bins_padded` (B, 100) carries the raw GT bin offsets (padded with
-1) for downstream mini-chart comparison.

### Train/val split

- Song-level grouping by `beatmapset_id`.
- Split seed: 42.
- Ratios: train = 0.9, val = 0.1.

### Augmentations

Applied on the training split only; never on val.

| Augmentation | Probability | Parameters |
|---|---|---|
| Time stretch | 30% | log-uniform in [1/1.4, 1.4] |
| Mel gain | 30% | +/-2 dB uniform |
| Mel noise | 15% | Gaussian sigma uniform in [0.1, 0.3] |
| Frequency roll | 15% | shift in {-3, ..., +3} |
| SpecAug freq mask | 20% | 1 mask, width <= 10 bands |
| SpecAug time mask | 20% | 1 mask, width <= 30 frames |
| Event jitter | 100% | global +/-3 bins + per-event +/-3 * recency-scaled (1.0 at oldest, 2.0 at newest) |
| Event deletion | 5% | drop 1-2 random past events |
| Event insertion | 3% | add 1 synthetic event between two reals |
| Partial metronome (recent half) | 2% | replace with evenly-spaced events |
| Partial adv metronome (older half) | 2% | replace with dominant-gap-spaced events |
| Large time shift | 2% | +/-50 bin shift on 2-4 recent events |
| Context truncation | 5% | keep only most recent 8-32 events |
| Conditioning jitter | 10% | +/-2% on density_mean / peak / std |

Order: time stretch first (defines "real content"), then audio augs,
then event augs, then conditioning jitter.

---

## Model architecture

Total parameters: ~21 M (16.35 M trunk + ~5 M head).

### 1. Conditioning MLP

```
conditioning (B, 3)
  -> Linear(3, 64) -> GELU
  -> Linear(64, 64)
  -> cond (B, 64)
```

### 2. Conv stem -- mel to audio tokens

4x downsample; 1000 mel frames -> 250 tokens.

```
mel (B, 80, 1000)
  -> Conv1d(in=80, out=192, kernel=7, stride=2, padding=3)
  -> GELU
  -> GroupNorm(num_groups=1, num_channels=192)
  -> Conv1d(in=192, out=384, kernel=7, stride=2, padding=3)
  -> GELU
  -> transpose -> (B, 250, 384)
  -> LayerNorm(384)
  -> + SinusoidalPosEmb(positions 0..249)       (d_model=384)
  -> FiLM(cond)
  -> x: (B, 250, 384)
```

Cursor sits at token index 125 (center).

### 3. Event embeddings

For each of 128 past events, build a feature vector from 5 components:

| Component | Dim | Description |
|---|---|---|
| presence | 384 | Learned parameter broadcast to every slot |
| gap_before | 384 | SinusoidalPosEmb of abs(offset[i] - offset[i-1]), clamped >= 1; placeholder 50 at i=0 |
| gap_after | 384 | SinusoidalPosEmb of abs(offset[i+1] - offset[i]), clamped >= 1; last valid event uses gap_before as proxy |
| gap_ratio_before | 384 | SinusoidalPosEmb of (gap_before[i-1] / gap_before[i]) * 50, clamped [0.1, 10] |
| gap_ratio_after | 384 | SinusoidalPosEmb of (gap_after[i+1] / gap_after[i]) * 50, clamped [0.1, 10] |

Concatenated: (B, 128, 1920) -> Linear(1920, 384) -> GELU -> Linear(384, 384) -> event_embs (B, 128, 384).

Scatter-add event embeddings onto audio tokens at their corresponding
mel-token positions. Only past events (token index < 125) are injected.

### 4. Transformer trunk

8 layers of pre-norm encoder blocks with FiLM after each:

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

The head operates on the bin axis (500 bins), not the token axis.

**Cursor token extraction:** cursor_token = x[:, 125, :]  (B, 384)

**Audio features extraction:** audio_features = x[:, 125:, :]  (B, 125, 384)
These are the future-half audio tokens after the transformer trunk.

**Per-bin channel assembly:**

| Channel | Dim | Construction |
|---|---|---|
| pos_embed | 32 | Sinusoidal positional embedding over 500 bin positions |
| audio_features | 384 | (B, 125, 384) transposed to (B, 384, 125), linearly interpolated to (B, 384, 500) |
| cursor_broadcast | 32 | Linear(384, 32)(cursor_token), broadcast to (B, 32, 500) |

Total input channels: 32 + 384 + 32 = 448.

**Conv stack:**

```
h: (B, 448, 500)
  -> Conv1d(448, 256, kernel=31, padding=15) -> GroupNorm(8, 256) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 256, kernel=15, padding=7)  -> GroupNorm(8, 256) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 256, kernel=15, padding=7)  -> GroupNorm(8, 256) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 1, kernel=1)
  -> squeeze(1)
  -> logits: (B, 500)
```

No FiLM conditioning in the head (no time step to modulate on -- this
is a single-shot predictor, not a denoiser).

`confidence_map = sigmoid(logits).detach()` -- produced alongside
logits in the forward pass but detached so downstream consumers
(decoder, metrics, diagnostics) never accidentally backprop through it.

### FiLM module (used in trunk only)

```
cond (B, 64)
  -> Linear(64, 2 * 384)
  -> split into (gamma, beta) each (B, 384)
output = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
```

Weight + bias initialized to zero (identity at init).

---

## Loss

**Binary cross-entropy with logits**, per-bin, with positive-class
upweighting.

### Target

Binary activation map (B, 500). Value 1.0 at GT onset bins, 0.0
elsewhere. No Gaussian smoothing.

### Positive-class weight

Per-sample: `pos_weight = clamp(n_neg / max(n_gt, 1), min=10, max=200)`.

For a typical window with ~5 GT onsets in 500 bins:
n_neg = 495, n_pos = 5, pos_weight = 99 (within clamp range).

### Loss formula

```
per_bin = F.binary_cross_entropy_with_logits(
    logits,                       # (B, 500) raw
    target_map_binary,            # (B, 500) {0, 1}
    weight=weight_map,            # (B, 500) pos_weight at GT bins, 1.0 elsewhere
    reduction="none",
)
loss = per_bin.mean()
```

### Hyperparameters

| Param | Value |
|---|---|
| pos_weight_clamp_min | 10.0 |
| pos_weight_clamp_max | 200.0 |
| canonical_threshold | 0.5 |
| canonical_tolerance_frames | 2 |

---

## Metrics

### Per-batch (loss-reported)

| Metric | Description |
|---|---|
| loss | Headline BCE loss |
| loss/pos_only | Unweighted BCE at GT-positive bins |
| loss/neg_only | Unweighted BCE at GT-negative bins |
| loss/pos_neg_ratio | pos_only / neg_only |
| frame/f1_tau_50_tol_2 | F1 at threshold 0.5, tolerance +/-2 bins |
| frame/precision_tau_50_tol_2 | Precision at canonical operating point |
| frame/recall_tau_50_tol_2 | Recall at canonical operating point |
| frame/auc_pr | Area under precision-recall curve |
| frame/auc_roc | Area under ROC curve |
| frame/mean_act_pos | Mean confidence at GT-positive bins |
| frame/mean_act_neg | Mean confidence at GT-negative bins |
| frame/separation | mean_act_pos - mean_act_neg |
| frame/pos_rate_pred_50 | Fraction of bins with confidence > 0.5 |
| frame/pos_rate_target | Fraction of GT-positive bins |
| frame/pred_hedge_frac | Fraction of predictions in [0.2, 0.8] |
| frame/brier | Brier score vs binary target |
| frame/conf_tp_median | Median confidence on true positives |
| frame/conf_fn_median | Median confidence on false negatives |
| frame/conf_fp_median | Median confidence on false positives |
| frame/conf_tn_median | Median confidence on true negatives |

### Per-eval (FramewiseMetric, accumulated across val pass)

All of the above plus:

| Metric | Description |
|---|---|
| frame/ece | Expected calibration error (10-bin) |
| frame/conf_tp_p10 | 10th percentile TP confidence |
| frame/conf_fp_p90 | 90th percentile FP confidence |
| frame/mini/tau{T}/matched_rate | Per-window matched_rate at threshold T%, tol=25ms |
| frame/mini/tau{T}/hallucination_rate | Per-window halluc rate at threshold T% |
| frame/mini/tau{T}/density_ratio | Per-window density ratio at threshold T% |
| frame/mini/tau{T}/error_median_ms | Per-window error median at threshold T% |
| frame/mini/tau{T}/matched_rate_at_tol_{M} | Per-window matched_rate at tol M ms |
| frame/mini/tau{T}/halluc_rate_at_tol_{M} | Per-window halluc rate at tol M ms |

Thresholds T: {30, 40, 50, 60, 70}. Tolerances M: {5, 10, 25, 50, 100} ms.

Mini-chart matching uses the same `gt_match_metrics` function as the
AR corpus comparison (greedy nearest-neighbor within tolerance, density
computed from event spacing). Bin offsets are converted to ms via
`bins_to_ms = 5.0` (= 1000 / 200 grid rate).

### Per-eval (InferCorpusHook, AR loop on 10% val)

| Metric | Description |
|---|---|
| corpus/gt_cond_cmp/matched_rate_mean | AR matched_rate (tol=25ms) |
| corpus/gt_cond_cmp/hallucination_rate_mean | AR halluc rate |
| corpus/gt_cond_cmp/density_ratio_mean | AR density ratio |
| corpus/gt_cond_cmp/error_median_ms_mean | AR error median ms |
| corpus/gt_cond_cmp/hi_pspace_mean | TaikoNation pattern-space overlap |
| corpus/gt_cond_cmp/dc_human_mean | Direct per-step match rate |

### Per-eval (benchmarks, all 10 modes)

Each benchmark mode produces the full metric set under
`bench/{mode_name}/*`. All 10 modes from the default benchmark suite
run per eval.

---

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clip (max_norm) | 1.0 |
| Batch size | 64 |
| Epochs | 15 |
| Scheduler | CosineAnnealingLR |
| Mixed precision | off |
| Balanced sampling | off (framewise targets make single-class weighting meaningless) |
| Evals per epoch | 4 |
| Watched metric | `loss` (lower is better) |
| Checkpoint cadence | every eval; `latest.pt` + `best.pt` |
| Train-noaug fraction | 0.05 (5% of train split, no augmentation, per eval) |
| Seed | 42 |

---

## Inference (autoregressive)

1. Initialize cursor at bin 0, past_onsets = empty.
2. At each step:
   a. Extract mel window [cursor - 500, cursor + 500]; pad with zeros
      past the audio edges.
   b. Gather up to 128 past onsets; encode as offsets from cursor.
   c. Build conditioning vector from user-supplied target density.
   d. Forward pass -> (1, 500) logits -> sigmoid -> confidence_map.
   e. Apply NMS: max_pool1d(kernel=3) on confidence_map; keep bins
      where value equals local max.
   f. Threshold at 0.5. Apply min_emit_gap = 1 bin.
   g. If no bins pass: cursor += 20 bins (100 ms hop on STOP).
      Else: emit all passing bins as onsets; cursor += last emitted bin.
3. Stop when cursor >= end of audio OR 10 000 onsets emitted.

### Decoder config

| Param | Value |
|---|---|
| decode_threshold | 0.5 |
| nms_kernel | 3 (odd; keeps local maxima only) |
| stop_hop_bins | 20 |
| min_emit_gap_bins | 1 |
| top_k_log | 5 (diagnostic extras per step) |

---

## Per-eval artifacts

Saved under `runs/exp_017_framewise_bce/eval_{step}/`:

| Artifact | Description |
|---|---|
| framewise_heatmap.png/.npz | Predicted confidence map vs GT for 64 representative windows |
| framewise_distribution.png/.npz | Histogram of predicted confidences at GT-positive vs GT-negative bins |
| per_bin_rate.png/.npz | Per-bin P(target=1), recall@bin, FPR@bin |
| value_hist_target.png/.npz | Histogram of target values, linear + log y-axis |
| value_hist_pred.png/.npz | Histogram of predicted confidences, linear + log y-axis |
| value_hist_combined.png | Side-by-side 2x2 grid: target vs prediction, linear vs log |
| confidence_by_outcome.png/.npz | Overlaid histograms of confidence for TP/FN/FP/TN |

---

## Dataset

- Name: taiko2_v1
- Source: osu!taiko .osz packs
- Mel bands: 80
- Charts: see manifest.json for exact count
- Split seed: 42
- Train/val ratio: 0.9 / 0.1

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

Note: experiments #001-#016b ran on Windows with the same GPU and
identical PyTorch / CUDA / numpy builds. This run is the first on
Linux. Python micro-version bumped from 3.13.12 to 3.13.13. No effect
on numerical results is expected.

---

## Addenda

> *2026-05-17: `value_hist_target.png` rendered with linear y-axis
> while `value_hist_pred.png` rendered with both linear and log panels.
> The visual mismatch is deceptive when comparing the two side-by-side.
> Fixed: both histograms now render as two-panel (linear + log) plots.
> Added `value_hist_combined.png` showing target and prediction
> side-by-side in a 2x2 grid. Use `cli/regenerate_diagnostics.py` to
> re-render from the saved NPZs.*

> *2026-05-17: `reliability.png` removed. The calibration plot used
> reservoir-sampled per-outcome data with per-class caps (50k each)
> that destroyed the true class ratios (~97% negative, ~3% positive
> -> each capped to 50k = ~50/50). This produced a misleading plot
> where low-confidence bins appeared to have 90%+ positive rate.
> The artifact and regeneration script no longer produce this plot.
> The `reliability.npz` from eval 1 should be ignored.*
