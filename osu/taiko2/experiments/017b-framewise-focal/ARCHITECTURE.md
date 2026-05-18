# Experiment 017b — Framewise focal loss -- Architecture

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

Identical to experiment 017:

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

---

## Model architecture

Total parameters: ~21.89 M. Identical to experiment 017.

### 1. Conditioning MLP

```
conditioning (B, 3) -> Linear(3, 64) -> GELU -> Linear(64, 64) -> cond (B, 64)
```

### 2. Conv stem

```
mel (B, 80, 1000) -> stride-2 Conv1d x2 -> LayerNorm -> (B, 250, 384)
  + SinusoidalPosEmb + FiLM(cond)
```

Cursor at token index 125.

### 3. Event embeddings

5-component feature vectors (presence + gap_before + gap_after +
gap_ratio_before + gap_ratio_after) per event, projected to d_model=384.
Scatter-added onto audio tokens at event positions.

### 4. Transformer trunk

8 layers: TransformerEncoderLayer(d_model=384, nhead=8, ffn=1536,
dropout=0.1, gelu, pre-norm) + FiLM(cond) after each.

### 5. Output head -- Conv1D-on-bin-axis

```
cursor_token = x[:, 125, :]           (B, 384)
audio_features = x[:, 125:, :]        (B, 125, 384)

Per-bin channels:
  pos_embed:       (B, 32, 500)   sinusoidal
  audio_features:  (B, 384, 500)  linearly interpolated from 125
  cursor_broadcast:(B, 32, 500)   Linear(384, 32) broadcast

h: (B, 448, 500)
  -> Conv1d(448, 256, k=31, p=15) -> GroupNorm(8) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 256, k=15, p=7)  -> GroupNorm(8) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 256, k=15, p=7)  -> GroupNorm(8) -> SiLU -> Dropout(0.1)
  -> Conv1d(256, 1, k=1)
  -> logits: (B, 500)
```

---

## Loss

**Focal loss** with positive-class upweighting.

### Formula

```
bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
p = sigmoid(logits)
p_t = p * target + (1 - p) * (1 - target)
focal_weight = (1 - p_t) ^ gamma
per_bin_loss = focal_weight * pos_weight_map * bce
loss = per_bin_loss.mean()
```

Where `pos_weight_map` has value `clamp(n_neg / max(n_gt, 1), [10, 200])`
at GT-positive bins and 1.0 at GT-negative bins.

### Effect of the focal modulation

| Example bin | p | y | p_t | focal_weight | Effect |
|---|---:|---:|---:|---:|---|
| Easy TN (confidence 0.02) | 0.02 | 0 | 0.98 | 0.0004 | Nearly zero gradient — removed from loss |
| Hard FP (metronomic beat, confidence 0.80) | 0.80 | 0 | 0.20 | 0.6400 | Strong gradient — model must learn to suppress |
| TP (confidence 0.90) | 0.90 | 1 | 0.90 | 0.0100 | Low gradient — already correct, don't over-optimize |
| Hard FN (missed onset, confidence 0.10) | 0.10 | 1 | 0.10 | 0.8100 | Strong gradient — model must learn to detect |

### Hyperparameters

| Param | Value |
|---|---|
| gamma | 2.0 |
| pos_weight_clamp_min | 10.0 |
| pos_weight_clamp_max | 200.0 |
| canonical_threshold | 0.5 |
| canonical_tolerance_frames | 2 |

### Diagnostics reported (in addition to all frame/* metrics from #017)

| Metric | Description |
|---|---|
| loss/focal_weight_pos | Mean focal modulation weight on GT-positive bins |
| loss/focal_weight_neg | Mean focal modulation weight on GT-negative bins |

---

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clip | 1.0 |
| Batch size | 64 |
| Epochs | 15 |
| Scheduler | CosineAnnealingLR |
| Mixed precision | off |
| torch.compile | on (triton backend) |
| Balanced sampling | off |
| Evals per epoch | 4 |
| Watched metric | loss (lower is better) |
| Train-noaug fraction | 0.05 |
| Benchmarks | all (10 modes) |
| Seed | 42 |

---

## Inference (autoregressive)

Identical to experiment 017.

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

---

## Dataset

- Name: taiko2_v1
- Mel bands: 80
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
