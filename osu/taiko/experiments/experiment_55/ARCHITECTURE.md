# Experiment 55 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position.

This experiment adds an auxiliary ratio head that predicts in log10-ratio space during training. The ratio head teaches the backbone proportional reasoning while the bin head provides concrete predictions for inference.

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1000) | Mel spectrogram, 80 bands, 1000 frames (500 past + 500 future at ~5ms/frame = 5.0s window) |
| event_offsets | (B, 128) | Past event positions relative to cursor (negative = past, int64) |
| event_mask | (B, 128) | Bool mask, True = padding (no event) |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] from chart metadata |

## Output

| Output | Shape | Description |
|---|---|---|
| onset_logits | (B, 251) | 250 onset bin offsets (0-249, ~0-1.25s ahead) + 1 STOP class (250) |
| ratio_logits | (B, 201) | 201 log10-ratio bins covering ratios 0.05x to 20.0x (training only) |

At inference, only onset_logits are used. ratio_logits are ignored.

## Window Configuration

| Parameter | Value | Description |
|---|---|---|
| A_BINS | 500 | Past audio context (2.5s) |
| B_BINS (B_AUDIO) | 500 | Future audio visible to model (2.5s) |
| B_PRED | 250 | Prediction range (1.25s) |
| N_CLASSES | 251 | 250 onset bins + 1 STOP |
| N_RATIO_BINS | 201 | Discretized log10-ratio space |
| WINDOW | 1000 | Total mel frames (A + B_AUDIO) |
| Tokens | 250 | Conv stem output (1000 // 4) |
| Cursor token | 125 | A_BINS // 4 |

## Model: EventEmbeddingDetector (with gap ratios + ratio head)

**Total parameters: ~16.3M** (~77K more than baseline from ratio head layers)

### 1. Conditioning MLP
```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. Conv Stem (mel → audio tokens)
```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
  → x: (B, 250, 384) audio tokens
```
4x downsample: 1000 mel frames → 250 tokens. Cursor at token 125.

### 3. Event Embeddings (with gap ratios)
For each of 128 context events, compute 5 features:
- **Presence embedding**: learned parameter (1, 384)
- **Gap before**: sinusoidal encoding of distance from previous event
- **Gap after**: sinusoidal encoding of distance to next event (last event uses gap_before as proxy to avoid target leakage)
- **Gap ratio before**: sinusoidal encoding of `gap_before[i-1] / gap_before[i]` scaled by 50, clamped [0.1, 10.0]
- **Gap ratio after**: sinusoidal encoding of `gap_after[i+1] / gap_after[i]` same scaling

```
[presence (384) | gap_before (384) | gap_after (384) | ratio_before (384) | ratio_after (384)]
→ Linear(1920, 384) → GELU → Linear(384, 384)
→ event_emb: (B, 128, 384)
```
Events are mapped to audio token positions: `token = (500 + offset) // 4`. Only events mapping to tokens 0-124 (past audio) are used. Event embeddings are scatter-added to the corresponding audio tokens.

### 4. Transformer (8 layers)
```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```
250 tokens attend to each other.

### 5. Onset Head (bin space)
```
cursor = x[:, 125, :]  # cursor token (B, 384)
onset_logits = Linear(384, 251)(LayerNorm(384)(cursor))
onset_logits = onset_logits + Conv1d_smooth(onset_logits)  # Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 251)
```

### 6. Ratio Head (log10-ratio space)
```
ratio_logits = Linear(384, 201)(LayerNorm(384)(cursor))
ratio_logits = ratio_logits + Conv1d_smooth(ratio_logits)  # Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 201)
```

Both heads read from the same cursor representation. The ratio head has its own LayerNorm and smooth conv (not shared with onset head).

### Ratio Space

201 bins uniformly covering log10 space from -1.301 to +1.301:
- Bin 0 = log10(0.05) = ratio 0.05x (20x shorter than prev gap)
- Bin 100 = log10(1.0) = ratio 1.0x (same as prev gap)
- Bin 200 = log10(20.0) = ratio 20.0x (20x longer than prev gap)
- Resolution: ~0.013 log10 units per bin = ~3% ratio per bin

Symmetry: bin 77 (ratio 0.5x) and bin 123 (ratio 2.0x) are equidistant from bin 100. A 2x overshoot and 0.5x undershoot get identical soft target distributions.

### FiLM Conditioning
Applied after conv stem and after every transformer layer:
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb
Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss

### Onset Loss (OnsetLoss, on all samples)

Mixed hard CE + soft target loss: `loss = 0.5 * hard_CE + 0.5 * soft_CE`

Soft targets: trapezoid in log-ratio space over 251 classes (asymmetric in bin space).

STOP samples get `stop_weight=1.5x` multiplier.

### Ratio Loss (RatioLoss, on valid samples only)

Pure soft CE in log10-ratio space: `loss = soft_CE` (hard_alpha=0.0)

Soft targets: trapezoid in log10 space (symmetric by construction):
- `good_zone = log10(1.03) ≈ 0.013` (3% ratio tolerance)
- `fail_zone = log10(1.20) ≈ 0.079` (20% ratio tolerance)
- Within good_zone: full credit (plateau)
- good_zone to fail_zone: linear ramp to zero
- Beyond fail_zone: zero

Masked for: STOP targets, <2 context events, prev_gap=0, target=0.

### Combined Loss

```
loss = onset_loss + 0.3 * ratio_loss
```

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset) |
| Balanced sampling | ON (1/count^0.5 weights) |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Ratio weight | 0.3 |

## Augmentation (~14% context corruption rate)

### Context Augmentation
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | Global ±3 bins + per-event ±3 bins * 1-2x recency scale |
| Event deletion | 5% | Drop 1-2 random events |
| Event insertion | 3% | Add 1 fake event between existing events |
| Partial metronome | 2% | Replace recent half of events with evenly-spaced |
| Partial adv metronome | 2% | Replace oldest half with dominant-gap metronome |
| Large time shift | 2% | Shift all events by ±50 bins |
| Context truncation | 5% | Keep only most recent 8-32 events |

### Audio Augmentation
| Aug | Rate | Params |
|---|---|---|
| Mel gain | 30% | ±2dB |
| Mel noise | 15% | Gaussian σ≤0.3 |
| Freq jitter | 15% | Roll mel bands ±3 |
| SpecAugment freq | 20% | 1 mask, 10 bands |
| SpecAugment time | 20% | 1 mask, 30 frames |

### Conditioning
| Aug | Rate | Params |
|---|---|---|
| Density jitter | 10% | ±2% on all 3 density values |

## Dataset: taiko_v2

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)
- With B_PRED=250: STOP is ~0.8% of samples

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor - 500 to cursor + 500 (1000 frames total)
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 251 onset logits (ratio logits ignored)
5. If argmax = 250 (STOP): hop cursor forward by hop_bins
6. If argmax = offset (0-249): place event at cursor + offset, move cursor there
7. Repeat until end of audio

The ratio head output is present but ignored at inference. The existing sampling pipeline (Top-U, temperature, metronome suppression) operates on onset logits only.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| cuDNN | 9.10.02 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB, compute 12.0) |
| OS | Windows 11 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |
