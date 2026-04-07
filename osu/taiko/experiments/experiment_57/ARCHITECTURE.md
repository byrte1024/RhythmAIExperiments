# Experiment 57 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position.

This experiment adds 128 virtual tokens (one per event slot) for collision-free out-of-window context representation.

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
| logits | (B, 251) | 250 onset bin offsets (0-249, ~0-1.25s ahead) + 1 STOP class (250). Model sees 500 bins of future audio but only predicts into 250. |

## Model: EventEmbeddingDetector (with gap ratios + 128 virtual tokens)

**Total parameters: ~16.1M**

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

### 3. Event Embeddings (with gap ratios)
For each of 128 context events, compute 5 features:
- **Presence embedding**: learned parameter (1, 384)
- **Gap before**: sinusoidal encoding of distance from previous event
- **Gap after**: sinusoidal encoding of distance to next event
- **Gap ratio before**: sinusoidal encoding of ratio to previous gap
- **Gap ratio after**: sinusoidal encoding of ratio to next gap

```
[presence (384) | gap_before (384) | gap_after (384) | ratio_before (384) | ratio_after (384)]
→ Linear(1920, 384) → GELU → Linear(384, 384)
→ event_emb: (B, 128, 384)
```

Events are classified into two groups:
- **In-window events** (offset >= -500): scatter-added to audio tokens at position `(500 + offset) // 4`
- **Out-of-window events** (offset < -500): placed directly into their dedicated virtual token (event slot i → vtoken i)

### 4. Virtual Tokens (128 tokens, 1:1 mode)

128 virtual tokens are prepended before the 250 audio tokens:

```
[128 virtual tokens] + [250 audio tokens] → (B, 378, 384)
```

Each virtual token is initialized as:
- **Learned watermark**: shared parameter (1, 1, 384), signals "this is context, not audio"
- **Sinusoidal position embedding**: positions -128 to -1 (negative time)
- **FiLM conditioning**: `cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384); x = x * (1 + scale) + shift`

**1:1 mapping (exp 57)**: When `n_virtual_tokens == max_events` (128 == 128), out-of-window event embeddings are placed directly at their event slot index. Event slot 5 → vtoken 5. No linear interpolation, no scatter collisions. Unused slots (masked events) remain as watermark + position only.

This differs from exp 49's approach (32 vtokens, linear mapping from offset to position) which caused collisions when many out-of-window events existed.

### 5. Transformer (8 layers)
```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```
All 378 tokens (128 virtual + 250 audio) attend to each other. Typically only 20-60 virtual tokens are active (filled with event embeddings); the rest are inert watermark tokens.

### 6. Output Head
```
cursor = x[:, 128 + 125, :]  # cursor at sequence position 253 (B, 384)
logits = Linear(384, 251)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)
→ (B, 251)
```

### FiLM Conditioning
Applied after conv stem, to virtual tokens, and after every transformer layer:
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

## Loss: OnsetLoss

Mixed hard CE + soft target loss: `loss = 0.5 * hard_CE + 0.5 * soft_CE`

Soft targets: trapezoid in log-ratio space (good_pct=3%, fail_pct=20%, frame_tolerance=2). STOP weight 1.5x.

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

## Augmentation (~14% context corruption rate)

### Context Augmentation
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | Global ±3 bins + per-event ±3 bins * 1-2x recency scale |
| Event deletion | 5% | Drop 1-2 random events |
| Event insertion | 3% | Add 1 fake event between existing events |
| Partial metronome | 2% | Replace recent half with evenly-spaced |
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
| Density jitter | **30%** | **±10%** on all 3 density values (exp 44's loose setting) |

## Dataset: taiko_v2

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor - 500 to cursor + 500 (1000 frames)
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 501 logits (128 vtokens + 250 audio tokens, cursor at position 253)
5. If argmax = 250 (STOP): hop cursor forward
6. If argmax = offset (0-249): place event, advance cursor
7. Repeat until end of audio

Out-of-window events (earlier than 500 bins before cursor) live in their dedicated virtual tokens, giving the model full context history without scatter collisions.

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
