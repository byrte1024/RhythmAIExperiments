# Experiment 54 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position.

Combines experiment 53-B's window configuration (A=500, B_AUDIO=500, B_PRED=250) with experiment 47-E's STOP query token architecture.

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
| onset_logits | (B, 250) | 250 onset bin offsets (0-249, ~0-1.25s ahead). STOP is handled separately. |
| stop_logit | (B,) | Scalar logit for binary STOP prediction |

Note: the model sees 500 bins (2.5s) of future audio but can only predict onsets within 250 bins (1.25s). If the next onset is beyond 250 bins, the STOP head predicts stop and the cursor hops forward.

## Window Configuration

| Parameter | Value | Description |
|---|---|---|
| A_BINS | 500 | Past audio context (2.5s) |
| B_BINS (B_AUDIO) | 500 | Future audio visible to model (2.5s) |
| B_PRED | 250 | Prediction range (1.25s) |
| N_CLASSES | 251 | Used for training infrastructure (250 onset bins + 1 STOP sentinel) |
| Onset head classes | 250 | STOP removed from onset classification |
| WINDOW | 1000 | Total mel frames (A + B_AUDIO) |
| Tokens | 250 audio + 1 STOP = 251 | Conv stem output + STOP query |
| Cursor token | 125 | A_BINS // 4 |

## Model: EventEmbeddingDetector (with gap ratios + stop_token)

**Total parameters: ~16.5M**

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

### 4. STOP Query Token

A learned parameter `stop_query` of shape (1, 1, 384) is appended after the 250 audio tokens:

```
[250 audio tokens] + [1 STOP query token] → (B, 251, 384)
```

The STOP query token participates in all 8 transformer layers of self-attention. It attends to every audio token, every event-marked token, and the cursor position. After 8 layers, it has built its own representation of "should I stop?" from the full context.

### 5. Transformer (8 layers)
```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```
All 251 tokens (250 audio + 1 STOP query) attend to each other. Future audio tokens (126-249) provide spectral context beyond the prediction range.

### 6. Onset Head
```
cursor = x[:, 125, :]  # cursor token (B, 384)
onset_logits = Linear(384, 250)(LayerNorm(384)(cursor))
onset_logits = onset_logits + Conv1d_smooth(onset_logits)  # Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 250)
```
250 classes, not 251. The onset head only predicts onset bin offsets. STOP is handled by the separate STOP head.

### 7. STOP Head
```
stop_repr = x[:, 250, :]  # STOP query token, last position (B, 384)
stop_logit = MLP(stop_repr):
    LayerNorm(384) → Linear(384, 96) → GELU → Linear(96, 1) → squeeze
→ (B,)
```
The STOP head is an independent MLP reading from the STOP query token's final representation.

### FiLM Conditioning
Applied after conv stem and after every transformer layer:
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb
Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss

Two separate losses combined:

### Onset Loss (OnsetLoss, applied to non-STOP samples only)

Mixed hard CE + soft target loss on the 250-class onset logits.

**Soft Targets (Trapezoid):**
For each non-STOP target at bin `t`, create a soft distribution over 250 onset bins:
- Compute `|log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Within 3% ratio (good_pct): full credit (plateau)
- 3% to 20% (fail_pct): linear ramp to zero
- Beyond 20%: zero
- Frame tolerance ±2: always get some credit regardless of ratio

```
onset_loss = 0.5 * hard_CE + 0.5 * soft_CE  (only on non-STOP samples)
```

### Stop Loss (BCE, averaged separately by class)

Binary cross-entropy on the stop logit:
- Target = 1.0 for STOP samples, 0.0 for onset samples
- Averaged separately for STOP and onset samples within each batch to prevent class dilution

```
stop_bce = BCE_with_logits(stop_logit, stop_target, reduction='none')
stop_loss_pos = stop_bce[is_stop].mean()      # avg over STOP samples
stop_loss_neg = stop_bce[~is_stop].mean()     # avg over onset samples
stop_loss = (stop_loss_pos + stop_loss_neg) / 2.0
```

### Combined Loss

```
loss = onset_loss + 1.5 * stop_loss
```

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset) |
| Balanced sampling | ON (1/count^0.5 weights) with **20x STOP boost** |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |

### 20x STOP Boost

With B_PRED=250, STOP is ~0.8% of samples (events beyond 250 bins from cursor). Standard balanced sampling at power=0.5 gives STOP modest upweighting, but still <2 STOP samples per batch. The 20x boost multiplies STOP's sampling weight by 20, bringing STOP to ~7-8 samples per batch of 48. Without this, most batches would have 0-1 STOP samples and the STOP token would receive insufficient gradient.

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
- With B_PRED=250: STOP is ~0.8% of samples (events beyond 250 bins from cursor)

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor - 500 to cursor + 500 (1000 frames total)
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 250 onset logits + 1 stop logit
5. If stop_logit > 0 (sigmoid > 0.5): hop cursor forward by hop_bins
6. If onset argmax = offset (0-249): place event at cursor + offset, move cursor there
7. Repeat until end of audio

At inference, the stop logit is scaled and placed into the STOP position of a padded logits tensor for compatibility with existing sampling pipeline: `logits[:, 250] = stop_logit * 5.0`.

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
