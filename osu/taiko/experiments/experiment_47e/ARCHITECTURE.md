# Experiment 47-E — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position.

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1000) | Mel spectrogram, 80 bands, 1000 frames (500 past + 500 future at ~5ms/frame = 5s window) |
| event_offsets | (B, 128) | Past event positions relative to cursor (negative = past, int64) |
| event_mask | (B, 128) | Bool mask, True = padding (no event) |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] from chart metadata |

## Output

| Output | Shape | Description |
|---|---|---|
| onset_logits | (B, 500) | 500 onset bin offsets (0-499, ~0-2.5s ahead). STOP is handled separately. |
| stop_logit | (B,) | Scalar logit for binary STOP prediction |

## Model: EventEmbeddingDetector (with stop_token=True)

**Total parameters: ~16.2M**

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

### 3. Event Embeddings

For each of 128 context events, compute:
- **Presence embedding**: learned parameter (1, 384)
- **Gap before**: sinusoidal encoding of distance from previous event
- **Gap after**: sinusoidal encoding of distance to next event (last event uses gap_before as proxy to avoid target leakage)

```
[presence (384) | gap_before (384) | gap_after (384)] → Linear(1152, 384) → GELU → Linear(384, 384)
→ event_emb: (B, 128, 384)
```

Events are mapped to audio token positions: `token = (500 + offset) // 4`. Only events mapping to tokens 0-124 (past audio) are used. Event embeddings are scatter-added to the corresponding audio tokens.

### 4. STOP Query Token

A learned parameter `stop_query` of shape (1, 1, 384) is appended after the 250 audio tokens, making the sequence 251 tokens total:

```
[250 audio tokens] + [1 STOP query token] → (B, 251, 384)
```

The STOP query token participates in all 8 transformer layers of self-attention. It can attend to every audio token, every event-marked token, and the cursor position. After 8 layers, it has built its own representation of "should I stop?" from the full context.

### 5. Transformer (8 layers)

```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```

Each layer: pre-norm self-attention + FFN with FiLM density conditioning. All 251 tokens (250 audio + 1 STOP query) attend to each other.

### 6. Onset Head

```
cursor = x[:, 125, :]  # cursor token (B, 384)
onset_logits = Linear(384, 500)(LayerNorm(384)(cursor))
onset_logits = onset_logits + Conv1d_smooth(onset_logits)  # 1d smoothing: Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 500)
```

Note: 500 classes, not 501. The onset head only predicts onset bin offsets. STOP is handled by the separate STOP head.

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

Mixed hard CE + soft target loss on the 500-class onset logits.

**Soft Targets (Trapezoid):**
For each non-STOP target at bin `t`, create a soft distribution over 500 bins:
- Compute `|log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Within 3% ratio: full credit (plateau)
- 3% to 20%: linear ramp to zero
- Beyond 20%: zero
- Frame tolerance +/-2: always get some credit regardless of ratio

```
onset_loss = 0.5 * hard_CE + 0.5 * soft_CE  (only on non-STOP samples)
```

### Stop Loss (BCE, averaged separately by class)

Binary cross-entropy on the stop logit:
- Target = 1.0 for STOP samples, 0.0 for onset samples
- Averaged separately for STOP and onset samples within each batch to prevent class dilution (47/48 onset samples would otherwise drown the 1/48 STOP signal)

```
stop_bce = BCE_with_logits(stop_logit, stop_target, reduction='none')
stop_loss_pos = stop_bce[is_stop].mean()      # avg over STOP samples
stop_loss_neg = stop_bce[~is_stop].mean()     # avg over onset samples
stop_loss = (stop_loss_pos + stop_loss_neg) / 2.0
```

### Combined Loss

```
loss = onset_loss + stop_weight * stop_loss
```

Default stop_weight = 1.5.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset, 5.25M samples) |
| Balanced sampling | ON (1/count^0.5 weights) with **20x STOP boost** |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |

### 20x STOP Boost

STOP is 15,866 / 5.25M samples (0.3%). Standard balanced sampling at power=0.5 gives STOP ~4.5x weight over common bins, but that is still less than 1 STOP sample per batch. The 20x boost multiplies STOP's sampling weight by 20, bringing STOP to ~16% sampling share (~7-8 STOP samples per batch of 48). Without this, most batches had zero STOP samples and the STOP token received no gradient.

## Augmentation (~14% context corruption rate)

### Context Augmentation (applied to event_offsets)
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | Global +/-3 bins + per-event +/-3 bins * 1-2x recency scale |
| Event deletion | 5% | Drop 1-2 random events |
| Event insertion | 3% | Add 1 fake event between existing events |
| Partial metronome | 2% | Replace recent half of events with evenly-spaced |
| Partial adv metronome | 2% | Replace oldest half with dominant-gap metronome |
| Large time shift | 2% | Shift all events by +/-50 bins |
| Context truncation | 5% | Keep only most recent 8-32 events |

### Audio Augmentation (applied to mel_window)
| Aug | Rate | Params |
|---|---|---|
| Mel gain | 30% | +/-2dB |
| Mel noise | 15% | Gaussian sigma<=0.3 |
| Freq jitter | 15% | Roll mel bands +/-3 |
| SpecAugment freq | 20% | 1 mask, 10 bands |
| SpecAugment time | 20% | 1 mask, 30 frames |

### Conditioning
| Aug | Rate | Params |
|---|---|---|
| Density jitter | 30% | +/-10% on all 3 density values |

## Dataset: taiko_v2

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor +/- 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 500 onset logits + 1 stop logit
5. If stop_logit > 0 (sigmoid > 0.5): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If onset argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Metrics (eval 4, stopped early)

| Metric | Value |
|---|---|
| HIT (<=3% or +/-1 frame) | 70.3% |
| MISS (>20% off) | — |
| stop_f1 | 0.469 |
| stop_precision | 0.338 |
| stop_recall | 0.766 |
| no_audio_stop | 81.5% |
| AR step0 | 70.8% |
