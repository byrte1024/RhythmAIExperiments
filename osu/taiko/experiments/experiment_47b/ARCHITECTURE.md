# Experiment 47-B — Full Architecture Specification (Failed)

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
| onset_logits | (B, 500) | 500 onset bin offsets (0-499, ~0-2.5s ahead). STOP is handled by separate gate. |
| gate_logit | (B,) | Scalar logit for binary STOP prediction. Gate sigmoid > threshold = STOP. |

## Model: EventEmbeddingDetector (with binary_stop=True)

**Total parameters: ~16.1M**

Base architecture is EventEmbeddingDetector (single-pathway: learned event embeddings added to audio tokens at event positions, unified self-attention) with binary STOP gate. Key improvements:
- **Flipped gate targets**: 1=stop, 0=onset (STOP is the positive/rare class)
- **Focal BCE (gamma=2)**: downweights easy negatives (confident onset predictions), focuses on hard positives (STOP boundaries)

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

### 4. Transformer (8 layers)

```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```

### 5. Gate Head (Binary STOP with Focal Loss)

```
cursor = x[:, 125, :]  # cursor token (B, 384)
gate_logit = MLP(cursor) → sigmoid → P(stop)
```

Gate target: 1=stop (rare, 0.3%), 0=onset (99.7%).

Gate loss: focal BCE (gamma=2):
```
focal_bce = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Where gamma=2 downweights easy negatives (confident onset predictions) and focuses gradient on hard positives (STOP boundaries). Loss averaged with `.mean()` reduction over the batch.

### 6. Onset Head (500-class)

```
cursor = x[:, 125, :]  # cursor token (B, 384)
logits = Linear(384, 500)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)
→ (B, 500)
```

Onset head only trained on non-STOP samples.

### Combined Loss

```
loss = gate_loss * gate_weight + onset_loss
```

gate_weight=2.0.

### FiLM Conditioning

Applied after conv stem and after every transformer layer:
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss: OnsetLoss + Focal BCE Gate Loss

### Onset Loss (Trapezoid, non-STOP samples only)
```
loss = 0.5 * hard_CE + 0.5 * soft_CE
```

### Gate Loss (Focal BCE)
```
gate_loss = focal_bce(gate_logit, target).mean()  # target: 1=stop, 0=onset
```

Additional logging: stop_pred_rate, stop_target_rate, stop_hallucinations, stop_misses.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset, ~5.25M samples) |
| Balanced sampling | ON (1/count^0.5 weights) |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| gate_weight | 2.0 |

## Augmentation (~14% context corruption rate)

### Context Augmentation (applied to event_offsets)
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | Global ±3 bins + per-event ±3 bins * 1-2x recency scale |
| Event deletion | 5% | Drop 1-2 random events |
| Event insertion | 3% | Add 1 fake event between existing events |
| Partial metronome | 2% | Replace recent half of events with evenly-spaced |
| Partial adv metronome | 2% | Replace oldest half with dominant-gap metronome |
| Large time shift | 2% | Shift 2-4 recent events by ±50 bins |
| Context truncation | 5% | Keep only most recent N events |

### Audio Augmentation (applied to mel_window)
| Aug | Rate | Params |
|---|---|---|
| Audio fade-in | 10% | Linear fade over 20-100 frames |
| Audio fade-out | 10% | Linear fade over 20-100 frames |
| Mel gain | 30% | ±2dB |
| Mel noise | 15% | Gaussian sigma 0.1-0.3 |
| Freq jitter | 15% | Roll mel bands ±3 |
| SpecAugment freq | 20% | 1 mask, up to 10 bands |
| SpecAugment time | 20% | 1 mask, up to 30 frames |

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

## Failure

Stopped at eval 1. **Stop pred rate 0.000% — gate almost never fires.**

Stop F1=0.066, precision=0.93, recall=0.03. The gate is extremely conservative — when it does predict STOP it is right, but it almost never does.

Root cause: focal BCE with `.mean()` reduction. With 99.7% onset samples, the mean is dominated by onset loss. Focal loss correctly downweights easy onsets per-sample (STOP gets 2.75x more gradient than onset), but the 300:1 class ratio means STOP's contribution is still invisible after averaging. Gate loss (~0.007) is tiny compared to onset CE loss (~2.5). With gate_weight=2.0, the gate contributes 0.014 to a total of ~2.5 — the optimizer cannot see it.
