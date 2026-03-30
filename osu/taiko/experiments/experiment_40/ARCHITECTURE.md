# Experiment 40 — Full Architecture Specification

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
| logits | (B, 501) | 500 onset bin offsets (0-499, ~0-2.5s ahead) + 1 STOP class (500) |

## Model: OnsetDetector (unified)

**Total parameters: ~16M**

Single-path unified architecture: AudioEncoder produces audio tokens, GapEncoder produces gap tokens from event context, both are concatenated and jointly attended through fusion self-attention layers.

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. AudioEncoder (mel → audio tokens)

```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
  → 4 TransformerEncoderLayers(d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each followed by FiLM(cond)
  → audio_tokens: (B, 250, 384)
```

4x downsample: 1000 mel frames → 250 tokens. Cursor at token 125.

### 3. Mel Ramp Event Embedding

Events are embedded directly into the mel spectrogram before the AudioEncoder processes it. For each audio frame, computes a "time since last event" exponential ramp:

```
ramp(t) = clamp(1 - (t - last_event_before_t) / gap_to_next_event, 0, 1)
```

Ramps are written to reserved mel bands (0-2 bottom, 77-79 top) with fading intensity inward. This injects event timing information into the audio signal itself.

### 4. GapEncoder (events → gap tokens)

For each of 128 context events, computes inter-onset intervals:
- **Gap encoding**: sinusoidal encoding of absolute gap between consecutive events
- **Audio snippets**: 10-frame mel snippets extracted at each event position, encoded via MLP

```
gap_features (B, 128, 384) + snippet_features (B, 128, 384)
  → + SequencePositionEmb(128)
  → 2 TransformerEncoderLayers(d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each followed by FiLM(cond)
  → gap_tokens: (B, 128, 384)
```

### 5. Fusion Transformer (4 layers)

```
concat([audio_tokens, gap_tokens]) → (B, 378, 384)
for each of 4 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```

### 6. Output Head

```
cursor = x[:, 125, :]  # cursor token (B, 384)
logits = Linear(384, 501)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)  # 1d smoothing: Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 501)
```

### FiLM Conditioning

Applied after conv stem, after each encoder layer, and after each fusion layer:
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss: OnsetLoss

Mixed hard CE + soft target loss.

### Soft Targets (Trapezoid)

For each non-STOP target at bin `t`, create a soft distribution over all 500 bins:
- Compute `|log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Within 3% ratio: full credit (plateau)
- 3% to 20%: linear ramp to zero
- Beyond 20%: zero
- Frame tolerance ±2: always get some credit regardless of ratio

### Combined Loss

```
loss = 0.5 * hard_CE + 0.5 * soft_CE
```

Where hard_CE = standard cross-entropy, soft_CE = KL divergence with soft targets.

STOP samples get `stop_weight=3.0x` multiplier.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset, ~5.25M samples) |
| **Balanced sampling** | **ON (1/count^0.7 weights)** |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |

This experiment uses **balance_power=0.7** (up from 0.5). This gives distant bins (200-500) 26.5% of training exposure (up from 11.5% at power 0.5) while common bins (10-25) drop from 21.3% to 12.2%.

## Augmentation

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

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor ± 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 501 logits
5. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Result

Killed after eval 1. Stronger balanced sampling (power=0.7) degraded short-range performance without helping distant predictions.

| Metric | Exp 40 eval 1 | Exp 35-C eval 1 |
|---|---|---|
| HIT | 63.4% | 66.2% |
| Miss | 35.9% | 33.1% |
| Frame err | 16.6 | 13.6 |
