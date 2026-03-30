# Experiment 46 — Parameter Sweep Specification

## Type

Parameter sweep experiment. Four sub-experiments testing different hard/soft loss ratios, all sharing the same architecture.

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

## Sweep Variable

**hard_alpha**: controls the ratio of hard cross-entropy to soft trapezoid targets in the loss function.

| Sub-exp | hard_alpha | Soft weight | Hard weight | Description |
|---|---|---|---|---|
| 46-A | 0.0 | 100% | 0% | Pure soft targets |
| 46-B | 0.25 | 75% | 25% | Mostly soft |
| 46-C | 0.75 | 25% | 75% | Mostly hard |
| 46-D | 1.0 | 0% | 100% | Pure hard CE |

Exp 44 (hard_alpha=0.5) serves as the baseline — not rerun.

## Common Architecture: EventEmbeddingDetector (with gap ratios)

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
- **Gap ratio before**: sinusoidal encoding of (gap_2_before / gap_before), scaled by 50x, clamped 0.1-10.0
- **Gap ratio after**: sinusoidal encoding of (gap_2_after / gap_after), scaled by 50x, clamped 0.1-10.0

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

Each layer: pre-norm self-attention + FFN with FiLM density conditioning.

### 5. Output Head

```
cursor = x[:, 125, :]  # cursor token (B, 384)
logits = Linear(384, 501)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)  # 1d smoothing: Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 501)
```

### FiLM Conditioning

Applied after conv stem and after every transformer layer:
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss: OnsetLoss (swept)

### Soft Targets (Trapezoid)

For each non-STOP target at bin `t`, create a soft distribution over all 500 bins:
- Compute `|log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Within 3% ratio: full credit (plateau)
- 3% to 20%: linear ramp to zero
- Beyond 20%: zero
- Frame tolerance ±2: always get some credit regardless of ratio

### Combined Loss

```
loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE
```

STOP samples get `stop_weight=3.0x` multiplier.

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

Each sub-experiment ran for 2 evals. Two configuration changes:
- Conditioning jitter tightened to ±2% at 10% rate (better AR density adherence)
- Gap ratio features enabled

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

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor ± 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 501 logits
5. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Results (eval 2 comparison)

| hard_alpha | HIT | Exact | AR s0 | AR s1 | Metronome delta |
|---|---|---|---|---|---|
| 0.00 | 68.2% | 40.1% | 70.6% | 25.6% | +4.0pp |
| 0.25 | 71.1% | 49.1% | 72.0% | 40.2% | +4.5pp |
| 0.50 (exp 44) | 70.9% | 52.0% | 73.3% | 40.3% | +14.5pp |
| 0.75 | 70.2% | 50.9% | 69.4% | 38.5% | +5.3pp |
| 1.00 | 68.6% | 50.8% | 67.7% | 39.6% | +9.5pp |

**Conclusion**: hard_alpha is a precision knob, not a behavior knob. Neither extreme offers a clear win over the 0.5 default. The extremes are clearly bad: pure soft loses 12pp exact accuracy; pure hard loses ~2pp HIT. The same error structure (ray bands, metronome lock-in, musical-ratio confusions) appears at every setting.
