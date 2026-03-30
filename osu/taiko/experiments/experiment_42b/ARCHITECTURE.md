# Experiment 42-B — Full Architecture Specification

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

## Model: EventEmbeddingDetector

**Total parameters: 16.1M**

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

## Loss: OnsetLoss (Pure Hard CE)

**This experiment uses hard_alpha=1.0 (pure hard CE, no soft trapezoid).**

```
loss = 1.0 * hard_CE + 0.0 * soft_CE = hard_CE
```

Standard cross-entropy with frame_tolerance=3 (±3 bins acceptable). No soft trapezoid distribution at all. The model gets credit only for hitting within 3 bins of the exact target, regardless of target distance.

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
| **hard_alpha** | **1.0 (pure hard CE)** |
| **frame_tolerance** | **3 (±3 bins)** |

Short diagnostic run (1-2 evals only).

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

## Key Results

Entropy slashed by 45% vs exp 42 — soft targets were the confidence bottleneck. But accuracy drops and skip rate increases.

| Metric | Exp 42 (soft) | Exp 42-B (hard CE) |
|---|---|---|
| Mean entropy | 2.390 | 1.320 (-45%) |
| Mean confidence | 0.355 | 0.535 (+51%) |
| HIT rate | 73.2% | 68.7% (-4.5pp) |
| Skip-1 rate | 11.1% | 13.2% (+2.1pp) |
| target_dist correlation | +0.582 | +0.363 (flatter) |
