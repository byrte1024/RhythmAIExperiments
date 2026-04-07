# Experiment 33 — Full Architecture Specification

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

## Model: InterleavedOnsetDetector

**Total parameters: ~25M**

Alternating self-attention and cross-attention layers. Each block: audio self-attention, gap self-attention, bidirectional cross-attention. 4 blocks = 12 attention layers total.

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. Audio Conv Stem

```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
  → audio: (B, 250, 384)
```

4x downsample: 1000 mel frames → 250 tokens. Cursor at token 125.

### 3. Gap Feature Extraction

Builds gap token representations without any transformer layers (those are in the interleaved blocks).

#### Gap Computation

```
gap_before[i] = event_offsets[i] - event_offsets[i-1]  (for consecutive valid events)
time_since_last = -event_offsets[-1]                    (cursor gap)
all_gaps = [gap_before_1, ..., gap_before_C-1, time_since_last]  (B, C)
```

#### Feature Building

```
gap_features = SinusoidalPosEmb(384)(all_gaps.abs())          (B, C, 384)
snippet_feat = snippet_encoder(mel_snippet_at_event_pos)       (B, C, 384)
  snippet_encoder: Linear(80*10, 384) → GELU → Linear(384, 384)
  (10-frame mel window at each event = ~50ms)

gap = gap_features + snippet_feat + Embedding(129, 384)(seq_pos)
```

Output: gap (B, C, 384), gap_mask (B, C)

### 4. Interleaved Blocks (4 blocks)

Each block has three stages:

```
for each of 4 blocks:
    # Stage 1: Audio self-attention
    audio = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(audio)
    audio = FiLM(cond)(audio)

    # Stage 2: Gap self-attention
    gap = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(gap, src_key_padding_mask=gap_mask)
    gap = FiLM(cond)(gap)

    # Stage 3: Bidirectional cross-attention (CrossAttentionFusionLayer)
    # audio cross-attends to gap
    a_norm = LayerNorm(384)(audio)
    a_cross = MultiheadAttention(384, 8, dropout=0.1)(
        query=a_norm, key=gap, value=gap, key_padding_mask=gap_mask
    )
    audio = audio + a_cross
    audio = audio + FFN(audio)     # LN → Linear(384,1536) → GELU → Linear(1536,384) → Dropout
    audio = FiLM(cond)(audio)

    # gap cross-attends to audio
    g_norm = LayerNorm(384)(gap)
    g_cross = MultiheadAttention(384, 8, dropout=0.1)(
        query=g_norm, key=audio, value=audio
    )
    gap = gap + g_cross * valid_mask
    gap = gap + FFN(gap) * valid_mask
    gap = FiLM(cond)(gap)

    # activation clamping for stability
    audio = audio.clamp(-1e4, 1e4)
    gap = gap.clamp(-1e4, 1e4)
```

### 5. Output Head

```
cursor = audio[:, 125, :]   (B, 384)

logits = Linear(384, 501)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)  # Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 501)
```

### FiLM Conditioning

Applied after conv stem, after every self-attention layer (audio and gap), and after every cross-attention layer:
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
- Frame tolerance +/-2: always get some credit regardless of ratio

### Combined Loss

```
loss = 0.5 * hard_CE + 0.5 * soft_CE
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

## Augmentation

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
4. Forward pass → 501 logits
5. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Metrics (eval 5)

| Metric | Value |
|---|---|
| HIT (<=3% or +/-1 frame) | 19.4% |
| MISS (>20% off) | 58.5% |
| Context delta | 1.8% |
| no_events accuracy | 5.8% |
| Unique predictions | 49 |

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
