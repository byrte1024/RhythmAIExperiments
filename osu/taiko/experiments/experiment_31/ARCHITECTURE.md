# Experiment 31 — Full Architecture Specification

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

## Model: DualStreamOnsetDetector

**Total parameters: ~23.3M**

Two independent streams process audio and context in parallel, then exchange information via bidirectional cross-attention in the final layers.

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. Audio Stream: AudioEncoder (conv stem + 4 transformer layers)

#### Conv Stem

```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
```

4x downsample: 1000 mel frames → 250 tokens. Cursor at token 125.

#### Transformer Layers (4 layers)

```
for each of 4 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```

Output: audio_tokens (B, 250, 384)

### 3. Context Stream: GapEncoder (4 transformer layers)

Deeper than prior experiments (4 layers instead of 2) to give context its own processing depth.

#### Gap Computation

```
gap_before[i] = event_offsets[i] - event_offsets[i-1]  (for consecutive valid events)
time_since_last = -event_offsets[-1]                    (cursor gap)
all_gaps = [gap_before_1, ..., gap_before_C-1, time_since_last]  (B, C)
```

#### Feature Extraction

```
gap_features = SinusoidalPosEmb(384)(all_gaps.abs())          (B, C, 384)
snippet_feat = snippet_encoder(mel_snippet_at_event_pos)       (B, C, 384)
  snippet_encoder: Linear(80*10, 384) → GELU → Linear(384, 384)
  (10-frame mel window at each event = ~50ms)

x = gap_features + snippet_feat + Embedding(129, 384)(seq_pos)
```

#### Self-Attention (4 layers)

```
for each of 4 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x, src_key_padding_mask=gap_mask)
    x = FiLM(cond)(x)
```

Output: gap_tokens (B, C, 384), gap_mask (B, C)

### 4. Cross-Attention Fusion (2 layers)

Each CrossAttentionFusionLayer performs bidirectional cross-attention:

```
for each of 2 CrossAttentionFusionLayer:
    # audio cross-attends to gap (audio queries, gap K/V)
    a_norm = LayerNorm(384)(audio)
    a_cross = MultiheadAttention(384, 8, dropout=0.1)(
        query=a_norm, key=gap, value=gap, key_padding_mask=gap_mask
    )
    audio = audio + a_cross
    audio = audio + FFN(audio)     # LN → Linear(384,1536) → GELU → Linear(1536,384) → Dropout
    audio = FiLM(cond)(audio)

    # gap cross-attends to audio (gap queries, audio K/V)
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
# save pre-fusion audio cursor
audio_pre_cursor = audio[:, 125, :]   (B, 384)

# after cross-attention fusion:
cursor = audio[:, 125, :] + audio_pre_cursor   (B, 384)  # skip connection

logits = Linear(384, 501)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)  # Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 501)
```

Note: The skip connection was added in exp 31-B/32. In exp 31 proper, cursor = audio[:, 125, :] without the skip.

### FiLM Conditioning

Applied after conv stem, after every AudioEncoder layer, every GapEncoder layer, and in every cross-attention fusion layer:
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

## Key Metrics (eval 2)

| Metric | Value |
|---|---|
| HIT (<=3% or +/-1 frame) | 50.8% |
| MISS (>20% off) | 40.8% |
| Context delta | 18.8% |
| no_events accuracy | 9.8% |
| Unique predictions | 82 |

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
