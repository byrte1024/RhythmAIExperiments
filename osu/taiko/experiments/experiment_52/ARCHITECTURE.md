# Experiment 52 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. This experiment sweeps the audio window size (A_BINS past, B_BINS future).

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, A+B) | Mel spectrogram, 80 bands, A_BINS + B_BINS frames at ~5ms/frame |
| event_offsets | (B, 128) | Past event positions relative to cursor (negative = past, int64) |
| event_mask | (B, 128) | Bool mask, True = padding (no event) |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] from chart metadata |

## Output

| Output | Shape | Description |
|---|---|---|
| logits | (B, B_BINS+1) | B_BINS onset bin offsets (0 to B_BINS-1) + 1 STOP class |

N_CLASSES = B_BINS + 1. Changes per sub-experiment.

## Sub-experiments

| Exp | A_BINS | B_BINS | Past | Future | N_CLASSES | Tokens | Status |
|---|---|---|---|---|---|---|---|
| 52-A | 250 | 250 | 1.25s | 1.25s | 251 | 125 | Done |
| 52-B | 500 | 250 | 2.5s | 1.25s | 251 | 188 | Pending |
| 52-C | 1000 | 250 | 5.0s | 1.25s | 251 | 312 | Pending |
| 52-D | 250 | 500 | 1.25s | 2.5s | 501 | 188 | Pending |
| 52-E | 500 | 500 | 2.5s | 2.5s | 501 | 250 | Baseline (exp 45) |
| 52-F | 1000 | 500 | 5.0s | 2.5s | 501 | 375 | Pending |
| 52-G | 250 | 1000 | 1.25s | 5.0s | 1001 | 312 | Pending |
| 52-H | 500 | 1000 | 2.5s | 5.0s | 1001 | 375 | Pending |
| 52-I | 1000 | 1000 | 5.0s | 5.0s | 1001 | 500 | Pending |
| 52-J | 500 | 125 | 2.5s | 0.625s | 126 | 156 | Pending |
| 52-K | 500 | 75 | 2.5s | 0.375s | 76 | 144 | Pending |
| 52-L | 500 | 33 | 2.5s | 0.165s | 34 | 133 | Running |

## Model: EventEmbeddingDetector (with gap ratios)

**Total parameters: ~16.5M** (head_proj size varies with N_CLASSES, negligible difference)

### 1. Conditioning MLP
```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. Conv Stem (mel → audio tokens)
```
mel (B, 80, A_BINS + B_BINS)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, (A+B)//4, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..(A+B)//4-1)
  → FiLM(cond)
  → x: (B, n_tokens, 384)
```
4x downsample. Cursor at token A_BINS // 4.

### 3. Event Embeddings (with gap ratios)
For each of 128 context events, compute 5 features:
- **Presence embedding**: learned parameter (1, 384)
- **Gap before**: sinusoidal encoding of distance from previous event
- **Gap after**: sinusoidal encoding of distance to next event (last event uses gap_before as proxy)
- **Gap ratio before**: sinusoidal encoding of `gap_before[i-1] / gap_before[i]` scaled by 50, clamped [0.1, 10.0]
- **Gap ratio after**: sinusoidal encoding of `gap_after[i+1] / gap_after[i]` same scaling

```
[presence (384) | gap_before (384) | gap_after (384) | ratio_before (384) | ratio_after (384)]
→ Linear(1920, 384) → GELU → Linear(384, 384)
→ event_emb: (B, 128, 384)
```
Events mapped to audio token positions: `token = (A_BINS + offset) // 4`. Only tokens 0 to (A_BINS//4 - 1) used. Scatter-added.

### 4. Transformer (8 layers)
```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```
Attention cost scales quadratically with token count: 500 tokens (1000/1000) costs 4x more than 250 tokens (500/500).

### 5. Output Head
```
cursor = x[:, A_BINS // 4, :]
logits = Linear(384, N_CLASSES)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)
→ (B, N_CLASSES)
```

### FiLM Conditioning
```
cond (B, 64) → Linear(64, 768) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

## Loss: OnsetLoss

Mixed hard CE + soft target loss: `loss = 0.5 * hard_CE + 0.5 * soft_CE`

Soft targets: trapezoid in log-ratio space (good_pct=3%, fail_pct=20%, frame_tolerance=2). STOP weight 1.5x.

N_CLASSES adapts to B_BINS + 1 — smaller B_BINS means fewer classes and easier classification.

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

Note: sample count changes with B_BINS. Smaller B_BINS → more STOP samples (targets beyond the window become STOP). At B_BINS=33, STOP is 41% of samples.

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

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor - A_BINS to cursor + B_BINS
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → N_CLASSES logits
5. If argmax = N_CLASSES-1 (STOP): hop cursor forward by hop_bins
6. If argmax = offset (0 to B_BINS-1): place event at cursor + offset, move cursor there
7. Repeat until end of audio

With small B_BINS (e.g. 33), the model hops frequently via STOP, scanning through the song in B_BINS-sized chunks.
