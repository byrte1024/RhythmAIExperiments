# Experiment 43 — Full Architecture Specification

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

EventEmbeddingDetector: single-pathway architecture where event context is injected by adding learned embeddings directly to audio tokens at event positions, then processed through unified self-attention. This experiment tests heavier augmentation rates and additional benchmarks.

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

## Augmentation (~43% context corruption rate — AGGRESSIVE)

This experiment uses much heavier context augmentation to simulate AR failure modes.

### Context Augmentation (applied to event_offsets)
| Aug | Rate | Params | Simulates |
|---|---|---|---|
| Event jitter | 100% | Global ±3 bins + per-event ±5 bins * 1-3x recency scale | Prediction errors |
| Event deletion | **15%** | Drop 1-4 random events | Skip errors |
| Event insertion | **10%** | Add 1-3 fake events | Hallucinations |
| **Metronome corruption** | **5%** | Replace recent half with evenly-spaced events | Model locks into repeating gap |
| **Advanced metronome** | **5%** | Replace oldest half with dominant-gap metronome | Right tempo, no variation |
| **Large time shift** | **5%** | Shift all events by ±100 bins | AR cursor drift |
| **Hallucination burst** | **3%** | Inject rapid spam section | Rapid spam |
| Context dropout | **5%** | Drop all events | Total context loss |
| Context truncation | **8%** | Keep only most recent 8-32 events | Partial context loss |

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

## New Benchmarks (introduced in this experiment)

### autoregress benchmark
32 consecutive AR predictions per sample:
- Feeds each prediction back as context (like real inference)
- Tracks: survival rate, entropy drift, prediction distribution drift, density comparison
- Graphs: survival curve, entropy over steps, prediction drift, density bar

### lightautoregress benchmark
32 consecutive predictions compared 1:1 to ground truth:
- pred[i] vs truth[i] — a cascade causes all future notes to misalign
- Tests whether the model can recover from early errors
- Graphs: HIT rate curve over steps, scatter at steps 0/4/8/16/31, frame error curve

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

Killed after eval 5. Augmentation too aggressive — model distrusted context entirely.

| Metric | Exp 43 (eval 5) | Exp 42 (eval 5) |
|---|---|---|
| HIT | 68.3% | 72.0% |
| Ctx delta | -0.4% | +4.3% |
| Unique preds (step 0) | 11 | 36 |

The ~43% context corruption rate taught the model to ignore context. Without context, it fell back to a narrow safe vocabulary of ~11 predictions, causing immediate metronome behavior — the exact failure mode the augmentation was designed to prevent.

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
