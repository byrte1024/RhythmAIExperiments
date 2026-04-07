# Experiment 38-C — Full Architecture Specification

## Task

Predict onset probability at every position in the future audio window. Per-token framewise detection with unweighted BCE loss.

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
| onset_probs | (B, 125) | Sigmoid probabilities for each future token (~20ms resolution per token) |

## Model: FramewiseOnsetDetector

**Total parameters: ~12M**

FramewiseOnsetDetector architecture (~12M params) with corrected causal mask and no teacher forcing. This experiment removes positive class weighting from the loss.

### Mel-Embedded Event Ramps (past half only)

Exponential decay ramps in past mel frames (0-499) only. Future half stays clean.

```
For each frame t in [0, 499]:
  ramp(t) = exp(-ln(2) * elapsed / half_life)
  elapsed = t - last_event_before_t (>= 0)
  half_life = 0.03 * gap_to_next_event (>= 0.5 frames)
  ramp[:, 500:] = 0

Audio scaling:
  Training: mel = mel * random(0.25, 0.75) + ramp * 10.0
  Eval:     mel = mel * 0.5 + ramp * 10.0
```

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. Conv Stem (mel → 250 tokens)

```
mel_with_ramps (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
```

Tokens 0-124: past audio with ramps. Tokens 125-249: future audio, clean.

### 3. Self-Attention with Causal Mask (6 layers)

```
for each of 6 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x, src_mask=causal_mask)
    x = FiLM(cond)(x)
```

Causal mask: past tokens see only past, future tokens see past + causal within future.

### 4. Per-Token Onset Head (future tokens only)

```
future_tokens = x[:, 125:, :]   # (B, 125, 384)
onset_logits = onset_head(future_tokens).squeeze(-1)   # (B, 125)
  onset_head: LayerNorm(384) → Linear(384, 192) → GELU → Linear(192, 1)
onset_probs = sigmoid(onset_logits)   # (B, 125)
```

### FiLM Conditioning

```
cond (B, 64) → Linear(64, 2*384) → split → scale, shift
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss: Unweighted BCE

```
loss = F.binary_cross_entropy(probs, target)
```

No positive class weighting (pos_weight removed). The 13% natural positive ratio is learnable without upweighting.

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
| pos_weight | none (1.0) |

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

### Mel Ramp Augmentation
| Aug | Rate | Params |
|---|---|---|
| Amplitude jitter | 100% (training) | Audio scaled by random(0.25, 0.75) per sample |

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

## Key Metrics (eval 1)

| Metric | Value |
|---|---|
| Event recall | 23.4% |
| Pred precision | 11.7% |
| F1 | 0.156 |
| Preds/window | 32.5 |
| Hallucination | 50.1% |

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
