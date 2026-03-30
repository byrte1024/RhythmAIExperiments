# Experiment 37 — Full Architecture Specification

## Task

Predict ALL onsets in the forward audio window of an osu!taiko rhythm game chart using per-bin sigmoid outputs. Each bin independently predicts P(onset here).

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
| logits | (B, 501) | 500 onset bin offsets (0-499) + 1 STOP class (500). Interpreted as independent sigmoid probabilities (not softmax). |

## Model: OnsetDetector (with exponential decay mel ramps + amplitude jitter)

**Total parameters: ~19M**

Same unified OnsetDetector architecture (~19M params): exponential decay ramps (sharp spikes at event positions) are embedded into the mel spectrogram, then processed through AudioEncoder (conv stem + 4 transformer layers), GapEncoder (2 transformer layers), and a 4-layer Fusion Transformer. The cursor token at position 125 feeds the output head producing (B, 501) logits. The change is in interpretation (sigmoid instead of softmax) and loss (per-bin BCE instead of cross-entropy).

### Mel-Embedded Event Ramps (exponential decay, full-band)

```
For each frame t:
  ramp(t) = exp(-ln(2) * elapsed / half_life)
  elapsed = t - last_event_before_t (>= 0)
  half_life = 0.03 * gap_to_next_event (>= 0.5 frames)

Audio scaling:
  Training: mel = mel * random(0.25, 0.75) + ramp * 10.0
  Eval:     mel = mel * 0.5 + ramp * 10.0
```

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. AudioEncoder (conv stem + 4 transformer layers)

#### Conv Stem

```
mel_with_ramps (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
```

#### Transformer Layers (4 layers)

```
for each of 4 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```

### 3. GapEncoder (2 transformer layers)

#### Gap Computation

```
gap_before[i] = event_offsets[i] - event_offsets[i-1]
time_since_last = -event_offsets[-1]
all_gaps = [gap_before_1, ..., gap_before_C-1, time_since_last]  (B, C)
```

#### Feature Extraction

```
gap_features = SinusoidalPosEmb(384)(all_gaps.abs())          (B, C, 384)
snippet_feat = snippet_encoder(mel_snippet_at_event_pos)       (B, C, 384)
  snippet_encoder: Linear(80*10, 384) → GELU → Linear(384, 384)

x = gap_features + snippet_feat + Embedding(129, 384)(seq_pos)
```

#### Self-Attention (2 layers)

```
for each of 2 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x, src_key_padding_mask=gap_mask)
    x = FiLM(cond)(x)
```

### 4. Fusion Transformer (4 layers)

```
x = concat([audio_tokens, gap_tokens], dim=1)   → (B, 250+C, 384)
mask = concat([zeros(B, 250), gap_mask], dim=1)

for each of 4 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x, src_key_padding_mask=mask)
    x = FiLM(cond)(x)
```

### 5. Output Head

```
cursor = x[:, 125, :]  # cursor token (B, 384)
logits = Linear(384, 501)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)  # Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 501)

probs = sigmoid(logits)  # NOT softmax — each bin is independent
```

### FiLM Conditioning

```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

## Loss: SigmoidMultiTargetLoss

Per-bin binary cross-entropy with soft trapezoid targets.

### Soft Targets

Same log-ratio trapezoid soft labels as OnsetLoss, but used as per-bin BCE targets (not a probability distribution):
- Bins near a real onset: target near 1.0
- Bins in ramp zone: interpolated target
- Bins far from any onset: target = 0.0

### Loss Computation

```
loss = F.binary_cross_entropy(sigmoid(logits), soft_targets, reduction='none')
loss = loss * pos_weight   where pos_weight=5.0 for bins with target > 0
loss = mean(loss)
```

pos_weight=5.0 upweights positive bins to handle class imbalance (~3-5 onsets per 500 bins). focal_gamma=0.0 (no focal modulation).

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
| pos_weight | 5.0 |
| focal_gamma | 0.0 |

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

## Inference (Threshold-based)

1. Start cursor at bin 0
2. Extract mel window: cursor +/- 500 bins
3. Forward pass → sigmoid(logits) → per-bin probabilities
4. All bins above threshold (0.05) are predicted onsets
5. Take earliest predicted onset as next event
6. Repeat until end of audio

## Key Metrics (eval 1)

| Metric | Value |
|---|---|
| Nearest HIT | 1.4% |
| Event recall | 99.9% |
| Pred precision | 3.5% |
| Hallucination | 96.5% |
| Preds/window | 468.5 |
