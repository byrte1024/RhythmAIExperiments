# Experiment 25 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position. Single unified model — audio and gap features fused via self-attention in one pathway.

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
| logits | (B, 501) | 500 onset bin offsets (0-499, ~0-2.5s ahead) + 1 STOP class (500). Single unified output. |

## Model: OnsetDetector

**Total parameters: ~19M (all trainable)**

Unified fusion architecture: AudioEncoder and GapEncoder produce separate token sequences that are concatenated and jointly processed by a FusionTransformer. No separate audio/context paths, no post-hoc combination. Single output from cursor token.

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. AudioEncoder — mel → audio tokens

```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
  → 4 TransformerEncoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each followed by FiLM(cond)
  → audio_tokens: (B, 250, 384)
```

4x downsample: 1000 mel frames → 250 tokens. Cursor at token 125.

### 3. GapEncoder — event gaps + mel snippets → gap tokens

Replaces the old EventEncoder. Computes inter-onset intervals, extracts mel snippets at event positions, and processes through self-attention. Operates at full d_model=384 (no bottleneck).

#### 3a. Gap Sequence Computation

```
event_offsets (B, 128)
  → gap_before[i] = offset[i] - offset[i-1] for i in 1..C-1   → (B, 127)
  → cursor_gap = -offset[-1] (time since last event)            → (B, 1)
  → all_gaps = cat([gap_before, cursor_gap])                     → (B, 128)
  → gap_valid[i] = event_valid[i] & event_valid[i-1]
  → all_gap_mask = ~gap_valid                                    → (B, 128)
```

#### 3b. Gap Feature Building

```
gap_features = SinusoidalPosEmb(384)(all_gaps.abs())  → (B, 128, 384)

mel snippet extraction at event positions:
  event_mel_frames = 500 + event_offsets → snippet frame centers
  for each gap position: extract 10-frame (~50ms) mel window → (B, 128, 80*10)
  → snippet_encoder: Linear(800, 384) → GELU → Linear(384, 384)
  → event_snippet_feat (B, 128, 384)

x = gap_features + event_snippet_feat
  → + nn.Embedding(129, 384) sequence-order positions
```

#### 3c. Gap Self-Attention

```
x (B, 128, 384)
  → NaN guard: if all gap positions masked, unmask last position as dummy
  → 2 TransformerEncoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each with src_key_padding_mask=gap_mask, followed by FiLM(cond)
  → gap_tokens (B, 128, 384), gap_mask (B, 128)
```

### 4. FusionTransformer — joint self-attention over audio + gap tokens

```
x = cat([audio_tokens, gap_tokens], dim=1) → (B, 378, 384)
  378 = 250 audio tokens + 128 gap tokens

fused_mask = cat([zeros(B, 250), gap_mask], dim=1) → (B, 378)
  audio tokens never masked, gap tokens use gap_mask

4 TransformerEncoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
  each with src_key_padding_mask=fused_mask, followed by FiLM(cond)

→ fused_tokens (B, 378, 384)
```

Every layer: audio attends to gaps and gaps attend to audio — bidirectional, deep interaction.

### 5. Output Head — cursor extraction → logits

```
cursor = fused_tokens[:, 125, :] → (B, 384)
  position 125 = center of audio window (audio tokens occupy positions 0-249)

logits = Linear(384, 501)(LayerNorm(384)(cursor)) → (B, 501)
logits = logits + Conv1d_smooth(logits)
```

Smoothing: `Conv1d(1, 8, k=5, pad=2) → GELU → Conv1d(8, 1, k=5, pad=2)`

### FiLM Conditioning

Applied after conv stem, after every AudioEncoder layer, every GapEncoder layer, and every FusionTransformer layer:
```
cond (B, 64) → Linear(64, 2*D) → split → scale (B, D), shift (B, D)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, dim).

## Loss

Single OnsetLoss on the unified logits:

```
loss = OnsetLoss(logits, targets)
```

### OnsetLoss: Mixed Hard CE + Soft Target

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

Where hard_CE = standard cross-entropy, soft_CE = KL divergence with soft targets.

STOP samples get `stop_weight=3.0x` multiplier.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 64 |
| Epochs | 5 (killed) |
| Scheduler | CosineAnnealingLR |
| Subsample | 4 (25% of training data) |
| Warm-start | None (trained from scratch) |
| Trainable params | ~19M (all) |
| Gradient clipping | 1.0 |
| Evals per epoch | 1 |

## Augmentation (Light)

Deliberately reduced augmentation to encourage context reliance.

### Audio Augmentation

| Aug | Description |
|---|---|
| Gain jitter | ±2dB @ 30% probability |
| Noise injection | sigma=0.1-0.3 @ 15% probability |
| SpecAugment freq mask | 1-8 bands, 1 mask @ 20% |
| SpecAugment time mask | 1-30 frames, 1 mask @ 20% |
| Fade in/out | 10% each |

### Context Augmentation (applied to event_offsets during training)

| Aug | Description |
|---|---|
| Event jitter | Recency-scaled per-event position noise |
| Global shift | Shift all events by a random offset |
| Event insertion/deletion | ~8% rate: add or remove random events |
| Event dropout | Random removal of individual events |
| Context truncation | Occasionally truncate context to fewer events |
| Conditioning jitter | Small perturbation to density metadata |

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
4. Forward pass → 501 logits (unified output)
5. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Metrics (E5 best)

| Metric | Value |
|---|---|
| HIT (<=3% or +/-1 frame) | 68.6% |
| MISS (>20% off) | 30.9% |
| Accuracy | 49.8% |
| Score | 0.330 |
| Stop F1 | 0.495 |
| Frame error mean | 11.9 |
| no_events acc | 47.5% |
| Context delta | 2.3% |
| Val loss | 2.665 |

## Context Status

Context contribution shrank over training: 6.8% (E1) to 2.3% (E5). The model converged to an audio-dominant solution. Gap tokens were drowned out by the 250 audio tokens (7:1 ratio) in self-attention. Overfitting from E2 onward (val loss 2.623 at E2 → 2.665 at E5). Matched exp 14's ceiling (~68.9% HIT) but did not exceed it. Lighter augmentation backfired — accelerated overfitting without improving context contribution.
