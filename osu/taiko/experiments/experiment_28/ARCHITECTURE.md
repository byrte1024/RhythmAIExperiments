# Experiment 28 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position. Single unified model with focal loss to redirect gradient toward hard (ambiguous) samples.

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

Unified fusion architecture: AudioEncoder and GapEncoder produce separate token sequences that are concatenated and jointly processed by a FusionTransformer. No separate audio/context paths. Single output from cursor token.

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

Computes inter-onset intervals, extracts mel snippets at event positions, and processes through self-attention. Operates at full d_model=384.

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

### 5. Output Head — cursor extraction → logits

```
cursor = fused_tokens[:, 125, :] → (B, 384)

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

Single OnsetLoss with focal modulation (gamma=2.0):

```
loss = OnsetLoss(logits, targets, gamma=2.0)
```

### OnsetLoss: Mixed Hard CE + Soft Target with Focal Modulation

### Soft Targets (Trapezoid)

For each non-STOP target at bin `t`, create a soft distribution over all 500 bins:
- Compute `|log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Within 3% ratio: full credit (plateau)
- 3% to 20%: linear ramp to zero
- Beyond 20%: zero
- Frame tolerance +/-2: always get some credit regardless of ratio

### Combined Loss with Focal Modulation

```
base_loss = 0.5 * hard_CE + 0.5 * soft_CE
pt = model's probability assigned to the correct class
focal_weight = (1 - pt)^gamma  where gamma = 2.0
loss = focal_weight * base_loss
```

Effect of gamma=2.0:
- 90% confident sample: loss multiplied by 0.01 (strongly downweighted)
- 50% confident sample: loss multiplied by 0.25
- 10% confident sample: loss multiplied by 0.81 (nearly full weight)

STOP samples get `stop_weight=3.0x` multiplier.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | ~3.25 (killed after eval 9) |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset, ~600K samples) |
| Evals per epoch | 4 |
| Focal gamma | 2.0 |
| Warm-start | None (trained from scratch) |
| Trainable params | ~19M (all) |
| Gradient clipping | 1.0 |

## Augmentation (Heavy)

Heavy audio augmentation (gain jitter ±3dB @ 50%, noise injection sigma=0.1-0.4 @ 30%, freq jitter ±1-5 bins @ 30%, SpecAugment freq/time masks @ 40%, temporal corruption @ 2%, fade in/out @ 10%).

### Audio Augmentation

| Aug | Description |
|---|---|
| Gain jitter | ±3dB @ 50% probability |
| Noise injection | sigma=0.1-0.4 @ 30% probability |
| Freq jitter | ±1-5 bins @ 30% — roll mel bands up/down, zero-fill edges |
| SpecAugment freq mask | 1-15 bands, 1-2 masks @ 40% |
| SpecAugment time mask | 1-50 frames, 1-2 masks @ 40% |
| Temporal corruption | 10-frame chunk shuffle @ 2% |
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
- Full dataset: ~600K training samples

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor +/- 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 501 logits (unified output)
5. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Metrics (eval 7 best HIT)

| Metric | Value |
|---|---|
| HIT (<=3% or +/-1 frame) | 68.6% |
| MISS (>20% off) | 31.0% |
| Accuracy | 50.8% |
| Score | 0.329 |
| Stop F1 | 0.552 (best ever) |
| Frame error mean | 11.8 |
| Context delta | ~0% (oscillated -0.7% to 1.0%) |
| Val loss | 1.613 |

## Context Status

Focal loss improved calibration (clean HIT/MISS entropy separation) and Stop F1 (0.552, best ever), but lowered the HIT ceiling to ~68.6% (1.2pp below exp 27's 69.8%). Context delta was zero or negative — worse than exp 27's already-low 1.5%. Focal loss redirected gradient to hard cases, but the model still solved them through audio, not context. Confirmed that the context problem requires structural forcing, not loss reweighting.

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
