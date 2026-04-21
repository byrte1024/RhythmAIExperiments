# Experiment 67b — Full Architecture Specification

**Only change from exp67:** Conv1d smoothing added to ratio head output to prevent banding.

## Task

Predict the next onset timing via ratio-based decomposition: divisor (base rhythm) × ratio (multiple) − offset (cursor position) = onset bin.

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1000) | Mel spectrogram, 80 bands, 1000 frames (500 past + 500 future) |
| event_offsets | (B, 128) | Past event positions relative to cursor |
| event_mask | (B, 128) | Bool mask, True = padding |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] |

## Output

| Output | Shape | Description |
|---|---|---|
| divisor_logits | (B, 250) | Dominant gap prediction |
| offset_logits | (B, 100) | Cursor-to-last-event distance |
| ratio_logits | (B, 256) | 255 ratio bins (0.125x-8.0x) + STOP |
| proposal_logits | (B, 250) | Stage 1 per-token onset confidence |
| derived_bin | (B,) | Final onset = divisor × ratio − offset |

## Window Configuration

| Parameter | Value |
|---|---|
| A_BINS | 500 |
| B_BINS | 500 |
| B_PRED | 250 |
| N_CLASSES | 251 (backbone, for S1 loss compatibility) |
| N_RATIO_CLASSES | 256 (255 ratio bins + STOP) |

## Ratio Bins

255 bins log-spaced from 0.125 to 8.0:

```
bins = exp(linspace(log(0.125), log(8.0), 255))
```

| Property | Value |
|---|---|
| R_MIN | 0.125 (1/8x) |
| R_MAX | 8.0 (8x) |
| Center bin (127) | 1.000x |
| Resolution | ~1.65% per bin |
| Bins per octave | ~42 |
| Total octaves | 6 |
| STOP | bin 255 |

Key positions: 0.25x=bin 42, 0.5x=bin 85, 1.0x=bin 127, 2.0x=bin 169, 4.0x=bin 212

## Model: ProposeSelectDetector + RatioHeads

**Backbone: ~23.5M params (ProposeSelectDetector, same as exp58)**
**RatioHeads: ~761K params**
**Total: ~24.3M params**

### Shared Conv Stem
```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → x: (B, 250, 384)
```

### Stage 1: Proposer (4 layers, pure audio)
```
x_proposer = x.clone()
for 4 transformer layers:
    x_proposer = TransformerEncoderLayer(d=384, heads=8, ffn=1536)(x_proposer)
proposal_logits = Linear(384, 1)(LayerNorm(384)(x_proposer)).squeeze(-1) → (B, 250)
proposal_conf = sigmoid(proposal_logits)
```

### Proposal Embedding
```
x = x + MLP(proposal_conf) → (B, 250, 384)
```

### Stage 2: Selector (8 layers with events + FiLM)
```
conditioning → FiLM
event_embeddings (gap ratios) → scatter-add to tokens
for 8 layers:
    x = TransformerEncoderLayer(d=384, heads=8, ffn=1536)(x)
    x = FiLM(cond)(x)
cursor_token = x[:, 125, :] → (B, 384)
```

### Head 1: Divisor
```
cursor_token (B, 384)
  → LayerNorm → Linear(384, 192) → GELU → Linear(192, 250)
  → divisor_logits (B, 250)

divisor_probs = softmax(divisor_logits)
divisor_value = sum(divisor_probs * [1, 2, ..., 250]) → (B, 1) soft expected value
```

### Head 2: Offset
```
cursor_token (B, 384)
  → LayerNorm → Linear(384, 192) → GELU → Linear(192, 100)
  → offset_logits (B, 100)

offset_probs = softmax(offset_logits)
offset_value = sum(offset_probs * [0, 1, ..., 99]) → (B, 1) soft expected value
```

### Head 3: Ratio (sees Head 1+2)
```
divisor_emb = Linear(1, 384) → GELU → Linear(384, 384) applied to divisor_value
offset_emb = Linear(1, 384) → GELU → Linear(384, 384) applied to offset_value

ratio_input = cursor_token + divisor_emb + offset_emb
→ LayerNorm → Linear(384, 384) → GELU → Linear(384, 256)
→ ratio_logits_raw (B, 256)
→ + Conv1d_smooth(ratio_logits_raw)   # NEW: Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ ratio_logits (B, 256)
```

The Conv1d smoothing correlates neighboring ratio bins, preventing the model from collapsing to a few isolated peaks (horizontal banding observed in exp67).

### Derived Position
```
ratio_probs = softmax(ratio_logits[:, :255])
expected_ratio = sum(ratio_probs * ratio_bins) → (B,)
is_stop = argmax(ratio_logits) == 255

derived_bin = round(divisor_value * expected_ratio - offset_value)
derived_bin = clamp(derived_bin, 0, 249)
derived_bin[is_stop] = 250  # STOP
```

## Loss

### Loss A (Heads 1+2, auxiliary, stop gradient from Loss B)
```
divisor_target = dominant gap from last 128 events (most frequent cluster)
offset_target = cursor_offset (0 normally, >0 with augmentation)

loss_A = CE(divisor_logits, divisor_target) + CE(offset_logits, offset_target)
```

### Loss B (Head 3, primary)
```
# OnsetLoss on derived position
onset_loss = OnsetLoss(derived_logits, target_bin)  # standard trapezoid + ramp

# Ratio hill loss (distance in log-ratio space)
dynamic_ratio_target = (target_bin + offset_pred) / divisor_pred
ratio_loss = compute_ratio_loss(ratio_logits, snapped_ratio_target)

loss_B = onset_loss + ratio_loss
```

### S1 Loss
```
s1_loss = focal_BCE(proposal_logits, proposal_targets, pos_weight=5.0, gamma=2.0)
```

### Combined
```
During warmup (eval 1, configurable): loss = s1_loss + 0.1 * loss_A
After warmup (eval 2+): loss = loss_B + 0.1 * loss_A + 0.5 * s1_loss
```

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Balanced sampling | ON |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Proposer freeze | 0 evals (S1 warm-started from exp67) |
| Ratio freeze | 0 evals (divisor+offset warm from exp67) |
| Warm-start | exp67 eval 1 (S1 + divisor + offset trained, ratio untrained) |
| ramp_alpha | 2.5 |
| Cursor offset aug | 30% (shift cursor between events) |

## Augmentation

### Context Augmentation
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | ±3 bins |
| Event deletion | 5% | Drop 1-2 events |
| Event insertion | 3% | Add 1 fake event |
| Partial metronome | 2% | Replace recent half |
| Partial adv metronome | 2% | Replace oldest half |
| Large time shift | 2% | ±50 bins |
| Context truncation | 5% | Keep 8-32 recent |
| **Cursor offset** | **30%** | **Shift cursor between events** |

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
| Density jitter | 30% | ±10% |

## Dataset: taiko_v2

- 10,048 charts, ~5.25M samples
- Train/val split: 90/10 by song (seed 42)
- Returns 8 elements: mel, events, mask, cond, target, ratio_target, proposals, cursor_offset

## Metrics

### Standard (on derived bin)
- HIT%, MISS%, GOOD%, accuracy, exact match
- Stop F1
- Frame error mean/median/p90

### Ratio-specific
- Divisor accuracy (exact, ±3 bins, mean error)
- Offset accuracy (exact, mean error)
- Ratio HIT (within 5% of correct ratio)
- Ratio MISS (>20% off in ratio space)
- Ratio stop F1

### Graphs
- Ratio target vs predicted scatter
- Ratio distribution (target vs predicted)
- Divisor distribution (predicted vs actual dominant gaps)
- Offset distribution (predicted vs actual)
- Divisor target vs predicted scatter
- Summary metrics text panel

## Inference

```python
divisor = argmax(Head1)  # or soft expectation
offset = argmax(Head2)
ratio = argmax(Head3)

if ratio == STOP:
    cursor += hop_bins
else:
    onset_bin = int(divisor * ratio_value - offset)
    onset_bin = clamp(onset_bin, 0, 249)
    cursor += onset_bin
```

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| OS | Windows 11 |
