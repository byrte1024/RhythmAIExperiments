# Experiment 65-S2v2 — Full Architecture Specification

## Task

Per-bin onset detection from rhythm pattern context alone. No audio input. For each of 250 prediction bins, output a confidence value indicating whether an onset exists at that position based on the preceding gap/ratio sequence.

## Input

| Input | Shape | Description |
|---|---|---|
| gap_sequence | (B, 128) | Past inter-onset gaps in bins. Padded with 0. |
| ratio_sequence | (B, 128) | Gap ratios: gap[i] / gap[i-1], clamped [0.1, 10.0]. Padded with 0. |
| event_mask | (B, 128) | Bool mask, True = padding |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] |

## Output

| Output | Shape | Description |
|---|---|---|
| bin_logits | (B, 250) | Per-bin onset logits (before sigmoid). Matches S1's output format. |

## Model: ContextProposer

**Total parameters: 13,043,329**

### 1. Event Token Encoding

```
For event i:
  feat_log_gap   = SinusoidalPosEmb(log(gap_i + 1))       # (d_model,)
  feat_log_ratio = SinusoidalPosEmb(log(ratio_i) * 50)    # (d_model,)
  feat_raw_gap   = SinusoidalPosEmb(gap_i)                 # (d_model,)

  token_i = Linear(3 * d_model, d_model)(concat) → GELU
  → (d_model,)
```

Padded events (mask=True) produce zero tokens.

### 2. Density Conditioning (FiLM)

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

Applied via FiLM to event tokens after encoding.

### 3. History Encoder: Bidirectional GRU

```
event_tokens (B, 128, d_model)
→ FiLM(cond)
→ Bidirectional GRU(d_model, 4 layers, dropout=0.1)
→ output (B, 128, 2 * d_model)
→ gather last valid hidden → Linear(2 * d_model, d_model) → GELU
→ context (B, d_model)
```

### 4. Per-Bin Expansion Head

```
context (B, d_model)
→ Linear(d_model, d_model * 2) → GELU
→ Linear(d_model * 2, b_pred * d_model // 4) → GELU
→ reshape → (B, b_pred, d_model // 4)
→ LayerNorm(d_model // 4) → Linear(d_model // 4, 1) → squeeze
→ bin_logits (B, 250)
```

The expansion head maps the single context vector to 250 independent bin features, each processed by a shared output projection. This is the key difference from S2: instead of competing classes (softmax), each bin makes an independent onset/no-onset decision (sigmoid).

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: scalar values. Output: (d_model,) vector.

### FiLM Conditioning

```
cond (B, 64) → Linear(64, 2 * d_model) → split → scale, shift
x = x * (1 + scale) + shift
```

## Loss: Focal BCE

```
bce = BCE_with_logits(logits, targets, pos_weight=5.0)
p_t = sigmoid(logits) * targets + (1 - sigmoid(logits)) * (1 - targets)
focal_weight = (1 - p_t) ^ gamma     # gamma=2.0
loss = mean(bce * focal_weight)
```

Same loss as S1. pos_weight=5.0 biases toward recall, focal_gamma=2.0 down-weights easy negatives.

### Target Construction

Binary vector of length B_PRED (250):
- 1.0 at bins where a real onset exists
- 0.5 at ±1 adjacent bins (soft label for annotation tolerance)
- 0.0 elsewhere

Same targets as S1 — both models learn the same detection task, just from different inputs.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 (no mel loading) |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Workers | 4 |
| pos_weight | 5.0 |
| focal_gamma | 2.0 |

## Augmentation

Same minimal augmentation as S2:
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | ±1 bin |
| Context truncation | 2% | Keep 32-128 most recent |
| Density jitter | 20% | ±5% |

## Dataset: taiko_v2

Same dataset, same train/val split (90/10 by song, seed 42), same samples as S1 and S2.

## Metrics

Same as S1:
- Precision, Recall, F1 at thresholds 0.3-0.7
- Average proposals per sample
- Onset/non-onset confidence means
- Confidence separation

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| OS | Windows 11 |
