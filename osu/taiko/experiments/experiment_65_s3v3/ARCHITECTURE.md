# Experiment 65-S3v3 — Full Architecture Specification

## Task

Single-onset prediction from S1+S2v2 confidence maps ONLY. No audio features, no event embeddings, no density conditioning. Tests whether the proposal signals alone are sufficient for onset prediction.

## Input

| Input | Shape | Description |
|---|---|---|
| s1_conf | (B, 250) | S1 per-bin onset confidence (sigmoid, frozen) |
| s2_conf | (B, 250) | S2v2 per-bin onset confidence (sigmoid, frozen) |

**NOT used:** mel spectrogram, event_offsets, event_mask, conditioning.

S1 (29.6M) and S2v2 (13M) run frozen to produce the confidence maps from mel + context. S3v3 only sees the resulting confidences.

## Output

| Output | Shape | Description |
|---|---|---|
| logits | (B, 251) | 250 onset bin offsets + STOP. Single onset via argmax. |

## Model: PureProposalFusion

**Estimated parameters: ~1-2M**

### 1. Per-Bin Token Construction

Each of 250 bins becomes a token with 2 features:

```
s1_conf (B, 250) + s2_conf (B, 250)
→ stack → (B, 250, 2)
→ Linear(2, d_model) → GELU → Linear(d_model, d_model)
→ + SinusoidalPosEmb(positions 0..249)
→ (B, 250, d_model) bin tokens
```

### 2. Transformer (4 layers)

```
for each of 4 layers:
    x = TransformerEncoderLayer(
        d_model=128, nhead=4, dim_feedforward=512,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
```

Self-attention over all 250 bins. Each bin can see every other bin's S1/S2v2 confidences. This enables cross-bin reasoning: "S1 peaks at bin 30, S2v2 peaks at bin 33, so the onset is around 31."

### 3. Output Head

```
cursor = x[:, 0, :]  # bin 0 = cursor position
→ LayerNorm(d_model) → Linear(d_model, d_model) → GELU → Linear(d_model, 251)
→ (B, 251) logits
```

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Tells the model which bin position each token represents.

## Hyperparameters

| Param | Value |
|---|---|
| d_model | 128 |
| Layers | 4 |
| Heads | 4 |
| FFN expansion | 4x (512) |
| Dropout | 0.1 |
| N_CLASSES | 251 |
| B_PRED | 250 |

## Loss

```
loss = OnsetLoss(hard_alpha=0.5, soft CE trapezoid, frame_tolerance=2, stop_weight=1.5, ramp_alpha=2.5)
```

Same as all other single-onset models.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 (tiny model, no mel in S3v3 itself) |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Balanced sampling | ON |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| S1 | Frozen (29.6M, runs on mel to produce s1_conf) |
| S2v2 | Frozen (13M, runs on context to produce s2_conf) |
| S3v3 trainable | ~1-2M |

### Proposal Augmentation (training only)

- 5% gaussian noise (σ=0.05) on S1/S2v2 confidences
- 5% blackout: 33% zero S1, 33% zero S2v2, 33% zero both

Note: S1/S2v2 still run on augmented inputs (mel aug + context jitter flow through), so their outputs vary per epoch.

## Dataset: taiko_v2

Same dataset, same split. Mel is loaded for S1 forward pass, events for S2v2 forward pass. S3v3 itself only sees the resulting 250-dim confidence vectors.

## Benchmarks

Standard benchmarks plus fusion-specific:
- **no_s1**: S1 confidence zeroed
- **no_s2**: S2v2 confidence zeroed
- **no_s1s2**: both zeroed
- **random_s1**: random S1 confidence
- **random_s2**: random S2v2 confidence

These directly measure dependence on each signal. Unlike S3v2 where the model could bypass proposals via audio features, S3v3 has no alternative — if no_s1 hurts, it's genuinely using S1.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4060 (8 GB) |
| OS | CachyOS (Linux) |
