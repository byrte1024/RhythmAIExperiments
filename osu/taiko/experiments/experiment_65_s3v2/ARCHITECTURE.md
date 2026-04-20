# Experiment 65-S3v2 — Full Architecture Specification

## Task

Single-onset prediction fusing S1 (audio) and S2v2 (context) confidence signals. Same paradigm as exp58 (251-class classification, argmax), but with S2v2 context signal added.

## References

- Exp58 ProposeSelectDetector: original 2-stage propose-select architecture
- Exp65-S1: ConformerProposer providing audio features + S1 confidence
- Exp65-S2v2: ContextProposer providing S2v2 per-bin confidence

## Input

| Input | Shape | Description |
|---|---|---|
| audio_features | (B, 250, 384) | From S1 conv stem (frozen, projected) |
| s1_conf | (B, 250) | S1 per-bin onset confidence (sigmoid, frozen) |
| s2_conf | (B, 250) | S2v2 per-bin onset confidence (sigmoid, frozen) |
| event_offsets | (B, 128) | Past event positions relative to cursor |
| event_mask | (B, 128) | True = padding |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] |

S1 (29.6M) and S2v2 (13M) are frozen. S3v2 (16.1M) is trainable.

## Output

| Output | Shape | Description |
|---|---|---|
| logits | (B, 251) | 250 onset bin offsets + STOP. Single onset via argmax. |

## Model: FusionClassifier

**Total parameters: 16,077,844**

### 1. Audio Feature Projection

```
audio_features (B, 250, 384) → Linear(384, 384) → (B, 250, 384)
```

### 2. S1/S2v2 Confidence Embedding

```
s1_conf (B, 250, 1) → Linear(1, 384) → GELU → Linear(384, 384) → s1_emb (B, 250, 384)
s2_conf (B, 250, 1) → Linear(1, 384) → GELU → Linear(384, 384) → s2_emb (B, 250, 384)
```

Both added to audio tokens in the B_PRED range (cursor to cursor + 250).

### 3. Token Construction

```
tokens = audio_proj + s1_emb + s2_emb + sinusoidal_pos
→ FiLM(cond)
→ + event_embeddings (scatter-add)
→ (B, 250, 384)
```

### 4. Event Embeddings (with gap ratios)

For each of 128 context events, 5 features:
- Presence embedding (learned, 384)
- Gap before: sinusoidal encoding
- Gap after: sinusoidal encoding
- Gap ratio before: sinusoidal encoding
- Gap ratio after: sinusoidal encoding

```
[presence | gap_before | gap_after | ratio_before | ratio_after]
→ Linear(1920, 384) → GELU → Linear(384, 384)
→ scatter-add to audio tokens
```

### 5. Transformer (8 layers with FiLM)

```
for each of 8 layers:
    x = TransformerEncoderLayer(d=384, heads=8, ffn=1536, gelu, norm_first)
    x = FiLM(cond)(x)
```

### 6. Output Head

```
cursor = x[:, 125, :]  # cursor token
logits = Linear(384, 251)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)
→ (B, 251)
```

### FiLM Conditioning

```
cond (B, 64) → Linear(64, 768) → split → scale (384), shift (384)
x = x * (1 + scale) + shift
```

### Proposal Augmentation (training only)

- 5% gaussian noise (σ=0.05) on S1/S2v2 confidences
- 5% blackout: 33% zero S1, 33% zero S2v2, 33% zero both

## Loss

```
loss = OnsetLoss(hard_alpha=0.5, soft CE trapezoid, frame_tolerance=2, stop_weight=1.5, ramp_alpha=2.5)
```

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| B_PRED | 250, N_CLASSES=251 |
| S1 | Frozen (29.6M) |
| S2v2 | Frozen (13M) |
| S3v2 trainable | 16.1M |
| Balanced sampling | ON |
| Gradient clipping | 1.0 |

## Dataset: taiko_v2

Same dataset, same split. S1/S2v2 run in real-time per sample with augmented inputs.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| OS | Windows 11 |
