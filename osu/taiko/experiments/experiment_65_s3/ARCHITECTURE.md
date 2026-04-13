# Experiment 65-S3 — Full Architecture Specification

## Task

Per-bin onset detection fusing audio (S1) and context (S2v2) signals with cross-bin coordination. Each bin's decision depends on all other bins' signals via self-attention.

## References

- Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020) — DETR: encoder-decoder with cross-attention, learned queries, auxiliary losses per decoder layer
- Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024) — RT-DETR: content-based query initialization beats learned embeddings
- Ye et al., "Sound Event Detection Transformer" (2021) — SEDT: 1D-DETR for audio, 3+3 layers, audio query for clip-level supervision
- Li et al., "Unified Audio Event Detection" (2024) — T-UAED: per-class queries, dot-product detection head, D=192
- Ahn, "Beat Tracking as Object Detection" (2025) — BeatFCOS: focal loss for sparse onset detection

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1000) | Mel spectrogram (for S1 conv stem) |
| s1_conf | (B, 250) | S1 per-bin onset confidence (sigmoid, frozen) |
| s2_conf | (B, 250) | S2v2 per-bin onset confidence (sigmoid, frozen) |
| event_offsets | (B, 128) | Past event positions relative to cursor |
| event_mask | (B, 128) | True = padding |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] |

S1 and S2v2 are frozen pretrained models. S3 receives their confidence outputs.

## Output

| Output | Shape | Description |
|---|---|---|
| bin_logits | (B, 250) | Per-bin onset logits. Sigmoid → probability. |
| aux_logits | (B, 3, 250) | Auxiliary logits from each decoder layer |

## Model: FusionSelector

**Total S3 parameters: ~3.7M** (excluding frozen S1 29.6M + S2v2 13M)

### 1. Audio Feature Extraction (from S1, frozen)

```
mel (B, 80, 1000)
  → S1.conv_stem (frozen) → (B, 250, 384)
  → Linear(384, 192) → (B, 250, 192)  # project to S3's D
  → audio_features
```

Reuse S1's conv stem as a feature extractor. Project from S1's d_model (384) to S3's D (192).

### 2. S1/S2v2 Confidence Embedding

```
s1_conf (B, 250, 1) → Linear(1, 192) → GELU → Linear(192, 192) → s1_emb (B, 250, 192)
s2_conf (B, 250, 1) → Linear(1, 192) → GELU → Linear(192, 192) → s2_emb (B, 250, 192)
```

Each scalar confidence is embedded into a D-dimensional vector. Separate MLPs for S1 and S2v2.

### 3. Token Construction

```
tokens = audio_features + s1_emb + s2_emb + sinusoidal_pos
→ (B, 250, 192)
```

All signals combined additively per token position.

### 4. Event Embeddings (scatter-add)

For each of 128 past events:
```
  presence_emb (learned, 192)
  gap_before_emb = SinusoidalPosEmb(gap_before)
  gap_after_emb = SinusoidalPosEmb(gap_after)

  event_emb = Linear(3 * 192, 192)(concat) → GELU → Linear(192, 192)
```

Events mapped to token positions via `token = (A_BINS + offset) // 4` and scatter-added.

### 5. Density Conditioning (FiLM)

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

Applied via FiLM after token construction and after each encoder layer.

```
cond → Linear(64, 2 * 192) → split → scale, shift
x = x * (1 + scale) + shift
```

### 6. Encoder (4 layers)

```
for each of 4 layers:
    x = x + MHSA(LayerNorm(x))       # self-attention, 8 heads
    x = FiLM(x, cond)
    x = x + FFN(LayerNorm(x))         # FFN: Linear(192, 768) → GELU → Dropout → Linear(768, 192) → Dropout
    x = x + sinusoidal_pos             # re-add position per layer (DETR finding)

→ Fenc (B, 250, 192)
```

Pre-norm transformer encoder with FiLM density conditioning and positional re-injection.

### 7. Decoder (3 layers)

**Query initialization** (content-based, per RT-DETR):
```
bin_queries = Fenc[:, cursor_token:cursor_token+B_PRED, :]  # (B, 250, 192) — slice prediction range
bin_queries = bin_queries + learned_bin_pos_emb              # (250, 192) learned per-bin
```

Each query starts as the encoder's representation at that bin position, enriched with a learned positional embedding.

```
for each of 3 decoder layers:
    # Self-attention among bin queries (cross-bin coordination)
    q = q + MHSA(LayerNorm(q))

    # Cross-attention: queries attend to full encoder output
    q = q + CrossAttention(
        query=LayerNorm(q),
        key=Fenc + sinusoidal_pos,
        value=Fenc
    )

    q = q + FFN(LayerNorm(q))

    # Auxiliary detection head (per DETR — critical for convergence)
    aux_logits_l = detection_head(q)    # shared head

→ Fdec (B, 250, 192)
```

### 8. Detection Head (shared across decoder layers)

```
Fdec (B, 250, 192) → LayerNorm(192) → Linear(192, 1) → squeeze → (B, 250)
```

Same head applied at every decoder layer for auxiliary losses.

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Re-added at every encoder layer and as key bias in cross-attention (per DETR finding that this outperforms input-only position).

## Loss

### Focal BCE (per BeatFCOS)

```
bce = BCE_with_logits(logits, targets, pos_weight=5.0)
p_t = sigmoid(logits) * targets + (1 - sigmoid(logits)) * (1 - targets)
focal_weight = (1 - p_t) ^ gamma     # gamma=2.0
loss = (bce * focal_weight).mean()
```

### Auxiliary Losses (per DETR)

```
total_loss = 0
for l in range(n_decoder_layers):
    total_loss += focal_bce(aux_logits[l], targets)
total_loss /= n_decoder_layers
```

Shared detection head + shared LayerNorm across layers (per DETR). This forces every decoder layer to be independently predictive and dramatically improves convergence.

### Target Construction

Binary vector of length B_PRED (250):
- 1.0 at onset bins
- 0.5 at ±1 adjacent bins (soft label)
- 0.0 elsewhere

Same targets as S1 and S2v2.

## Training

| Param | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (lr=1e-4, wd=1e-4) | DETR recommendation for decoder training |
| Batch size | 48 | Needs mel for S1 feature extraction |
| Epochs | 50 | |
| Scheduler | CosineAnnealingLR | |
| Gradient clipping | 1.0 | |
| Dropout | 0.1 | DETR default |
| Init | Xavier for all transformer params | DETR recommendation |
| S1 | Frozen (29.6M) | Pretrained proposer |
| S2v2 | Frozen (13M) | Pretrained proposer |
| S3 trainable | ~3.7M | Fusion selector |
| Evals per epoch | 4 | |
| pos_weight | 5.0 | High recall bias |
| focal_gamma | 2.0 | Down-weight easy negatives |

## Augmentation

### Audio (applied to mel before S1)
| Aug | Rate | Params |
|---|---|---|
| Mel gain | 30% | ±2dB |
| Mel noise | 15% | Gaussian σ≤0.3 |
| Freq jitter | 15% | Roll mel bands ±3 |
| SpecAugment freq | 20% | 1 mask, 10 bands |
| SpecAugment time | 20% | 1 mask, 30 frames |

### Context (applied to gap sequences before S2v2)
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | ±1 bin |
| Context truncation | 2% | Keep 32-128 |
| Density jitter | 20% | ±5% |

## Dataset: taiko_v2

Same dataset, same train/val split (90/10 by song, seed 42).

## Metrics

### Primary
- Per-bin F1 at thresholds 0.3-0.7
- Precision, Recall at best threshold
- Confidence separation

### Comparison targets
| Model | F1 |
|---|---|
| S1 alone | 0.712 |
| S2v2 alone | 0.727 |
| Average S1+S2v2 | 0.752 |
| **S3 target** | **>0.752** |

### Per-decoder-layer F1
Track F1 improvement across decoder layers 1→2→3.

### Disagree analysis
What fraction of S1/S2v2 disagreement bins does S3 resolve correctly?

### Both-miss recovery
Of the 16.4% both-miss onset bins, does S3 recover any via cross-signal fusion of moderate confidences?

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| OS | Windows 11 |
