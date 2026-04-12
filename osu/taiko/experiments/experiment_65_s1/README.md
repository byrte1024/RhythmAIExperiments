# Experiment 65-S1 — Dedicated Audio Onset Proposer

## Purpose

Build a standalone S1 model for the 3-stage architecture. Improves on the current proposer (4 transformer layers inside ProposeSelectDetector) with:
1. Conformer blocks (conv inside transformer) for local transient detection
2. 8 layers instead of 4
3. Per-bin output (250 bins) instead of per-token (250 tokens covering 4 bins each)

The output matches S2's 250-bin space exactly, enabling direct comparison and fusion.

## Architecture: ConformerProposer

### Design Principle

Tokens for thinking, bins for output:
- **Self-attention operates on tokens** (250 tokens, efficient, meaningful receptive fields)
- **Conformer conv captures local transients** (onset edges within each token's neighborhood)
- **Upsample head recovers per-bin resolution** (250 bins, matches S2 output space)

### Pipeline

```
mel (B, 80, A+B)
  → Conv Stem (4x downsample) → tokens (B, 250, d_model)
  → 8 Conformer Blocks → enriched tokens (B, 250, d_model)
  → Upsample Head (4x) → per-bin features (B, 1000, d_model)
  → slice to B_PRED → Linear → sigmoid
  → per-bin confidence (B, 250)
```

### Conv Stem
```
mel (B, 80, 1000)
  → Conv1d(80, d_model//2, k=7, stride=2, pad=3) → GELU → GroupNorm
  → Conv1d(d_model//2, d_model, k=7, stride=2, pad=3) → GELU
  → LayerNorm → + SinusoidalPosEmb
  → (B, 250, d_model)
```
Same as current. 4x downsample keeps attention at 250 tokens.

### Conformer Block (×8)

Each block follows the Conformer architecture (Gulati et al., 2020):
```
x → FFN(half-step) → MHSA → DepthwiseConv → FFN(half-step) → LayerNorm → x + residual
```

Specifically:
```
# Half-step FFN
x = x + 0.5 * FFN(LayerNorm(x))

# Multi-head self-attention
x = x + MHSA(LayerNorm(x))

# Convolution module
x = x + ConvModule(LayerNorm(x))
  where ConvModule = Pointwise(d→2d) → GLU → DepthwiseConv1d(k=31, pad=15) → BatchNorm → Swish → Pointwise(d→d) → Dropout

# Half-step FFN
x = x + 0.5 * FFN(LayerNorm(x))

# Final LayerNorm
x = LayerNorm(x)
```

The depthwise conv (kernel=31) at token level covers 31×4 = 124 mel frames ≈ 620ms. This captures:
- Onset transient edges (sharp energy changes)
- Local rhythmic patterns (note clusters)
- Attack/decay envelopes

### Upsample Head

Recover per-bin resolution from tokens:
```
tokens (B, 250, d_model)
  → ConvTranspose1d(d_model, d_model, k=4, stride=4) → GELU
  → (B, 1000, d_model)
  → slice [:, cursor_token*4 : cursor_token*4 + B_PRED]  # extract B_PRED bins
  → LayerNorm → Linear(d_model, 1)
  → sigmoid
  → (B, 250) per-bin onset confidence
```

Each output bin now maps to exactly one mel frame position (within B_PRED), vs current S1 where each token covers 4 bins.

### Parameters

| Param | Value | Rationale |
|---|---|---|
| d_model | 384 | Same as current |
| n_layers | 8 | Up from 4, SOTA range for audio |
| n_heads | 8 | Same as current |
| conv_kernel | 31 | Standard conformer, ~620ms at token level |
| FFN expansion | 4x (1536) | Standard |
| Dropout | 0.1 | Standard |
| A_BINS | 500 | Same |
| B_BINS | 500 | Same |
| B_PRED | 250 | Same, but now output is per-bin not per-token |
| Output | (B, 250) sigmoid | Per-bin confidence, matches S2 space |

**Actual params: 29,568,577**

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Loss | Focal BCE (same as current S1) |
| pos_weight | 5.0 |
| focal_gamma | 2.0 |
| Targets | Binary: 1 at bins where onsets exist in B_PRED window |
| Balanced sampling | OFF (binary targets, use pos_weight instead) |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |

### Target Construction

For each sample, build a binary vector of length B_PRED (250):
- 1 at bins where a real onset exists (within ±1 bin tolerance for soft targets)
- 0 elsewhere

This is different from current S1 which uses per-token targets. Per-bin targets give finer supervision signal.

### No context, no density

S1 sees ONLY the mel spectrogram. No event offsets, no density conditioning, no FiLM. Pure audio → onset confidence.

## Dataset

Same taiko_v2, same train/val split. Only loads mel spectrograms (no event augmentation needed for S1).

Audio augmentation (same as current):
| Aug | Rate | Params |
|---|---|---|
| Mel gain | 30% | ±2dB |
| Mel noise | 15% | Gaussian σ≤0.3 |
| Freq jitter | 15% | Roll mel bands ±3 |
| SpecAugment freq | 20% | 1 mask, 10 bands |
| SpecAugment time | 20% | 1 mask, 30 frames |

## Metrics

### Per-bin detection metrics
- Precision, Recall, F1 at various thresholds (0.3, 0.4, 0.5, 0.6, 0.7)
- Average number of proposals (bins above threshold)
- Onset confidence (mean conf at actual onset bins)
- Non-onset confidence (mean conf at non-onset bins)
- Confidence separation

### For overlap analysis
- Per-bin confidence vector saved for direct comparison with S2
- Same 250-bin output space as S2 → element-wise overlap trivial

## Comparison Targets

| Model | Type | Output | Params |
|---|---|---|---|
| Current S1 (inside exp58) | 4 transformer layers | per-token (250) | ~4M |
| **New S1 (this experiment)** | 8 conformer layers + upsample | per-bin (250) | ~12M |
| S2 (exp 65-S2) | 4 GRU layers | per-bin (251 logits) | 4.9M |

## References

- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (Interspeech 2020) — Conformer block design
- Thapa & Lee, "Dual-Path Beat Tracking" (Applied Sciences 2024) — TCN + Transformer parallel for beat detection
- Zaman et al., "Transformers and Audio Detection Tasks" (DSP 2025) — Survey showing Conformer outperforms pure transformer for audio

## Result

*Pending*

## Lesson

*Pending*
