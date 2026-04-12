# Experiment 65-S1 — Full Architecture Specification

## Task

Per-bin onset detection from audio alone. No events, no context, no density conditioning. Output a confidence value for each of 250 prediction bins indicating whether an onset exists at that position.

## References

- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (Interspeech 2020) — Conformer block design: FFN-MHSA-Conv-FFN with half-step residuals
- Zaman et al., "Transformers and Audio Detection Tasks: An Overview" (DSP 2025) — Conformer outperforms pure transformer for audio sequence tasks
- Thapa & Lee, "Dual-Path Beat Tracking" (Applied Sciences 2024) — local conv + global attention complementary for rhythmic features
- Zehren et al., "ADTOF: A Large Dataset for Automatic Drum Transcription" (ISMIR 2021) — CRNN baseline for onset detection from rhythm game data

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1000) | Mel spectrogram, 80 bands, 1000 frames (500 past + 500 future at ~5ms/frame = 5.0s window) |

No event offsets. No event mask. No conditioning. Pure audio.

## Output

| Output | Shape | Description |
|---|---|---|
| bin_logits | (B, 250) | Per-bin onset logits (before sigmoid). One value per prediction bin in B_PRED range. |

After sigmoid: per-bin onset confidence in [0, 1].

## Window Configuration

| Parameter | Value | Description |
|---|---|---|
| A_BINS | 500 | Past audio context (2.5s) |
| B_BINS | 500 | Future audio context (2.5s) |
| B_PRED | 250 | Prediction range (1.25s) — output bins |
| WINDOW | 1000 | Total mel frames (A + B) |
| Tokens | 250 | Conv stem output (1000 // 4) |
| Cursor token | 125 | A_BINS // 4 |

## Model: ConformerProposer

**Total parameters: 29,568,577**

### 1. Conv Stem (4x downsample)

```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → (B, 250, 384) audio tokens
```

4x downsample: 1000 mel frames → 250 tokens. Keeps attention efficient.

### 2. Conformer Blocks (×8)

Each block follows Gulati et al. (2020):

```
# Half-step FFN 1
x = x + 0.5 * FFN(LayerNorm(x))
  where FFN = Linear(384, 1536) → SiLU → Dropout → Linear(1536, 384) → Dropout

# Multi-Head Self-Attention
x = x + Dropout(MHSA(LayerNorm(x)))
  where MHSA = MultiheadAttention(384, 8 heads, dropout=0.1)

# Convolution Module
x = x + ConvModule(LayerNorm(x))
  where ConvModule =
    Pointwise Linear(384, 768)
    → GLU (split in half, gate)
    → DepthwiseConv1d(384, 384, kernel=31, padding=15, groups=384)
    → BatchNorm1d(384)
    → SiLU (Swish)
    → Pointwise Linear(384, 384)
    → Dropout

# Half-step FFN 2
x = x + 0.5 * FFN(LayerNorm(x))

# Final LayerNorm
x = LayerNorm(x)
```

The depthwise conv (kernel=31) at token level covers 31 × 4 = 124 mel frames = ~620ms. This captures onset transient edges, local rhythmic patterns, and attack/decay envelopes that pure self-attention misses.

8 blocks × ~3.4M params each = 27.3M params for the conformer stack.

### 3. Upsample Head (4x upsample → per-bin output)

```
tokens (B, 250, 384)
  → transpose → (B, 384, 250)
  → ConvTranspose1d(384, 384, kernel=4, stride=4) → GELU
  → (B, 384, 1000)
  → Conv1d(384, 384, kernel=7, padding=3) → GELU    # refine / smooth artifacts
  → transpose → (B, 1000, 384)
  → slice [cursor_frame : cursor_frame + b_pred]     # extract B_PRED range
  → (B, 250, 384)
  → LayerNorm(384) → Linear(384, 1)
  → squeeze → (B, 250) per-bin logits
```

The ConvTranspose1d recovers mel-frame resolution from tokens. The refine Conv1d smooths transpose convolution checkerboard artifacts. Only the B_PRED slice (250 bins starting at cursor) is kept.

## Loss: Focal BCE

```
bce = BCE_with_logits(logits, targets, pos_weight=5.0)
p_t = sigmoid(logits) * targets + (1 - sigmoid(logits)) * (1 - targets)
focal_weight = (1 - p_t) ^ gamma     # gamma=2.0
loss = mean(bce * focal_weight)
```

pos_weight=5.0 biases toward recall (finding all onsets). focal_gamma=2.0 down-weights easy negatives.

### Target Construction

For each sample, binary vector of length B_PRED (250):
- 1.0 at bins where a real onset exists
- 0.5 at ±1 adjacent bins (soft label for annotation tolerance)
- 0.0 elsewhere

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset) |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Workers | 3 |
| pos_weight | 5.0 |
| focal_gamma | 2.0 |
| Train samples | 5,249,035 |
| Val samples | 594,268 |

## Augmentation (audio only)

| Aug | Rate | Params |
|---|---|---|
| Mel gain | 30% | ±2dB |
| Mel noise | 15% | Gaussian σ≤0.3 |
| Freq jitter | 15% | Roll mel bands ±3 |
| SpecAugment freq | 20% | 1 mask, 10 bands |
| SpecAugment time | 20% | 1 mask, 30 frames |

No context augmentation — there is no context input.

## Dataset: taiko_v2

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song)

## Metrics

### Per-threshold (0.3, 0.4, 0.5, 0.6, 0.7)
- Precision, Recall, F1
- Average proposals per sample (bins above threshold)

### Confidence analysis
- Onset confidence mean (mean sigmoid at actual onset bins)
- Non-onset confidence mean
- Confidence separation (onset_conf - non_onset_conf)

### Eval graphs
- `_conf_dist.png` — confidence histogram: onset bins vs non-onset bins
- `_pr_curve.png` — precision/recall/F1 vs threshold sweep
- `_bin_profile.png` — mean confidence and mean target density by bin position

## Comparison Targets

| Model | Type | Output | Params |
|---|---|---|---|
| Current S1 (inside exp58) | 4 transformer layers | per-token (250 tokens) | ~4M |
| **This S1** | 8 conformer layers + upsample | per-bin (250 bins) | 29.6M |
| S2 (exp 65-S2) | 4 GRU layers | per-class (251 logits) | 4.9M |

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
