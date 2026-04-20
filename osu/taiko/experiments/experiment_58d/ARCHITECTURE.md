# Experiment 58d — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Training data filtered to star rating 4.0-6.0 only (Oni to Inner Oni difficulty).

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
| onset_logits | (B, 251) | 250 onset bin offsets (0-249) + STOP at 250 |
| proposal_logits | (B, 250) | Stage 1: per-audio-token onset confidence (before sigmoid) |

## Window Configuration

| Parameter | Value |
|---|---|
| A_BINS | 500 |
| B_BINS | 500 |
| B_PRED | 250 |
| N_CLASSES | 251 |
| WINDOW | 1000 |
| Tokens | 250 |
| Cursor token | 125 |

## Model: ProposeSelectDetector

**Total parameters: ~23.5M**

### Shared Conv Stem
```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → x: (B, 250, 384) audio tokens
```

### Stage 1: Proposer (pure audio, no events, no density)
```
x_proposer = x.clone()
for 4 transformer layers:
    x_proposer = TransformerEncoderLayer(d=384, heads=8, ffn=1536, gelu, norm_first)(x_proposer)
proposal_logits = Linear(384, 1)(LayerNorm(384)(x_proposer)).squeeze(-1)
→ (B, 250) raw logits
proposal_conf = sigmoid(proposal_logits) → (B, 250)
```

### Proposal Embedding
```
proposal_conf (B, 250, 1) → Linear(1, 384) → GELU → Linear(384, 384)
x = x + proposal_embedding
```

### Stage 2: Selector (events + density + proposals)

#### Conditioning MLP
```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
x = FiLM(cond)(x)
```

#### Event Embeddings (with gap ratios)
For each of 128 context events, 5 features:
- Presence embedding (learned, 384)
- Gap before: sinusoidal encoding
- Gap after: sinusoidal encoding
- Gap ratio before: sinusoidal encoding of ratio × 50, clamped [0.1, 10.0]
- Gap ratio after: sinusoidal encoding

```
[presence | gap_before | gap_after | ratio_before | ratio_after]
→ Linear(1920, 384) → GELU → Linear(384, 384) → event_emb (B, 128, 384)
```
Events mapped to audio tokens via `token = (500 + offset) // 4` and scatter-added.

#### Selector Transformer (8 layers)
```
for each of 8 layers:
    x = TransformerEncoderLayer(d=384, heads=8, ffn=1536, gelu, norm_first)(x)
    x = FiLM(cond)(x)
```

#### Output Head
```
cursor = x[:, 125, :]
logits = Linear(384, 251)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)
→ (B, 251)
```

### FiLM Conditioning
```
cond (B, 64) → Linear(64, 768) → split → scale (384), shift (384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb
Standard sinusoidal positional encoding.

## Loss

### Stage 1: Focal BCE
```
focal_bce = BCE_with_logits(proposal_logits, proposal_target, pos_weight=5.0)
focal_weight = (1 - p_t)^2.0
s1_loss = mean(focal_bce * focal_weight)
```

### Stage 2: OnsetLoss
```
loss = 0.5 * hard_CE + 0.5 * soft_CE + 2.5 * |log(E[pred]/target)|
```
- Hard CE: standard cross-entropy at exact target bin
- Soft CE: trapezoid in log-ratio space (good_pct=3%, fail_pct=20%, frame_tolerance=2)
- Distance ramp: ramp_alpha=2.5, ramp_exp=1.0 (from exp44e)
- STOP weight: 1.5x

### Combined
```
During freeze (first 2 evals): loss = s1_loss
After freeze: loss = s2_loss + 0.5 * s1_loss
```

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Balanced sampling | ON (1/count^0.5) |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Proposer layers | 4 |
| Selector layers | 8 |
| Proposer freeze | 2 evals |
| S1 pos_weight | 5.0 |
| S1 focal gamma | 2.0 |
| Gap ratios | ON |
| Density jitter | ±10% at 30% |
| ramp_alpha | 2.5 |

## Data Filter

| | Full dataset | This experiment |
|---|---|---|
| Filter | None | **star_rating >= 4.0 AND < 6.0** |
| Charts | 10,048 | **2,548** |
| Samples | ~5.25M | **~2.5M** |
| Density | 0.5-14.8 | 3.0-10.5 |
| Stars | 0-13.8 | 4.0-6.0 |
| Tiers | All | Oni + lower Inner Oni |

## Augmentation (~14% context corruption rate)

### Context Augmentation
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | Global ±3 bins + per-event ±3 bins |
| Event deletion | 5% | Drop 1-2 events |
| Event insertion | 3% | Add 1 fake event |
| Partial metronome | 2% | Replace recent half with evenly-spaced |
| Partial adv metronome | 2% | Replace oldest half with dominant-gap metronome |
| Large time shift | 2% | Shift all events by ±50 bins |
| Context truncation | 5% | Keep only 8-32 most recent |

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

## Dataset: taiko_v2 (filtered)

- Source: 10,048 charts from osu!taiko
- Filter: 4.0 ≤ star_rating < 6.0
- Result: 2,548 charts
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887)
- Train/val split: 90/10 by song (seed 42), filter applied after split
- MIN_CURSOR_BIN = 6000

## Evaluation

Per-difficulty metrics computed on val set:
- **Low (stars < 4)**: out-of-distribution easy charts
- **Mid (stars 4-6)**: in-distribution
- **High (stars ≥ 6)**: out-of-distribution hard charts
- **Total**: all charts

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 4060 (8 GB) |
| OS | CachyOS (Linux) |
