# Experiment 58-B — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Two-stage propose-select architecture where Stage 1 identifies audio-supported onset positions and Stage 2 uses context to select the correct one.

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1000) | Mel spectrogram, 80 bands, 1000 frames (500 past + 500 future at ~5ms/frame = 5.0s window) |
| event_offsets | (B, 128) | Past event positions relative to cursor (negative = past, int64) |
| event_mask | (B, 128) | Bool mask, True = padding (no event) |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] from chart metadata |

## Output

| Output | Shape | Description |
|---|---|---|
| onset_logits | (B, 251) | Stage 2: 250 onset bin offsets + STOP (standard prediction) |
| proposal_logits | (B, 250) | Stage 1: per-audio-token onset confidence (before sigmoid) |

## Window Configuration

| Parameter | Value | Description |
|---|---|---|
| A_BINS | 500 | Past audio context (2.5s) |
| B_BINS (B_AUDIO) | 500 | Future audio visible to model (2.5s) |
| B_PRED | 250 | Prediction range (1.25s) |
| N_CLASSES | 251 | 250 onset bins + 1 STOP |
| WINDOW | 1000 | Total mel frames (A + B_AUDIO) |
| Tokens | 250 | Conv stem output (1000 // 4) |
| Cursor token | 125 | A_BINS // 4 |

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
for 4 transformer layers (no FiLM, no events):
    x_proposer = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x_proposer)
proposal_logits = Linear(384, 1)(LayerNorm(384)(x_proposer)).squeeze(-1)
→ (B, 250) raw logits
proposal_conf = sigmoid(proposal_logits) → (B, 250) per-token onset confidence
```

Stage 1 sees ONLY audio. No events, no density conditioning, no FiLM. Each of the 250 audio tokens independently answers: "does the audio at my position support an onset?"

### Proposal Embedding
```
proposal_conf (B, 250, 1) → Linear(1, 384) → GELU → Linear(384, 384) → (B, 250, 384)
x = x + proposal_embedding
```
Audio tokens are enriched with "Stage 1 thinks onset here with X% confidence."

### Stage 2: Selector (events + density + proposals)

#### Conditioning MLP
```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
x = FiLM(cond)(x)
```

#### Event Embeddings (with gap ratios)
For each of 128 context events, compute 5 features:
- **Presence embedding**: learned parameter (1, 384)
- **Gap before**: sinusoidal encoding of distance from previous event
- **Gap after**: sinusoidal encoding of distance to next event
- **Gap ratio before**: sinusoidal encoding of ratio to previous gap, scaled by 50, clamped [0.1, 10.0]
- **Gap ratio after**: sinusoidal encoding of ratio to next gap

```
[presence (384) | gap_before (384) | gap_after (384) | ratio_before (384) | ratio_after (384)]
→ Linear(1920, 384) → GELU → Linear(384, 384)
→ event_emb: (B, 128, 384)
```
Events mapped to audio token positions: `token = (500 + offset) // 4`. Scatter-added to audio tokens at event positions.

#### Selector Transformer (8 layers)
```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```

#### Output Head
```
cursor = x[:, 125, :]  # cursor token (B, 384)
logits = Linear(384, 251)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits)  # Conv1d(1,8,k=5) → GELU → Conv1d(8,1,k=5)
→ (B, 251)
```

### FiLM Conditioning
Applied to Stage 2 only (after conv stem and after every selector transformer layer). Stage 1 has NO conditioning.
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb
Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss

### Stage 1 Loss: Focal BCE (precision-focused)
```
focal_bce = BCE_with_logits(proposal_logits, proposal_target, pos_weight=2.0)
focal_weight = (1 - p_t)^gamma, gamma=2.0
s1_loss = mean(focal_bce * focal_weight)
```

pos_weight=2.0 (vs 5.0 in exp 58). Lower weight means false negatives are penalized less relative to false positives, producing fewer but higher-quality proposals.

Proposal targets: binary vector (250 tokens) marking ALL onset positions in the full audio window — past events and ALL future onsets within B_BINS. Precomputed in dataset.

### Stage 2 Loss: OnsetLoss (standard)
```
onset_loss = 0.5 * hard_CE + 0.5 * soft_CE
```
Soft targets: trapezoid in log-ratio space (good_pct=3%, fail_pct=20%, frame_tolerance=2). STOP weight 1.5x.

### Combined Loss
```
During freeze period (first 2 evals):
    loss = s1_loss  (Stage 2 forward skipped entirely)

After freeze:
    loss = s2_loss + 0.5 * s1_loss
```

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset) |
| Balanced sampling | ON (1/count^0.5 weights) |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Proposer layers | 4 |
| Selector layers | 8 |
| Stage 2 freeze | First 2 evals |
| S1 pos_weight | **2.0** |
| S1 focal gamma | 2.0 |

## Augmentation (~14% context corruption rate)

### Context Augmentation
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | Global ±3 bins + per-event ±3 bins * 1-2x recency scale |
| Event deletion | 5% | Drop 1-2 random events |
| Event insertion | 3% | Add 1 fake event between existing events |
| Partial metronome | 2% | Replace recent half with evenly-spaced |
| Partial adv metronome | 2% | Replace oldest half with dominant-gap metronome |
| Large time shift | 2% | Shift all events by ±50 bins |
| Context truncation | 5% | Keep only most recent 8-32 events |

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
| Density jitter | 30% | ±10% on all 3 density values |

## Dataset: taiko_v2

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor - 500 to cursor + 500 (1000 frames)
3. Gather up to 128 past events as offsets from cursor
4. Forward pass: Stage 1 proposes onset positions, proposals embedded into tokens, Stage 2 predicts from cursor → 251-class softmax
5. If argmax = 250 (STOP): hop cursor forward by hop_bins
6. If argmax = offset (0-249): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Stage 1 Metrics

| Metric | Description |
|---|---|
| s1_recall | % of GT onset tokens detected (conf >= 0.5) |
| s1_precision | % of proposals that match a GT onset token |
| s1_f1 | Harmonic mean |
| s1_avg_proposals | Mean proposals per sample |
| s1_onset_conf / s1_non_onset_conf | Confidence separation |
| s2_picks_s1 | % of S2 predictions on S1 proposals |
| s2_agree_accuracy / s2_override_accuracy | S2 accuracy when agreeing/disagreeing with S1 |
| s1_naive_accuracy | Baseline: take S1's earliest proposal above threshold |
