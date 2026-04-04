# Experiment 58 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Two-stage propose-select architecture where Stage 1 identifies audio-supported onset positions and Stage 2 uses context to select the correct one.

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1000) | Mel spectrogram, 80 bands, 1000 frames (500 past + 500 future) |
| event_offsets | (B, 128) | Past event positions relative to cursor (negative = past, int64) |
| event_mask | (B, 128) | Bool mask, True = padding (no event) |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] from chart metadata |

## Output

| Output | Shape | Description |
|---|---|---|
| onset_logits | (B, 251) | Stage 2: 250 onset bin offsets + STOP (standard prediction) |
| proposal_logits | (B, 250) | Stage 1: per-audio-token onset confidence (before sigmoid) |

## Model: ProposeSelectDetector

**Total parameters: ~23.5M** (16.1M selector + 7.4M proposer)

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
x_proposer = x.clone()  # detached from selector path
for 4 transformer layers (no FiLM):
    x_proposer = TransformerEncoderLayer(x_proposer)
proposal_logits = Linear(384, 1)(LayerNorm(384)(x_proposer)).squeeze(-1)
→ (B, 250) raw logits
proposal_conf = sigmoid(proposal_logits) → (B, 250) per-token onset confidence
```

Stage 1 sees ONLY audio. No events, no density conditioning, no FiLM. It answers: "does the audio at this position support an onset?"

### Proposal Embedding
```
proposal_conf (B, 250, 1) → Linear(1, 384) → GELU → Linear(384, 384) → (B, 250, 384)
x = x + proposal_embedding  # enrich audio tokens with "Stage 1 thinks onset here"
```

### Stage 2: Selector (events + density + proposals)
```
cond = CondMLP(conditioning) → (B, 64)
x = FiLM(cond)(x)  # density conditioning

Event embeddings scatter-added to audio tokens (same as EventEmbeddingDetector)

for 8 transformer layers:
    x = TransformerEncoderLayer(x)
    x = FiLM(cond)(x)

cursor = x[:, 125, :]  # cursor token
logits = Linear(384, 251)(LayerNorm(384)(cursor)) + smooth_conv
→ (B, 251) onset prediction
```

Stage 2 sees: audio features + Stage 1 proposal confidences + event context + density. It picks from the audio-supported candidates.

### Event Embeddings (with gap ratios)
Same as EventEmbeddingDetector: 5 features (presence, gap_before, gap_after, ratio_before, ratio_after) → Linear(1920, 384) → GELU → Linear(384, 384). Scatter-added to audio tokens at event positions.

### FiLM Conditioning
Applied to Stage 2 only (after conv stem and after every selector transformer layer). Stage 1 has NO conditioning.

## Loss

### Stage 1 Loss: Focal BCE (recall-focused)
```
focal_bce = BCE_with_logits(proposal_logits, proposal_target, pos_weight=5.0)
focal_weight = (1 - p_t)^gamma, gamma=2.0
s1_loss = mean(focal_bce * focal_weight)
```

Proposal targets: binary vector marking tokens containing:
- Past events (from event_offsets, mapped to token positions)
- The target onset (the GT next onset)
- STOP samples → all zeros (no onset in window)

### Stage 2 Loss: OnsetLoss (standard)
```
s2_loss = 0.5 * hard_CE + 0.5 * soft_CE (trapezoid in log-ratio space)
```

### Combined Loss
```
During freeze period (first 2 evals):
    loss = s1_loss  # only proposer trains

After freeze:
    loss = s2_loss + 0.5 * s1_loss  # joint training
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
| Proposer layers | 4 |
| Selector layers | 8 |
| Stage 2 freeze | First 2 evals |
| Density jitter | ±10% at 30% (loose) |
| Gap ratios | ON |

## Stage 1 Metrics

| Metric | Description |
|---|---|
| s1_recall | % of GT onset tokens detected (conf >= 0.5) |
| s1_precision | % of proposals that match a GT onset token |
| s1_f1 | Harmonic mean |
| s1_avg_proposals | Mean proposals per sample |
| s1_onset_conf | Mean confidence at GT onset positions |
| s1_non_onset_conf | Mean confidence at non-onset positions |

## Dataset: taiko_v2

Standard: 10,048 charts, 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz), ~5ms/frame, 90/10 val split by song (seed 42).

## Inference

At inference, both stages run but only Stage 2's output is used:
1. Conv stem → audio tokens
2. Stage 1 → proposal confidences (embedded into tokens)
3. Stage 2 → 251-class prediction from cursor
4. Standard AR loop (argmax or sampling)

The proposal embedding enriches audio tokens but the final prediction is the standard softmax over 251 classes.
