# Experiment 64 — Full Architecture Specification

## Task

Predict the next N onset timings simultaneously in an osu!taiko rhythm game chart, using **delta encoding**: each onset predicts the gap to the next event rather than an absolute offset from the cursor.

## Input

| Input | Shape | Description |
|---|---|---|
| mel | (B, 80, 1750) | Mel spectrogram, 80 bands, 1750 frames (500 past + 1250 future at ~5ms/frame = 8.7s window) |
| event_offsets | (B, 128) | Past event positions relative to cursor (negative = past, int64) |
| event_mask | (B, 128) | Bool mask, True = padding (no event) |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] from chart metadata |

## Output

| Output | Shape | Description |
|---|---|---|
| onset_logits | (B, 4, 251) | 4 onset predictions, each 251-class (delta bins 0-249 + STOP at 250) |
| proposal_logits | (B, 438) | Stage 1: per-audio-token onset confidence (before sigmoid) |

### Delta Encoding

Each onset predicts the **gap from the previous onset** (or cursor for o1):
- o1: distance from cursor to first event
- o2: distance from o1's position to second event
- o3: distance from o2's position to third event
- o4: distance from o3's position to fourth event

A simple metronome at gap=20 becomes targets [20, 20, 20, 20] instead of [20, 40, 60, 80].

### STOP Cascade

If delta[i] >= B_PRED (250), that onset and all subsequent become STOP:

```
cursor, [onsets]               → deltas                → targets
0, [50, 100, 150, 200]        → [50, 50, 50, 50]      → [50, 50, 50, 50]
0, [1, 2, 3, 4]               → [1, 1, 1, 1]          → [1, 1, 1, 1]
0, [100, 1000, 1200, 1400]    → [100, 900, 200, 200]   → [100, STOP, STOP, STOP]
1300, [1500, 1700, 1750, 1800] → [200, 200, 50, 50]    → [200, 200, 50, 50]
500, [600, 700, 1700, 1833]   → [100, 100, 1000, 133]  → [100, 100, STOP, STOP]
100, [500, 600, 700, 800]     → [400, 100, 100, 100]   → [STOP, STOP, STOP, STOP]
```

## Window Configuration

| Parameter | Value | Description |
|---|---|---|
| A_BINS | 500 | Past audio context (2.5s) |
| B_BINS | 1252 | Future audio visible to model (6.25s), auto: (4+1) × 250, rounded for conv alignment |
| B_PRED | 250 | Prediction range per onset (1.25s) |
| N_CLASSES | 251 | 250 delta bins + 1 STOP |
| N_ONSETS | 4 | Simultaneous onset predictions |
| WINDOW | 1752 | Total mel frames (A + B) |
| Tokens | 438 | Conv stem output (1752 // 4) |
| Cursor token | 125 | A_BINS // 4 |

### Why B_BINS = (n_onsets + 1) × B_PRED

With deltas, worst case: all 4 onsets predict delta=249. Total offset from cursor = 4 × 249 = 996. The model needs audio coverage for all possible predictions. B_BINS = 5 × 250 = 1250 guarantees full coverage, with one B_PRED buffer past the last possible onset for STOP decisions.

## Model: ProposeSelectDetector (delta multi-onset)

**Total parameters: ~24M** (slightly more than exp 62 due to larger audio token count)

### Shared Conv Stem
```
mel (B, 80, 1752)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 438, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..436)
  → x: (B, 438, 384) audio tokens
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
→ (B, 438) raw logits — covers the full audio window
proposal_conf = sigmoid(proposal_logits) → (B, 438) per-token onset confidence
```

### Proposal Embedding
```
proposal_conf (B, 438, 1) → Linear(1, 384) → GELU → Linear(384, 384) → (B, 438, 384)
x = x + proposal_embedding
```

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
- **Gap after**: sinusoidal encoding of distance to next event (last event uses gap_before as proxy)
- **Gap ratio before**: sinusoidal encoding of `gap_before[i-1] / gap_before[i]` scaled by 50, clamped [0.1, 10.0]
- **Gap ratio after**: sinusoidal encoding of `gap_after[i+1] / gap_after[i]`, same scaling

```
[presence (384) | gap_before (384) | gap_after (384) | ratio_before (384) | ratio_after (384)]
→ Linear(1920, 384) → GELU → Linear(384, 384)
→ event_emb: (B, 128, 384)
```
Events mapped to audio token positions via `token = (500 + offset) // 4` and scatter-added.

#### Selector Transformer (8 layers)
```
for each of 8 layers:
    x = TransformerEncoderLayer(
        d_model=384, nhead=8, dim_feedforward=1536,
        dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )(x)
    x = FiLM(cond)(x)
```

#### Multi-Onset Output Head
```
cursor = x[:, 125, :]  # cursor token (B, 384)
raw = Linear(384, 251 * 4)(LayerNorm(384)(cursor))  # (B, 1004)
logits = raw.view(B, 4, 251)  # (B, 4, 251)
```

No smooth conv for multi-onset mode.

### FiLM Conditioning
Applied to Stage 2 only (after conv stem and after every selector transformer layer). Stage 1 has NO conditioning.
```
cond (B, 64) → Linear(64, 2*384) → split → scale (B, 384), shift (B, 384)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb
Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, 384).

## Loss

### Stage 1 Loss: Focal BCE
```
focal_bce = BCE_with_logits(proposal_logits, proposal_target, pos_weight=5.0)
focal_weight = (1 - p_t)^gamma, gamma=2.0
s1_loss = mean(focal_bce * focal_weight)
```

Proposal targets: binary vector (438 tokens) marking ALL onset positions in the full audio window.

### Stage 2 Loss: Multi-Onset OnsetLoss
```
total_onset_loss = 0
for i in range(4):
    total_onset_loss += OnsetLoss(logits[:, i], target[:, i])
total_onset_loss /= 4
```

OnsetLoss per onset: `0.5 * hard_CE + 0.5 * soft_CE` (trapezoid in log-ratio space, good_pct=3%, fail_pct=20%, frame_tolerance=2). STOP weight 1.5x. Each onset's target is an independent delta value — the soft trapezoid applies to each delta independently.

### Combined Loss
```
During freeze period (first 0 evals — S1 warm-started):
    loss = s1_loss  (Stage 2 forward skipped)

After freeze:
    loss = s2_multi_onset_loss + 0.5 * s1_loss
```

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 36 (reduced from 48 due to larger audio window) |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 (full dataset) |
| Balanced sampling | ON (1/count^0.5 weights) |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| N onsets | 4 |
| Delta onsets | ON |
| Proposer layers | 4 |
| Selector layers | 8 |
| Stage 2 freeze | 2 evals (S1 trains alone first) |
| Warm-start | None (training from scratch — different window size and target encoding) |
| S1 pos_weight | 5.0 |
| S1 focal gamma | 2.0 |
| Density jitter | ±10% at 30% |

## Dataset Target Construction (Delta Encoding)

For each sample (cursor position in a chart):
1. Find all future events within `n_onsets × B_PRED` (1000 bins) of cursor
2. Sort by absolute position
3. Convert to deltas: `delta[0] = event[0] - cursor`, `delta[i] = event[i] - event[i-1]`
4. Scan left-to-right: if `delta[i] >= B_PRED` (250), set it and all following to STOP
5. Fill remaining slots with STOP

Example: cursor=0, events at [50, 100, 150, 200]:
- deltas = [50, 50, 50, 50] → targets = [50, 50, 50, 50]

Example: cursor=500, events at [600, 700, 1700, 1833]:
- deltas = [100, 100, 1000, 133] → targets = [100, 100, STOP, STOP] (1000 >= 250)

Example: cursor=100, events at [500, 600, 700, 800]:
- deltas = [400, 100, 100, 100] → targets = [STOP, STOP, STOP, STOP] (400 >= 250)

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

## Inference (Autoregressive, Delta Mode)

1. Start cursor at bin 0
2. Extract mel window: cursor - 500 to cursor + 1250 (1750 frames)
3. Gather up to 128 past events as offsets from cursor
4. Forward pass: Stage 1 proposes, Stage 2 predicts 4 delta onsets → `(B, 4, 251)`
5. Decode deltas cumulatively:
   - prev_pos = cursor
   - For each onset i:
     - If delta_i = STOP: stop scanning (cascade)
     - Else: place event at prev_pos + delta_i, update prev_pos
6. Hop cursor to the last placed event (or hop_bins if all STOP)
7. Repeat until end of audio

## Metrics

### Per-onset step (i = 1, 2, 3, 4):
- `onset_i_hit_rate`, `onset_i_miss_rate`, `onset_i_accuracy`, `onset_i_score`

### Averaged:
- `multi_onset_avg_hit`, `multi_onset_avg_miss`, `multi_onset_avg_score`

### Stage 1 metrics:
- `s1_precision`, `s1_recall`, `s1_f1`, `s1_avg_proposals`
- `s2_picks_s1`, `s2_agree_accuracy`, `s2_override_accuracy`

### Per-onset S1 agreement:
- `onset_N_s2_picks_s1`, `onset_N_s2_agree_acc`, `onset_N_s2_override_acc`

### Per-onset STOP distribution:
- `onset_N_stop_pred_rate`, `onset_N_stop_target_rate`

### Per-onset bin histograms:
- `onset_N_pred_pct_X_Y`, `onset_N_tgt_pct_X_Y` (bins: 0-10, 10-25, 25-50, 50-100, 100-200, 200-250)

### TaikoNation patterning metrics (from AR benchmark):
- `tn_over_pspace` — unique 8-step pattern diversity (% of 256 possible)
- `tn_hi_pspace` — % of GT patterns found in predicted chart
- `tn_dc_human` — direct binary match at 23ms resolution
- `tn_dc_rand` — similarity to random noise (~50% = structured)

### Multi-onset structural metrics:
- `strict_increasing` — % of predictions where o1 < o2 < o3 < o4. With delta encoding, this should be ~100% (any positive delta maintains order).
- `strict_stop_violation_rate` — % with onset-after-STOP. Should be ~0%.
- `all_stop_rate` — % where all 4 onsets are STOP.
- `all_stop_target_rate` — same for targets.

### Eval graphs:
Full graph set generated per onset: `eval_XXX_o1_*.png`, `eval_XXX_o2_*.png`, `eval_XXX_o3_*.png`, `eval_XXX_o4_*.png`, `eval_XXX_oA_*.png` (pooled)

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
