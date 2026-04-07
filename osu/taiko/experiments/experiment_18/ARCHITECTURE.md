# Experiment 18 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position.

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
| logits | (B, 501) | Scattered selection logits at top-K candidate positions, -100 elsewhere. |
| audio_logits | (B, 501) | Raw 501-way audio logits from AudioPath. |
| selection_logits | (B, 20) | K-way context selection logits over top-K candidates. |
| top_k_indices | (B, 20) | Bin indices of the top-K audio candidates. |

## Model: Exp18OnsetDetector

**Total parameters: ~22.5M** (Context: 9.0M, 40%)

Two-path architecture with stop-gradient isolation and a two-stage context path. Audio path produces 501-way logits (unchanged). Context path uses two explicit stages: event understanding (self-attention only) then candidate selection (cross-attention to candidates). All shared encoder outputs are detached before entering the context path.

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. AudioEncoder (mel → audio tokens)

```
mel (B, 80, 1000)
  → Conv1d(80, 192, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 192)
  → Conv1d(192, 384, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
  → 4 TransformerEncoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each followed by FiLM(cond)
  → audio_tokens: (B, 250, 384)
```

4x downsample: 1000 mel frames → 250 tokens. Cursor at token 125.

### 3. EventEncoder (event offsets → event tokens)

Operates at a bottleneck dimension d_event=128, then projects up to d_model=384.

```
event_offsets (B, 128) → SinusoidalPosEmb(128) → Linear(128, 128) → (B, 128, 128)
  → + nn.Embedding(128, 128) sequence-order positions
  → 2 TransformerEncoderLayers (d_model=128, nhead=4, ffn=512, dropout=0.1, gelu, pre-norm)
    each with src_key_padding_mask=event_mask, followed by FiLM(cond)
  → Linear(128, 384) → event_tokens: (B, 128, 384)
```

NaN guard: if all events are masked for a sample, the last position is unmasked as a dummy.

### 4. AudioPath (audio self-attention + cross-attention to events → audio logits)

```
audio_tokens (B, 250, 384) as tgt
event_tokens (B, 128, 384) as memory
  → 2 TransformerDecoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each with memory_key_padding_mask=event_mask, followed by FiLM(cond)
  → cursor = x[:, 125, :] (B, 384)
  → LayerNorm(384) → Linear(384, 501) → audio_logits (B, 501)
  → audio_logits = audio_logits + Conv1d_smooth(audio_logits)
```

Smoothing: `Conv1d(1, 8, k=5, pad=2) → GELU → Conv1d(8, 1, k=5, pad=2)`

### 5. Exp18ContextPath (two-stage with stop-gradient isolation)

All inputs to the context path are detached: `event_tokens.detach()`, `audio_tokens.detach()`, `cond.detach()`. Selection loss trains only the context path parameters.

#### 5a. Candidate Feature Building

```
audio_logits (B, 501) → detach → topk(20) → top_k_scores (B, 20), top_k_indices (B, 20)
  (STOP class forced into top-K if not already present)

For each of K=20 candidates:
  bin_emb = SinusoidalPosEmb(384)(top_k_indices)                    → (B, 20, 384)
  score_feat = Linear(2, 384) → GELU ([audio_score, normalized_rank]) → (B, 20, 384)
  audio_feat = audio_tokens gathered at candidate temporal positions   → (B, 20, 384)

candidate_feat = Linear(384*3, 384) → GELU → LayerNorm(384)
  (cat [bin_emb, score_feat, audio_feat])                            → (B, 20, 384)
```

#### 5b. Stage 1 — Event Understanding (pure self-attention)

Events reason about temporal patterns before seeing any candidates. Non-causal — every event sees every other event.

```
event_tokens.detach() (B, 128, 384)
  → + nn.Embedding(128, 384) sequence-order positions
  → NaN guard (unmask dummy if all masked)
  → 2 TransformerEncoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each with src_key_padding_mask=safe_mask, followed by FiLM(cond)
  → event_repr: (B, 128, 384)
```

#### 5c. Stage 2 — Candidate Selection (cross-attention to candidates)

```
event_repr (B, 128, 384)
  → append learned query_token (1, 384) + position embedding → (B, 129, 384)
  → 2 TransformerDecoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each with:
      tgt_key_padding_mask = event_mask + False for query
      memory = candidate_feat (B, 20, 384) — cross-attention to candidates
      NO causal mask (events already have global context from stage 1)
    followed by FiLM(cond)
  → query_out = x[:, -1, :] (B, 384)
```

#### 5d. Scoring

```
q = Linear(384, 64)(query_out)                    → (B, 64)
k = Linear(384, 64)(candidate_feat)               → (B, 20, 64)
selection_logits = (k @ q.unsqueeze(-1)) / sqrt(64) → (B, 20)
```

### 6. Final Prediction

```
logits = full_like(audio_logits, -100.0)
logits.scatter_(1, top_k_indices, selection_logits) → (B, 501)
```

### Gradient Isolation

Audio loss trains: `audio_path` + `audio_encoder` + `event_encoder` + `cond_mlp`.
Selection loss trains: `context_path` ONLY (all encoder outputs detached).

### FiLM Conditioning

Applied after conv stem, after every encoder layer, and after every decoder layer:
```
cond (B, 64) → Linear(64, 2*D) → split → scale (B, D), shift (B, D)
x = x * (1 + scale) + shift
```

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, dim).

## Loss

Two-term loss: hard CE selection loss on context + onset loss on audio.

```
loss = F.cross_entropy(selection_logits, sel_target) + 1.0 * OnsetLoss(audio_logits, targets)
```

### Selection Loss (Hard CE)

Simple hard CE. Find which of the K candidates is closest to the true target — that is the correct class. `F.cross_entropy(sel_logits, sel_target)`. Uses hard cross-entropy rather than soft trapezoid K-way CE.

### OnsetLoss: Mixed Hard CE + Soft Target

### Soft Targets (Trapezoid)

For each non-STOP target at bin `t`, create a soft distribution over all 500 bins:
- Compute `|log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Within 3% ratio: full credit (plateau)
- 3% to 20%: linear ramp to zero
- Beyond 20%: zero
- Frame tolerance +/-2: always get some credit regardless of ratio

### Combined Loss

```
loss = 0.5 * hard_CE + 0.5 * soft_CE
```

Where hard_CE = standard cross-entropy, soft_CE = KL divergence with soft targets.

STOP samples get `stop_weight=1.5x` multiplier.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 |
| Epochs | 2 (killed — context actively harmful) |
| Scheduler | CosineAnnealingLR |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 1 |

## Augmentation (Context — AR-Simulating)

Context augmentation applied to event_offsets during training:

| Aug | Description |
|---|---|
| Recency-scaled jitter | Per-event noise scales from 1x (oldest) to 3x (most recent), simulating how AR errors are larger for recent predictions. Plus a global ±3 bin shift for systematic drift. |
| Random deletion (8%) | Drop 1 to N/6 individual events, simulating missed beats during inference |
| Random insertion (8%) | Add 1 to N/6 spurious events at random positions, simulating false positives |
| Full dropout | 5% chance: mask all events |
| Truncation | 10% chance: keep only a random prefix of events |

No audio augmentation beyond basic processing. No density augmentation.

## Dataset: taiko_v2 (corrected BIN_MS)

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887 — corrected from 5.0)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor +/- 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → audio_logits (501-way), selection_logits (K-way), top_k_indices
5. Final logits: scatter selection_logits onto 501-way tensor at top_k positions
6. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
7. If argmax = offset (0-499): place event at cursor + offset, move cursor there
8. Repeat until end of audio

## Key Metrics (Epoch 2)

| Metric | Value |
|---|---|
| Audio HIT | 67.4% |
| Final HIT | 66.4% |
| Context delta | -0.94pp |
| Override rate | 6.0% |
| Override accuracy | 35.2% |
| Rescued rate | 29.2% |
| Damaged rate | 44.8% |
| Rank 0 selection | 93.9% |

## Context Status

Stop-gradient successfully protected audio (HIT 67.5% at E1, stable). Two-stage architecture did not prevent rubber-stamping — 94% rank-0 selection. Hard CE with imbalanced 20-way classification where class 0 is correct ~95% of the time led to "mostly pick 0, occasionally pick wrong." Override accuracy 35% — worse than coin flip, worse than exp 17's 52%. Context overrides were net-harmful and worsening (damaged 45% > rescued 29%).

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
