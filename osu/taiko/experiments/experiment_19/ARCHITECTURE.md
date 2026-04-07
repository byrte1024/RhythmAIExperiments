# Experiment 19 — Full Architecture Specification

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

## Model: RerankerOnsetDetector

**Total parameters: ~16.0M** (Context: 2.5M, 16%)

Two-path architecture where context operates entirely in "gap space" (inter-onset intervals) with its own encoders, independent of the shared audio/event encoders. Audio path uses the standard AudioEncoder (conv stem + 4 transformer layers + FiLM conditioning) producing 501-way logits. Context uses gap representation + local mel snippets + own gap encoder and snippet encoder.

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

### 3. EventEncoder (event offsets → event tokens) — used by audio path only

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

### 5. RerankerContextPath (gap-based K-way reranker with own encoders)

Context operates entirely in gap space with its own encoders (d_ctx=192). Gradient isolation is structural — context does not use shared encoder outputs. Conditioning is detached by the caller.

#### 5a. Gap Computation

```
event_offsets (B, 128), event_mask (B, 128)
  → gap_before[i] = offset[i+1] - offset[i]   for consecutive valid events → (B, 127)
  → cursor_gap = -offset[-1]                   time since last event        → (B, 1)
  → all_gaps = cat(gap_before, cursor_gap)                                  → (B, 128)
  → all_gap_mask = ~(gap_valid for both events)                             → (B, 128)
```

#### 5b. Snippet Encoder (shared for events and candidates)

Extracts ~50ms mel spectrogram windows at each position and encodes them.

```
snippet_frames = 10 (half = 5 frames each side)
mel snippet at position p: mel[:, :, p-5:p+5] → flatten → (80 * 10 = 800)

snippet_encoder:
  Linear(800, 192) → GELU → Linear(192, 192) → snippet_feat (B, N, 192)
```

Events outside the mel window (>2.5s ago) get zero snippets. STOP candidates get a learned embedding (`stop_snippet_emb`, 192-dim) instead.

#### 5c. Gap Encoder (rhythm pattern processing)

```
gap_features = SinusoidalPosEmb(192)(all_gaps.abs())          → (B, 128, 192)
event_snippet_feat = snippet_encoder(mel snippets at events)   → (B, 128, 192)

x = gap_features + event_snippet_feat
  → + nn.Embedding(129, 192) sequence-order positions
  → NaN guard (unmask dummy if all masked)
  → 2 TransformerEncoderLayers (d_model=192, nhead=6, ffn=768, dropout=0.1, gelu, pre-norm)
    each with src_key_padding_mask=safe_mask, followed by FiLM(cond)
  → rhythm_repr: (B, 128, 192)
```

#### 5d. Candidate Embedding Building

```
audio_logits (B, 501) → detach → topk(20) → top_k_scores (B, 20), top_k_indices (B, 20)
  (STOP class forced into top-K if not already present)

proposed_gaps = top_k_indices - last_event_offset             → (B, 20)
cand_gap_emb = SinusoidalPosEmb(192)(proposed_gaps.abs())     → (B, 20, 192)
cand_snippet_feat = snippet_encoder(mel at candidate positions) → (B, 20, 192)
  (STOP candidates get stop_snippet_emb instead)

audio_probs = softmax(audio_logits_det)
cand_probs = gather at top_k positions                        → (B, 20)
score_feat = Linear(1, 192) → GELU (cand_probs)              → (B, 20, 192)

candidate_feat = Linear(192*3, 192) → GELU → LayerNorm(192)
  (cat [cand_gap_emb, cand_snippet_feat, score_feat])         → (B, 20, 192)
```

Training-time candidate shuffling: candidates are randomly permuted per sample to prevent positional bias (k=0 always being audio's #1).

#### 5e. Selection Head (cross-attention to candidates)

```
rhythm_repr (B, 128, 192)
  → append learned query_token (1, 192) + position embedding → (B, 129, 192)
  → 2 TransformerDecoderLayers (d_model=192, nhead=6, ffn=768, dropout=0.1, gelu, pre-norm)
    each with:
      tgt_key_padding_mask = gap_mask + False for query
      memory = candidate_feat (B, 20, 192) — cross-attention to candidates
    followed by FiLM(cond)
  → query_out = x[:, -1, :] (B, 192)
```

#### 5f. Scoring

```
q = Linear(192, 64)(query_out)                    → (B, 64)
k = Linear(192, 64)(candidate_feat)               → (B, 20, 64)
selection_logits = (k @ q.unsqueeze(-1)) / sqrt(64) → (B, 20)
```

### 6. Final Prediction

```
logits = full_like(audio_logits, -100.0)
logits.scatter_(1, top_k_indices, selection_logits) → (B, 501)
```

### Gradient Isolation

Structural: context path has its own gap encoder and snippet encoder, does not use audio_tokens or event_tokens from shared encoders. Conditioning is detached by the caller. Audio loss trains: `audio_path` + `audio_encoder` + `event_encoder` + `cond_mlp`. Selection loss trains: `context_path` ONLY.

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

Find which of the K candidates is closest to the true target — that is the correct class. `F.cross_entropy(sel_logits, sel_target)`. Audio loss trains encoders + audio path. Selection loss trains context path only.

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
| Epochs | 3 (killed — context still net-negative) |
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

## Key Metrics (Epoch 3)

| Metric | Value |
|---|---|
| Audio HIT | 67.3% |
| Final HIT | 66.9% |
| Context delta | -0.44pp |
| Override rate | 6.1% |
| Override accuracy | 38.2% |
| Rescued rate | 32.5% |
| Damaged rate | 39.7% |
| True Top1 | 64.5% |
| False Top1 | 29.4% |

## Context Status

Most promising context results across all experiments. Gap-based representation showed positive signal during training (context outperformed audio: 54.2% vs 52.5% HIT at one point). Own encoders with direct gradient worked. Override accuracy (36-40%) was the best achieved. Context delta reached -0.18pp at E2 — closest to break-even ever. However, training advantage did not fully transfer to validation. Key issue: audio instability during training meant context learned patterns about bad proposals that became irrelevant as audio improved. Next direction: warm-start audio from a pretrained checkpoint to provide stable proposals from the start.

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
