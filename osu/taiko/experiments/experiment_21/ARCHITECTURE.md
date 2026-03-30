# Experiment 21 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position. Context path reranks audio's top-K candidates using gap-based rhythm features with a relative quality selection loss.

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
| audio_logits | (B, 501) | 500 onset bin offsets (0-499) + 1 STOP class (500). From frozen audio path. |
| selection_logits | (B, 20) | K-way reranking scores over audio's top-20 candidates |
| top_k_indices | (B, 20) | Which 20 bins from audio_logits were selected as candidates |

Final prediction: `top_k_indices[argmax(selection_logits)]`.

## Model: RerankerOnsetDetector

**Total parameters: ~16M (2.5M trainable, 13.5M frozen)**

Two-path architecture: a frozen audio path produces 501-way logits, a trainable context path reranks the top-20 candidates.

### 1. Conditioning MLP (FROZEN)

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. AudioEncoder (FROZEN) — mel → audio tokens

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

### 3. EventEncoder (FROZEN) — event offsets → event tokens

Operates at d_event=128, projects up to d_model=384.

```
event_offsets (B, 128) → SinusoidalPosEmb(128) → Linear(128, 128) → (B, 128, 128)
  → + nn.Embedding(128, 128) sequence-order positions
  → 2 TransformerEncoderLayers (d_model=128, nhead=4, ffn=512, dropout=0.1, gelu, pre-norm)
    each with src_key_padding_mask=event_mask, followed by FiLM(cond)
  → Linear(128, 384) → event_tokens: (B, 128, 384)
```

NaN guard: if all events are masked for a sample, the last position is unmasked as a dummy.

### 4. AudioPath (FROZEN) — audio self-attention + cross-attention to events → audio logits

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

### 5. RerankerContextPath (TRAINABLE) — gap-based K-way reranker

Context operates entirely in "gap space" — inter-onset intervals rather than absolute positions. Uses its own encoders (d_ctx=192), fully gradient-isolated from audio.

#### 5a. Gap Encoder — rhythm pattern understanding

```
event_offsets (B, 128)
  → compute gap sequence: gap_before[i] = offset[i] - offset[i-1], append cursor_gap = -offset[-1]
  → all_gaps (B, 128), all_gap_mask (B, 128)

gap_features = SinusoidalPosEmb(192)(all_gaps.abs())  → (B, 128, 192)

mel snippet extraction at each event position:
  event_mel_frames = 500 + event_offsets → snippet centers
  for each gap position: extract 10-frame (~50ms) mel window → (B, 128, 80*10)
  → snippet_encoder: Linear(800, 192) → GELU → Linear(192, 192)
  → event_snippet_feat (B, 128, 192)

x = gap_features + event_snippet_feat
  → + nn.Embedding(129, 192) sequence-order positions
  → 2 TransformerEncoderLayers (d_model=192, nhead=6, ffn=768, dropout=0.1, gelu, pre-norm)
    each with src_key_padding_mask=gap_mask, followed by FiLM(cond)
  → rhythm_repr (B, 128, 192)
```

#### 5b. Candidate Building — top-K with score/rank features

```
audio_logits (B, 501) → detached
  → top_k_scores, top_k_indices = topk(20)  → (B, 20) each
  → force-include STOP (class 500) if not in top-K

For each of K=20 candidates:
  proposed_gap = candidate_bin - last_event_offset → (B, 20)
  cand_gap_emb = SinusoidalPosEmb(192)(proposed_gaps.abs())  → (B, 20, 192)
  cand_snippet_feat = snippet_encoder(mel snippets at candidate positions) → (B, 20, 192)
    STOP candidates get learned stop_snippet_emb (192,) instead
  score_feat = score_proj: Linear(1, 192) → GELU → (B, 20, 192)
    input: softmax probability per candidate

candidate_feat = candidate_combine: Linear(192*3, 192) → GELU → LayerNorm(192) → (B, 20, 192)
```

Candidates are NOT shuffled in exp 21 (shuffling introduced in exp 22).

#### 5c. Selection Head — cross-attend to candidates → K-way scores

```
rhythm_repr (B, 128, 192)
  → append learned query_token (1, 192) → (B, 129, 192)
  → + nn.Embedding(129, 192) position for query
  → tgt_key_padding_mask = gap_mask + False for query

2 TransformerDecoderLayers (d_model=192, nhead=6, ffn=768, dropout=0.1, gelu, pre-norm)
  tgt = [rhythm_repr; query], memory = candidate_feat
  each followed by FiLM(cond)

query_out = x[:, -1, :] → (B, 192)

q = Linear(192, 64)(query_out)        → (B, 64)
k = Linear(192, 64)(candidate_feat)    → (B, 20, 64)
selection_logits = bmm(k, q) * (64^-0.5) → (B, 20)
```

### FiLM Conditioning

Applied after conv stem, after every encoder layer, and after every decoder layer:
```
cond (B, 64) → Linear(64, 2*D) → split → scale (B, D), shift (B, D)
x = x * (1 + scale) + shift
```

Context path receives `cond.detach()` — no gradient flows from context loss back to cond_mlp.

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, dim).

## Loss

### SelectionLoss — Relative Quality (new in exp 21)

Replaces hard cross-entropy with a relative quality loss operating in trapezoid ratio space:

**Step 1 — Quality scoring:** For each of K=20 candidates, compute quality = closeness to true target using the trapezoid (1.0 within 3%, linear ramp to 0 at 20%, frame floor ±1).

**Step 2 — Relative soft targets:** Build a probability distribution over K candidates:
- Candidates at or above audio's #1 quality: weight = their quality score
- Candidates below #1 quality: weight = 0 (suppressed)
- Normalize to sum to 1

**Step 3 — Soft CE:** Standard cross-entropy against the soft target distribution.

**Step 4 — Asymmetric miss penalty:** Scale up loss by `miss_penalty=2.0` when:
- Context chose to keep #1 (no override)
- #1 was bad (quality < 0.5)
- A significantly better candidate existed (best quality > #1 quality + 0.1)

**Step 5 — Skip impossible samples:** If no candidate has any quality (all zero weight), the sample is excluded from the loss.

Audio loss: none (audio path is frozen).

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 |
| Epochs | 2 (killed) |
| Scheduler | CosineAnnealingLR |
| Subsample | 4 (25% of training data) |
| Warm-start | exp 14 best.pt (audio components) |
| Freeze | AudioEncoder, EventEncoder, AudioPath, cond_mlp |
| Trainable params | 2.5M (context path only) |
| miss_penalty | 2.0 |
| Gradient clipping | 1.0 |
| Evals per epoch | 1 |

## Augmentation

### Context Augmentation (applied to event_offsets during training)

| Aug | Description |
|---|---|
| Event jitter | Recency-scaled per-event position noise |
| Global shift | Shift all events by a random offset |
| Event insertion/deletion | ~8% rate: add or remove random events |

### Audio Augmentation

Standard audio augmentation: gain jitter, noise injection, SpecAugment, fade in/out.

## Dataset: taiko_v2

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 4.9887)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor +/- 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → audio_logits (501), selection_logits (20), top_k_indices (20)
5. Selected bin = top_k_indices[argmax(selection_logits)]
6. If selected = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
7. If selected = offset (0-499): place event at cursor + offset, move cursor there
8. Repeat until end of audio

## Key Metrics (E2 best)

| Metric | Value |
|---|---|
| Audio HIT | 69.5% (frozen) |
| Final HIT | 68.5% |
| Delta | -0.95pp |
| Override rate | 36.5% |
| Override accuracy | 61.4% |
| Override F1 | 45.7% |
| true_topK | 22.4% |
| false_topK | 23.3% |

## Context Status

Relative quality loss transformed context behavior — override F1 doubled (22% to 46%), override accuracy above coin flip for the first time (61.4%). But delta remained negative because the loss design has a conservatism bias: when #1 is correct (70% of the time), all other candidates are suppressed to zero weight, making "keep #1" the dominant gradient signal.
