# Experiment 22 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position. Context path reranks audio's top-K candidates using gap-based rhythm features, with candidates shuffled and audio scores removed.

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
| selection_logits | (B, 20) | K-way reranking scores over audio's top-20 candidates (shuffled order) |
| top_k_indices | (B, 20) | Which 20 bins from audio_logits were selected as candidates (shuffled to match) |

Final prediction: `top_k_indices[argmax(selection_logits)]`.

## Model: RerankerOnsetDetector (modified)

**Total parameters: ~16M (~2.3M trainable, ~13.7M frozen)**

Two-path architecture: a frozen audio path produces 501-way logits, a trainable context path reranks the top-20 candidates. Audio score features removed and candidates shuffled during training.

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

### 5. RerankerContextPath (TRAINABLE) — blind gap-based K-way reranker

Context operates entirely in "gap space." Uses its own encoders (d_ctx=192). **No audio score features; candidates shuffled during training.**

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

#### 5b. Candidate Building — top-K WITHOUT score features

```
audio_logits (B, 501) → detached
  → top_k_scores, top_k_indices = topk(20)  → (B, 20) each
  → force-include STOP (class 500) if not in top-K

For each of K=20 candidates:
  proposed_gap = candidate_bin - last_event_offset → (B, 20)
  cand_gap_emb = SinusoidalPosEmb(192)(proposed_gaps.abs())  → (B, 20, 192)
  cand_snippet_feat = snippet_encoder(mel snippets at candidate positions) → (B, 20, 192)
    STOP candidates get learned stop_snippet_emb (192,) instead

candidate_feat = candidate_combine: Linear(192*2, 192) → GELU → LayerNorm(192) → (B, 20, 192)
```

**No score_proj layer** — candidates have only gap embedding + mel snippet. Input to candidate_combine is `d_ctx*2` (was `d_ctx*3` in exp 20-21).

**Candidate shuffle during training:** Per sample, the K=20 candidates are randomly permuted. Context cannot learn positional bias (e.g., "k=0 is usually right"). At inference, candidates are unshuffled.

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

Context path receives `cond.detach()`.

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, dim).

## Loss

### Simplified SelectionLoss (new in exp 22)

~20 lines, no baseline comparison, no miss penalty, no asymmetric scaling:

1. Compute quality of each candidate: trapezoid closeness to target (1.0 within 3%, linear ramp to 0 at 20%, frame floor ±1)
2. If no candidate is a HIT (quality > 0): skip the sample
3. Normalize quality scores to soft probability distribution
4. Soft CE: standard cross-entropy against that distribution

The loss simply says "pick whichever candidate is closest to the truth, with partial credit for near-misses."

Audio loss: none (audio path is frozen).

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 |
| Epochs | 15 (killed) |
| Scheduler | CosineAnnealingLR |
| Subsample | 4 (25% of training data) |
| Warm-start | exp 14 best.pt (audio components) |
| Freeze | AudioEncoder, EventEncoder, AudioPath, cond_mlp |
| Trainable params | ~2.3M (slightly less without score_proj) |
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

## Key Metrics (E14 best)

| Metric | Value |
|---|---|
| Audio HIT | 69.5% (frozen) |
| Final HIT | 64.8% |
| Delta | -4.6pp |
| Override rate | 52.5% |
| Override accuracy | 52.3% |
| Override F1 | 49.6% |
| false_top1 | 9.2% |
| true_topK | 27.4% |
| false_topK | 31.4% |

## Context Status

Shuffling eliminated positional bias and context made reasonable picks (low inaccurate_topK 12-14%). But without audio confidence, context overrode 50%+ of predictions uniformly — it could not tell when audio was confident and likely correct vs uncertain and worth overriding. Delta slowly improved over 15 epochs (-10.6pp to -4.6pp) but remained deeply negative.

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
