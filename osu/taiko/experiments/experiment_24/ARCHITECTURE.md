# Experiment 24 — Full Architecture Specification

## Task

Predict the next onset timing in an osu!taiko rhythm game chart, given audio + past event context. Autoregressive: each prediction advances the cursor to the predicted onset position. Context path produces its own 501-way logit distribution, added to audio logits before softmax — soft influence instead of hard override.

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
| logits | (B, 501) | audio_logits + context_logits. 500 onset bin offsets (0-499) + 1 STOP class (500). |
| audio_logits | (B, 501) | From frozen audio path only. |
| context_logits | (B, 501) | From trainable context path only. |

Final prediction: `argmax(logits)` = `argmax(audio_logits + context_logits)`.

## Model: AdditiveOnsetDetector

**Total parameters: ~15M (~1.2M trainable, ~13.8M frozen)**

Two-path architecture: a frozen audio path produces 501-way logits, a trainable context path produces independent 501-way logits, summed for the final output.

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

### 5. ContextPath (TRAINABLE) — gap-based additive 501-way output

Context produces its own 501-way logit distribution from gap patterns and mel snippets. Added to audio logits for soft influence.

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

#### 5b. Cursor Token + Output Head

```
rhythm_repr (B, 128, 192)
  → append learned cursor_token (1, 192) at position C
  → + nn.Embedding(129, 192) position for cursor
  → one more self-attention pass (re-uses last gap layer) so cursor attends to rhythm
    src_key_padding_mask = [gap_mask, False]
  → FiLM(cond)
  → cursor_out = x[:, -1, :] → (B, 192)

output_head: Linear(192, 192) → GELU → Linear(192, 501) → context_logits (B, 501)
```

### 6. Logit Combination

```
logits = audio_logits + context_logits → (B, 501)
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

Two independent OnsetLoss terms on audio and context logits:

```
loss = OnsetLoss(audio_logits, targets) + OnsetLoss(context_logits, targets)
```

Note: audio loss has no effect (audio path frozen, no gradient). Only context loss trains the context path. Combined logits are not directly trained — they emerge from the sum.

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

STOP samples get `stop_weight=3.0x` multiplier.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 |
| Epochs | 5 (killed) |
| Scheduler | CosineAnnealingLR |
| Subsample | 4 (25% of training data) |
| Warm-start | exp 14 best.pt (audio components) |
| Freeze | AudioEncoder, EventEncoder, AudioPath, cond_mlp |
| Trainable params | ~1.2M (context path only, smaller than reranker's 2.5M) |
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
4. Forward pass → logits (audio + context), audio_logits, context_logits
5. If argmax(logits) = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax(logits) = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Metrics (E4 best)

| Metric | Value |
|---|---|
| Audio HIT | 69.5% (frozen) |
| Context HIT | 53.8% (standalone) |
| Combined HIT | 68.8% |
| Delta | -0.64pp |
| context_helped | 5.52% |
| context_hurt | 6.15% |

## Context Status

Best delta ever for any experiment (-0.64pp at E4). Context learned real signal: 53.8% standalone HIT from gaps + snippets alone. But context logit magnitudes grew at E5, causing hurt to spike. The additive paradigm is safer than reranking but context operating blindly (only gaps + snippets, no audio access) causes helped ~ hurt, guaranteeing net-negative delta.
