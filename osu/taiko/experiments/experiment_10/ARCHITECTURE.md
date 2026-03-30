# Experiment 10 — Full Architecture Specification

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
| logits | (B, 501) | 500 onset bin offsets (0-499, ~0-2.5s ahead) + 1 STOP class (500). Sum of audio_logits + context_logits. |

## Model: LegacyOnsetDetector

**Total parameters: ~21M**

Two-path architecture: an audio path and a context path each produce 501-way logits, which are summed. Audio path proposes candidates, context path selects among them. In practice, a NaN bug in the EventEncoder caused the context path to produce NaN for any sample with no prior events, corrupting training on those samples.

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

**NaN Bug (no guard):** When all events are masked for a sample, PyTorch's TransformerEncoderLayer computes softmax over all -inf attention scores, producing NaN. This NaN propagated through the EventEncoder into both paths via cross-attention. No dummy-unmasking guard was present in this experiment.

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

### 5. LegacyContextPath (causal event attention + cross-attention to audio → context logits)

```
event_tokens (B, 128, 384)
  → append learned query_token (1, 384) → seq (B, 129, 384)
  → + nn.Embedding(129, 384) sequence-order positions
  → 3 TransformerDecoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each with:
      tgt_mask = causal mask (129x129)
      tgt_key_padding_mask = event_mask + False for query position
      memory = audio_tokens (B, 250, 384)
    followed by FiLM(cond)
  → query_out = x[:, -1, :] (B, 384)
  → LayerNorm(384) → Linear(384, 501) → context_logits (B, 501)
  → context_logits = context_logits + Conv1d_smooth(context_logits)
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

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, dim).

## Loss

Two-term loss: main logits + auxiliary audio logits.

```
loss = OnsetLoss(logits, targets) + 0.2 * OnsetLoss(audio_logits, targets)
```

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
| Epochs | 6 |
| Scheduler | CosineAnnealingLR |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 1 |

## Augmentation (Context Only)

Context augmentation applied to event_offsets during training:

| Aug | Description |
|---|---|
| Event jitter | Uniform per-event position noise (±4 bins) |
| Full dropout | 5% chance: mask all events |
| Truncation | 10% chance: keep only a random prefix of events |

No audio augmentation beyond basic processing. No density augmentation.

## Dataset: taiko_v2 (misaligned BIN_MS)

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 5.0 — incorrect, should be 4.9887; causes progressive drift)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor +/- 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 501 logits (audio_logits + context_logits)
5. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Metrics (Epoch 6)

| Metric | Value |
|---|---|
| val_loss | NaN (all epochs) |
| accuracy | 40.8% |
| HIT (<=3% or +/-1 frame) | 60.6% |
| top-10 HIT | 93.1% |
| unique preds | 438 |
| STOP F1 | 0.381 |
| no_events acc | 0.0% (broken — NaN) |
| no_audio acc | 38.0% (rising — bad) |

## NaN Bug Status

val_loss was NaN in every epoch because samples with zero prior events produced NaN through the EventEncoder. `best.pt` was never saved (NaN < best_val_loss = False). The no_events and no_events_no_audio benchmarks were completely broken (0% accuracy). NaN gradients on affected samples meant the model never learned what to do when events are absent. The fix (applied in exp 11) is to unmask a dummy position when all events are masked.
