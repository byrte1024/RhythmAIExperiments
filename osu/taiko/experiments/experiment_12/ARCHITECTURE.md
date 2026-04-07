# Experiment 12 — Full Architecture Specification

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

## Model: LegacyOnsetDetector (modified)

**Total parameters: ~24.5M** (+17% over exp 11's ~21M, invested in context side)

Two-path architecture with a widened event encoder and deeper context path compared to exp 11. Both paths produce 501-way logits, which are summed. This experiment also returns context_logits separately for the new context auxiliary loss.

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

### 3. EventEncoder (event offsets → event tokens) — WIDENED

Operates at a widened bottleneck d_event=192 (was 128 in exp 11), with 3 layers (was 2) and 6 heads (was 4).

```
event_offsets (B, 128) → SinusoidalPosEmb(192) → Linear(192, 192) → (B, 128, 192)
  → + nn.Embedding(128, 192) sequence-order positions
  → 3 TransformerEncoderLayers (d_model=192, nhead=6, ffn=768, dropout=0.1, gelu, pre-norm)
    each with src_key_padding_mask=event_mask, followed by FiLM(cond)
  → Linear(192, 384) → event_tokens: (B, 128, 384)
```

NaN guard: if all events are masked for a sample, the last position is unmasked as a dummy.

Parameter increase: ~0.5M → ~1.5M.

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

### 5. LegacyContextPath (causal event attention + cross-attention to audio → context logits) — DEEPENED

4 decoder layers (was 3 in exp 11).

```
event_tokens (B, 128, 384)
  → append learned query_token (1, 384) → seq (B, 129, 384)
  → + nn.Embedding(129, 384) sequence-order positions
  → 4 TransformerDecoderLayers (d_model=384, nhead=8, ffn=1536, dropout=0.1, gelu, pre-norm)
    each with:
      tgt_mask = causal mask (129x129)
      tgt_key_padding_mask = event_mask + False for query position
      memory = audio_tokens (B, 250, 384)
    followed by FiLM(cond)
  → query_out = x[:, -1, :] (B, 384)
  → LayerNorm(384) → Linear(384, 501) → context_logits (B, 501)
  → context_logits = context_logits + Conv1d_smooth(context_logits)
```

Parameter increase: ~7.5M → ~9.9M.

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

Three-term loss: main logits + audio auxiliary + context auxiliary.

```
loss = OnsetLoss(logits, targets) + 0.1 * OnsetLoss(audio_logits, targets) + 0.1 * OnsetLoss(context_logits, targets)
```

Audio aux reduced from 0.2 to 0.1 (redistributed to context aux). This change was identified as the primary cause of failure.

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
| Epochs | 2 (killed early — collapsed) |
| Scheduler | CosineAnnealingLR |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 1 |

## Augmentation (Context — AR-Simulating)

Redesigned event augmentation to simulate autoregressive inference errors:

| Aug | Description |
|---|---|
| Recency-scaled jitter | Per-event noise scales from 1x (oldest) to 3x (most recent), simulating how AR errors are larger for recent predictions |
| Global shift | Uniform ±3 bin shift to all events simultaneously, simulating systematic timing drift |
| Random deletion (8%) | Drop 1 to N/6 individual events, simulating missed beats during inference |
| Random insertion (8%) | Add 1 to N/6 spurious events at random positions, simulating false positives |
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

## Key Metrics (Epoch 2)

| Metric | Value |
|---|---|
| val_loss | 3.787 |
| accuracy | 11.9% |
| HIT (<=3% or +/-1 frame) | 25.3% |
| top-3 HIT | ~42% |
| top-10 HIT | 65.7% |
| unique preds | 288 |
| no_events acc | 11.8% |
| no_audio acc | 5.4% |

## Failure Mode

Severe mode collapse — prediction distribution was spiky, concentrated on a handful of "safe" bins (~15, ~25, ~50, ~65). The scatter plot showed horizontal banding (same few y-values regardless of target). The audio proposer was crippled — top-10 only 65.7% (exp 11 E2 was 95%). Root cause: reducing audio aux from 0.2 to 0.1 halved the direct gradient to the audio path, starving the proposer to feed the selector. The bigger context path had nothing useful to select from.

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
