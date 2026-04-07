# Experiment 15 — Full Architecture Specification

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

Two-path architecture. Both paths produce 501-way logits, which are summed. This experiment adds context auxiliary loss (0.1 weight) and new density ablation benchmarks.

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

Three-term loss: main logits + audio auxiliary + context auxiliary.

```
loss = OnsetLoss(logits, targets) + 0.2 * OnsetLoss(audio_logits, targets) + 0.1 * OnsetLoss(context_logits, targets)
```

Audio aux weight: 0.2. Context aux weight: 0.1. Total aux weight: 0.3.

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
| Epochs | 4 (stopped — context path remained dormant) |
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
- ~5ms per mel frame (BIN_MS = 4.9887 — corrected from 5.0 in exp 14)
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

## Key Metrics (Epoch 4)

| Metric | Value |
|---|---|
| val_loss | 2.701 |
| accuracy | 48.6% |
| HIT (<=3% or +/-1 frame) | 66.4% |
| MISS (>20% off) | 32.8% |
| STOP F1 | 0.440 |
| p99 error | 151 frames |
| no_events acc | 48.5% |
| no_audio acc | 0.2% |
| Metronome benchmark | 45.0% |

## Density Benchmark Discovery

New ablation benchmarks revealed density conditioning is load-bearing:

| Benchmark | E4 |
|---|---|
| full accuracy | 48.6% |
| zero_density (conditioning = [0,0,0]) | 21.8% |
| random_density (randomized conditioning) | 40.0% |
| full - zero gap | 26.8pp |

Zeroing the density vector halves accuracy, and the gap increased over training. The model deeply relies on FiLM conditioning.

## Context Status

The context path remained **dormant**: no_events accuracy (48.5%) matched full accuracy (48.6%), meaning context contributes nothing measurable. The 0.1 context aux CE loss pushes the context path to independently predict the correct answer, but the path's optimal strategy remains "copy audio's top choice." Standard CE has no mechanism to reward overriding audio when audio's #2 or #3 is correct. Consistently ~1% behind exp 14 — the context aux added gradient noise without benefit.

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
