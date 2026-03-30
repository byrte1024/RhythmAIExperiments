# Experiment 05 — Full Architecture Specification

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
| logits | (B, 501) | 500 onset bin offsets (0-499, ~0-2.5s ahead) + 1 STOP class (500) |

## Model: OnsetDetector (single-path)

**Total parameters: ~47M**

Single-path architecture: AudioEncoder produces audio tokens, EventDecoder cross-attends to audio and produces logits via a query token. No separate audio/context paths — one unified decoder.

### 1. Conditioning MLP

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

### 2. AudioEncoder (mel → audio tokens)

```
mel (B, 80, 1000)
  → Conv1d(80, 256, kernel=7, stride=2, padding=3) → GELU → GroupNorm(1, 256)
  → Conv1d(256, 512, kernel=7, stride=2, padding=3) → GELU
  → transpose → (B, 250, 512)
  → LayerNorm(512)
  → + SinusoidalPosEmb(positions 0..249)
  → FiLM(cond)
  → 6 TransformerEncoderLayers (d_model=512, nhead=8, ffn=2048, dropout=0.1, gelu, pre-norm)
    each followed by FiLM(cond)
  → audio_tokens: (B, 250, 512)
```

4x downsample: 1000 mel frames → 250 tokens. Cursor at token 125.

### 3. EventDecoder (events + cross-attention to audio → logits)

```
event_offsets (B, 128) → SinusoidalPosEmb(512) → Linear(512, 512) → (B, 128, 512)
  → append learned query_token (1, 512) → seq (B, 129, 512)
  → + nn.Embedding(129, 512) sequence-order positions
  → TransformerDecoder: 8 TransformerDecoderLayers (d_model=512, nhead=8, ffn=2048, dropout=0.1, gelu, pre-norm)
    with:
      tgt_mask = causal mask (129x129)
      tgt_key_padding_mask = event_mask + False for query position
      memory = audio_tokens (B, 250, 512)
  → query_out = x[:, -1, :] (B, 512)
  → LayerNorm(512) → Linear(512, 501) → logits (B, 501)
```

### FiLM Conditioning

Applied after conv stem and after every AudioEncoder layer:
```
cond (B, 64) → Linear(64, 2*D) → split → scale (B, D), shift (B, D)
x = x * (1 + scale) + shift
```

FiLM is zero-initialized so it starts as identity: (1+0)*x + 0 = x.

Note: FiLM was newly added in this experiment. EventDecoder does NOT have FiLM — only the AudioEncoder path is conditioned.

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, dim).

## Loss: Gaussian Soft Targets in Log-Ratio Space

**This is the key change in experiment 05.** Unlike later experiments that used trapezoid soft targets, exp 05 used Gaussian soft targets with log_sigma=0.04.

### Gaussian Soft Target Construction

For each non-STOP target at bin `t`, create a soft distribution over all 500 bins:
- Compute `log_ratio = |log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Apply Gaussian kernel: `weight = exp(-0.5 * (log_ratio / log_sigma)^2)` with log_sigma=0.04
- Normalize to sum to 1.0

The Gaussian has infinite tails, meaning harmonic predictions (2x, 0.5x the correct gap) receive partial credit through the bell curve.

### Combined Loss

```
loss = 0.5 * hard_CE + 0.5 * soft_CE
```

Where hard_CE = standard cross-entropy, soft_CE = KL divergence with Gaussian soft targets.

STOP samples get `stop_weight=3.0x` multiplier.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 |
| Epochs | 5 |
| Scheduler | CosineAnnealingLR |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Subsample | 4 (every 4th sample, ~150K training samples) |
| Balanced sampling | Yes (WeightedRandomSampler inversely weighted by class frequency) |
| Evals per epoch | 1 |

## Augmentation

Context and audio augmentation applied during training:

| Aug | Rate | Description |
|---|---|---|
| Event jitter | Always | +/-4 bins random per-event offset |
| Context dropout | 8% | Remove all past events |
| Context truncation | 10% | Keep random suffix of events |
| Context time-warp | 15% | Scale event positions by musical fraction (0.5x, 2x, 1/3x, etc.) |
| Context gap-shuffle | 10% | Randomize inter-event gap order |
| Audio fade-in | 15% | Linear fade over 20-100 frames at start |
| Audio fade-out | 15% | Linear fade over 20-100 frames at end |
| Mel gain | 50% | +/-3 dB uniform shift |
| Mel noise | 30% | Additive Gaussian noise (sigma 0.1-0.5) |
| SpecAugment freq mask | 40% | Zero out 1-8 consecutive mel bands |
| SpecAugment time mask | 40% | Zero out 1-40 consecutive time frames |
| Conditioning jitter | 50% | Scale density values by 0.85-1.15x |

## Dataset: taiko_v1 (BIN_MS = 5.0)

- 10,048 charts from osu!taiko
- Audio: 22050 Hz mono, mel spectrogram (80 bands, hop=110, n_fft=2048, 20-8000 Hz)
- ~5ms per mel frame (BIN_MS = 5.0 -- later discovered to be wrong; actual frame duration is 4.9887ms)
- Events: int32 arrays of mel frame indices per chart
- Train/val split: 90/10 by unique song (random seed 42)
- MIN_CURSOR_BIN = 6000 (~30s into song, skip early events)

**Note**: The BIN_MS=5.0 bug caused progressive timing misalignment between mel frames and event labels (0.01134ms/frame, compounding to ~408ms at 3 minutes). This was discovered in exp 13 and fixed in exp 14. All experiments 05-13 trained on this misaligned dataset, which capped accuracy at ~46%.

## Inference (Autoregressive)

1. Start cursor at bin 0
2. Extract mel window: cursor +/- 500 bins
3. Gather up to 128 past events as offsets from cursor
4. Forward pass → 501 logits
5. If argmax = 500 (STOP): hop cursor forward by hop_bins (default 20 = ~100ms)
6. If argmax = offset (0-499): place event at cursor + offset, move cursor there
7. Repeat until end of audio

## Key Metrics (Epoch 5 best)

| Metric | Value |
|---|---|
| Accuracy | 33.0% |
| HIT (<=3% or +/-1 frame) | 53.2% |
| STOP F1 | 0.075 |
| Frame error median | 1.0 |
| Within 2 frames | 54.9% |

## Key Finding

The model became a metronome, not a beat detector. Scatter plots showed clear lines at musical ratios (1/2, 2/1, 1/4, 4/1). The Gaussian soft targets' infinite tails gave partial credit for harmonic predictions, so the model learned to predict at the correct BPM interval rather than detecting actual onsets. Loss kept decreasing (the metronome strategy was being rewarded) but inference showed steady-rhythm output regardless of audio content.
