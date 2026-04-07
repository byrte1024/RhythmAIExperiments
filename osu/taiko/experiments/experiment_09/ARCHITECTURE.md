# Experiment 09 — Full Architecture Specification

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

Single-path architecture: AudioEncoder produces audio tokens, EventDecoder cross-attends to audio and produces logits via a query token. Identical model architecture to experiments 05, 06, and 07.

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

Note: FiLM applies only to the AudioEncoder. The EventDecoder has no FiLM conditioning.

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: integer positions. Output: (B, T, dim).

## Loss: Trapezoid Soft Targets (OnsetLoss)

Trapezoid soft targets with hard cutoffs (no Gaussian tails), with reduced stop_weight.

### Soft Targets (Trapezoid)

For each non-STOP target at bin `t`, create a soft distribution over all 500 bins:
- Compute `|log((bin+1)/(t+1))|` for each bin (proportional distance in ratio space)
- Within 3% ratio (`log_good = log(1.03)`): full credit (plateau)
- 3% to 20% (`log_fail = log(1.20)`): linear ramp to zero
- Beyond 20%: zero (hard cutoff)
- Frame tolerance +/-2: always get some credit regardless of ratio

### Combined Loss

```
loss = 0.5 * hard_CE + 0.5 * soft_CE
```

Where hard_CE = standard cross-entropy, soft_CE = KL divergence with soft targets.

**STOP weight reverted to 1.5** (down from 3.0 in exp 07, which was too aggressive and caused the model to default to STOP when uncertain).

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 |
| Epochs | 2 (killed after regression at E2) |
| Scheduler | CosineAnnealingLR |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Subsample | 4 (every 4th sample, ~150K training samples) |
| Balanced sampling | Yes (WeightedRandomSampler inversely weighted by class frequency) |
| Evals per epoch | 1 |

## Augmentation (LIGHT — Reverted to Original)

**The key change in this experiment was fully reverting to the original light augmentation rates** from experiments 05/06, after heavy augmentation (exp 07) killed the model:

| Aug | Rate | Description |
|---|---|---|
| Event jitter | Always | +/-4 bins random per-event offset |
| Context dropout | **5%** (reverted from 25%) | Remove all past events |
| Context truncation | **10%** | Keep random suffix of events |
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

## Key Metrics

| Metric | E1 | E2 |
|---|---|---|
| val_loss | 3.295 | 3.593 |
| Accuracy | 26.4% | 23.4% |
| HIT (<=3% or +/-1 frame) | 49.8% | 44.7% |
| STOP F1 | 0.234 | 0.049 |
| Frame error median | 2.0 | 4.0 |

## Ablation Benchmark Results (Epoch 1)

| Benchmark | Accuracy |
|---|---|
| no_events | 7.3% |
| no_audio | 26.5% |
| random_events | 2.4% |
| static_audio | 26.0% |
| no_events_no_audio | 3.4% |
| metronome | 5.5% |
| time_shifted | 5.6% |
| advanced_metronome | 21.5% |

## Key Finding

Training was unstable — epoch 2 regressed on every metric. The audio/event imbalance persisted exactly as in exp 06: no_audio (26.5%) dominated no_events (7.3%). Three consecutive experiments of augmentation tuning (07, 08, 09) confirmed the same pattern at every augmentation level. The model trusts events over audio even when events are completely wrong (metronome=5.5% vs no_audio=26.5%). Dead zones in the prediction distribution (certain bins never predicted) were also observed. This experiment concluded the augmentation-based approach: the audio/event imbalance is architectural, not a training data problem. The single-path decoder architecture needs structural redesign to enforce audio as the primary signal.

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
