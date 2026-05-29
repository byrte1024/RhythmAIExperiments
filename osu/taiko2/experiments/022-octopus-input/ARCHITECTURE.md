# Experiment NNN — {title} · Architecture

> **This document is self-contained.** It must describe everything needed to
> reproduce the experiment from scratch: data pipeline, model, loss,
> training schedule, inference procedure, environment. No cross-references
> to other experiments, other documents, or external URLs. Links and
> citations belong in `README.md`.

---

## Task

{One paragraph stating what the model predicts given what inputs, and at
what time resolution. Example: "Predict the bin offset from the current
cursor to the next onset, in 5-ms bins, up to 2.5 s ahead. Emit a STOP
class if no onset falls within that range."}

## Inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| mel | (B, 80, 1000) | float32 | Log-mel spectrogram; 80 bands, 1000 time frames at 5 ms/frame (500 past + 500 future). |
| event_offsets | (B, 128) | int64 | Last 128 onset bin positions relative to the cursor (negative = past). Padded at the start with zeros where fewer than 128 real events exist. |
| event_mask | (B, 128) | bool | True = padding, False = real event. |
| conditioning | (B, 3) | float32 | [density_mean, density_peak, density_std] from chart metadata. |

## Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| logits | (B, 501) | float32 | 500 bin-offset classes (0-499) + 1 STOP class (500). |

---

## Data pipeline

### Audio preprocessing

| Param | Value |
|---|---|
| Sample rate | 22 000 Hz |
| FFT size | 2048 samples |
| Hop length | 110 samples → 5.000 ms/frame |
| Mel bands | 80 |
| Frequency range | 20 Hz - 8 000 Hz |
| Power spectrum | power=2.0 |
| Amplitude→dB | top_db=80 |
| On-disk dtype | float16 |
| Served dtype | float32 |

### Event encoding

| Param | Value |
|---|---|
| Bin duration | 5.000 ms |
| Grid rate | 200 bins/second (exact integer) |
| Onset kinds retained | DON, KA, BIG_DON, BIG_KA, DRUMROLL, SPINNER |
| Bin index formula | floor(time_ms / bin_ms) |

Kind codes in the stored .npz arrays:

| Code | Kind |
|---|---|
| 0 | DON |
| 1 | KA |
| 2 | BIG_DON |
| 3 | BIG_KA |
| 4 | DRUMROLL |
| 5 | SPINNER |
| 6 | UNKNOWN |

### Sample construction

| Param | Value |
|---|---|
| Past audio bins (A_BINS) | 500 |
| Future audio bins (B_BINS) | 500 |
| Past events (C_EVENTS) | 128 |
| Future events (D_EVENTS) | 1 |
| Cursor at first event (ei=0) | max(0, evt[0] - 500) |
| Cursor at 1 ≤ ei ≤ N-1 | evt[ei - 1] |
| Cursor at ei = N | evt[N-1] |
| Min cursor bin (skip warmup) | 6000 |
| Allowed overlap forward | 500 bins |
| Allowed overlap backward | 500 bins |
| Past-event padding | start (oldest-first; front-padded) |
| Future-event padding | end (nearest-first; back-padded) |

### Train/val split

- Song-level grouping by `beatmapset_id`.
- Split seed: `42`.
- Ratios: `train = 0.9`, `val = 0.1`.

### Augmentations

Applied on the training split only; never on val.

| Augmentation | Probability | Parameters |
|---|---|---|
| Mel gain | 30% | ±2 dB uniform |
| Mel noise | 15% | Gaussian σ uniform in [0.1, 0.3] |
| Frequency roll | 15% | shift in {-3, …, +3} |
| SpecAug freq mask | 20% | 1 mask, width ≤ 10 bands |
| SpecAug time mask | 20% | 1 mask, width ≤ 30 frames |
| Event jitter | 100% | global ±3 bins + per-event ±3 * recency-scaled (1.0 at oldest, 2.0 at newest) |
| Event deletion | 5% | drop 1-2 random past events |
| Event insertion | 3% | add 1 synthetic event between two reals |
| Partial metronome (recent half) | 2% | replace with evenly-spaced events |
| Partial adv metronome (older half) | 2% | replace with dominant-gap-spaced events |
| Large time shift | 2% | ±50 bin shift on 2-4 recent events |
| Context truncation | 5% | keep only most recent 8-32 events |
| Conditioning jitter | 10% | ±2% on density_mean / peak / std |

Order in the augmentation pipeline: audio augs first (they don't depend
on event layout), then event augs, then conditioning jitter.

---

## Model architecture

Total parameters: {N}.

### 1. Conditioning MLP

```
conditioning (B, 3)
  → Linear(3, 64) → GELU
  → Linear(64, 64)
  → cond (B, 64)
```

### 2. Conv stem — mel → audio tokens

4× downsample; 1000 mel frames → 250 tokens.

```
mel (B, 80, 1000)
  → Conv1d(in=80, out=192, kernel=7, stride=2, padding=3)
  → GELU
  → GroupNorm(num_groups=1, num_channels=192)
  → Conv1d(in=192, out=384, kernel=7, stride=2, padding=3)
  → GELU
  → transpose → (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)       (d_model=384)
  → FiLM(cond)
  → x: (B, 250, 384)
```

Cursor sits at token index 125 (center).

### 3. Event embeddings

For each of 128 past events, build a feature vector from {features}:
{Describe each feature vector component explicitly, with dimensions.}

### 4. Transformer trunk

{N} layers of pre-norm encoder blocks with FiLM after each:

```
for each of {N} layers:
    x = TransformerEncoderLayer(
            d_model=384, nhead=8, dim_feedforward=1536,
            dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )(x)
    x = FiLM(cond)(x)
```

### 5. Output head

```
cursor = x[:, 125, :]               # (B, 384)
logits = Linear(384, 501)(LayerNorm(384)(cursor))
logits = logits + Conv1d_smooth(logits.unsqueeze(1)).squeeze(1)
    where Conv1d_smooth = Conv1d(1, 8, k=5, p=2) → GELU → Conv1d(8, 1, k=5, p=2)
```

### FiLM module (used throughout)

```
cond (B, 64)
  → Linear(64, 2 * d_model)
  → split into (γ, β) each (B, d_model)
output = x * (1 + γ.unsqueeze(1)) + β.unsqueeze(1)
```

---

## Loss

**Mixed hard-CE + trapezoid-soft-CE**, weighted by a STOP-class multiplier.

### Soft targets (trapezoid over log-ratio space)

For a ground-truth bin `t` (not STOP), the soft target over the 500 non-
STOP bins is built from `d_i = |log((i + 1) / (t + 1))|`:

- `d_i ≤ log(1 + 0.03)` → full credit (plateau).
- `log(1 + 0.03) < d_i ≤ log(1 + 0.20)` → linear ramp from 1 down to 0.
- `d_i > log(1 + 0.20)` → zero.
- **Floor:** any bin within ±2 frames of `t` always gets full credit,
  regardless of the ratio distance.

### STOP target

For STOP samples the soft target is a one-hot on class 500 multiplied
by `stop_weight = 1.5`.

### Hyperparameters

| Param | Value |
|---|---|
| hard_alpha (mix weight on hard CE) | 0.5 |
| good_pct (plateau width, ratio) | 0.03 |
| fail_pct (cutoff, ratio) | 0.20 |
| frame_tolerance (±frames with guaranteed credit) | 2 |
| stop_weight | 1.5 |

### Loss formula

`loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE`

where `hard_CE = F.cross_entropy(logits, target_int)` and `soft_CE`
uses the trapezoid-soft target described above.

---

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clip (max_norm) | 1.0 |
| Batch size | 48 |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Mixed precision | off |
| Balanced sampling | on (weights ∝ 1 / count^0.5) |
| Evals per epoch | 4 |
| Watched metric | `val/single/loss` (lower is better) |
| Checkpoint cadence | every eval; `latest.pt` + `best.pt` |
| Seed | 42 |

---

## Inference (autoregressive)

1. Initialize cursor at bin 0, past_onsets = empty.
2. At each step:
   a. Extract mel window [cursor - 500, cursor + 500]; pad with zeros
      past the audio edges.
   b. Gather up to 128 past onsets; encode as offsets from cursor.
   c. Build conditioning vector from user-supplied target density.
   d. Forward pass → 501 logits.
   e. Argmax (or temperature-sampled, see below) → class.
   f. If class == 500 (STOP): cursor += 20 bins (≈100 ms).
      Else: cursor += class; record an onset at the new cursor.
3. Stop when cursor ≥ end of audio OR 10 000 onsets emitted.

### Decoder options

| Decoder | Config |
|---|---|
| Argmax | deterministic |
| TopK + temperature | K, temperature |
| TopUnique (cluster within 5% ratio, then sample) | tolerance=0.05, temperature |

---

## Dataset

- Name: `{dataset name}`
- Source: osu!taiko .osz packs
- Charts: {N}
- Total onsets: {N}
- Val charts: {M}
- Train charts: {N - M}

---

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| torchaudio | 2.11.0.dev20260227+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | {card + vram} |
| OS | {os} |
| numpy | 2.4.2 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |

---

## Addenda

{Dated entries if the run deviated from the original spec. Example:}

> *2026-04-22: Mid-run, reduced batch size 48 → 32 after OOM at epoch 12.
> Learning rate held constant; eval cadence unchanged.*
