# Experiment 002 — exp 45 full recreation · Architecture

> **This document is self-contained.** Everything needed to reproduce
> this experiment is written inline: audio preprocessing, event
> encoding, sample construction, augmentations (with rates and ranges),
> the model layer-by-layer with shapes, loss math, training schedule,
> inference procedure, dataset, and environment versions. No links,
> no "see elsewhere."

---

## Task

Given a mel-spectrogram window around a cursor plus the last 128
onsets as bin-offsets from the cursor, predict the bin offset from
the cursor to the next onset — or `STOP` if no onset falls within
the 500-bin (≈ 2.5 s) prediction range.

One cursor per training sample. The cursor is placed one of three
ways (see **Sample construction** below) and the target is derived
from the `d_events=1` next-event slot.

## Inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `mel` | `(B, 80, 1000)` | float32 | Log-mel spectrogram, 80 bands, 1000 frames = 500 past (`a_bins`) + 500 future (`b_bins`) at 5.000 ms / frame. Zero-padded at chart edges. |
| `event_offsets` | `(B, 128)` | int64 | Last 128 onset bin offsets relative to the cursor, back-aligned (newest at index 127, leading slots padded with zeros). Offsets are ≤ 0. |
| `event_mask` | `(B, 128)` | bool | True ⇒ slot is padding. |
| `conditioning` | `(B, 3)` | float32 | `[density_mean, density_peak, density_std]` from the chart metadata. |

## Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `logits` | `(B, 501)` | float32 | 500 bin-offset classes (indices 0..499) + 1 STOP class (index 500 = `b_pred`). |

---

## Data pipeline

### Audio preprocessing

| Param | Value |
|---|---|
| Sample rate | 22 000 Hz |
| FFT size | 2048 samples |
| Hop length | 110 samples → **5.000 ms / frame** |
| Mel bands | 80 |
| Frequency range | 20 Hz – 8 000 Hz |
| Power spectrum | power = 2.0 |
| Amplitude → dB | `top_db = 80` |
| On-disk dtype | float16 |
| Served dtype | float32 |

### Event encoding

| Param | Value |
|---|---|
| Bin duration | 5.000 ms (event sampler divisor = 200) |
| Grid rate | 200 bins / second (exact integer) |
| Onset kinds retained | DON, KA, BIG_DON, BIG_KA, DRUMROLL, SPINNER |
| Bin-index formula | `floor(time_ms / bin_ms)` |

Kind-code table in the on-disk `.npz`:

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
| Past audio bins (`a_bins`) | 500 |
| Future audio bins (`b_bins`) | 500 |
| Past events context (`c_events`) | 128 |
| Future events stored (`d_events`) | 1 |
| Cursor at first event (`ei = 0`) | `max(0, bins[0] - 500)` |
| Cursor at `1 ≤ ei ≤ N-1` | `bins[ei - 1]` |
| Cursor at trailing `ei = N` | `bins[N-1]` |
| Min cursor bin filter | 6000 (taiko1 default) |
| Allowed overlap forward | 500 bins |
| Allowed overlap backward | 500 bins |
| Past-event padding | Start (oldest-first, back-aligned) |
| Future-event padding | End |
| **Subsample** | **1 (full dataset)** |

### Train / val split

- Song-level grouping by `beatmapset_id`.
- Ratios: `train = 0.9`, `val = 0.1`.
- Seed: 42.

### Target derivation

```
stop_idx = b_pred  # = 500
if future_events_mask[0]:
    target = stop_idx
elif future_events[0].cursor_offset < 0 or future_events[0].cursor_offset >= b_pred:
    target = stop_idx
else:
    target = future_events[0].cursor_offset
```

### Class-balanced sampling

Per-sample weight:

```
count[c]   = number of training samples with target class c
weight[i]  = min(1.0, 1 / (count[target(i)] + 1) ** 0.5)   # sqrt inverse
```

Each epoch draws `N` indices with replacement from the weight-
normalized distribution. `N` = number of training samples.

### Augmentations (training only)

Applied to each sample via a post-sample pipeline. Rates and
parameters are the exact exp 45 set.

| Name | Probability | Parameters |
|---|---|---|
| MelGainJitter | 30 % | `delta_dB ~ U(-2, +2)` added to every mel value |
| MelGaussianNoise | 15 % | additive Gaussian, `σ ~ U(0.1, 0.3)` per call |
| MelFreqJitter | 15 % | `shift ∈ {-3, …, +3}`, `np.roll` the mel bands |
| SpecAugFreq | 20 % | one mask, width uniform in `[1, 10]` bands |
| SpecAugTime | 20 % | one mask on either past or future side (50/50), width `[1, 30]` frames |
| EventJitter | 100 % | global `shift ∈ {-3, …, +3}` + per-event noise `{-3, …, +3} × scale`, where `scale` is linear from 1.0 (oldest) to 2.0 (newest) |
| EventDropout | 5 % | drop 1–2 random real past events |
| EventInsertion | 3 % | add 1 synthetic event between two reals (uniform random offset inside that range) |
| PartialMetronome | 2 % | replace recent half with events spaced uniformly at `gap ∈ [10, 80]` bins |
| PartialAdvMetronome | 2 % | replace older half with events spaced at the sample's dominant gap (mode of diffs quantized to steps of 3), jittered ±1 bin |
| LargeTimeShift | 2 % | shift 2–4 most recent events by `∈ {-50, …, +50}` bins |
| ContextTruncation | 5 % | keep only the 8–32 most recent real events |
| ConditioningJitter | 10 % | each of `density_mean / peak / std` multiplied independently by `U(0.98, 1.02)` |

Pipeline order: audio augs first, then event augs, then conditioning jitter.

---

## Model architecture

**Total parameters: ≈ 16.5 M** (exact value printed on `model.n_params`
at construction; verified in the model's unit tests in range 15–18 M).

### 1. Conditioning MLP

```
conditioning (B, 3)
  → Linear(3, 64)  → GELU
  → Linear(64, 64)
  → cond (B, 64)
```

### 2. Conv stem — mel → audio tokens (4× downsample)

```
mel (B, 80, 1000)
  → Conv1d(in=80,  out=192, k=7, s=2, p=3)  → GELU
  → GroupNorm(num_groups=1, num_channels=192)
  → Conv1d(in=192, out=384, k=7, s=2, p=3)  → GELU
  → transpose      →  (B, 250, 384)
  → LayerNorm(384)
  → + SinusoidalPosEmb(positions 0..249)     (d_model = 384)
  → FiLM(cond)
  → x: (B, 250, 384)
```

Cursor sits at token index `a_bins // 4 = 125`.

### 3. Event embeddings (injected into audio tokens)

For each of the 128 event slots, build a `d_model`-dimensional
feature from these concatenated parts (5 × 384 = 1920 total, projected
back down to 384):

1. **Presence** — a single learned parameter `(1, d_model)` broadcast to every slot.
2. **`gap_before[i]`** — `|event_offsets[i] - event_offsets[i-1]|`, clamped to ≥ 1; for `i = 0` a placeholder value of 50 is used. Encoded via sinusoidal position embedding at `d_model`.
3. **`gap_after[i]`** — `|event_offsets[i+1] - event_offsets[i]|`, clamped to ≥ 1. For the **last valid** event in each row `gap_after` would span the target gap; we overwrite it with that event's `gap_before` as a structure-preserving proxy.
4. **`gap_ratio_before[i]`** — `gap_before[i-1] / gap_before[i]`, clamped to `[0.1, 10.0]`, multiplied by 50. Sinusoidal-encoded.
5. **`gap_ratio_after[i]`**  — `gap_after[i+1]  / gap_after[i]`, same clamp + 50× scale. Sinusoidal-encoded.

```
parts   = concat([presence, gb, ga, rb, ra], dim=-1)        # (B, 128, 5 * 384)
event_embs = Linear(5*384, 384) → GELU → Linear(384, 384)   # (B, 128, 384)
```

Mel-frame → token-index mapping (critical):

```
mel_frame[i] = a_bins + event_offsets[i]      # event_offsets ≤ 0
token_pos[i] = mel_frame[i] // 4              # conv stride is 4
in_window[i] = valid[i] AND  0 ≤ token_pos[i] < cursor_token  (= 125)
```

For every row `b`, every in-window event `i`:

```
x[b].scatter_add_(dim=0, index=token_pos[b, i], src=event_embs[b, i])
```

Out-of-window events (those with `token_pos < 0`) are ignored in this
experiment (no virtual tokens).

### 4. Transformer trunk — 8 layers, per-layer FiLM

```
for layer in range(8):
    x = TransformerEncoderLayer(
            d_model       = 384,
            nhead         = 8,
            dim_feedforward = 1536,           # 4 × d_model
            dropout       = 0.1,
            activation    = "gelu",
            batch_first   = True,
            norm_first    = True,
        )(x)
    x = FiLM(cond)(x)
```

### 5. Output head

```
cursor_tok = x[:, 125, :]                     # (B, 384)
logits     = Linear(384, 501)(LayerNorm(384)(cursor_tok))
logits     = logits + Conv1d_smooth(logits.unsqueeze(1)).squeeze(1)
    where Conv1d_smooth = Conv1d(1, 8, k=5, p=2) → GELU → Conv1d(8, 1, k=5, p=2)
```

### FiLM module (used throughout)

```
cond (B, 64)
  → Linear(64, 2 * d_model)      # weight + bias initialized to zero
  → split → (γ, β), each (B, d_model)
out = x * (1 + γ.unsqueeze(1)) + β.unsqueeze(1)
```

Zero-init means FiLM starts as the identity — no destabilization from
random conditioning modulation in early training.

---

## Loss

`loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE`, then per-
sample multiplied by `stop_weight` where the target is STOP.

### Hyperparameters

| Param | Value |
|---|---|
| `hard_alpha` | 0.5 |
| `good_pct` (plateau width) | 0.03 |
| `fail_pct` (ramp-to-0 cutoff) | 0.20 |
| `frame_tolerance` | 2 |
| `stop_weight` | 1.5 |

### Hard CE

`F.cross_entropy(logits, target, reduction="none")` → `(B,)`.

### Soft CE

Per non-STOP target `t` build a trapezoid distribution over the 500
non-STOP bins:

```
d_i           = |log((i + 1) / (t + 1))|                  # ratio distance
log_good      = log(1 + good_pct)     ≈ 0.02956           # plateau edge
log_fail      = log(1 + fail_pct)     ≈ 0.18232           # zero edge
ratio_weight  = clip((log_fail - d_i) / (log_fail - log_good), 0, 1)

frame_dist    = |i - t|
frame_weight  = clip((frame_tolerance + 1 - frame_dist) / (frame_tolerance + 1), 0, 1)

weight_i      = max(ratio_weight, frame_weight)          # ±2-frame floor
soft_target   = weight / sum(weight)                      # renormalized to a proper distribution
```

For STOP targets the soft distribution is a pure one-hot at index 500.

```
soft_CE = -(soft_target * log_softmax(logits)).sum(-1)    # (B,)
```

### STOP weighting

`per_sample_multiplier = stop_weight when target == 500 else 1.0`.

Returned `LossResult.metrics` carries four detached floats for logging:
`loss`, `hard_ce`, `soft_ce`, `stop_rate`.

---

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.01 |
| Gradient clip (max norm) | 1.0 |
| Batch size | 64 |
| Epochs | 50 |
| LR scheduler | CosineAnnealingLR (`T_max = steps_per_epoch × epochs`) |
| Mixed precision | off |
| Balanced sampling | on (weights ∝ 1 / (count + 1) ^ 0.5) |
| Evals per epoch | 4 |
| Watched metric | `onset/miss` (lower is better) |
| Checkpoint cadence | every eval — `latest.pt` rewritten; `best.pt` on new best |
| Step log cadence | every step into `metrics.jsonl` |
| Eval artifacts | heatmap / distributions / ratio_error / error_hist / ratio_hit / metronome PNG + raw (.npy or .npz) under `{run_dir}/eval_{step}/` |
| Seed | 42 |

### Metrics reported per eval

All computed by `OnsetMetric` on the full val pass:

- `onset/exact`, `onset/fhit`, `onset/fgood`, `onset/fmiss`
- `onset/rhit`, `onset/rgood`, `onset/rmiss`
- `onset/hit`, `onset/good`, `onset/miss`
- `onset/ihit`, `onset/igood`, `onset/imiss` (only when `all_future_bins` has ≥ 1 real event)
- `onset/stop_precision`, `onset/stop_recall`, `onset/stop_f1`
- `onset/frame_err_mean`, `onset/frame_err_median`, `onset/frame_err_p90`
- `onset/pred_stop_rate`, `onset/n_total`, `onset/n_nonstop`, `onset/n_stop_target`

Thresholds:

- **FHIT**: `|pred − target| ≤ 2` bins.
- **FGOOD**: `|pred − target| ≤ 7` bins.
- **RHIT**: `|log((pred + 1) / (target + 1))| < log(100/97)` (≈ 3 %).
- **RGOOD**: same ratio check with `log(100/90)` (≈ 10 %).

---

## Inference (autoregressive)

Not run in this experiment. The trained checkpoint writes config
snapshots sufficient for an inference script to reconstruct:
`EventEmbeddingDetector` + `ArgmaxDecoder(b_pred=500)` +
`DetectionARInputBuilder(a_bins=500, b_bins=500, c_events=128)`.

AR loop (from `AutoregressivePredictor.predict`):

1. Initialize `cursor = 0`, `past_onsets = []`, `step = 0`.
2. While `cursor < max_bin` and `step < max_events`:
   - Slice mel `[cursor - 500, cursor + 500)` with zero-padding.
   - Back-align up to 128 past onsets as `event_offsets` / `event_mask`.
   - Forward → 501 logits → `ArgmaxDecoder.decode` → `ARDecision`.
   - `STOP` (empty `bin_offsets`) → `cursor += hop_bins_on_stop` (default 20).
   - Else → emit onset at `cursor + bin_offset`, advance `cursor` there.
3. Return a new `Chart` with the accumulated onsets.

---

## Dataset

- Name: `taiko2_v1`.
- Source: parsed osu!taiko `.osz` packs (`osu/taiko/charts/`).
- Charts: 10 048.
- Total onsets: 6 934 185.
- Train split (seed 42, 90 %): ≈ 9 090 charts.
- Val split (seed 42, 10 %): ≈ 958 charts.
- Raw training samples after overlap filter (500 / 500) and
  `min_cursor_bin = 6000`: ≈ 360 k (the 6000-bin filter trims the
  early-silence portion).
- **No subsample: ≈ 360 k training samples**.
- Val samples (same filter, full split): ≈ 36 k.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.13.13 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| torchaudio | 2.11.0.dev20260227+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB, compute 12.0) |
| OS | Windows 11 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.3 |

---

## Addenda

_(None before the run.)_
