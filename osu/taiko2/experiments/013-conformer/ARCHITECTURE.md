# Experiment 013 — Conformer trunk · Architecture

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

### Sample construction

| Param | Value |
|---|---|
| Past audio bins (`a_bins`) | 500 |
| Future audio bins (`b_bins`) | 500 |
| Past events context (`c_events`) | 128 |
| Future events stored (`d_events`) | 1 |
| Min cursor bin filter | 6000 |
| Allowed overlap forward | 0 bins |
| Allowed overlap backward | 0 bins |
| Past-event padding | Start (oldest-first, back-aligned) |
| Future-event padding | End |
| Subsample | 1 (full dataset) |

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

Target derivation happens **after** augmentation, so the post-stretch
`cursor_offset` is what the adapter sees. An in-range target stretched
to ≥ 500 becomes STOP.

### Class-balanced sampling

Per-sample weight:

```
count[c]   = number of training samples with target class c
weight[i]  = min(1.0, 1 / (count[target(i)] + 1) ** 0.5)
```

Each epoch draws `N` indices with replacement from the weight-
normalized distribution. Sampling weights are computed pre-aug
against the source target.

### Augmentations (training only)

`TimeStretch` is inserted **first** in the post-sample pipeline so
every subsequent aug operates on the stretched sample.

| Order | Name | Probability | Parameters |
|---:|---|---|---|
| 1 | TimeStretch | 30 % | `max_scale = 1.4`; per-call `s ~ log-Uniform(1/1.4, 1.4)` |
| 2 | MelGainJitter | 30 % | `delta_dB ~ U(-2, +2)` added to every mel value |
| 3 | MelGaussianNoise | 15 % | additive Gaussian, `σ ~ U(0.1, 0.3)` per call |
| 4 | MelFreqJitter | 15 % | `shift ∈ {-3, …, +3}`, `np.roll` the mel bands |
| 5 | SpecAugFreq | 20 % | one mask, width uniform in `[1, 10]` bands |
| 6 | SpecAugTime | 20 % | one mask on either past or future side (50/50), width `[1, 30]` frames |
| 7 | EventJitter | 100 % | global `shift ∈ {-3, …, +3}` + per-event noise `{-3, …, +3} × scale`, where `scale` is linear from 1.0 (oldest) to 2.0 (newest) |
| 8 | EventDropout | 5 % | drop 1–2 random real past events |
| 9 | EventInsertion | 3 % | add 1 synthetic event between two reals (uniform random offset inside that range) |
| 10 | PartialMetronome | 2 % | replace recent half with events spaced uniformly at `gap ∈ [10, 80]` bins |
| 11 | PartialAdvMetronome | 2 % | replace older half with events spaced at the sample's dominant gap (mode of diffs quantized to steps of 3), jittered ±1 bin |
| 12 | LargeTimeShift | 2 % | shift 2–4 most recent events by `∈ {-50, …, +50}` bins |
| 13 | ContextTruncation | 5 % | keep only the 8–32 most recent real events |
| 14 | ConditioningJitter | 10 % | each of `density_mean / peak / std` multiplied independently by `U(0.98, 1.02)` |

### TimeStretch — complete semantics

**Per-call draw:** `s = exp(u)` where `u ~ Uniform(-log 1.4, log 1.4)`.
Log-uniform so `s = 0.75` and `s = 1.33` are equally likely.

**Audio.** The mel window is treated as a single `(80, a_bins + b_bins)
= (80, 1000)` tensor with the cursor pinned at frame index `a_bins =
500`. For each output frame `t`, the source index is
`src_idx(t) = a_bins + (t - a_bins) / s`. Linear interpolation between
`floor(src_idx)` and `floor(src_idx) + 1` produces the output value;
the result is then split back into `(80, 500) + (80, 500)`.

When `s < 1` (speed up), some output frames require source indices
outside `[0, total - 1]` — those positions are zero-padded.

When `s > 1` (slow down), the required source range is a subset of
the original window — no padding is needed.

**Events.** For each past / future `RelativeOnset`, the new
`cursor_offset` is `round(cursor_offset * s)`. Past events whose new
offset falls outside `[-a_bins, 0]` are dropped. Survivors are
dedupe'd — two events that post-round onto the same integer offset
collapse to one, keeping the older. The list is then re-padded at
the start so its length stays `c_events`.

**Determinism.** The `TimeStretch` instance owns a `random.Random`
seeded from `TrainerConfig.seed`; the per-sample draw is deterministic
under a fixed trainer seed.

---

## Model architecture

**Total parameters: 29,474,894 (29.47 M)** [computed by instantiating
`ConformerDetector(config)` with `experiments/013-conformer/config/model.json`
and summing `p.numel() for p in model.parameters()`].

### Conditioning MLP

`(B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → (B, 64)`.

### Conv stem

Input `(B, n_mels=80, T=1000)`:
1. `Conv1d(80, 192, kernel_size=7, stride=2, padding=3)` → `(B, 192, 500)`.
2. `GroupNorm(num_groups=8, num_channels=192)`.
3. `Conv1d(192, 384, kernel_size=7, stride=2, padding=3)` → `(B, 384, 250)`.
4. Transpose to `(B, 250, 384)`.
5. Add sinusoidal positional embedding (`d_model=384`, position indices
   `0..249`).
6. `LayerNorm(384)`.
7. FiLM conditioning: `gamma, beta = Linear(64, 2*384)(cond_vec).chunk(2, dim=-1)`,
   apply as `x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)`. The FiLM
   `Linear` is **zero-initialized** so it is identity at training start.

Output: `(B, 250, 384)` audio tokens. Cursor token index: `a_bins // 4 = 125`.

### Event embeddings

For each of the 128 past-event slots, build a 5-component feature:
- `presence`: a learned parameter `(1, 384)`, initialized `N(0, 0.02²)`,
  broadcast to every slot.
- `gap_before[i]`: `|offsets[i] - offsets[i-1]|`, clamped to ≥ 1; for
  `i = 0` use placeholder = 50. Sinusoidal embed in dim `d_model`.
- `gap_after[i]`: `|offsets[i+1] - offsets[i]|`, clamped to ≥ 1. For
  the LAST valid event in each row, `gap_after` would leak the target
  offset; overwrite with that event's `gap_before` as a proxy.
  Sinusoidal embed in dim `d_model`.
- `gap_ratio_before[i]`: `gap_before[i-1] / gap_before[i]`, clamped to
  `[0.1, 10.0]` then × 50. Sinusoidal embed.
- `gap_ratio_after[i]`: `gap_after[i+1] / gap_after[i]`, same clamp + scale.
  Sinusoidal embed.

Concatenate to a `(B, 128, 5 * 384) = (B, 128, 1920)` tensor and project:
`Linear(1920, 384) → GELU → Linear(384, 384)`. Result: `(B, 128, 384)`.

### Audio + event mixer

Compute per-event token positions:
```
mel_frame[i] = a_bins + event_offsets[i]      # offsets are ≤ 0
token_pos[i] = mel_frame[i] // 4              # conv stride 4
in_window[i] = (not event_mask[i]) AND 0 ≤ token_pos[i] < cursor_token
```

For each batch row, scatter-add the per-event embeddings into the audio
token sequence at their `token_pos` positions, restricted to the past
audio side (`token_pos < 125`). Output shape unchanged: `(B, 250, 384)`.

### Conformer trunk (THE ONLY DIFFERENCE FROM #007)

8 × `ConformerBlock(d_model=384, n_heads=8, ffn_dim=1536, depthwise_kernel=31, dropout=0.1, use_group_norm=True)`,
with per-block FiLM conditioning applied AFTER each block.

Each block:

```
x_in = x                                # (B, 250, 384)

# Macaron FFN-1, half-residual
y = LayerNorm(x_in)
y = Linear(384, 1536)(y)
y = SiLU(y)                             # Swish
y = Dropout(0.1)(y)
y = Linear(1536, 384)(y)
y = Dropout(0.1)(y)
x1 = x_in + 0.5 * y                     # (B, 250, 384)

# Multi-head self-attention (pre-norm + post-residual dropout)
qkv = LayerNorm(x1)
attn_out, _ = MultiheadAttention(embed=384, heads=8, dropout=0.1,
                                 batch_first=True)(qkv, qkv, qkv,
                                                   need_weights=False)
x2 = x1 + Dropout(0.1)(attn_out)        # (B, 250, 384)

# Convolution module
y = LayerNorm(x2)
y = transpose(y, -1, -2)                # (B, 384, 250)
y = Conv1d(384, 768, kernel_size=1)(y)  # pointwise expand
y = GLU(dim=1)(y)                       # halves channels → (B, 384, 250)
y = Conv1d(384, 384, kernel_size=31, groups=384, padding=15,
          bias=False)(y)                # depthwise, RF = 31 tokens = 620 ms
y = GroupNorm(num_groups=1, num_channels=384)(y)
y = SiLU(y)                             # Swish
y = Conv1d(384, 384, kernel_size=1)(y)  # pointwise
y = Dropout(0.1)(y)
y = transpose(y, -1, -2)                # (B, 250, 384)
x3 = x2 + y

# Macaron FFN-2, half-residual
y = LayerNorm(x3)
y = Linear(384, 1536)(y)
y = SiLU(y)
y = Dropout(0.1)(y)
y = Linear(1536, 384)(y)
y = Dropout(0.1)(y)
x4 = x3 + 0.5 * y

# Final LayerNorm
x_out = LayerNorm(x4)                   # (B, 250, 384)
```

After each block, FiLM is applied:
```
gamma, beta = FiLM_Linear(64, 2*384)(cond_vec).chunk(2, -1)   # zero-init
x_out = x_out * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
```

Stacked 8 times. Output: `(B, 250, 384)`.

### Output head

```
cursor_tok = x[:, 125, :]                          # (B, 384)
h = LayerNorm(cursor_tok)
logits_raw = Linear(384, 501)(h)                   # (B, 501)
smooth = Conv1d(1, 8, kernel_size=5, padding=2)(logits_raw.unsqueeze(1))
smooth = GELU(smooth)
smooth = Conv1d(8, 1, kernel_size=5, padding=2)(smooth)
logits = logits_raw + smooth.squeeze(1)            # (B, 501)
```

The conv-smooth is an additive residual over the bin axis; it
slightly spreads adjacent-bin probability mass.

### FiLM module

```
class FiLM(nn.Module):
    def __init__(self, cond_dim=64, d_model=384):
        self.proj = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
```

Zero-init means the module is identity at start; FiLM gradually learns
to modulate as training proceeds.

### Why Conformer-specific values

- `d_model = 384`: matched to #007 baseline for variable isolation.
- `n_heads = 8` (head_dim 48): matched to #007.
- `ffn_dim = 1536 = 4 × d_model`: macaron expansion, paper-canonical.
- `depthwise_kernel_size = 31` (odd): canonical Conformer kernel; covers
  31 tokens × 20 ms / token = 620 ms of receptive field per block;
  ~2 beats at 200 BPM, ~half a beat at 60 BPM.
- `use_group_norm = True`: GroupNorm with `num_groups=1` (LayerNorm-
  equivalent over channels per time step). Paper used BatchNorm; modern
  audio-paper convention favors GroupNorm/LayerNorm for small-batch
  stability and eval-time consistency.
- `dropout = 0.1`: matched to #007.
- `convolution_first = False`: attention before convolution per block,
  paper default.

---

## Loss

`OnsetLoss` — mixed hard + trapezoid-soft CE with a ±2-frame floor and
STOP weighting.

### Hyperparameters

| Param | Value |
|---|---|
| `hard_alpha` | 0.5 |
| `good_pct` (plateau width) | 0.03 |
| `fail_pct` (ramp-to-0 cutoff) | 0.20 |
| `frame_tolerance` | 2 |
| `stop_weight` | 1.5 |

### Forward

`loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE`, then per-
sample multiplied by `stop_weight` where the target is STOP.

Hard CE: `F.cross_entropy(logits, target, reduction="none")`.

Soft CE: per non-STOP target `t`, build a trapezoid over bins `0..499`:

```
d_i           = |log((i + 1) / (t + 1))|
log_good      = log(1 + 0.03)  ≈ 0.02956
log_fail      = log(1 + 0.20)  ≈ 0.18232
ratio_weight  = clip((log_fail - d_i) / (log_fail - log_good), 0, 1)

frame_dist    = |i - t|
frame_weight  = clip((2 + 1 - frame_dist) / (2 + 1), 0, 1)

weight_i      = max(ratio_weight, frame_weight)
soft_target   = weight / sum(weight)
soft_CE       = -(soft_target * log_softmax(logits)).sum(-1)
```

For STOP targets the soft distribution is a pure one-hot at index 500.
Per-sample multiplier `= stop_weight when target == 500 else 1.0`.

Returned metrics: `loss`, `hard_ce`, `soft_ce`, `stop_rate`.

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
| Eval artifacts | heatmap / distributions / ratio_error / error_hist / ratio_hit / metronome PNG + raw (.npy or .npz) under `{run_dir}/eval_{step}/`; **also** under `{run_dir}/eval_{step}/train_noaug/` for the augmentation-off pass |
| Seed | 42 |

### Metrics reported per eval

All computed by `OnsetMetric` on the full val pass and additionally on
the 5 %-of-train augmentation-off pass + 5 %-of-val × 10 benchmark-mode
passes:

- `onset/exact`, `onset/fhit`, `onset/fgood`, `onset/fmiss`
- `onset/rhit`, `onset/rgood`, `onset/rmiss`
- `onset/hit`, `onset/good`, `onset/miss`
- `onset/ihit`, `onset/igood`, `onset/imiss`
- `onset/stop_precision`, `onset/stop_recall`, `onset/stop_f1`
- `onset/frame_err_mean`, `onset/frame_err_median`, `onset/frame_err_p90`
- `onset/pred_stop_rate`, `onset/pred_stop_fp_rate`
- `onset/n_total`, `onset/n_nonstop`, `onset/n_stop_target`

Thresholds: FHIT `≤ 2 bins`, FGOOD `≤ 7 bins`, RHIT `log ratio
< log(100/97)`, RGOOD `< log(100/90)`.

### Auxiliary eval passes

- **`train_noaug`** — 5 % of the train split, fetched with augmentations
  OFF. Metrics prefixed `val/single/train_noaug/*`. Artifacts saved
  under `{run_dir}/eval_{step}/train_noaug/`.
- **`benchmarks`** — 10 input-distortion modes on 5 % of val per eval:
  `normal`, `no_audio`, `no_future_audio`, `no_past_audio`,
  `static_audio`, `no_context`, `random_context`, `metronome`,
  `advanced_metronome`, `context_time_shifted`.
- **AR corpus hook** — every eval runs AR inference on 10 % of val
  charts using the LIVE model (no checkpoint reload), both `gt` and
  `fixed` conditioning modes.

---

## Inference (autoregressive)

Reconstructs at inference time:
- `ConformerDetector` (the model).
- `ArgmaxDecoder(b_pred=500)` — raw argmax over all 501 logits.
- `DetectionARInputBuilder(a_bins=500, b_bins=500, c_events=128)`.
- `MelSampler(sample_rate=22000, n_fft=2048, hop_divisor=200, n_mels=80, ...)`.

AR loop:
1. Initialize `cursor = 0`, `past_onsets = []`, `step = 0`.
2. While `cursor < max_bin` and `step < max_events`:
   - Slice mel `[cursor - 500, cursor + 500)` with zero-padding.
   - Back-align up to 128 past onsets as `event_offsets` / `event_mask`.
   - Forward → 501 logits → `ArgmaxDecoder.decode` → `ARDecision`.
   - `STOP` → `cursor += hop_bins_on_stop` (default 20).
   - Else → emit onset at `cursor + bin_offset`, advance `cursor` there.
3. Return a new `Chart` with the accumulated onsets.

---

## Dataset

- Name: `taiko2_v1` (same as #007; 80-row log-mel features).
- Source: parsed osu!taiko `.osz` packs.
- Charts: 10,048.
- Total onsets: 6,934,185.
- Train split (seed 42, 90 %): ≈ 9,090 charts.
- Val split (seed 42, 10 %): ≈ 958 charts.
- With `allowed_overlap_forward = allowed_overlap_back = 0`, every
  event cursor is a valid sample. `min_cursor_bin = 6000` filters
  early-silence cursors.
- **Subsample = 1 (full dataset).**

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
