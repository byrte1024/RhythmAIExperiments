# Experiment 014 — Diffusion output head · Architecture

> **This document is self-contained.** Everything needed to reproduce
> this experiment is written inline: audio preprocessing, event
> encoding, sample construction, augmentations (with rates and ranges),
> the model layer-by-layer with shapes, the diffusion stack, loss math,
> training schedule, inference procedure (including the sampler step
> equations), dataset, and environment versions. No links, no "see
> elsewhere."

---

## Task

Given a mel-spectrogram window around a cursor plus the last 128
onsets as bin-offsets from the cursor, predict the bin offset from
the cursor to the next onset — or `STOP` if no onset falls within
the 500-bin (≈ 2.5 s) prediction range. Prediction is performed by
running an iterative denoising-diffusion sampler against the model's
cursor-token representation; the final sampler output is decoded
to an integer bin via argmax.

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
| `cursor_token` | `(B, 384)` | float32 | The trunk's cursor-token vector at audio token index 125. **Only output of `predict()`.** |
| Inference logits | `(B, 501)` | float32 | Output of the AR decoder's sampler (not directly the model). 500 bin-offset classes (0..499) + 1 STOP class (500 = `b_pred`). |

The model emits `cursor_token`; the decoder turns it into logits via
the diffusion sampler. Training mode also emits the structured
`(model_out, loss_target, t, x_t)` tuple consumed by `DiffusionLoss`.

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

```
count[c]   = number of training samples with target class c
weight[i]  = min(1.0, 1 / (count[target(i)] + 1) ** 0.5)
```

Each epoch draws `N` indices with replacement from the weight-
normalized distribution. Sampling weights are computed pre-aug
against the source target.

### Augmentations (training only)

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

---

## Trunk architecture

**Total parameters: 23,699,011 (23.70 M)** [computed by instantiating
`DiffusionDetector(config)` with `experiments/014-diffusion/config/model.json`
and summing `p.numel() for p in m.parameters()`]. Of these, 7,344,629
(7.34 M) are in the diffusion denoiser and the remainder (≈ 16.35 M)
are the `EventEmbeddingDetector` trunk.

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

### Transformer trunk

8 × `nn.TransformerEncoderLayer(d_model=384, nhead=8, dim_feedforward=1536,
dropout=0.1, activation="gelu", batch_first=True, norm_first=True)`,
with per-block FiLM conditioning applied AFTER each block:

```
for layer, film in zip(self.layers, self.film_layers):
    x = layer(x)                              # (B, 250, 384)
    x = film(x, cond_vec)                     # FiLM modulation
```

Output: `(B, 250, 384)`. Cursor token = `x[:, 125, :]`, shape `(B, 384)`.

The parent class's `head_norm`, `head_proj`, `head_smooth` modules
remain in the parameter dict but are **never called** during the
training-time `forward_diffusion()` or inference-time `predict()`
paths. They receive zero gradient.

---

## Diffusion head

Takes `cursor_token (B, 384)` and produces a final per-bin logit
distribution `(B, 501)` either by one denoiser forward (training) or
by running `T_inf` reverse-diffusion steps (inference).

### Notation

- `T = 64` — number of training timesteps.
- `t ∈ {0, …, T-1}` — discrete timestep. `t = 0` is the cleanest end
  (closest to a clean one-hot); `t = T-1` is the noisiest end (closest
  to the prior).
- `b = b_pred + 1 = 501` — number of bins.

### Cosine noise schedule

`f(t) = cos((t / T + s) / (1 + s) · π / 2) ²`, with `s = 0.008`.

```
ab(t)        = f(t) / f(0)                    # clamped to [0, 1]
beta(t)      = clamp(1 - ab(t) / ab(t-1), max_beta=0.999)
alpha(t)     = 1 - beta(t)
ab_cum(t)    = product of alpha(0..t)         # = ab(t) by construction
```

`ab_cum`, `1 - ab_cum`, `sqrt(ab_cum)`, `sqrt(1 - ab_cum)` are cached
on the schedule object at construction.

### x0-parameterized continuous Gaussian process

The forward process at timestep `t` is
```
x_t = sqrt(ab_cum(t)) * x_0 + sqrt(1 - ab_cum(t)) * eps,    eps ~ N(0, I)
```

with `x_0 = encode_x0(target_bin)`:
```
x_0 ∈ R^501,  x_0[target_bin] = 2.0,  x_0[other] = 0.0
                                  ^^^                    (x0_scale)
```

Decoding back to logits divides by the scale: `logits = x_0_hat / 2.0`.

### MLP denoiser

Inputs:
- `cursor_token` `(B, 384)`,
- `x_t`         `(B, 501)`,
- `t`           `(B,)` int64.

Build a sinusoidal time embedding from `t`:
```
half = time_embed_dim / 2 = 64
freqs = exp(-log(10000) * arange(half) / half)
emb   = concat([sin(t[:, None] * freqs), cos(t[:, None] * freqs)], dim=-1)
                                                                   # (B, 128)
```

Project: `time_proj = Linear(128, 256) → SiLU → Linear(256, 256)`,
yielding `(B, 256)`.

Concatenate the conditioning sources:
```
h = concat([cursor_token, time_proj_out, x_t], dim=-1)
                                                  # (B, 384 + 256 + 501 = 1141)
```

3-layer MLP:
```
h = Linear(1141, 1536) → SiLU → Dropout(0.1)
h = Linear(1536, 1536) → SiLU → Dropout(0.1)
h = Linear(1536, 1536) → SiLU → Dropout(0.1)
out = Linear(1536, 501)         # the predicted x_0 (already on x0_scale)
```

Output: `model_out (B, 501)` — interpreted as `x_0_hat * x0_scale`,
matching the `encode_x0` scale.

**Denoiser parameter count: 7,344,629.**

### Loss target (x0 parameterization)

`loss_target = x_0` (the encoded scaled one-hot). The denoiser is
trained to output `x_0` directly; per-sample distance is
`MSE(model_out, x_0)`.

### Training-time forward (`forward_diffusion`)

```
B  = cursor_token.size(0)
device = cursor_token.device

x_0    = encode_x0(target_bin)                  # (B, 501)
t      = randint(0, T, (B,))                    # uniform sample
eps    = randn_like(x_0)                        # (B, 501)
x_t    = sqrt(ab_cum[t]) * x_0
       + sqrt(1 - ab_cum[t]) * eps              # (B, 501)
mout   = denoiser(cursor_token, x_t, t)         # (B, 501)
ltgt   = x_0                                    # x0-param loss target
x0_hat = mout                                   # already x_0-shaped
logits = x0_hat / 2.0                           # decode_to_logits

return DiffusionModelOutput(
    logits=logits, cursor_token=cursor_token,
    model_out=mout, loss_target=ltgt, t=t, x_t=x_t,
)
```

`logits` here is a placeholder used only for the `argmax_match`
diagnostic metric; **it is not the inference-time prediction** (which
requires the full sampler).

### Inference-time forward (`predict`)

```
cursor_token = trunk(input)               # (B, 384)
return DiffusionModelOutput(
    logits=zeros(B, 501),                 # placeholder; decoder fills
    cursor_token=cursor_token,
)
```

The decoder runs the sampler on `cursor_token` to produce the real
logits.

---

## Loss

`DiffusionLoss` — MSE on the structured `DiffusionModelOutput`.

### Hyperparameters

| Param | Value |
|---|---|
| `loss_type` | `"mse"` |
| `snr_weighting` | `false` (reported as a diagnostic only) |
| `snr_gamma` | `5.0` |
| `stop_weight` | `1.5` |
| `n_t_buckets` | `4` |

### Forward

```
per_sample = ((mout - loss_target) ** 2).mean(dim=-1)        # (B,)
is_stop    = (target_bin == 500)
mult       = where(is_stop, 1.5, 1.0)                        # STOP weight
weighted   = per_sample * mult
loss       = weighted.mean()
```

### Diagnostic metrics reported per step

- `loss` — the headline scalar.
- `loss/per_t_q0..3` — per-sample loss bucketed into `n_t_buckets`
  equal-width quartiles of the sampled `t` (0..63). Reveals whether
  the model is good at the easy or hard end of the schedule.
- `loss/snr_weighted` — `(per_sample * snr_w).mean()` where
  `snr_w(t) = min(snr(t), 5.0) / snr(t)` and
  `snr(t) = ab_cum(t) / (1 - ab_cum(t))`. Always computed as a
  diagnostic, even when `snr_weighting=False`. The schedule's
  `alphas_cumprod()` is bound to the loss at trainer-construction
  time via `loss.bind_schedule(model.schedule.alphas_cumprod())`.
- `stop_rate` — fraction of training samples whose target is STOP.
- `argmax_match` — `(logits.argmax(-1) == target_bin).float().mean()`.
  **At-sampled-t** — diagnostic only; not directly comparable to
  inference-time argmax.

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
| Watched metric | `loss` (lower is better) |
| Checkpoint cadence | every eval — `latest.pt` rewritten; `best.pt` on new best |
| Step log cadence | every step into `metrics.jsonl` |
| Eval artifacts | per-eval AR-corpus output under `{run_dir}/infer_corpus/eval_{step}/` from the `InferCorpusHook`; train_noaug pass under `{run_dir}/eval_{step}/train_noaug/` |
| Seed | 42 |

### Auxiliary eval passes

- **`train_noaug`** — 5 % of the train split, fetched with
  augmentations OFF. Loss + diagnostics reported under
  `val/single/train_noaug/*`.
- **`benchmarks`** — 10 input-distortion modes on 5 % of val per
  eval: `normal`, `no_audio`, `no_future_audio`, `no_past_audio`,
  `static_audio`, `no_context`, `random_context`, `metronome`,
  `advanced_metronome`, `context_time_shifted`. Same diffusion
  loss, diagnostic only.
- **AR-corpus hook** — every eval runs the full AR pipeline
  (including the 16-step DDIM sampler) on 10 % of val charts using
  the LIVE model (no checkpoint reload), both `gt_cond` and
  `fixed_cond` conditioning modes.

---

## Inference (autoregressive)

Reconstructs at inference time:
- `DiffusionDetector` (the model).
- `DiffusionDecoder(b_pred=500, decode_strategy="argmax", n_samples=1, top_k_log=5,
   sampler_config=DDIMSamplerConfig(n_inference_steps=16, eta=0.0,
   timestep_spacing="linspace"))`.
- `DetectionARInputBuilder(a_bins=500, b_bins=500, c_events=128)`.
- `MelSampler(sample_rate=22000, n_fft=2048, hop_divisor=200, n_mels=80,
   f_min=20, f_max=8000, power=2.0, top_db=80.0)`.
- `FixedRateEventSampler(divisor=200)`.

After model load, `assemble_predictor` calls
`decoder.bind_model(model)` which constructs `DDIMSampler(sampler_config,
model.process, model.denoiser)`.

### DDIM sampler (16 steps, eta = 0)

Pre-compute the inference-time timestep schedule by `linspace`-spacing
16 steps over `[0, T-1] = [0, 63]`, producing a strictly-decreasing
sequence `t_15 > t_14 > … > t_0 = 0`.

Initialize `x = randn(1, 501)` (this is `x_T`).

For `i = 15, …, 0`:

```
t      = ts[i]                            # current step
t_prev = ts[i - 1] if i > 0 else -1       # next step (cleaner end)

ab_t      = ab_cum(t)
ab_prev   = ab_cum(t_prev) if t_prev >= 0 else 1.0     # clean

# Predict x_0 from the denoiser at this step:
mout      = denoiser(cursor_token, x, tensor([t]))     # (1, 501)
x0_hat    = mout                                       # x0-param

# Recover the implied noise:
eps_hat   = (x - sqrt(ab_t) * x0_hat) / sqrt(1 - ab_t)

# DDIM update with eta = 0 (deterministic):
sigma     = 0.0                # eta * sqrt((1 - ab_prev) * (1 - ab_t / ab_prev))
                               #   evaluates to 0 when eta = 0
mean_pred = sqrt(ab_prev) * x0_hat
          + sqrt(1 - ab_prev - sigma**2) * eps_hat
x         = mean_pred                                  # no stochastic term
```

After the loop, `logits = x / 2.0` (decode_to_logits, dividing by
`x0_scale`). With `n_samples = 1` this is the final logit vector
fed to the argmax. With `n_samples = N > 1`, the sampler is run
`N` times from independent `x_T` draws and the resulting per-draw
softmaxes are averaged: `logits_final = log(mean(softmax(logits_i)))`.

The decoder argmaxes (`decode_strategy = "argmax"`); a class index
of 500 is STOP, otherwise the AR loop emits an onset at
`cursor + cls`.

### Sampler ablation (post-run)

Run `cli.diffusion_sampler_ablation` against
`config/ablation_matrix.json`. Each variant overrides
`decoder.config.sampler_config.{n_inference_steps, eta}` and
`decoder.config.n_samples`. The matrix probes:

- DDIM 4 / 8 / 16 / 32 steps at `eta = 0` — quality vs steps knee.
- DDIM 16 steps at `eta = 1` — stochastic resampling.
- DDIM 16 steps at `eta ∈ {0, 1}` × `n_samples = 4` — multi-draw
  marginalization benefit.
- DDPM 64-step reference (full-schedule reverse process) — upper
  bound on this stack's quality.

### AR loop

1. Initialize `cursor = 0`, `past_onsets = []`, `step = 0`.
2. While `cursor < max_bin` and `step < max_events`:
   - Slice mel `[cursor - 500, cursor + 500)` with zero-padding.
   - Back-align up to 128 past onsets as `event_offsets` /
     `event_mask`.
   - Trunk forward → `cursor_token (1, 384)`.
   - DDIM sampler runs 16 reverse-diffusion steps starting from
     `x_T ~ N(0, I)`, each step calling the denoiser with
     `(cursor_token, x_i, t_i)`.
   - With `n_samples > 1`, repeat the sampler call N times and
     average softmaxes.
   - Argmax → class `cls`.
   - `cls == 500` (STOP) → `cursor += hop_bins_on_stop = 20`.
     Else → emit onset at `cursor + cls`, advance `cursor` there.
3. Return a new `Chart` with the accumulated onsets.

`hop_bins_on_stop = 20`, `max_events = 10000`, `min_onset_gap_bins = 1`,
`default_kind = "DON"`.

---

## Dataset

- Name: `taiko2_v1` (80-row log-mel features).
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
