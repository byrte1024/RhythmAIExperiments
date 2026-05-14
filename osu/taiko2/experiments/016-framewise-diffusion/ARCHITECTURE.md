# Experiment 016 — Framewise activation-map diffusion · Architecture

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
onsets as bin-offsets from the cursor, predict an **activation map**
over the next 500 bins (= 2.5 seconds at 5 ms/bin). Each bin `b ∈
[0, 500)` has a predicted activation `M_0_hat[b] ∈ [0, 1]` ≈ "is
there an onset at this bin offset from the cursor". A threshold-based
decoder emits all bins above the threshold as onsets. No STOP class
— an empty positive set causes the autoregressive cursor to advance
by a fixed `hop_bins_on_stop = 20` bins ≈ 100 ms.

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
| `cursor_token` | `(B, 384)` | float32 | The trunk's cursor-token vector at audio token index 125. |
| `audio_features` | `(B, 125, 384)` | float32 | The future-half audio token sequence (indices 125..249), used as per-bin conditioning for the denoiser. |
| Predicted activation map (inference) | `(B, 500)` | float32 in [0, 1] | The output of the AR decoder's diffusion sampler after `T_inf` DDIM steps and clipping. A threshold-based decoder turns this into a list of bin offsets to emit as onsets. |

The model emits `(cursor_token, audio_features)` from `predict()`;
the decoder runs the diffusion sampler against both to produce the
final activation map. Training mode emits the structured
`(model_out, loss_target, t, x_t, audio_features, cursor_token)`
tuple consumed by `FramewiseDiffusionLoss`.

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
| Future events stored (`d_events`) | 100 (changed from #015's 1; framewise needs all future events in the window) |
| Min cursor bin filter | 6000 |
| Allowed overlap forward | 0 bins |
| Allowed overlap backward | 0 bins |
| Past-event padding | Start (oldest-first, back-aligned) |
| Future-event padding | End — invalid slots get bin_offset = -1 sentinel |
| Subsample | 1 (full dataset) |

### Train / val split

- Song-level grouping by `beatmapset_id`.
- Ratios: `train = 0.9`, `val = 0.1`.
- Seed: 42.

### Target derivation

For each sample, gather the future-events list. Each future onset
has a bin offset `b ∈ ℤ` relative to the cursor. Filter to
`b ∈ [0, 500)` (in-window). Build two target tensors:

```
M_target_binary[b] = 1.0 if any GT onset at bin b, else 0.0     # (B, 500)
M_target_smoothed[b] = max over GT onsets g of
                           exp(-(b - g_bin)² / (2 * sigma²))    # (B, 500)
where sigma = 2 frames (= 10 ms).
```

The smoothed target is what the model is trained against (MSE
target); the binary target is used for metric computation
(precision/recall/F1) and decoder thresholding.

Edge cases:
- No future events in window → all-zero target.
- Multiple onsets at the same bin → still 1.0 (max).
- Adjacent onsets (e.g., bin 100 + bin 101) → smoothed map is
  pointwise max, peaks at both bins remain 1.0.

### Class-balanced sampling

**Disabled** for #016. Class balancing in #002/#005/#007/#014/#015
was over the next-target-bin distribution (concentrated mass at
small bin offsets); under framewise output every sample produces
a B_PRED-dim target so there is no analogous discrete class to
balance.

Random uniform sampling within each train epoch.

### Augmentations (training only)

Identical to #007/#014/#015 — 14 augmentations:

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

Note: event-level augmentations (EventJitter, EventDropout, etc.)
operate on PAST events. The future events used to build M_target
are jittered only by TimeStretch (which moves all events by the
same factor).

---

## Trunk architecture

**Total parameters: 22,265,519 (22.27 M)** [computed by
instantiating `FramewiseDiffusionDetector(config)` with
`experiments/016-framewise-diffusion/config/model.json` and summing
`p.numel() for p in m.parameters()`]. Of these, 5,911,137 (5.91 M)
are in the diffusion denoiser and the remainder (≈ 16.35 M) are
the `EventEmbeddingDetector` trunk. The 1D Conv denoiser is
lighter than the MLP denoiser used in earlier diffusion experiments
because the convolutional inductive bias replaces dense parameter
sharing.

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

Output: `(B, 250, 384)`. From this:
- `cursor_token = x[:, 125, :]`, shape `(B, 384)`.
- `audio_features = x[:, 125:250, :]`, shape `(B, 125, 384)` — the
  future-half audio tokens.

The parent class's `head_norm`, `head_proj`, `head_smooth` modules
remain in the parameter dict but are **never called**. They receive
zero gradient.

---

## Diffusion head

The diffusion head takes `(cursor_token, audio_features)` and
produces a final per-bin activation map `(B, 500)` either by one
denoiser forward (training) or by running `T_inf` reverse-diffusion
steps (inference).

### Notation

- `T = 64` — number of training timesteps.
- `t ∈ {0, …, T-1}` — discrete timestep. `t = 0` is the cleanest end
  (closest to a clean activation map); `t = T-1` is the noisiest end
  (close to N(0, I)).
- `n_bins = 500` — bin count = b_pred = B_PRED. There is **no STOP
  class**; the activation map encodes "no onset here" as low
  activation.

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

### Framewise activation Gaussian process (x_0-parameterized)

The forward (q) process at timestep `t` is

```
x_t = sqrt(ab_cum(t)) * x_0 + sqrt(1 - ab_cum(t)) * eps,    eps ~ N(0, I)
```

where `x_0 = M_target_smoothed` ∈ [0, 1]^500 (the Gaussian-smoothed
GT activation map). Unlike #014/#015's process, **no encoding step
is needed** — the activation map IS the x_0 representation; the
process treats it as a continuous tensor without scaling.

Decoding: `decode_to_logits(x_0_hat) = clamp(x_0_hat, 0.0, 1.0)`.
No `logit_scale` or `x0_scale` constants — argmax-style decoding
is not meaningful for framewise (which threshold-decodes), and the
clamp keeps the output in the valid probability-like range.

Only `parameterization = "x0"` is supported by
`FramewiseActivationProcess`. Predicting noise or v would require
inverse formulas that overshoot [0, 1] and need re-clipping every
step, which defeats the parameterization.

### Conv1D denoiser (with self-conditioning and audio context)

Inputs:
- `cursor_token` `(B, 384)`,
- `x_t` `(B, 500)`,
- `t` `(B,)` int64,
- `prev_x0_hat` `(B, 500)` — previous reverse step's predicted x_0
  (zeros on the first step or during the training-time no-self-cond
  pass),
- `audio_features` `(B, 125, 384)` — future-half audio tokens.

Per-bin input channels (concatenated along the channel axis of a
`(B, C, 500)` tensor):

```
x_t                  (B, 1, 500)
prev_x0_hat          (B, 1, 500)            # only if self_cond=True
positional embed     (B, 32, 500)           # sinusoidal over bin index, precomputed
audio features       (B, 384, 500)          # F.interpolate(audio_features.transpose(1,2), 500, mode="linear")
cursor projection    (B, 32, 500)           # broadcast Linear(384 → 32) of cursor_token
time projection      (B, 32, 500)           # broadcast of Linear(128 → 32) of sinusoidal_time_embedding(t, 128)
```

Total input channels: 1 + 1 + 32 + 384 + 32 + 32 = **482**.

Time embedding builder:
```
time_emb_raw = sinusoidal_time_embedding(t, dim=128)             # (B, 128)
time_emb     = Linear(128, 32) → SiLU → Linear(32, 32)(time_emb_raw)  # (B, 32)
```

Cursor projection:
```
cursor_emb = Linear(384, 32)(cursor_token)                       # (B, 32)
```

Audio upsampling:
```
audio = audio_features.transpose(1, 2)                           # (B, 384, 125)
audio = F.interpolate(audio, size=500, mode="linear", align_corners=False)  # (B, 384, 500)
```

Conv stack (3 blocks):

```
h: (B, 482, 500)
for k in [31, 15, 15]:
    h = Conv1d(in_ch, 256, kernel_size=k, padding=k//2)(h)
    h = GroupNorm(num_groups=8, num_channels=256)(h)
    gamma, beta = Linear(32+32, 2*256)([cursor_emb || time_emb]).chunk(2, dim=-1)
    h = h * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
    h = SiLU(h)
    h = Dropout(0.1)(h)
    in_ch = 256

out = Conv1d(256, 1, kernel_size=1)(h)                           # (B, 1, 500)
out = out.squeeze(1)                                             # (B, 500)
```

FiLM linear layers are **zero-initialized** so the conv stack is
identity at training start.

Receptive field of the kernel = 31 (= 155 ms at 5 ms/bin) for the
first conv block, accumulating to 31 + 14 + 14 = 59 bins ≈ 295 ms
total — sufficient to capture local musical structure around each
predicted bin.

**Denoiser parameter count: 5,911,137.**

### Loss target (x0-parameterization)

`loss_target = M_target_smoothed` (the Gaussian-σ=2 smoothed map).
The denoiser is trained to output the clean smoothed map directly;
per-bin distance is `(M_0_hat[b] - M_target_smoothed[b])²`.

### Training-time forward (`forward_diffusion`)

Self-conditioning at training time uses the Analog-Bits two-pass
recipe: with probability `self_cond_prob = 0.5` per sample, a first
no-grad denoise pass produces the `prev_x0_hat` input for the
second (gradient-tracked) pass; otherwise `prev_x0_hat = 0` (no
prior estimate). Matches the inference distribution where the
first reverse step has no prior estimate but every subsequent
step does.

```
B          = cursor_token.size(0)
x_0        = M_target_smoothed                          # (B, 500), identity in process
t          = randint(0, T, (B,))                        # uniform
eps        = randn_like(x_0)                            # (B, 500)
x_t        = sqrt(ab_cum[t]) * x_0
           + sqrt(1 - ab_cum[t]) * eps                  # (B, 500)

# Two-pass self-conditioning (training only, when self_cond=True)
mask       = (rand(B) < 0.5)
prev_x0    = zeros(B, 500)
if mask.any():
    with no_grad():
        sc_out = denoiser(cursor_token, x_t, t,
                          prev_x0_hat=None,
                          audio_features=audio_features)
        sc_x0  = process.predict_x0(sc_out, x_t, t).detach()
    prev_x0 = where(mask, sc_x0, zeros_like(sc_x0))

mout       = denoiser(cursor_token, x_t, t,
                      prev_x0_hat=prev_x0,
                      audio_features=audio_features)
ltgt       = x_0                                        # x0-param loss target
x0_hat     = mout                                       # x0-param
logits     = clamp(x0_hat, 0.0, 1.0)                    # decode_to_logits

return FramewiseModelOutput(
    logits=logits, cursor_token=cursor_token,
    audio_features=audio_features,
    model_out=mout, loss_target=ltgt, t=t, x_t=x_t,
)
```

### Inference-time forward (`predict`)

```
cursor_token, audio_tokens = trunk(input)                # full token sequence
audio_features              = audio_tokens[:, 125:250, :] # future-half

return FramewiseModelOutput(
    logits=zeros(B, 500),                                # placeholder; sampler fills
    cursor_token=cursor_token,
    audio_features=audio_features,
)
```

The decoder runs the sampler on `(cursor_token, audio_features)` to
produce the real activation map.

---

## Loss

`FramewiseDiffusionLoss` — weighted MSE on the predicted activation
map vs the smoothed target.

### Hyperparameters

| Param | Value |
|---|---|
| `loss_type` | `"mse"` |
| `snr_weighting` | `true` (Min-SNR-γ applied as the loss weight) |
| `snr_gamma` | `5.0` |
| `pos_weight_clamp_min` | `10.0` |
| `pos_weight_clamp_max` | `200.0` |
| `n_t_buckets` | `4` |
| `canonical_threshold` | `0.5` |
| `canonical_tolerance_frames` | `2` |

No `stop_weight` — there is no STOP class to weight.

### Forward

```
# (B, 500) per-bin squared error
per_bin = (mout - loss_target) ** 2

# Per-sample positive-class upweighting: balance gradient
# contribution between positive and negative bins.
n_gt           = target.n_gt                            # (B,) int
n_neg          = n_bins - n_gt                          # (B,) int
pos_w          = (n_neg / clamp(n_gt, min=1)).float()   # (B,)
pos_w          = pos_w.clamp(min=10.0, max=200.0)       # (B,)

# Per-bin weight: pos_w at positive bins, 1.0 at negatives.
pos_mask       = (target.target_map_binary > 0.5)       # (B, 500) bool
weight         = where(pos_mask, pos_w[:, None], ones_like(pos_mask).float())

# Min-SNR-γ weighting per sample (Hang et al. 2023).
snr_t          = ab_cum(t) / (1 - ab_cum(t))            # (B,)
snr_w          = min(snr_t, 5.0) / snr_t                # (B,) — equivalent to min(SNR, γ)
                                                        #         on the x_0-MSE term

loss           = (per_bin * weight * snr_w[:, None]).mean()
```

### Diagnostic metrics reported per step

- `loss` — the headline scalar.
- `loss/snr_weighted` — `(per_bin * snr_w).mean()` reported always (independent of whether `snr_weighting` is on as the actual loss path).
- `loss/per_t_q0..3` — per-sample loss bucketed into `n_t_buckets`
  equal-width quartiles of the sampled `t` (0..63).
- `loss/pos_only` — MSE summed over positive-target bins only.
- `loss/neg_only` — MSE summed over negative-target bins only.
- `loss/pos_neg_ratio` — `pos_only / neg_only` (a value < 1 means
  the model is paying more attention to negatives than positives).
- `frame/precision_τ_50_tol_2`, `recall_τ_50_tol_2`, `f1_τ_50_tol_2`
  — single-point F1 at the canonical operating point (`τ=0.5`,
  tolerance ±2 frames). Computed via 1D max-pool dilation:
  a predicted positive at bin `b` is TP iff `max_pool1d(target, k=5, stride=1)[b] == 1` (i.e., some target within ±2 frames).
- `frame/auc_pr`, `frame/auc_roc` — threshold-free integrated
  quality. Sorted-prediction trapezoidal integration.
- `frame/mean_act_pos`, `frame/mean_act_neg`, `frame/separation` —
  mean predicted activation at GT bins vs non-GT bins;
  `separation = mean_pos - mean_neg`.
- `frame/pos_rate_pred_50` — fraction of bins with M_0_hat > 0.5
  (model-emitted positive rate at the canonical threshold).
- `frame/pos_rate_target` — fraction of target bins == 1.0 (≈ 1.5
  % at typical density).

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
| Balanced sampling | **off** (no class to balance under framewise) |
| Evals per epoch | 4 |
| Watched metric | `loss` (lower is better) |
| Checkpoint cadence | every eval — `latest.pt` rewritten; `best.pt` on new best |
| Step log cadence | every step into `metrics.jsonl` |
| Eval artifacts | per-eval AR-corpus output under `{run_dir}/infer_corpus/eval_{step}/`; train_noaug pass under `{run_dir}/eval_{step}/train_noaug/`; full T_inf-step rollout under `{run_dir}/eval_{step}/rollout_maps.npz` plus convergence plots and GIFs |
| Seed | 42 |

### Auxiliary eval passes (5 per training eval)

1. **EVAL_1** — sampled-t (random `t` per sample), single denoiser
   forward. Full val 5% subset. Cheap. Tracks training health
   (loss curves, per-t-quartile, framewise F1 at canonical op).
2. **NOAUG_1** — same machinery on a deterministic train-no-aug 5%
   subset. Diagnoses overfit (gap between val and train_noaug).
3. **EVAL_K** — full `T_inf`-step rollout from `x_T ~ N(0, I)`. On
   ~32 charts × ~5 random windows = ~150 samples. Tracks per-step
   convergence (F1 at each k, MSE, mass distribution). Mini-chart
   metrics at step `K` (threshold-decode the final M_0_hat, compare
   emissions to GT events in the window at 101 thresholds × 5
   tolerances). Saves `rollout_maps.npz` with full per-step M_k
   tensors. Renders 5 representative GIFs (best/p75/p50/p25/worst
   by final F1) + a population-summary GIF + convergence plots.
4. **NOAUG_K** — same on train_noaug subset.
5. **AR-corpus** — full AR loop on 10% of val charts. Uses
   `FramewiseDiffusionDecoder` (threshold + multi-emit + cursor-to-
   last). Produces chart-level `matched_rate`, `error_median_ms`,
   `density_ratio`, etc. via the standard `comparisons_summary.json`
   machinery, extended to 5 tolerances (5/10/20/40/100 ms).

### Per-eval artifacts

```
runs/exp_016_framewise_diffusion/eval_{step}/
  ├── checkpoint.pt
  ├── eval.json                       # scalar metric summary
  ├── curves.npz                      # 101-threshold × 5-tolerance grids
  ├── rollout_maps.npz                # per-sample per-step M_k tensors
  ├── noaug_rollout_maps.npz          # train_noaug equivalent
  ├── convergence_curves.png          # mean+p10/p25/p75/p90 over k, 3 panels
  ├── convergence_by_density.png      # 4 sub-panels for density buckets
  ├── convergence_by_star.png         # 4 sub-panels for star buckets
  ├── convergence_by_kind.png         # 6 sub-panels per onset kind
  ├── rollout_gifs/
  │   ├── sample_best_chart-X_window-Y_f1-Z.gif
  │   ├── sample_p75_*.gif
  │   ├── sample_p50_*.gif
  │   ├── sample_p25_*.gif
  │   ├── sample_worst_*.gif
  │   └── summary_histogram.gif        # population-level histogram evolution
  ├── train_noaug/                     # mirror of the above for noaug
  └── (legacy artifacts disabled in framewise mode)
```

---

## Inference (autoregressive)

Reconstructs at inference time:
- `FramewiseDiffusionDetector` (the model).
- `FramewiseDiffusionDecoder(b_pred=500, decode_threshold=0.5, nms_kernel=1,
   stop_hop_bins=20, min_emit_gap_bins=1, top_k_log=5,
   sampler_config=DDIMSamplerConfig(n_inference_steps=16, eta=0.0,
   timestep_spacing="linspace", time_offset=0.0))`.
- `DetectionARInputBuilder(a_bins=500, b_bins=500, c_events=128)`.
- `MelSampler(sample_rate=22000, n_fft=2048, hop_divisor=200, n_mels=80,
   f_min=20, f_max=8000, power=2.0, top_db=80.0)`.
- `FixedRateEventSampler(divisor=200)`.

After model load, `assemble_predictor` calls
`decoder.bind_model(model)` which constructs `DDIMSampler(sampler_config,
model.process, model.denoiser)`. The sampler reads
`model.denoiser.config.self_cond` to decide whether to thread
`prev_x0_hat` through the reverse loop.

### DDIM sampler (16 steps, eta = 0, self-conditioning, time_offset = 0)

Pre-compute the inference-time timestep schedule by `linspace`-spacing
16 steps over `[0, T-1] = [0, 63]`, producing a strictly-decreasing
sequence `t_15 > t_14 > … > t_0 = 0`.

Initialize `x = randn(1, 500)` (= `x_T ~ N(0, I)`).
`prev_x0_hat = None` (treated as zeros on the first denoiser call).

For `i = 15, …, 0`:

```
t        = ts[i]                          # current step
t_prev   = ts[i - 1] if i > 0 else -1     # next step (cleaner end)

ab_t     = ab_cum(t)
ab_prev  = ab_cum(t_prev) if t_prev >= 0 else 1.0

t_query  = clamp(t + time_offset, 0, T - 1)  # asymmetric time intervals

# Predict x_0 from the denoiser at this step, with self-cond.
mout     = denoiser(cursor_token, x, tensor([t_query]),
                    prev_x0_hat=prev_x0_hat,
                    audio_features=audio_features)
x0_hat   = mout                           # x0-param identity
prev_x0_hat = x0_hat.detach()             # for next step

# Recover the implied noise:
eps_hat  = (x - sqrt(ab_t) * x0_hat) / sqrt(1 - ab_t)

# DDIM update with eta = 0 (deterministic):
sigma    = 0.0
mean_pred = sqrt(ab_prev) * x0_hat
          + sqrt(1 - ab_prev - sigma**2) * eps_hat
x        = mean_pred
```

After the loop, `M_0_hat = clamp(x, 0.0, 1.0)` (`decode_to_logits`).

### Decoder: threshold + (optional) NMS + min-gap

```
scores = M_0_hat[0]                                       # (500,)

# Optional 1-D max-pool NMS (kept only when scores[b] == local max).
if nms_kernel > 1:
    pooled    = max_pool1d(scores.view(1,1,-1),
                            kernel_size=nms_kernel,
                            stride=1, padding=nms_kernel // 2)
    local_max = scores >= pooled.view(-1) - 1e-9
else:
    local_max = ones_like(scores, dtype=bool)

above   = scores > decode_threshold
keep    = above & local_max
bins    = sorted(keep.nonzero()[0].tolist())

# Min-emit-gap enforcement: greedy, lower-bin wins.
final   = []
last    = -10**9
for b in bins:
    if b - last >= min_emit_gap_bins:
        final.append(b)
        last = b

if not final:
    return ARDecision(bin_offsets=(), ...)       # empty -> STOP_HOP
return ARDecision(
    bin_offsets=tuple(final),
    confidences=tuple(scores[b] for b in final),
    ...,
)
```

### AR loop

1. Initialize `cursor = 0`, `past_onsets = []`, `step = 0`.
2. While `cursor < max_bin` and `step < max_events`:
   - Slice mel `[cursor - 500, cursor + 500)` with zero-padding.
   - Back-align up to 128 past onsets as `event_offsets` /
     `event_mask`.
   - Trunk forward → `cursor_token (1, 384)`, `audio_features (1, 125, 384)`.
   - DDIM sampler runs 16 reverse-diffusion steps starting from
     `x_T ~ N(0, I)`, each step calling the denoiser with
     `(cursor_token, x_i, t_i, prev_x0_hat=M_(i-1)_hat, audio_features)`.
   - Decode: threshold + optional NMS + min-gap → list of positive
     bin offsets.
   - If empty (`bin_offsets == ()`): cursor advances by
     `hop_bins_on_stop = 20` bins.
   - Else: for each bin `b` in `bin_offsets`, emit an onset at
     `cursor + b`. Advance cursor to `cursor + max(bin_offsets)`
     (the predictor's standard semantics — the next AR step starts
     AT the last emitted onset, and `min_onset_gap_bins = 1` prevents
     repeat emission).
3. Return a new `Chart` with the accumulated onsets.

`hop_bins_on_stop = 20`, `max_events = 10000`, `min_onset_gap_bins = 1`,
`default_kind = "DON"`.

### Sampler / decoder ablation (post-run)

Run `cli.diffusion_sampler_ablation` against
`config/ablation_matrix.json`. 10 variants:

- Threshold sweep: τ ∈ {0.3, 0.5, 0.7} at default sampler config.
- NMS variants: nms_kernel ∈ {3, 5} at τ ∈ {0.3, 0.5}.
- DDIM step count: {4, 8, 16, 32} at τ=0.5.
- Asymmetric time offset: time_offset ∈ {0, 1} at 4-step DDIM.
- DDPM-64: full-schedule reverse process reference.
- Combined Pareto: 4-step + offset=1 + NMS-3 + threshold tuned.

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
| Pillow | 12.2.0 (used for GIF rendering — no imageio dependency) |
| tqdm | 4.67.3 |

---

## Addenda

_(None before the run.)_
