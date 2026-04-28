# Experiment 009b — MDN with capped sigma · Architecture

> **This document is self-contained.** Everything needed to reproduce
> this experiment is written inline. No links, no "see elsewhere."

---

## Task

Given mel + 128 past events + density conditioning, predict the next
onset bin offset (0..499) or STOP. The model outputs K=3 Gaussian
components instead of a 501-class softmax. Sigma is capped at 3.0
bins to prevent inflation.

## Inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `mel` | `(B, 80, 1000)` | float32 | Log-mel, 80 bands, 500+500 frames @ 5 ms |
| `event_offsets` | `(B, 128)` | int64 | Past onset offsets, back-aligned |
| `event_mask` | `(B, 128)` | bool | True = padding |
| `conditioning` | `(B, 3)` | float32 | density_mean, peak, std |

## Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `raw` | `(B, 10)` | float32 | 1 STOP gate + 3 × (mu_raw, log_sigma, log_pi) |

Parsed as:
```
stop_logit = raw[:, 0]                              # (B,)
params = raw[:, 1:].reshape(B, 3, 3)                # (B, K, 3)
mu    = sigmoid(params[:, :, 0]) * 500              # (B, K) in [0, 500]
sigma = clamp(softplus(params[:, :, 1]) + 1.0,      # (B, K) in [1.0, 3.0]
              max=3.0)
pi    = softmax(params[:, :, 2], dim=-1)            # (B, K) sums to 1
```

The sigma clamp at 3.0 is the only change from #009. It forces all
components to be sharp (~±3 bins = ±15 ms) and place mu precisely.
The value 3.0 slightly exceeds the ±2-bin FHIT tolerance — tight
enough to force precision, loose enough that a small mu positioning
error doesn't catastrophically spike the NLL.

---

## Data pipeline

Audio preprocessing:

| Param | Value |
|---|---|
| Sample rate | 22 000 Hz |
| FFT size | 2048 samples |
| Hop length | 110 samples → 5.000 ms / frame |
| Mel bands | 80 |
| Frequency range | 20 Hz – 8 000 Hz |
| Power spectrum | power = 2.0 |
| Amplitude → dB | top_db = 80 |
| On-disk dtype | float16 |
| Served dtype | float32 |

Sample construction:

| Key param | Value |
|---|---|
| a_bins / b_bins | 500 / 500 |
| c_events | 128 |
| d_events | 1 |
| min_cursor_bin | 6000 |
| allowed_overlap forward / back | 0 / 0 |
| subsample | 1 |
| split | train 90 % / val 10 %, seed 42, song-grouped |

Target derivation:
```
stop_idx = 500
if future_events_mask[0]:       target = stop_idx
elif cursor_offset < 0 or >= 500: target = stop_idx
else:                           target = cursor_offset
```

Class-balanced sampling: `weight[i] = min(1.0, 1 / (count[target(i)] + 1) ^ 0.5)`.

14 augmentations in order: TimeStretch (30 %, max_scale=1.4),
MelGainJitter (30 %, ±2 dB), MelGaussianNoise (15 %, σ 0.1–0.3),
MelFreqJitter (15 %, shift ±3), SpecAugFreq (20 %, ≤10 bands),
SpecAugTime (20 %, ≤30 frames), EventJitter (100 %, global ±3 +
recency 1–2×), EventDropout (5 %, drop 1–2), EventInsertion (3 %),
PartialMetronome (2 %), PartialAdvMetronome (2 %), LargeTimeShift
(2 %, ±50 bins), ContextTruncation (5 %, keep 8–32),
ConditioningJitter (10 %, ±2 %).

---

## Model architecture

**Total parameters: ~16.2 M.**

Backbone (identical to all prior experiments):
- Conditioning MLP: 3 → 64 (GELU) → 64.
- Conv stem: Conv1d(80→192, k=7, s=2, p=3) → GELU → GroupNorm(1, 192) →
  Conv1d(192→384, k=7, s=2, p=3) → GELU → transpose → (B, 250, 384) →
  LayerNorm(384) → + SinusoidalPosEmb → FiLM(cond).
  Cursor token index = 125 (= a_bins // 4).
- Event embeddings: 5 parts per slot (presence, gap_before, gap_after,
  gap_ratio_before, gap_ratio_after) each sinusoidal at d_model=384 →
  concat (5×384=1920) → Linear(1920, 384) → GELU → Linear(384, 384) →
  scatter-add into audio tokens at `(a_bins + cursor_offset) // 4`.
- Transformer: 8 layers TransformerEncoderLayer(d_model=384, nhead=8,
  dim_feedforward=1536, dropout=0.1, gelu, norm_first=True) with
  per-layer FiLM(cond_dim=64, d_model=384). FiLM is zero-initialized.

### Output head

```
cursor_tok = x[:, 125, :]                   # (B, 384)
raw = Linear(384, 10)(LayerNorm(384)(cursor_tok))  # (B, 10)
```

No Conv1d smoothing — smoothness is built into each Gaussian. The
10 output dims encode: 1 STOP gate logit + 3 components ×
(mu_raw, log_sigma, log_pi).

---

## Loss — MdnLoss

### Hyperparameters

| Param | Value |
|---|---|
| `n_components` (K) | 3 |
| `b_pred` | 500 |
| `stop_weight` | 1.5 |
| `min_sigma` | 1.0 |
| **`max_sigma`** | **3.0** |

### Forward — bin-target sample (target != 500)

```
P_stop = sigmoid(stop_logit)
mu, sigma, pi = parse_mdn_params(raw, max_sigma=3.0)
  # sigma = clamp(softplus(raw) + 1.0, max=3.0)

# Log-likelihood of each Gaussian component at the target.
log_comp_k = log N(t | mu_k, sigma_k)       for k = 0..K-1

# Mixture log-likelihood (log-sum-exp weighted by pi).
log_mixture = logsumexp(log_comp_k + log(pi_k))

# Total NLL includes the "not STOP" gate.
loss = -log(1 - P_stop) - log_mixture
```

The loss only requires SOME component to cover `t`. Other components
can sit at any position without penalty. With sigma capped at 3.0,
the only way to achieve low NLL is to place mu within ~3 bins of the
target — no component can inflate to cover a wide region cheaply.

### Forward — STOP-target sample (target == 500)

```
loss = -log(P_stop) * stop_weight
```

### Diagnostics per batch

- `loss`, `mixture_nll`, `stop_bce`, `stop_rate`
- `mdn/coverage_2bin` — % of samples where min_k |mu_k - t| <= 2
- `mdn/coverage_5bin` — same, <= 5
- `mdn/dominant_weight` — mean of max(pi_k) per sample
- `mdn/n_active_components` — mean count of components with pi > 0.1
- `mdn/mean_sigma` — mean sigma across all components
- `mdn/correct_component_weight` — mean pi of closest component

---

## Inference (autoregressive)

**MdnDecoder**: at each AR step:
1. Parse MDN params from the `(1, 10)` output.
2. If `sigmoid(stop_logit) > 0.5` → STOP.
3. Else → `bin = round(mu_{argmax(pi)})` (highest-weight component).

Per-step extras: `p_stop`, `comp{k}_mu`, `comp{k}_sigma`,
`comp{k}_pi` for all K.

AR loop: cursor=0, predict, emit onset at cursor+bin_offset or
STOP (cursor += 20). Max 10,000 events. Default kind DON.

---

## Per-eval artifacts

Standard (all MDN-aware): heatmap, distributions, ratio_error,
error_hist, ratio_hit, metronome.

MDN-specific (`{eval_dir}/mdn/`):
- `comp{k}_heatmap.png` — target vs mu_k (pi-weighted), K=3.
- `comp{k}_ratio_error.png` — log-ratio error per component (pi-weighted), K=3.
- `combined_heatmap.png` — argmax-pi prediction.
- `combined_ratio_error.png` — argmax-pi ratio error.
- `mdn_components.npz` — raw arrays for post-hoc analysis.

Also under `{eval_dir}/train_noaug/` and `train_noaug/mdn/`.

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
| LR scheduler | CosineAnnealingLR (T_max = steps_per_epoch × epochs) |
| Mixed precision | off |
| Balanced sampling | on (sqrt-inverse) |
| Evals per epoch | 4 |
| Watched metric | `onset/miss` (lower is better) |
| Seed | 42 |

---

## Dataset

`taiko2_v1`. 10,048 charts, 6,934,185 onsets. Train 90 % (~9,090
charts), val 10 % (~958 charts). Song-grouped by beatmapset_id,
seed 42. `allowed_overlap = 0`, `min_cursor_bin = 6000`,
`subsample = 1`.

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
