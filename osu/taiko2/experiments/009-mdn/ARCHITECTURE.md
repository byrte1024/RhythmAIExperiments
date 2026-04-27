# Experiment 009 — Mixture Density Network · Architecture

> **This document is self-contained.** Everything needed to reproduce
> this experiment is written inline. No links, no "see elsewhere."

---

## Task

Identical to prior experiments: given mel + 128 past events +
density conditioning, predict the next onset bin offset (0..499) or
STOP. The change is in the output head and loss — the model outputs
K=3 Gaussian components instead of a 501-class softmax.

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
sigma = softplus(params[:, :, 1]) + 1.0             # (B, K) >= 1.0
pi    = softmax(params[:, :, 2], dim=-1)            # (B, K) sums to 1
```

---

## Data pipeline

Identical to #007. Audio preprocessing, event encoding, sample
construction, train/val split, target derivation, class-balanced
sampling, and augmentation pipeline (including TimeStretch at 30%
probability, max_scale=1.4) are all unchanged.

| Key param | Value |
|---|---|
| a_bins / b_bins | 500 / 500 |
| c_events | 128 |
| min_cursor_bin | 6000 |
| allowed_overlap | 0 / 0 |
| subsample | 1 |
| split | 90/10, seed 42 |

14 augmentations in order: TimeStretch (30%), MelGainJitter (30%),
MelGaussianNoise (15%), MelFreqJitter (15%), SpecAugFreq (20%),
SpecAugTime (20%), EventJitter (100%), EventDropout (5%),
EventInsertion (3%), PartialMetronome (2%), PartialAdvMetronome (2%),
LargeTimeShift (2%), ContextTruncation (5%), ConditioningJitter (10%).

---

## Model architecture

Identical to #007 except the output head.

**Total parameters: ~16.2 M** (~190k fewer than the 501-class head
because `Linear(384, 10)` replaces `Linear(384, 501) + Conv1d`).

Backbone: conditioning MLP (3→64→64), conv stem (80→192→384, 4x
downsample, LN + sinusoidal PosEmb + FiLM → 250 tokens), event
embeddings (5-part, scatter-add into audio tokens), 8-layer
transformer (d=384, 8 heads, ff=1536, dropout=0.1, gelu,
norm_first, per-layer FiLM).

### Output head (changed)

```
cursor_tok = x[:, 125, :]                   # (B, 384)
raw = Linear(384, 10)(LayerNorm(384)(cursor_tok))  # (B, 10)
```

No Conv1d smoothing — each Gaussian component is inherently smooth.
The 10 output dims encode: 1 STOP gate logit + 3 components ×
(mu_raw, log_sigma, log_pi).

---

## Loss — MdnLoss

### Hyperparameters

| Param | Value |
|---|---|
| `n_components` (K) | 3 |
| `b_pred` | 500 |
| `stop_weight` | 1.5 |

### Forward — bin-target sample (target != 500)

```
P_stop = sigmoid(stop_logit)
mu, sigma, pi = parse_mdn_params(raw)

# Log-likelihood of each Gaussian component at the target.
log_comp_k = log N(t | mu_k, sigma_k)       for k = 0..K-1

# Mixture log-likelihood (log-sum-exp weighted by pi).
log_mixture = logsumexp(log_comp_k + log(pi_k))

# Total NLL includes the "not STOP" gate.
loss = -log(1 - P_stop) - log_mixture
```

The loss only requires SOME component to cover `t`. Other components
can sit at any position — `2t`, `t/2`, `3t` — without penalty. This
is the fundamental difference from softmax CE: **no gradient pushes
secondary peaks toward zero**.

### Forward — STOP-target sample (target == 500)

```
loss = -log(P_stop) * stop_weight
```

Pure sigmoid BCE on the STOP gate, weighted. No mixture term.

### Diagnostics per batch

The loss reports:
- `loss`, `mixture_nll`, `stop_bce`, `stop_rate`
- `mdn/coverage_2bin` — % of samples where min_k |mu_k - t| <= 2
- `mdn/coverage_5bin` — same, <= 5
- `mdn/dominant_weight` — mean of max(pi_k) per sample
- `mdn/n_active_components` — mean count of components with pi > 0.1
- `mdn/mean_sigma` — mean sigma across all components
- `mdn/correct_component_weight` — mean pi of the closest component

### Why MDN

Previous experiments (#005, #007, #008) showed that the `±log 2`
ridges are NOT loss-fixable — three loss families produced identical
ridge patterns. The model's hidden states carry multi-modal
information (taiko1 top-K oracle 91.8 %) but every loss tried forces
a single-peak output, which makes the model hedge. MDN lets the
model express K explicit peaks — if the components specialize, we
learn WHERE the ambiguity lives and can design a targeted decoder or
auxiliary head to resolve it.

---

## Inference (autoregressive)

Same AR loop as prior experiments. Decoder changed:

**`MdnDecoder`**: at each step:
1. Parse MDN params from the `(1, 10)` output.
2. If `sigmoid(stop_logit) > 0.5` → STOP.
3. Else → `bin = round(mu_{argmax(pi)})` (highest-weight component).

Per-step extras: `p_stop`, `comp{k}_mu`, `comp{k}_sigma`,
`comp{k}_pi` for all K — full per-step diagnostic trace.

---

## Per-eval artifacts

Standard set (all MDN-aware via `decode_pred_bins`):
- `heatmap.png`, `distributions.png`, `ratio_error.png`,
  `error_hist.png`, `ratio_hit.png`, `metronome.png`

Plus MDN-specific (`{eval_dir}/mdn/`):
- `comp{k}_heatmap.png` — target vs mu_k, color = log(1 + pi-weighted count).
  One per component (3 total). Shows component specialization.
- `comp{k}_ratio_error.png` — log-ratio error per component,
  pi-weighted. One per component (3 total).
- `combined_heatmap.png` — argmax-pi prediction heatmap.
- `combined_ratio_error.png` — argmax-pi ratio error.
- `mdn_components.npz` — raw (targets, mus, sigmas, pis) for
  post-hoc analysis.

Also under `{eval_dir}/train_noaug/` and
`{eval_dir}/train_noaug/mdn/` — same set for the augmentation-off
train pass (inherits the artifact list from the eval loop).

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
| LR scheduler | CosineAnnealingLR |
| Mixed precision | off |
| Balanced sampling | on (sqrt-inverse) |
| Evals per epoch | 4 |
| Watched metric | `onset/miss` (lower is better) |
| Seed | 42 |

---

## Dataset

`taiko2_v1`. 10,048 charts, 6.9M onsets. Train 90 %, val 10 %,
seed 42. `allowed_overlap = 0`, `subsample = 1`.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.13.13 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| OS | Windows 11 |

---

## Addenda

_(None before the run.)_
