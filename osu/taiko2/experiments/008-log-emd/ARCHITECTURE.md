# Experiment 008 — Log-ratio EMD loss · Architecture

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
| `mel` | `(B, 80, 1000)` | float32 | Log-mel spectrogram, 80 bands, 1000 frames = 500 past + 500 future at 5.000 ms / frame. Zero-padded at chart edges. |
| `event_offsets` | `(B, 128)` | int64 | Last 128 onset bin offsets relative to the cursor, back-aligned. Offsets are ≤ 0. |
| `event_mask` | `(B, 128)` | bool | True ⇒ slot is padding. |
| `conditioning` | `(B, 3)` | float32 | `[density_mean, density_peak, density_std]`. |

## Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `logits` | `(B, 501)` | float32 | 500 bin classes (0..499) + 1 STOP class (index 500). |

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
| Onset kinds retained | DON, KA, BIG_DON, BIG_KA, DRUMROLL, SPINNER |
| Bin formula | `floor(time_ms / 5)` |

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
- Ratios: `train = 0.9`, `val = 0.1`. Seed 42.

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

Target derivation runs after augmentation, so the post-stretch
`cursor_offset` is what the adapter sees.

### Class-balanced sampling

Per-sample weight (identical to #002 / #007):

```
count[c]   = number of training samples with target class c
weight[i]  = min(1.0, 1 / (count[target(i)] + 1) ** 0.5)
```

Weights are computed pre-aug; time-stretch perturbs the realized
batch target distribution modestly without correction.

### Augmentations (training only)

Pipeline (post-sample, applied in order):

| Order | Name | Probability | Parameters |
|---:|---|---|---|
| 1 | TimeStretch | 30 % | `max_scale = 1.4`; per-call `s ~ log-Uniform(1/1.4, 1.4)` |
| 2 | MelGainJitter | 30 % | `delta_dB ~ U(-2, +2)` |
| 3 | MelGaussianNoise | 15 % | `σ ~ U(0.1, 0.3)` |
| 4 | MelFreqJitter | 15 % | `shift ∈ {-3, …, +3}` |
| 5 | SpecAugFreq | 20 % | one mask, width `[1, 10]` bands |
| 6 | SpecAugTime | 20 % | one mask, width `[1, 30]` frames |
| 7 | EventJitter | 100 % | global ±3 bins + per-event recency-weighted noise |
| 8 | EventDropout | 5 % | drop 1–2 random real past events |
| 9 | EventInsertion | 3 % | add 1 synthetic event between two reals |
| 10 | PartialMetronome | 2 % | replace recent half with uniform-gap events |
| 11 | PartialAdvMetronome | 2 % | replace older half with dominant-gap events |
| 12 | LargeTimeShift | 2 % | shift 2–4 recent events by ±50 bins |
| 13 | ContextTruncation | 5 % | keep only 8–32 most recent real events |
| 14 | ConditioningJitter | 10 % | each density field × `U(0.98, 1.02)` |

Identical to #007's pipeline.

---

## Model architecture

Identical to #002 / #007. **Total parameters: ≈ 16.5 M.**

- Conditioning MLP: 3 → 64 → 64.
- Conv stem: Conv1d(80→192, k=7, s=2) → GroupNorm → Conv1d(192→384,
  k=7, s=2) → LN + sinusoidal PosEmb + FiLM, output `(B, 250, 384)`.
  Cursor token = index 125.
- Event embeddings: 5 parts per slot (presence, gap_before,
  gap_after, gap_ratio_before, gap_ratio_after) → linear projection
  to `d_model = 384` → scatter-add into the 250-token sequence.
- Transformer trunk: 8 layers `TransformerEncoderLayer(d_model=384,
  nhead=8, dim_feedforward=1536, dropout=0.1, gelu, norm_first=True)`
  with per-layer FiLM.
- Head: cursor-token LayerNorm → Linear to 501 → additive
  `Conv1d(1→8, k=5, p=2) → GELU → Conv1d(8→1, k=5, p=2)` smoothing.
- FiLM: zero-initialized `Linear(64, 2*d_model)` → split into γ, β
  → `x * (1 + γ) + β`.

---

## Loss

### LogEmdLoss

Per-sample loss combines hard cross-entropy with a log-ratio Earth-
Mover Distance term over the bin part of the softmax.

### Hyperparameters

| Param | Value |
|---|---|
| `hard_alpha` | 0.5 |
| `exponent` | 1 (linear log-ratio EMD) |
| `stop_weight` | 1.5 |

### Forward — bin-target sample (`target != stop_idx`)

Let `logits ∈ ℝ^{501}`, `t ∈ {0,…,499}`, `n = stop_idx = 500`.

```
hard_ce = -log(softmax(logits)[t])                          # standard CE on full 501-way
P_bin   = softmax(logits)[:n]                               # mass on bin classes only
log_dist[i] = |log((i + 1) / (t + 1))|                      # for i in 0..n-1
log_emd = sum_i (P_bin[i] * log_dist[i])

per_sample_loss = hard_alpha * hard_ce + (1 - hard_alpha) * log_emd
```

`P_bin` is sliced from the full softmax (so it does NOT renormalize
over bins-only) — mass on `STOP` for a bin-target sample is invisible
to `log_emd` but punished by `hard_ce`.

### Forward — STOP-target sample (`target == stop_idx`)

```
hard_ce = -log(softmax(logits)[stop_idx])
log_emd = 0                                                 # no bin target ⇒ no log-ratio distance
per_sample_loss = hard_alpha * hard_ce + 0 = hard_alpha * hard_ce
```

A pure-hard-CE-on-STOP signal up to weighting (below).

### STOP weighting

Apply per-sample multiplier:

```
multiplier = stop_weight if target == stop_idx else 1.0
per_sample_loss *= multiplier
```

Final batch loss = mean of per-sample losses.

### Reported metrics

`LossResult.metrics` carries: `loss`, `hard_ce` (batch mean over all
samples), `log_emd` (batch mean — zero contribution from STOP samples
flattens this slightly), `stop_rate` (= mean of `target == stop_idx`).

### Why this loss shape

- **No entropy floor on bin term.** `log_emd` minimum is 0 iff
  `P_bin = δ_t`. There is no flat region; mass at every bin position
  costs proportionally to its log-ratio distance from the target.
  The trapezoid soft CE in #002 saturates at its entropy floor
  outside the trapezoid support, providing zero gradient on octave-
  distance mass — visible in the loss-landscape heatmap analysis.
- **Perception-correct.** `|log((i+1)/(t+1))|` is symmetric in
  log-space: `i = 2t` and `i = t/2` cost equal in the EMD. Matches
  the human-rhythm-perception observation that octaves up and down
  feel equally wrong.
- **Punishes bimodal hedging.** A 50/50 split between mass at `t/2`
  and `2t` scores `log 2 ≈ 0.69` in the EMD (full octave penalty).
  A sharp `1.3·t` prediction scores `log 1.3 ≈ 0.26`. The hedging
  failure mode #002's training converged to is specifically and
  proportionally punished.
- **Hard CE preserved.** `α = 0.5` keeps the strong sharpening
  signal that hard CE provides near the optimum. log-EMD's gradient
  is smoothest near the optimum, so mixing prevents under-sharpening.

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
| Eval artifacts | val + train_noaug under `{run_dir}/eval_{step}/` and `{run_dir}/eval_{step}/train_noaug/` |
| Seed | 42 |

### Auxiliary eval passes

- **`train_noaug`** — 5 % of the train split, augmentations OFF.
  Metrics + artifacts under `train_noaug/`.
- **`benchmarks`** — 10 input-distortion modes on 5 % of val per
  eval: `normal`, `no_audio`, `no_future_audio`, `no_past_audio`,
  `static_audio`, `no_context`, `random_context`, `metronome`,
  `advanced_metronome`, `context_time_shifted`.
- **AR corpus hook** — every eval runs AR inference on 10 % of val
  charts using the LIVE model, both `gt` and `fixed` conditioning
  modes.

### Metrics

All computed by `OnsetMetric`:

- `onset/exact`, `onset/fhit`, `onset/fgood`, `onset/fmiss`
- `onset/rhit`, `onset/rgood`, `onset/rmiss`
- `onset/hit`, `onset/good`, `onset/miss`
- `onset/ihit`, `onset/igood`, `onset/imiss`
- `onset/stop_precision`, `onset/stop_recall`, `onset/stop_f1`
- `onset/frame_err_mean`, `onset/frame_err_median`, `onset/frame_err_p90`
- `onset/pred_stop_rate` — total STOP preds / total samples
- `onset/pred_stop_fp_rate` — FP STOP / non-STOP samples (legacy)
- `onset/n_total`, `onset/n_nonstop`, `onset/n_stop_target`

Thresholds: FHIT `≤ 2 bins`, FGOOD `≤ 7 bins`, RHIT `log ratio
< log(100/97)`, RGOOD `< log(100/90)`.

---

## Inference (autoregressive)

Identical to #007. Reconstructs `EventEmbeddingDetector` +
`ArgmaxDecoder(b_pred=500)` + `DetectionARInputBuilder(a_bins=500,
b_bins=500, c_events=128)`.

AR loop:
1. Initialize `cursor = 0`, `past_onsets = []`, `step = 0`.
2. While `cursor < max_bin` and `step < max_events`:
   - Slice mel `[cursor - 500, cursor + 500)` with zero-padding.
   - Back-align up to 128 past onsets.
   - Forward → 501 logits → argmax → `ARDecision`.
   - STOP → `cursor += 20`.
   - Bin → emit onset at `cursor + bin_offset`, advance there.
3. Return `Chart` with accumulated onsets.

The argmax decoder takes argmax over the full 501 logits, which
matches how `LogEmdLoss` computes hard_ce (also over the full 501
softmax). No mismatch between training and inference decoding.

---

## Dataset

- Name: `taiko2_v1`. Source: parsed osu!taiko `.osz` packs.
- Charts: 10 048. Total onsets: 6 934 185.
- Train split (seed 42, 90 %): ≈ 9 090 charts.
- Val split (seed 42, 10 %): ≈ 958 charts.
- `allowed_overlap_forward = allowed_overlap_back = 0` — every
  cursor a valid sample. `min_cursor_bin = 6000` filters early
  silence.
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
