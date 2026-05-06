# Experiment 012 — Onset-feature channels appended to mel input · Architecture

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
| `mel` | `(B, 84, 1000)` | float32 | **Augmented input.** Rows 0..79 are log-mel spectrogram (80 bands); rows 80..83 are sub-band spectral flux (4 bands), range-matched to log-mel dB scale `[-80, 0]`. 1000 frames = 500 past (`a_bins`) + 500 future (`b_bins`) at 5.000 ms / frame. Zero-padded at chart edges. |
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
| Onset sub-bands | 4 (mel-band groups of 20 each) |
| Onset normalization | per-chart 99th-percentile, then linear stretch to `[-80, 0]` dB |
| On-disk dtype | float16 |
| Served dtype | float32 |
| Output rows | 80 (log-mel) + 4 (sub-band SF) = **84 total** |

#### Sub-band spectral flux computation

After the standard mel + AmpToDB pipeline produces an `(80, T)`
log-mel-in-dB tensor, four onset rows are derived from it on the
same time grid:

1. Compute per-mel-band positive half-wave-rectified time
   difference: `flux[b, t] = max(0, mel[b, t] - mel[b, t-1])`,
   `flux[b, 0] = 0`. Result is `(80, T)` float32.
2. Split the 80 mel bands into 4 contiguous groups of 20 bands
   each (group 0 = bands 0..19, group 1 = bands 20..39, ...).
   Sum within each group along the band axis. Result is `(4, T)`
   float32, the per-group sub-band spectral flux.
3. Per-row, divide by the row's 99th-percentile value (clamped
   to ≥ 1e-9 to avoid div-by-zero on silent rows). Most values
   land in `[0, 1]`; the top 1 % can exceed 1.
4. Clamp to `[0, 1]`, then linear-stretch to `[-80, 0]` dB:
   `sb_dB = -80 + clamped * 80`. Silent frames map to -80 dB
   (matching silent log-mel); 99th-percentile activations map
   to 0 dB (matching loud log-mel).
5. Concatenate the `(4, T)` dB-scale tensor under the `(80, T)`
   log-mel along the feature axis. Final tensor: `(84, T)`
   float32.

The sub-band flux is computed at dataset-build time and cached
to disk as float16 alongside the log-mel rows. No training-time
or inference-time computation cost.

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
| Allowed overlap forward | 0 bins (taiko1 parity) |
| Allowed overlap backward | 0 bins (taiko1 parity) |
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
to ≥ 500 becomes STOP; the reverse is not possible (only real events
start in-range, and `TimeStretch` never materialises new ones).

### Class-balanced sampling

Per-sample weight (identical to #002):

```
count[c]   = number of training samples with target class c
weight[i]  = min(1.0, 1 / (count[target(i)] + 1) ** 0.5)
```

Each epoch draws `N` indices with replacement from the weight-
normalized distribution. Sampling weights are computed pre-aug
against the source target, so time-stretch distorts the
realized-batch target distribution modestly — within ±40% log-
stretch most targets stay within their original balance bucket; the
distortion is considered acceptable and not corrected.

### Augmentations (training only)

Pre-pipeline: `TimeStretch` is inserted **first** in the post-sample
pipeline so every subsequent aug operates on the stretched sample.
Remaining augs are #002's exact exp 45 set, unchanged.

| Order | Name | Probability | Parameters |
|---:|---|---|---|
| 1 | **TimeStretch** | **30 %** | `max_scale = 1.4`; per-call `s ~ log-Uniform(1/1.4, 1.4)` |
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

**Audio.** The mel window is treated as a single `(84, a_bins + b_bins)
= (84, 1000)` tensor with the cursor pinned at frame index `a_bins = 500`.
TimeStretch operates on all 84 rows uniformly — both the log-mel rows
and the appended onset rows are rescaled together, since the onset rows
are precomputed at the same time grid and have the same temporal
semantics as the log-mel.
For each output frame `t`, the source index is
`src_idx(t) = a_bins + (t - a_bins) / s`. Linear interpolation between
`floor(src_idx)` and `floor(src_idx) + 1` produces the output value;
the result is then split back into `(84, 500) + (84, 500)`.

When `s < 1` (speed up), some output frames require source indices
outside `[0, total - 1]` — those positions are zero-padded. At
`s = 1/1.4 ≈ 0.714`, the outermost ~28 % of each side pulls from
beyond the source window and reads as zero. This matches how the
model already sees zero-padded edges at song boundaries in normal
training, so it is not a distribution-shift in pad behaviour.

When `s > 1` (slow down), the required source range is a subset of
the original window — no padding is needed; interpolation always
has valid source data.

**Events.** For each past / future `RelativeOnset`, the new
`cursor_offset` is `round(cursor_offset * s)`. Because time-stretch
preserves the cursor, past offsets stay ≤ 0 and future offsets stay
≥ 0.

For past events, events whose new offset falls outside `[-a_bins, 0]`
are dropped (they fell out of the audio window). Survivors are
dedupe'd — two events that post-round onto the same integer offset
collapse to one, keeping the older (earlier in the oldest-first
sequence, i.e. smaller list index post-rebuild). The list is then
re-padded at the start so its length stays `c_events`.

For future events, offsets are scaled but the mask is NOT touched;
the adapter (see Target derivation) handles STOP flips from the
post-stretch `cursor_offset` alone.

Each rebuilt `RelativeOnset` has `cursor_offset` set to the new
value, and `bin = cursor_offset`, `time_ms = cursor_offset * 5` so
the three fields stay mutually consistent. Absolute chart positions
(the original `time_ms` / `bin` values) are not recoverable after
stretching; the adapter and model only read `cursor_offset`.

**Determinism.** The `TimeStretch` instance owns a `random.Random`
seeded from `TrainerConfig.seed`; the per-sample draw is deterministic
under a fixed trainer seed.

---

## Model architecture

Identical to #007 except the conv stem's first-layer input channel
count expands from 80 to 84 to accept the augmented mel input.
Total trainable parameters: **16,359,758** (16.36 M)
[computed by instantiating `OnsetAugmentedDetector(config)` with
`config/model.json` and summing `p.numel() for p in model.parameters()`].
The conv stem grows by 4 × 192 × 7 = 5,376 weights vs #007's
16,354,382 [#007 instantiated from `007-time-stretch/config/model.json`],
which is +0.033 % of total — confirming the architecture is
otherwise identical.

Conditioning MLP (3 → 64 → 64), conv stem
(**Conv1d(84→192, k=7, s=2)** — the only change from #007 →
GroupNorm → Conv1d(192→384, k=7, s=2) → LN + sinusoidal PosEmb +
FiLM, transposed to `(B, 250, 384)`). Cursor token index = 125.
Event embeddings built from 5 concatenated parts per slot (presence,
gap_before, gap_after, gap_ratio_before, gap_ratio_after) projected
to `d_model = 384`, scatter-added into the 250-token sequence at
`token_pos = (a_bins + cursor_offset) // 4`. Transformer trunk = 8
layers of `TransformerEncoderLayer(d_model=384, nhead=8,
dim_feedforward=1536, dropout=0.1, gelu, norm_first=True)` with
per-layer FiLM conditioning. Head: cursor-token LayerNorm → Linear
to 501, then additive `Conv1d(1→8, k=5, p=2) → GELU → Conv1d(8→1,
k=5, p=2)` smoothing of the logits over bin axis.

FiLM module: `cond (B, 64) → Linear(64, 2*d_model)` (zero-init
weight + bias) → split into `(γ, β)`; apply `x * (1 + γ) + β`.
Zero-init = identity at start.

---

## Loss

Identical to #002. `OnsetLoss` — mixed hard + trapezoid-soft CE with
a ±2-frame floor and STOP weighting.

### Hyperparameters

| Param | Value |
|---|---|
| `hard_alpha` | 0.5 |
| `good_pct` (plateau width) | 0.03 |
| `fail_pct` (ramp-to-0 cutoff) | 0.20 |
| `frame_tolerance` | 2 |
| `stop_weight` | 1.5 |

### Forward

`loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE`, then
per-sample multiplied by `stop_weight` where the target is STOP.

Hard CE: `F.cross_entropy(logits, target, reduction="none")`.

Soft CE: per non-STOP target `t`, build a trapezoid over bins
`0..499`:

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

For STOP targets the soft distribution is a pure one-hot at index
500. Per-sample multiplier `= stop_weight when target == 500 else 1.0`.

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

All computed by `OnsetMetric` on the full val pass and additionally
on the 5 %-of-train augmentation-off pass + 5 %-of-val × 10
benchmark-mode passes:

- `onset/exact`, `onset/fhit`, `onset/fgood`, `onset/fmiss`
- `onset/rhit`, `onset/rgood`, `onset/rmiss`
- `onset/hit`, `onset/good`, `onset/miss`
- `onset/ihit`, `onset/igood`, `onset/imiss`
- `onset/stop_precision`, `onset/stop_recall`, `onset/stop_f1`
- `onset/frame_err_mean`, `onset/frame_err_median`, `onset/frame_err_p90`
- `onset/pred_stop_rate` — total STOP predictions / total samples
- `onset/pred_stop_fp_rate` — FP STOP / non-STOP (legacy)
- `onset/n_total`, `onset/n_nonstop`, `onset/n_stop_target`

Thresholds: FHIT `≤ 2 bins`, FGOOD `≤ 7 bins`, RHIT `log ratio
< log(100/97)`, RGOOD `< log(100/90)`.

### Auxiliary eval passes

- **`train_noaug`** — 5 % of the train split, fetched with
  augmentations OFF. Metrics prefixed `val/single/train_noaug/*`.
  Artifacts saved under `{run_dir}/eval_{step}/train_noaug/` for
  direct side-by-side comparison with val.
- **`benchmarks`** — 10 input-distortion modes on 5 % of val per
  eval: `normal`, `no_audio`, `no_future_audio`, `no_past_audio`,
  `static_audio`, `no_context`, `random_context`, `metronome`,
  `advanced_metronome`, `context_time_shifted` (renamed from
  `time_shifted` — it only rescales past-event offsets, not audio).
- **AR corpus hook** — every eval runs AR inference on 10 % of val
  charts using the LIVE model (no checkpoint reload), both `gt` and
  `fixed` conditioning modes. Per-mode averaged scalars merged under
  `val/single/corpus/{mode}_cond[_cmp]/*_mean`.

---

## Inference (autoregressive)

Not run in the main eval pass (but the corpus hook runs AR on 10 %
of val charts). Reconstructs:
- `EventEmbeddingDetector` (the model).
- `ArgmaxDecoder(b_pred=500)` — raw argmax over all 501 logits.
- `DetectionARInputBuilder(a_bins=500, b_bins=500, c_events=128)`.

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

- Name: `taiko2_v1_onset`.
- Source: parsed osu!taiko `.osz` packs (same packs as
  `taiko2_v1`).
- Audio sampler: `MelOnsetSampler` (subclass of `MelSampler`) —
  identical mel pipeline + appended sub-band-spectral-flux rows
  (see Audio preprocessing → Sub-band spectral flux computation
  above). Output features: `(84, T)` float32 → cached as float16.
- Event sampler: identical to `taiko2_v1` — `FixedRateEventSampler`
  at `divisor = 200` (5 ms per bin).
- Charts: 10,048 (same packs as `taiko2_v1`; expected to match
  exactly).
- Total onsets: 6,934,185 (same).
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
