# Experiment 010c — Ratio decomposition (128 bins) · Architecture

> **This document is self-contained.** Everything needed to reproduce
> this experiment is written inline. No links, no "see elsewhere."

---

## Task

Given a mel-spectrogram window around a cursor plus the last 128
onsets as bin-offsets from the cursor, predict the next onset by
decomposing the prediction into three sequential questions:

1. **Divisor** — what is the dominant rhythmic gap (the "beat")?
2. **Offset** — how far is the cursor from the last event?
3. **Ratio** — what multiple of the divisor is the next onset?

Final position: `predicted_bin = divisor × ratio_value − offset`.

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
| packed | `(B, 729)` | float32 | `[div_logits(500) | off_logits(100) | ratio_logits(129)]` |

Parsed as:
```
div_logits   = output[:, :500]          # (B, 500) — divisor head
off_logits   = output[:, 500:600]       # (B, 100) — offset head
ratio_logits = output[:, 600:729]       # (B, 129) — 128 ratio bins + STOP
```

---

## Data pipeline

### Audio preprocessing

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

### Sample construction

| Param | Value |
|---|---|
| a_bins / b_bins | 500 / 500 |
| c_events / d_events | 128 / 1 |
| min_cursor_bin | 6000 |
| allowed_overlap | 0 / 0 |
| subsample | 1 |
| split | train 90 % / val 10 %, seed 42, song-grouped |

### Target derivation

For the standard onset target (used by OnsetMetric for derived-bin
evaluation):
```
stop_idx = 500
if future_events_mask[0]:        target = stop_idx
elif cursor_offset < 0 or >= 500: target = stop_idx
else:                            target = cursor_offset
```

For the ratio heads (computed in the adapter when `ratio_mode=True`):

**Divisor target** — dominant gap from past events:
- Compute gaps between consecutive non-padded past events.
- Quantize gaps to nearest 3 bins for stability.
- Take the mode (most frequent quantized gap).
- **Valid** only when ≥2 past events exist AND the mode count ≥2
  (clear peak). Invalid samples get div_ce masked to 0.

**Offset target** — cursor distance from last event:
- `max(0, -last_real_event.cursor_offset)`.
- Normally 0 (cursor at last event). >0 after CursorShift aug or
  STOP hops at inference.
- **Valid** only when ≥1 real past event exists. Invalid samples
  get off_ce masked to 0.

**Ratio target** — DYNAMIC, computed from the model's own predictions:
```
div_val = soft_argmax(div_logits).detach()    # predicted divisor
off_val = soft_argmax(off_logits).detach()    # predicted offset
ratio_float = (target_bin + off_val) / max(div_val, 1.0)
ratio_target = nearest_log_bin(ratio_float)   # index into 255 ratio bins
```
STOP targets map to ratio_target = 255 (the STOP class).

### Class-balanced sampling

Same as #007: `weight[i] = min(1.0, 1 / (count[target(i)] + 1) ^ 0.5)`.

### Augmentations (training only)

15 augmentations in order:

| Order | Name | Probability | Parameters |
|---:|---|---|---|
| **0** | **CursorShift** (pre-sample) | **30 %** | shifts cursor forward between current and next event — trains offset head |
| 1 | TimeStretch | 30 % | max_scale = 1.4 |
| 2 | MelGainJitter | 30 % | ±2 dB |
| 3 | MelGaussianNoise | 15 % | σ 0.1–0.3 |
| 4 | MelFreqJitter | 15 % | shift ±3 |
| 5 | SpecAugFreq | 20 % | ≤10 bands |
| 6 | SpecAugTime | 20 % | ≤30 frames |
| 7 | EventJitter | 100 % | global ±3 + recency 1–2× |
| 8 | EventDropout | 5 % | drop 1–2 |
| 9 | EventInsertion | 3 % | 1 synthetic |
| 10 | PartialMetronome | 2 % | uniform gap |
| 11 | PartialAdvMetronome | 2 % | dominant gap |
| 12 | LargeTimeShift | 2 % | ±50 bins |
| 13 | ContextTruncation | 5 % | keep 8–32 |
| 14 | ConditioningJitter | 10 % | ±2 % |

CursorShift is a **pre-sample** augmentation (shifts the cursor
before audio loading), so the shifted cursor drives audio extraction
and event-offset computation natively — no post-hoc audio rolling.

---

## Model architecture

**`RatioDetector`** inherits the full `EventEmbeddingDetector`
backbone. Only the output head differs.

**Total parameters: ~16.9 M** (~500k more than the 501-class head
for the 3-head MLPs).

### Backbone (inherited, identical to #002–#009)

- Conditioning MLP: 3 → 64 (GELU) → 64.
- Conv stem: Conv1d(80→192, k=7, s=2, p=3) → GELU → GroupNorm →
  Conv1d(192→384, k=7, s=2, p=3) → GELU → transpose → (B, 250, 384)
  → LayerNorm → sinusoidal PosEmb → FiLM(cond). Cursor token = 125.
- Event embeddings: 5-part sinusoidal (presence, gap_before,
  gap_after, gap_ratio_before, gap_ratio_after) → Linear(1920, 384)
  → GELU → Linear(384, 384) → scatter-add at audio token positions.
- Transformer: 8 layers (d=384, 8 heads, ff=1536, dropout=0.1, gelu,
  norm_first) with per-layer FiLM (zero-initialized).

### Output heads (replaced)

**Head 1 — Divisor (auxiliary):**
```
cursor_tok (B, 384)
  → LayerNorm(384) → Linear(384, 192) → GELU → Linear(192, 500)
  → div_logits (B, 500)
```

**Head 2 — Offset (auxiliary):**
```
cursor_tok (B, 384)
  → LayerNorm(384) → Linear(384, 192) → GELU → Linear(192, 100)
  → off_logits (B, 100)
```

**Soft expectations (detached, no backprop into div/off heads):**
```
div_val = Σ_i softmax(div_logits)_i × i    → (B, 1), detached
off_val = Σ_i softmax(off_logits)_i × i    → (B, 1), detached
div_emb = Linear(1, 384)(div_val)
off_emb = Linear(1, 384)(off_val)
```

**Head 3 — Ratio (primary, receives div+off embeddings):**
```
ratio_input = cursor_tok + div_emb + off_emb
  → LayerNorm(384) → Linear(384, 384) → GELU → Linear(384, 256)
  → + Conv1d(1→8→1, k=5, p=2) smoothing
  → ratio_logits (B, 129)   # 128 ratio bins + 1 STOP  [CHANGED: was 256]
```

Conv1d smoothing prevents the ratio-collapse pathology taiko1 exp 67
observed (model snapping to ~5 favorite ratio values from 255 bins).

### Ratio bin table

**128 bins** log-spaced from 0.125× to 8.0× (6 octaves):
```
centers = exp(linspace(log(0.125), log(8.0), 128))
```
- 21 bins per octave, ~3.3% resolution per bin. [CHANGED: was 255/42/1.65%]
- bin 0 = 0.125×, bin 42 ≈ 0.5×, bin 63 ≈ 1.0×, bin 84 ≈ 2.0×,
  bin 127 = 8.0×.
- bin 128 = STOP (the 129th class).
- All standard musical ratios within ~1.5% of a bin center.
- Each bin gets ~2× more training signal than at 255 bins.

---

## Loss — RatioLoss

### Hyperparameters

| Param | Value |
|---|---|
| divisor_bins | 500 |
| offset_bins | 100 |
| ratio_bins | 128 |
| aux_weight | 0.1 |
| stop_weight | 1.5 |
| ratio_freeze_evals | 1 |

### Three components

**Divisor CE (auxiliary, weight 0.1):**
```
div_ce = CE(div_logits, div_target) × div_valid
```
Fixed GT target (IOI mode from past events). Masked to 0 when GT is
unreliable (< 2 past events or no clear peak). Stop-gradient: div_ce
does NOT backprop through the ratio head path.

**Offset CE (auxiliary, weight 0.1):**
```
off_ce = CE(off_logits, off_target) × off_valid
```
Fixed GT target (cursor − last_event). Masked when no real past event.

**Ratio CE (primary):**
```
ratio_target = find_nearest_ratio_bin((target + off_val) / div_val)
  # where div_val, off_val = detached soft expectations from heads 1+2
  # STOP targets → ratio_target = 255
ratio_ce = CE(ratio_logits, ratio_target)
```
Dynamic target from the model's own predictions. If the divisor head
predicts 2× the true beat, the ratio target becomes 0.5× to
compensate — the system is self-correcting.

**Total:**
```
per_sample = ratio_ce + aux_weight × (div_ce + off_ce)
per_sample *= stop_weight where target is STOP
loss = per_sample.mean()
```

### Warmup

For the first `ratio_freeze_evals` evals (~20,674 steps for
`evals_per_epoch=4`), the ratio CE is zeroed — only div + off heads
train. This lets the divisor and offset stabilize before the ratio
head sees their soft expectations. After warmup, all three heads
train jointly.

### Metrics reported

- `loss`, `ratio_ce`, `div_ce`, `off_ce`, `stop_rate`
- `ratio/rhit` — predicted ratio within ±3 % of true ratio (log-ratio)
- `ratio/rgood` — within ±10 %
- `ratio/rmiss` — 1 − rgood
- `ratio/div_acc` — exact divisor match
- `ratio/div_acc_3` — divisor within ±3 bins
- `ratio/off_acc` — exact offset match
- `ratio/frozen` — 1.0 during warmup, 0.0 after

### Loss curves

The loss graph shows `loss` (total) with companion curves for
`ratio_ce`, `div_ce`, and `off_ce` — each loss component tracked
separately so convergence of the three heads is visible.

---

## Per-eval artifacts

Standard set (all ratio-aware via `decode_pred_bins` — derives bin
from `divisor × ratio − offset`): heatmap, distributions,
ratio_error, error_hist, ratio_hit, metronome.

Ratio-specific (`{eval_dir}/ratio/`):
- `divisor_heatmap.png` — GT divisor vs predicted divisor.
- `offset_heatmap.png` — GT offset vs predicted offset.
- `ratio_heatmap.png` — GT ratio bin vs predicted ratio bin
  (dynamic target from model's own div/off predictions).
- `ratio_error.png` — histogram of log(pred_ratio / true_ratio)
  with reference lines at 1×, 2×, 0.5×, 3×, 1/3×.
- `ratio_decomp.npz` — raw arrays for post-hoc analysis.

Also under `{eval_dir}/train_noaug/` and
`{eval_dir}/train_noaug/ratio/` — same set for the augmentation-off
train pass.

---

## Inference (autoregressive)

**RatioDecoder**: at each AR step:
1. Parse the `(1, 856)` packed output into div/off/ratio logits.
2. Compute soft expectations: `div_val`, `off_val`.
3. Ratio argmax → if STOP class (index 255), emit STOP.
4. Else: `ratio_val = centers[argmax]`, `predicted_bin =
   round(div_val × ratio_val − off_val)`.

Per-step extras: `divisor`, `offset`, `ratio_val`, `ratio_idx`,
`div_argmax`, `off_argmax`.

AR loop: cursor=0, predict, emit onset at `cursor + predicted_bin`
or STOP (`cursor += 20`). Max 10,000 events. Default kind DON.

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
