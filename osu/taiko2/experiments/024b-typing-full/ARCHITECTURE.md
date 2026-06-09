# Architecture -- Experiment 024b: Typing model full run

## Task

Given a sequence of known onset positions in a taiko chart, predict
each onset's **type** (DON or KA) and **strength** (normal or big).
DRUMROLL and SPINNER are excluded. The model operates one onset at a
time in autoregressive fashion: past D/K labels are known, future
D/K labels are not.

## Dataset

- Source: `taiko2_v1` (10,048 charts, 6,934,185 hit events after
  filtering DRUMROLL/SPINNER/UNKNOWN).
- Split: 90/10 song-grouped, seed 42.
- Subsample: 1 (full dataset).
- Train samples: ~6,265,000. Val samples: ~653,000.
- Events file: `events/{chart_id}.npz` -- `bins` (int32), `times_ms`
  (int32), `kind_ids` (uint8). Kind mapping: 0=DON, 1=KA, 2=BIG_DON,
  3=BIG_KA, 4=DRUMROLL, 5=SPINNER, 6=UNKNOWN.
- Features file: `features/{audio_stem}.npy` -- (80, T) float16
  log-mel spectrogram, 80 bands, 200 fps (5 ms/frame), sr=22000,
  n_fft=2048, hop=110, f_min=20, f_max=8000, power=2, top_db=80.

## Sample construction

Each training sample is one onset (the "target") with its surrounding
context window.

### Filtering

Only hit onsets (kind_id in {0, 1, 2, 3}) are kept. Charts with
fewer than 3 hits are skipped.

### Window geometry

- Past context: 16 onsets immediately preceding the target, with
  known D/K and big/normal labels. Padded at the start if the
  target is near the beginning of the chart.
- Target: 1 onset. Type and strength are the prediction targets.
- Future context: 16 onsets immediately following the target. Their
  positions and IOIs are known; their types are marked UNKNOWN.
  Padded at the end if near chart boundary.

Total window: 33 onset tokens.

### Per-onset features

For each of the 33 onsets in the window:

1. **Mel patch**: 5 frames centered on the onset bin from the cached
   (80, T) feature array. Shape: (80, 5) = 400 values, flattened.
   Zero-padded if near audio boundary.

2. **IOI features**: 3 scalars.
   - `log_ioi_before = log1p(|bin[i] - bin[i-1]|)`. Zero for the
     first onset.
   - `log_ioi_after = log1p(|bin[i+1] - bin[i]|)`. Zero for the
     last onset.
   - `log_ioi_ratio = log_ioi_before - log_ioi_after`. Zero when
     either IOI is zero.

3. **Kind label**: integer.
   - Past onsets: 0 = DON (includes BIG_DON), 1 = KA (includes
     BIG_KA). D/K merges big into normal for the type label.
   - Target onset: 2 = UNKNOWN.
   - Future onsets: 2 = UNKNOWN.

4. **Big label**: integer.
   - Past onsets: 0 = normal, 1 = big.
   - Target onset: 2 = UNKNOWN.
   - Future onsets: 2 = UNKNOWN.

5. **Position**: integer index 0..32 mapping to relative position
   -16..0..+16 within the window.

### Padding

Padded tokens (near chart boundaries) have:
- Mel patch = zeros.
- IOI features = zeros.
- Kind label = UNKNOWN (2).
- Big label = UNKNOWN (2).
- Padding mask = True (masked in transformer attention).

## Augmentation

Five augmentations, applied in the adapter during batching (training
only):

| Augmentation | Scope | Probability / Parameter | Effect |
|---|---|---|---|
| D/K flip | per sample | 50 % | Swap all D/K labels in the window (past labels + target GT). Strength labels unchanged. Enforces corpus-wide D/K symmetry. |
| Future context dropout | per sample | 10 % | Mask all 16 future tokens as padded. Forces prediction from past context alone. |
| Past label dropout | per token | 15 % | Set individual past kind/big labels to UNKNOWN. Simulates early-chart AR where few past predictions exist. |
| IOI jitter | per token | sigma=0.05 | Additive gaussian noise on all 3 log-IOI features. Simulates onset detector position noise. |
| Mel noise | per token | sigma=0.1 | Additive gaussian noise on flattened mel patch. Simulates audio variation. |

Past label dropout, IOI jitter, and mel noise are new in 024b
(024 had only the D/K flip and future dropout). They address the
teacher-forced-to-AR gap by injecting the kinds of noise the model
will encounter at inference time.

## Model

### Architecture

Bidirectional transformer encoder over 33 onset tokens.

### Input encoding

Each of the 33 tokens is encoded as:

```
mel_patch (400) -> Linear(400, 32) -> ReLU -> (32,)
ioi_features (3) -> Linear(3, 16) -> ReLU -> (16,)
kind_label -> Embedding(3, 16) -> (16,)
big_label -> Embedding(3, 16) -> (16,)
position -> Embedding(33, 32) -> (32,)

Concatenate: (112,) -> Linear(112, 64) -> (64,)
```

### Transformer

```
Type:           nn.TransformerEncoder
d_model:        64
n_layers:       3
n_heads:        4 (head_dim = 16)
d_ff:           256
dropout:        0.1
norm_first:     True (pre-LN)
batch_first:    True
Attention mask: None (full bidirectional)
Padding mask:   src_key_padding_mask from sample padding (True = ignore)
```

Input shape: (B, 33, 64).
Output shape: (B, 33, 64).

### Output heads

Extract the center token (index 16, the target onset):

```
target_features = output[:, 16, :]    # (B, 64)

type_head:     Linear(64, 1) -> squeeze -> (B,)   # raw logit
strength_head: Linear(64, 1) -> squeeze -> (B,)   # raw logit
```

At inference: `sigmoid(type_logit) > 0.5` -> DON, else KA.
`sigmoid(strength_logit) > threshold` -> BIG, else NORMAL.
Threshold for strength is determined by per-eval sweep (not fixed
at 0.5).

### Parameter count

| Component | Parameters |
|---|---:|
| Mel projection (Linear 400->32 + bias) | 12,832 |
| IOI projection (Linear 3->16 + bias) | 64 |
| Kind embedding (3 x 16) | 48 |
| Big embedding (3 x 16) | 48 |
| Position embedding (33 x 32) | 1,056 |
| Input projection (Linear 112->64 + bias) | 7,232 |
| Transformer (3 layers) | ~149,000 |
| Type head (Linear 64->1 + bias) | 65 |
| Strength head (Linear 64->1 + bias) | 65 |
| **Total** | **~171,000** |

## Loss

Two independent binary cross-entropy terms plus an entropy penalty
on the type head:

```
type_bce = BCEWithLogits(type_logit, type_target)
type_entropy = mean(H(sigmoid(type_logit)))
    where H(p) = -p*log(p) - (1-p)*log(1-p)
strength_bce = BCEWithLogits(strength_logit, strength_target,
                             pos_weight=17.0)
total_loss = type_bce + 0.1 * type_entropy + strength_bce
```

- `type_target`: 1.0 = DON, 0.0 = KA. After D/K flip augmentation,
  the target is flipped correspondingly.
- `strength_target`: 1.0 = BIG, 0.0 = NORMAL. Unaffected by D/K
  flip.
- `pos_weight=17.0` for strength: BIG is ~5.5 % of hits, so
  17 approximates 94.5/5.5 to balance the gradient.
- `entropy_weight_type=0.1`: penalizes uncertain type predictions
  (sigmoid near 0.5). H(p) is maximized at 0.693 nats (p=0.5) and
  minimized at 0 (p=0 or p=1). The penalty pushes the type sigmoid
  toward commitment. Combined with BCE pulling toward the correct
  class, the total loss rewards confident-and-correct predictions
  more than uncertain-but-correct ones.
- `entropy_weight_strength=0.0`: strength head already has high
  decisive mass (75.7 % at 024 E6); no penalty needed.

### Per-batch metrics (logged every step)

```
loss, type_loss, strength_loss, type_entropy, strength_entropy,
type_acc, strength_acc, combined_acc
```

## Training schedule

- Optimizer: AdamW, lr=3e-4, weight_decay=0.01.
- Scheduler: CosineAnnealingLR, T_max = total_steps.
- Gradient clipping: max_norm=1.0.
- Epochs: 10.
- Batch size: 128.
- Evals per epoch: 2 (20 evals total).
- Seed: 42.
- Metric to watch: `typing/type/accuracy` (higher is better).
- No weighted sampling (D/K is balanced; strength pos_weight handles
  BIG imbalance in the loss).

## Eval metrics

### TypingMetric (computed per eval on full val pass)

**Type head (D/K):**
- accuracy, precision_D, recall_D, f1_D, precision_K, recall_K, f1_K
- conf_correct, conf_wrong, conf_mean, conf_std
- entropy_mean, entropy_std
- mass_decisive (fraction with max confidence > 0.9)
- mass_conflicted (fraction with max confidence 0.4-0.6)
- Threshold sweep at 9 thresholds (0.30..0.70): accuracy, f1_D, f1_K
  per threshold. best_threshold, best_f1 tracked as scalars.

**Strength head (normal/big):**
- accuracy, precision_BIG, recall_BIG, f1_BIG, precision_NORMAL,
  recall_NORMAL, f1_NORMAL
- conf_correct, conf_wrong, conf_mean, conf_std
- entropy_mean, entropy_std
- mass_decisive, mass_conflicted
- Threshold sweep at 12 thresholds (0.05..0.80): accuracy,
  precision_BIG, recall_BIG, f1_BIG per threshold.
  best_threshold, best_f1_BIG tracked as scalars.

**Combined 4-class (DON/KA/BDON/BKA):**
- accuracy
- Per-class precision, recall, f1 for all 4 classes.

Same metric class used for both train and val -- the training loop
handles namespacing.

## Eval artifacts (saved per eval to eval_{step}/typing/)

14 plots:
- type_confusion.png (2x2)
- strength_confusion.png (2x2)
- combined_confusion.png (4x4)
- type_confidence_dist.png (bimodal histogram, correct vs wrong)
- strength_confidence_dist.png
- type_calibration.png (predicted prob vs actual positive rate)
- strength_calibration.png
- type_entropy_dist.png (correct vs wrong entropy)
- strength_entropy_dist.png
- type_conf_vs_acc.png (accuracy per confidence bin)
- strength_conf_vs_acc.png
- type_threshold_sweep.png
- strength_threshold_sweep.png

2 data files:
- type_predictions.npz (probs, targets, preds arrays)
- strength_predictions.npz

## AR evaluation hook

Every 2 evals, `TypingARHook` runs the typing model autoregressively
over 100 val charts using `inference.typing_pass.type_chart` -- the
same code path `cli/infer.py --typing-config` uses. The hook:

1. Builds a stub Chart with DON-only onsets from the val sampler's
   event bins.
2. Calls `type_chart(model, stub, features, device, config)` which
   predicts one onset at a time, feeding own past predictions as
   context.
3. Compares the predicted D/K + BIG sequence against GT.
4. Saves per-chart CSV + aggregate JSON + 4 plots under
   `{run_dir}/typing_ar/eval_{step}/`.
5. Merges headline scalars into `val_metrics` for auto-graphing.

Per-chart metrics:
- type_accuracy, type_accuracy_sym (max of raw and all-flipped)
- type_f1_D, type_f1_K
- type_pattern_match_4, type_pattern_match_8 (fraction of N-onset
  windows matching GT or GT-flipped)
- type_ngram_tvd_2, type_ngram_tvd_4 (total variation distance
  between predicted and GT n-gram distributions)
- type_transition_tvd (TVD between 2x2 transition matrices)
- type_alternation_rate_pred, type_alternation_rate_gt, delta
- type_run_length_tvd (TVD between run-length histograms)
- strength_accuracy, strength_precision/recall/f1_BIG
- big_ratio_pred, big_ratio_gt, delta

Strength threshold for AR is read from
`val_metrics["typing/strength/best_threshold"]` (the per-eval sweep
result), so AR uses the calibrated operating point.

Plots saved per AR eval:
- ar_type_accuracy_hist.png (per-chart accuracy_sym distribution)
- ar_alternation_scatter.png (pred vs GT alternation rate)
- ar_strength_f1_hist.png (per-chart strength F1 distribution)
- ar_ngram_tvd_hist.png (per-chart 4-gram TVD distribution)

## Inference

The typing model runs as a second pass after the onset detector.

### Single chart (`cli/infer.py`)

```
--typing-config osu/taiko2/experiments/024b-typing-full/config/infer.json
```

Typing spec JSON:
```json
{
  "checkpoint": "osu/taiko2/runs/exp_024b_typing_full/checkpoints/best.pt",
  "config": {
    "__class__": "osu.taiko2.inference.typing_pass:TypingInferConfig",
    "strength_threshold": 0.8,
    "bin_ms": 5.0
  }
}
```

### Corpus eval (`InferCorpusConfig`)

Set `typing_spec` in the corpus config JSON to the same spec path.

### AR loop

For each onset in order:
1. Past 16 onsets with their predicted D/K + big labels.
2. Target onset with UNKNOWN labels.
3. Future 16 onsets with positions/IOIs only, UNKNOWN labels.
4. Forward pass through the 171K-param transformer -> sigmoid ->
   threshold.
5. Assign kind = _DK_BIG_TO_KIND[(dk, big)] from the prediction.
6. Add to history, advance.

500 onsets at 171K params on 33 tokens each = well under 1 second
per chart on GPU.

## Environment

- Python 3.13, PyTorch nightly (from pyproject.toml + uv.lock).
- Training on CUDA.
- Dataset: taiko2_v1, pre-computed mel features on disk.
