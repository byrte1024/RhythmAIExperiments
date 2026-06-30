# Architecture -- Experiment 025: Typing model with temporal attention bias

## Task

Given a sequence of known onset positions in a taiko chart, predict
each onset's **type** (DON or KA) and **strength** (normal or big).
DRUMROLL and SPINNER are excluded. The model operates one onset at a
time in autoregressive fashion: past D/K labels are known, future
D/K labels are not.

## Dataset

- Source: `taiko2_v1` (10,048 charts, 6,918,036 hit events after
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

- Past context: 32 onsets immediately preceding the target, with
  known D/K and big/normal labels. Padded at the start if the
  target is near the beginning of the chart.
- Target: 1 onset. Type and strength are the prediction targets.
- Future context: 32 onsets immediately following the target.
  Their positions and IOIs are known; their types are marked UNKNOWN.
  Padded at the end if near chart boundary.

Total window: 65 onset tokens.

### Per-onset features

For each of the 65 onsets in the window:

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

5. **Position**: integer index 0..64 mapping to relative position
   within the window.

6. **Onset bin**: integer, the absolute bin position (frame index)
   of the onset in the feature array. Used by the temporal attention
   bias to compute pairwise temporal distances. Zero for padded
   tokens.

### Padding

Padded tokens (near chart boundaries) have:
- Mel patch = zeros.
- IOI features = zeros.
- Kind label = UNKNOWN (2).
- Big label = UNKNOWN (2).
- Onset bin = 0.
- Padding mask = True (masked in transformer attention).

## Augmentation

Five augmentations, applied in the adapter during batching (training
only):

| Augmentation | Scope | Probability / Parameter | Effect |
|---|---|---|---|
| D/K flip | per sample | 50 % | Swap all D/K labels in the window (past labels + target GT). Strength labels unchanged. |
| Future context dropout | per sample | 10 % | Mask all 32 future tokens as padded. |
| Past label dropout | per token | 15 % | Set individual past kind/big labels to UNKNOWN. |
| IOI jitter | per token | sigma=0.05 | Additive gaussian noise on log-IOI features. |
| Mel noise | per token | sigma=0.1 | Additive gaussian noise on flattened mel patch. |

## Model

### Architecture

Bidirectional transformer encoder over 65 onset tokens with
**Gaussian temporal attention bias**.

### Input encoding

Each of the 65 tokens is encoded as:

```
mel_patch (400) -> Linear(400, 32) -> ReLU -> (32,)
ioi_features (3) -> Linear(3, 16) -> ReLU -> (16,)
kind_label -> Embedding(3, 16) -> (16,)
big_label -> Embedding(3, 16) -> (16,)
position -> Embedding(65, 32) -> (32,)

Concatenate: (112,) -> Linear(112, 64) -> (64,)
```

### Temporal attention bias

The temporal bias is injected via a **custom transformer
implementation** (`TemporalTransformerEncoder`) that replaces
PyTorch's `nn.TransformerEncoder`. The custom implementation gives
full control over the attention computation, avoiding numerical
issues that arise when combining PyTorch's `src_mask` with
`src_key_padding_mask`.

Each `TemporalTransformerLayer` contains:
- `TemporalMultiheadAttention`: custom MHA that computes QKV via a
  single fused linear, applies the temporal bias to attention scores
  before softmax, and handles padding by masking key columns to
  `-inf` separately from the temporal bias.
- Pre-LN residual connection.
- Feed-forward block (Linear -> GELU -> Dropout -> Linear -> Dropout).

The temporal bias is computed once per forward pass:

```
diff_ij = onset_bin_i - onset_bin_j            # signed bin distance
dist_sq_ij = diff_ij^2                         # squared distance
sigma = clamp(exp(log_sigma), min=1, max=500)  # learnable, clamped
bias_ij = -dist_sq_ij / (2 * sigma^2)          # Gaussian decay

attn_ij = (Q_i . K_j^T) / sqrt(d_k) + bias_ij
```

Where:
- `onset_bin_i` is the absolute frame index of onset i.
- `sigma` is a **learnable parameter**, initialized at 50.0 bins
  (250 ms at 5 ms/bin). Stored as `log(sigma)` for numerical
  stability. Clamped to [1, 500] to prevent collapse or explosion.
- The bias is computed as a (B, W, W) matrix, broadcast across all
  heads (shared sigma for all heads and layers).
- Padding is handled separately: padded key columns receive `-inf`
  in the attention scores via `masked_fill`, after the temporal
  bias has been added.

The bias encodes: "onsets close in time attend strongly to each
other; onsets far apart attend weakly." At sigma=50 bins:
- Onsets 1 beat apart (~34 bins at 170 BPM): bias = -0.23
  (moderate attenuation)
- Onsets 4 beats apart (~136 bins): bias = -3.7 (strong attenuation)
- Onsets in the same beat subdivision (~10 bins): bias = -0.02
  (near-full attention)

### Transformer

```
Type:           TemporalTransformerEncoder (custom)
d_model:        64
n_layers:       3
n_heads:        4 (head_dim = 16)
d_ff:           256
dropout:        0.1
Normalization:  Pre-LN (LayerNorm before attention and FF)
Activation:     GELU
Attention:      Custom TemporalMultiheadAttention with Gaussian bias
Padding:        masked_fill on key columns (True = ignore)
```

Input shape: (B, 65, 64).
Output shape: (B, 65, 64).

### Output heads

Extract the target token (index 32, after all past tokens):

```
target_features = output[:, 32, :]    # (B, 64)

type_head:     Linear(64, 1) -> squeeze -> (B,)   # raw logit
strength_head: Linear(64, 1) -> squeeze -> (B,)   # raw logit
```

At inference: `sigmoid(type_logit) > 0.5` -> DON, else KA.
`sigmoid(strength_logit) > threshold` -> BIG, else NORMAL.
Threshold for strength is determined by per-eval sweep.

### Parameter count

| Component | Parameters |
|---|---:|
| Mel projection (Linear 400->32 + bias) | 12,832 |
| IOI projection (Linear 3->16 + bias) | 64 |
| Kind embedding (3 x 16) | 48 |
| Big embedding (3 x 16) | 48 |
| Position embedding (65 x 32) | 2,080 |
| Input projection (Linear 112->64 + bias) | 7,232 |
| Transformer (3 layers) | ~149,000 |
| Type head (Linear 64->1 + bias) | 65 |
| Strength head (Linear 64->1 + bias) | 65 |
| Temporal log_sigma | 1 |
| **Total** | **~172,000** |

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
- `pos_weight=17.0` for strength.
- `entropy_weight_type=0.1`: penalizes uncertain type predictions.
- `entropy_weight_strength=0.0`: strength head already decisive.

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
- No weighted sampling.

## Eval metrics

Same as 024c: TypingMetric (type P/R/F1, strength P/R/F1, combined
4-class, confidence, entropy, threshold sweeps). 14 plots + 2 npz
per eval. TypingARHook fires every eval on 100 val charts.

## AR evaluation hook

Every eval, `TypingARHook` runs the typing model autoregressively
over 100 val charts using `inference.typing_pass.type_chart`. The
function reads `past_context`, `future_context`, and
`temporal_bias` from `model.config`, so AR inference automatically
uses the temporal bias when the model was trained with it.

## Inference

Same spec pattern as 024c. `config/infer.json` points to the
best checkpoint. The temporal bias is part of the model architecture
stored in the checkpoint -- no config changes needed at inference.

## Environment

- Python 3.13, PyTorch nightly, CUDA.
- Dataset: taiko2_v1, pre-computed mel features on disk.
