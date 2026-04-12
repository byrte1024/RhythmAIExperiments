# Experiment 65-S2 — Full Architecture Specification

## Task

Predict the next onset bin from rhythm pattern context alone. No audio input of any kind. The model receives the last 128 inter-onset gaps and gap ratios, plus density conditioning, and outputs a probability distribution over 251 classes (250 bin offsets + STOP).

## References

Architecture informed by:
- Bosser & Ben Taieb, "On the Predictive Accuracy of Neural Temporal Point Process Models for Continuous-time Event Data" (TMLR 2023) — GRU encoder outperforms self-attention for inter-arrival time prediction; log-transform of gaps is critical; LogNormMix decoder best for NLL.
- Panos, "Decomposable Transformer Point Processes" (NeurIPS 2024) — inter-event time prediction needs only recent context (Markov property); separating time prediction from pattern prediction improves both.
- Shchur et al., "Neural Temporal Point Processes: A Review" (IJCAI 2021) — standard 3-step TPP pipeline: encode events, compress history, parameterize next-event distribution. GRU + flexible decoder is simplest viable design.

## Input

| Input | Shape | Description |
|---|---|---|
| gap_sequence | (B, 128) | Past inter-onset gaps in bins (int64). gap[i] = event[i] - event[i-1]. Padded with 0 for missing events. |
| ratio_sequence | (B, 128) | Gap ratios: gap[i] / gap[i-1], clamped [0.1, 10.0]. First event ratio = 1.0. Padded with 0. |
| event_mask | (B, 128) | Bool mask, True = padding (no event at this position) |
| conditioning | (B, 3) | [density_mean, density_peak, density_std] from chart metadata |

## Output

| Output | Shape | Description |
|---|---|---|
| logits | (B, 251) | 250 onset bin offsets (0-249) + 1 STOP class (250) |

## Model: ContextPredictor

**Estimated parameters: ~3-4M**

### 1. Event Token Encoding

Each of 128 past events encoded into a d_model-dimensional token using three sinusoidal features concatenated and projected:

```
For event i:
  gap_i     = gap_sequence[i]
  ratio_i   = ratio_sequence[i]

  feat_log_gap   = SinusoidalPosEmb(log(gap_i + 1))       # (d_model,) — log-scaled gap (TPP best practice)
  feat_log_ratio = SinusoidalPosEmb(log(ratio_i) * 50)    # (d_model,) — log-ratio scaled to spread range
  feat_raw_gap   = SinusoidalPosEmb(gap_i)                 # (d_model,) — raw gap for absolute scale

  token_i = Linear(3 * d_model, d_model)(concat([feat_log_gap, feat_log_ratio, feat_raw_gap]))
  token_i = GELU(token_i)
  → (d_model,)
```

Padded events (mask=True) produce zero tokens.

### 2. Density Conditioning (FiLM)

```
conditioning (B, 3) → Linear(3, 64) → GELU → Linear(64, 64) → cond (B, 64)
```

Applied via FiLM after encoding and after each GRU/Transformer layer.

### 3. History Encoder: Bidirectional GRU

```
event_tokens (B, 128, d_model)
→ FiLM(cond)
→ Bidirectional GRU(input_size=d_model, hidden_size=d_model, num_layers=4, dropout=0.1)
→ output (B, 128, 2 * d_model)
→ take final hidden state from both directions
→ Linear(2 * d_model, d_model) → GELU
→ context (B, d_model)
```

Bidirectional GRU chosen over Transformer per TPP benchmark findings: GRU outperforms self-attention on average across 15 real-world event sequence datasets, is more stable without careful time encoding, and is computationally cheaper.

### 4. Output Head

```
context (B, d_model)
→ LayerNorm(d_model)
→ Linear(d_model, d_model) → GELU → Linear(d_model, N_CLASSES)
→ logits (B, 251)
```

Two-layer MLP head for expressivity. Direct 251-class prediction (no per-candidate scoring in v1).

### FiLM Conditioning

```
cond (B, 64) → Linear(64, 2 * d_model) → split → scale (B, d_model), shift (B, d_model)
x = x * (1 + scale) + shift
```

Applied to event tokens after encoding and to context vector before the output head.

### SinusoidalPosEmb

Standard sinusoidal positional encoding. Input: scalar values (gaps, ratios). Output: (d_model,) vector. Same implementation as the main detection model.

## Hyperparameters

| Param | Value | Rationale |
|---|---|---|
| d_model | 256 | Smaller than audio models — less data to process |
| GRU layers | 4 | Matches depth of current S1 proposer |
| GRU bidirectional | Yes | Sees full past context in both directions |
| GRU dropout | 0.1 | Standard |
| History length | 128 events | Same as current C_EVENTS |
| N_CLASSES | 251 | 250 bins + STOP (same as B_PRED=250 configs) |
| FiLM cond_dim | 64 | Same as current |

## Loss

```
loss = OnsetLoss(
    hard_alpha=0.5,
    good_pct=0.03,
    fail_pct=0.20,
    frame_tolerance=2,
    stop_weight=1.5,
    ramp_alpha=2.5,
    ramp_exp=1.0,
)
```

3-component loss:
- 0.5 * hard CE (exact bin precision)
- 0.5 * soft CE (trapezoid in log-ratio space, ±3% good, ±20% fail, ±2 frame floor)
- 2.5 * distance ramp (|log(E[pred]/target)|, gradient everywhere)

STOP weight 1.5x. Same loss as current best practice (exp 44e).

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 (no mel = lightweight samples) |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Subsample | 1 |
| Balanced sampling | ON (1/count^0.5 weights) |
| AMP | OFF |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Workers | 4 |

Expected training speed: 5-10x faster than mel-based models due to no spectrogram I/O.

## Dataset

Reuses taiko_v2 event data. No mel spectrogram loading.

For each sample:
1. Cursor position = previous event position (same as main training)
2. Gather up to 128 past events
3. Compute gaps: gap[i] = event[i] - event[i-1]
4. Compute ratios: ratio[i] = gap[i] / gap[i-1], clamped [0.1, 10.0]
5. Target: offset to next event, or STOP if beyond B_PRED

Same samples, same targets, same train/val split (90/10 by song, seed 42) as the main training pipeline.

## Augmentation

Minimal — context is the ONLY input, destructive augmentation is fatal.

| Aug | Rate | Params | Rationale |
|---|---|---|---|
| Event jitter | 100% | ±1 bin | Gentle timing noise, preserves ratios |
| Context truncation | 2% | Keep 32-128 most recent | Simulates sparse early-song context |
| Density jitter | 20% | ±5% | Lighter than audio models |

All destructive augmentations disabled: no deletion, insertion, metronome replacement, or large time shifts. These destroy the gap/ratio sequence which is the model's entire input.

## Metrics

### Standard per-sample
- HIT% (≤3% or ±1 frame), MISS% (>20%), accuracy, exact match
- Stop F1, precision, recall
- Frame error mean/median/p90
- Model score

### Context-specific
- **HIT by streak length**: streak 1-3, 4-7, 8+ — does S2 improve with clearer patterns?
- **HIT by ratio bucket**: 1.0x (repeat), 0.5x (double), 2.0x (halve), other
- **Metronome accuracy**: when target IS "continue the pattern"
- **Anti-metronome accuracy**: when target is NOT "continue the pattern"

### Benchmarks
| Benchmark | What it tests |
|---|---|
| normal | Standard performance |
| random_events | Random past events — should collapse to near-random |
| no_events | No past events — should be near-random |
| metronome | Fake metronome context — should predict continuation |
| advanced_metronome | Dominant-gap metronome — should predict dominant gap |
| zero_density | Zero density conditioning — density dependence |
| random_density | Random density — density specificity |

## Overlap Analysis (post-training)

The decisive experiment. Run both S1 (current best proposer) and S2 on the same val set:

| Category | Meaning | Desired |
|---|---|---|
| S1 correct, S2 correct | Both know the answer | Common cases |
| S1 correct, S2 wrong | Audio knows, context doesn't | Audio-dependent samples |
| S1 wrong, S2 correct | **Context knows, audio doesn't** | **The prize — S2's unique value** |
| S1 wrong, S2 wrong | Neither knows | Structurally hard samples |

If "S1 wrong, S2 correct" is >5% of samples, S2 adds meaningful signal for S3.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB) |
| OS | Windows 11 |
