# Experiment 65-S2 — Context-Only Onset Predictor

## Purpose

Can a model predict the next onset from rhythm patterns alone, with zero audio? This is the critical validation for the 3-stage architecture. If S2 can achieve meaningful accuracy from context alone, it proves context carries real predictive signal that our current models waste.

## The Question

Given:
- Last 128 inter-onset gaps (in bins): [33, 33, 66, 33, 33, 66, ...]
- Last 128 gap ratios (gap[i] / gap[i-1]): [1.0, 2.0, 0.5, 1.0, 2.0, ...]
- Density conditioning: [density_mean, density_peak, density_std]

Predict: which of 251 bins (0-249 offset + STOP) is the next onset?

No mel spectrogram. No audio of any kind. Pure pattern continuation.

## What Success Looks Like

For reference, our models achieve:
- Random baseline: ~0.4% accuracy (1/251)
- Audio-only (S1 proposer, exp58 eval 1): ~57.5% F1
- Full model (exp44): 73.7% HIT
- Full model (exp58): 74.6% HIT

S2 expectations:
- **>20% HIT would be significant** — proving context is genuinely predictive
- **>40% HIT would be extraordinary** — context alone captures nearly as much as the audio/context gap (exp44 no_events benchmark is ~68% HIT, context delta is ~5pp, so context contributes ~5pp on top of audio)
- **The miss overlap analysis is the real goal** — even 20% HIT S2 is valuable if it hits different samples than S1

## Model Architecture

### Input Encoding

Each of 128 past events encoded as a token:

```
For event i (i=0 is oldest, i=127 is most recent):
  gap_i = distance to previous event (bins)
  ratio_i = gap_i / gap_{i-1} (clamped to [0.1, 10.0])

  token_i = Linear(3 * d_model, d_model)(concat([
      SinusoidalEmb(log(gap_i + 1)),        # log-gap (TPP best practice)
      SinusoidalEmb(log(ratio_i) * 50),      # log-ratio scaled
      SinusoidalEmb(gap_i),                   # raw gap for absolute scale
  ]))
```

Padding mask for events that don't exist (fewer than 128 past events).

### Density Conditioning

Same FiLM approach as our current models:
```
cond = [density_mean, density_peak, density_std]
→ Linear(3, 64) → GELU → Linear(64, 64) → FiLM
```

Density IS context — the model needs to know "this is a dense chart" to predict appropriate gaps.

### History Encoder

**Primary: Bidirectional GRU** (TPP benchmark winner for time prediction)

```
event_tokens (B, 128, d_model)
→ Bidirectional GRU(d_model, n_layers=4)
→ hidden states (B, 128, 2 * d_model)
→ take last hidden → Linear(2 * d_model, d_model) → context (B, d_model)
```

**Alternative: Causal Transformer** (if GRU underperforms)

```
event_tokens (B, 128, d_model)
→ + positional encoding (position in sequence)
→ 6 causal transformer layers (d_model, 8 heads)
→ FiLM conditioning each layer
→ last token → context (B, d_model)
```

### Output Head

```
context (B, d_model)
→ LayerNorm → Linear(d_model, N_CLASSES)
→ logits (B, 251)
```

Simple direct head. No per-candidate encoding for v1 — keep it minimal.

### Parameters

| Param | Value |
|---|---|
| d_model | 256 |
| GRU layers | 4 (bidirectional) |
| n_heads (if transformer) | 8 |
| History length | 128 events |
| N_CLASSES | 251 (250 bins + STOP) |
| FiLM cond_dim | 64 |

**Estimated params: ~3-4M** (lightweight — fast to train)

## Dataset

Reuses taiko_v2 but only needs event data, no mel spectrograms.

For each sample (cursor position in a chart):
1. Gather up to 128 past events as offsets from cursor
2. Compute gaps: `gap[i] = event[i] - event[i-1]`
3. Compute ratios: `ratio[i] = gap[i] / gap[i-1]`
4. Target: offset to next event (same as current), or STOP

### Key difference from current dataset
- **No mel loading** — dramatically faster training (mel I/O is the bottleneck)
- **No audio augmentation** — only context augmentations apply
- Samples and targets are identical to the main training

### Context augmentation (much lighter than audio models)

Since S2 has NO audio fallback, destructive augmentations are fatal. Only use gentle noise that preserves the pattern structure.

| Aug | Rate | Params | Rationale |
|---|---|---|---|
| Event jitter | 100% | ±1 bin (NOT ±3) | Gentle timing noise, preserves ratios |
| Event deletion | OFF | — | Destroys gap sequence entirely |
| Event insertion | OFF | — | Creates fake gaps that don't exist |
| Partial metronome | OFF | — | Replaces real patterns with fake |
| Partial adv metronome | OFF | — | Same problem |
| Large time shift | OFF | — | Changes all gaps — fatal for ratio model |
| Context truncation | 2% | Keep 32-128 most recent | Mild, simulates sparse context |
| Density jitter | 20% | ±5% on all 3 values | Lighter than usual, density is important input |

## Loss

```
loss = OnsetLoss(hard_alpha=0.5, good_pct=0.03, fail_pct=0.20, 
                 frame_tolerance=2, stop_weight=1.5, ramp_alpha=2.5)
```

Same loss as current best (with ramp from exp44e). The soft trapezoid is important here — context predictions won't be frame-accurate, so credit for being close matters.

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Batch size | 256 (no mel = much more memory) |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Balanced sampling | ON |
| Gradient clipping | 1.0 |
| Evals per epoch | 4 |
| Workers | 4 |

Batch size 256 should be feasible since each sample is just 128 integers + conditioning, no mel tensor.

## Metrics

### Standard per-sample
- HIT% (≤3% or ±1 frame), MISS% (>20%), accuracy, exact match
- Stop F1, precision, recall
- Frame error mean/median/p90

### Context-specific
- **HIT by streak length**: Does S2 improve when there's a clear streak? (streak 1-3, 4-7, 8+)
- **HIT by ratio bucket**: Accuracy for 1.0x (repeat), 0.5x (double), 2.0x (halve), other
- **HIT by density**: Low-density (sparse) vs high-density (dense) charts
- **Metronome accuracy**: When target IS "continue the pattern," how often does S2 get it?
- **Anti-metronome accuracy**: When target is NOT "continue the pattern," how often does S2 get it?

### For overlap analysis (post-training)
- Run S2 on the same val set as S1
- For each sample, record: S1 correct?, S2 correct?, both correct?, neither correct?
- **Key metric: S2_correct AND S1_wrong** — samples where context knows the answer but audio doesn't
- **Also: S1_correct AND S2_wrong** — samples where audio knows but context doesn't

## Benchmarks

### Adapted from current benchmarks (no audio variants removed)
| Benchmark | What it tests |
|---|---|
| normal | Standard performance |
| random_events | Random past events (context destroyed) — should collapse to ~0% |
| no_events | No past events at all — should be near-random |
| metronome | Fake metronome context — should predict metronome continuation |
| advanced_metronome | Dominant-gap metronome — should predict dominant gap |
| zero_density | Zero density conditioning — density dependence |
| random_density | Random density — density specificity |

`random_events` is the critical check — if S2 performs the same with random events as real events, it hasn't learned context at all.

## Expected Timeline

Since there's no mel loading, training should be **5-10x faster** than our normal experiments. A full 50-epoch run might finish in a few hours on a single GPU.

## Outcome Scenarios

### Scenario A: S2 HIT > 40%
Context is strongly predictive. Proceed with full 3-stage architecture. S2 will contribute meaningful signal to S3.

### Scenario B: S2 HIT 20-40%
Context has moderate predictive power. S2 is useful for specific cases (high-streak, clear patterns) but not universally. Overlap analysis determines if it's worth the complexity.

### Scenario C: S2 HIT < 20%
Context alone can't predict onsets meaningfully. The gap between audio bins is too ambiguous without audio guidance. Pivot: S2 may still work as a reranker (given S1's candidates, which is most likely?) rather than independent predictor.

### Regardless of S2 accuracy:
The overlap analysis (S1 miss ∩ S2 hit) is the decisive metric. Even 15% HIT S2 is valuable if those 15% are samples S1 gets wrong.
