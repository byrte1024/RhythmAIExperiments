# Experiment 39-C — Full Analysis Specification

## Task

Diagnostic analysis: extend exp 39-B reranking with an entropy-weighted term that only applies proximity bias when the model is uncertain (top candidates close in confidence).

## Analysis Type

Post-hoc inference-time reranking of a trained model's predictions. No training performed.

## Model Analyzed

Exp 35-C eval 8 checkpoint (OnsetDetector with exponential decay mel ramps, ~19M params, 71.6% HIT).

## Method

### Step 1: Collect top-10 predictions

Run the exp 35-C eval 8 checkpoint on the validation set. For each non-STOP sample, collect the top-10 predictions with their softmax probabilities.

### Step 2: Entropy-weighted reranking score

```
score = conf_w * confidence + pos_w * (1 - bin/500) + ent_w * (confidence / top1_confidence)
```

Three terms:
- `conf_w * confidence`: trust the model's confidence
- `pos_w * (1 - bin/500)`: prefer closer predictions
- `ent_w * (confidence / top1_confidence)`: confidence-relative bonus. High ent_w = strongly prefer candidates close to #1 in confidence (conservative). Low ent_w = let position override even when #1 is more confident.

### Step 3: Three-way weight sweep

Sweep conf_w, pos_w, and ent_w across a grid of weight combinations. For each, measure HIT rate, improvements, regressions, and net gain.

## Scripts Used

Analysis script extending exp 39-B with the entropy weight term. Same infrastructure for checkpoint loading and top-K extraction.

## Data

- Validation set from taiko_v2 (~10% of 10,048 charts)
- 74,075 non-STOP validation samples analyzed
- Top-10 predictions per sample

## Key Results

```
Baseline: 71.6%
Best: 72.6% (conf=0.7, pos=3.0, ent=0.5)
Improvement: +1.0pp
  Improved: 2,269
  Regressed: 1,559
  Net: +710
```

The entropy weight adds only +0.1pp over exp 39-B's +0.9pp. The regression ratio (~2:3) is unchanged across all weight combinations. All weight combinations cluster around the same +0.9-1.0pp ceiling.

Post-hoc reranking of the model's own top-K is fundamentally limited to ~+1pp. The model's confidence ranking is already strongly correlated with correctness.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| cuDNN | 9.10.02 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB, compute 12.0) |
| OS | Windows 11 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |
