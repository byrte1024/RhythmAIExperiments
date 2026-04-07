# Experiment 39-B — Full Analysis Specification

## Task

Diagnostic analysis: attempt to recover overprediction errors from exp 39 by reranking the model's top-K candidates using a weighted score combining confidence rank and position proximity.

## Analysis Type

Post-hoc inference-time reranking of a trained model's predictions. No training performed.

## Model Analyzed

Exp 35-C eval 8 checkpoint (OnsetDetector with exponential decay mel ramps, ~19M params, 71.6% HIT).

## Method

### Step 1: Collect top-10 predictions

Run the exp 35-C eval 8 checkpoint on the validation set. For each non-STOP sample, collect the top-10 predictions with their softmax probabilities.

### Step 2: Reranking score function

For each weight combination (conf_w, pos_w), rerank the top-10 candidates:

```
score = conf_w * normalized_confidence + pos_w * (1 - bin/500)
```

- `normalized_confidence`: softmax probability normalized within top-10
- `(1 - bin/500)`: linear proximity bias (closer bins score higher)
- Higher conf_w = trust model's ranking
- Higher pos_w = prefer closer predictions

### Step 3: Sweep weight combinations

Sweep conf_w in {0, 0.1, 0.3, 0.5, 0.7, 1.0} and pos_w in {0, 0.5, 1.0, 2.0, 3.0, 5.0}.

For each combination, measure:
- **HIT rate** after reranking
- **Improvements**: samples that changed from miss to hit
- **Regressions**: samples that changed from hit to miss
- **Net gain**: improvements - regressions

### Step 4: Track regressions

Critical metric: how many current HITs become misses from reranking. A method that improves 2000 but breaks 1500 is not practical.

## Scripts Used

Analysis script run in the experiment directory, using:
- `detection_train.py` evaluation infrastructure for checkpoint loading
- Custom reranking code with weight sweep over the top-10 predictions per sample

## Data

- Validation set from taiko_v2 (~10% of 10,048 charts)
- 74,075 non-STOP validation samples analyzed
- Top-10 predictions per sample

## Key Results

```
Baseline (argmax): 71.6%
Best reranked: 72.5% (conf_w=0.5, pos_w=2.0)
Improvement: +0.9pp
  Improved: 1,994 (miss → hit)
  Regressed: 1,299 (hit → miss)
  Net: +695
```

conf_w=0 (pure proximity): catastrophic -59.3%, picks smallest bin regardless of confidence.

The theoretical ceiling from exp 39 is +14.9pp, but reranking only recovers +0.9pp. Global position weight helps overpredictions but hurts correct distant predictions.

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
