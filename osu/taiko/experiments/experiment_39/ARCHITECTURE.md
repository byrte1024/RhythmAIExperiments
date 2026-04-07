# Experiment 39 — Full Analysis Specification

## Task

Diagnostic analysis: determine what fraction of the exp 35-C model's overpredictions (predicting a bin further than the nearest target) actually match real future onsets in the window.

## Analysis Type

Post-hoc evaluation of a trained model checkpoint. No training performed.

## Model Analyzed

Exp 35-C eval 8 checkpoint (OnsetDetector with exponential decay mel ramps, ~19M params). This was the ATH model at 71.6% HIT.

## Method

### Step 1: Run inference on validation set

Load the exp 35-C eval 8 checkpoint and run forward passes on the full validation set. For each sample, collect:
- Top-1 prediction (argmax of softmax logits)
- Target (nearest onset bin)
- All future onsets in the forward window (bins 0-499)

### Step 2: Classify predictions

For each non-STOP sample:
- **HIT**: prediction within 3% ratio or +/-1 frame of nearest target
- **Overprediction**: pred > target (predicted further than nearest onset)
- **Underprediction**: pred < target

### Step 3: Match overpredictions to future onsets

For each overprediction, check if it matches ANY real onset in the forward window (not just the nearest). A match uses the same 3% ratio or +/-1 frame tolerance.

### Step 4: Top-K analysis

For K = 1, 2, 3, 5, 10, collect the top-K predictions and check each against all future onsets. Report cumulative match rates.

## Scripts Used

Analysis script run in the experiment directory, using:
- `detection_train.py` evaluation infrastructure for checkpoint loading and forward passes
- Custom analysis code computing overprediction matching against all future onsets per sample
- Top-K extraction from softmax logits

## Data

- Validation set from taiko_v2 (~10% of 10,048 charts, split by unique song)
- 74,075 non-STOP validation samples analyzed

## Key Results

| Metric | Value |
|---|---|
| Total non-STOP samples | 74,075 |
| HIT (nearest) | 53,038 (71.6%) |
| MISS | 21,037 (28.4%) |
| Overpredictions | 13,269 (17.9%) |
| Underpredictions | 7,768 (10.5%) |
| Overpredictions matching ANY future onset | 11,041 (83.2%) |
| Overpredictions matching 2nd onset specifically | 9,042 (68.1%) |
| Overpredictions matching NO onset | 2,228 (16.8%) |
| Theoretical HIT if overpred counted | 86.5% (+14.9pp) |

### Top-K Analysis

| K | Nearest match | Any future match | Gain |
|---|---|---|---|
| 1 | 71.6% | 86.5% | +14.9% |
| 2 | 138.4% | 169.3% | +30.9% |
| 3 | 187.9% | 235.2% | +47.3% |
| 5 | 241.7% | 315.6% | +74.0% |
| 10 | 285.8% | 422.8% | +137.0% |

(Cumulative percentages: each K adds new matches from that rank position across all samples.)

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
