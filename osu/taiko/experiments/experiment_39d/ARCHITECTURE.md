# Experiment 39-D — Full Analysis Specification

## Task

Diagnostic analysis: investigate how many HIT-matching candidates exist per sample in the top-10, and when the model is wrong, how close in confidence the correct answer is to the chosen wrong answer.

## Analysis Type

Post-hoc analysis of a trained model's top-K prediction distribution. No training performed.

## Model Analyzed

Exp 35-C eval 8 checkpoint (OnsetDetector with exponential decay mel ramps, ~19M params, 71.6% HIT).

## Method

### Analysis 1: HIT count per sample in top-10

For each non-STOP sample:
1. Extract top-10 predictions
2. Count how many of the 10 candidates match the target (within 3% ratio or +/-1 frame)
3. Report distribution of HIT counts

### Analysis 2: When wrong, where is the correct answer?

For each sample where top-1 is wrong (28.4% of samples):
1. Check if any of top-10 candidates match the target
2. Record the rank position of the first correct candidate
3. Record the confidence (softmax probability) of the correct candidate vs the chosen wrong candidate
4. Compute the confidence ratio

## Scripts Used

Analysis script run in the experiment directory, using:
- `detection_train.py` evaluation infrastructure for checkpoint loading
- Custom analysis code for top-K HIT counting and confidence gap measurement

## Data

- Validation set from taiko_v2 (~10% of 10,048 charts)
- 74,075 non-STOP validation samples analyzed
- Top-10 predictions per sample with softmax probabilities

## Key Results

### Analysis 1: HIT count distribution in top-10

| HITs in top-10 | Samples | % |
|---|---|---|
| 0 | 3,048 | 4.1% |
| 1 | 3,147 | 4.2% |
| 2 | 7,041 | 9.5% |
| 3 | 55,212 | 74.5% |
| 4+ | 5,627 | 7.6% |

Mean: 2.86, Median: 3. The model produces a tight cluster of ~3 adjacent correct bins, not scattered candidates.

### Analysis 2: Wrong predictions (21,037 samples, 28.4%)

- Correct answer in top-10: 17,989 (85.5%)
- Correct answer NOT in top-10: 3,048 (14.5%)

Rank distribution of correct answer (when in top-10):

| Rank | Count | % |
|---|---|---|
| 1 (2nd choice) | 7,910 | 44.0% |
| 2 (3rd choice) | 5,352 | 29.8% |
| 3-4 | 2,529 | 14.1% |
| 5-9 | 2,198 | 12.2% |

### Confidence gap analysis

| Metric | Value |
|---|---|
| Correct answer mean confidence | 0.108 (10.8%) |
| Wrong chosen mean confidence | 0.284 (28.4%) |
| Confidence ratio (correct/wrong) | 0.43 |
| Nearly equal (>0.9 ratio) | 6.5% |
| Model very sure of wrong (<0.1 ratio) | 9.9% |

The model is 2.3x more confident in the wrong answer, explaining why reranking only gets +1pp.

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
