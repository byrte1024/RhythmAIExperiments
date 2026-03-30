# Experiment 39-E — Full Analysis Specification

## Task

Diagnostic analysis: compare audio features (mel energy, spectral flux, onset strength) at the target position vs the predicted position for failure cases. Determine if the model picks the more acoustically prominent onset instead of the nearest one.

## Analysis Type

Post-hoc audio analysis of a trained model's failure cases. No training performed.

## Model Analyzed

Exp 35-C eval 8 checkpoint (OnsetDetector with exponential decay mel ramps, ~19M params, 71.6% HIT).

## Method

### Step 1: Identify fixable failure cases

From the validation set, select samples where the model is wrong AND the correct answer exists in the top-K predictions.

### Step 2: Extract audio features at target and predicted positions

For each failure case, compute three audio features at both the target mel frame and the predicted mel frame:

1. **Mel energy**: Mean mel value across all 80 bands in a +/-5 frame window around the position
2. **Spectral flux**: Frame-to-frame change in mel magnitude (L2 norm of frame difference), maximum in a +/-3 frame window
3. **Onset strength**: Maximum spectral flux in a +/-3 frame window (captures transient sharpness)

### Step 3: Compare target vs predicted positions

For each failure case:
- Record which position has higher mel energy, spectral flux, and onset strength
- Split analysis by overpredictions (pred > target) and underpredictions (pred < target)

### Step 4: Visual examples

Export 10 mel spectrogram windows with target and predicted positions marked for visual inspection.

## Scripts Used

Analysis script run in the experiment directory, using:
- `detection_train.py` evaluation infrastructure for checkpoint loading and mel extraction
- Custom audio analysis code computing mel energy, spectral flux, and onset strength
- Matplotlib visualization for sample mel windows

## Data

- Validation set from taiko_v2 (~10% of 10,048 charts)
- Failure cases where the correct answer is in the top-K predictions
- Mel spectrograms at 80 bands, hop=110, n_fft=2048, 20-8000 Hz

## Key Results

### Overall (all failure cases)

| Metric | At target | At predicted | Pred higher % |
|---|---|---|---|
| Mel energy | 22.21 | 22.17 | 50.0% (equal) |
| Spectral flux | 35.16 | 45.24 | 60.6% |
| Onset strength | 49.97 | 62.32 | 61.2% |

### Overpredictions (63.3% of failures)

| Metric | Pred higher % |
|---|---|
| Mel energy | 47.8% (equal) |
| Spectral flux | 77.8% |
| Onset strength | 81.0% |

### Underpredictions (36.7% of failures)

| Metric | Pred higher % |
|---|---|
| Mel energy | 53.9% (equal) |
| Spectral flux | 31.0% (target is sharper) |

### Interpretation

- Raw energy is identical at both positions (both are real onsets with similar loudness)
- For overpredictions, 78-81% land on a position with sharper transients (stronger spectral flux)
- The model picks the most "onset-like" audio event, not the nearest one
- The further onset is often the start of a new rhythmic group with a harder attack
- ~20% of overpredictions have weaker transients at the predicted position, indicating other failure causes
