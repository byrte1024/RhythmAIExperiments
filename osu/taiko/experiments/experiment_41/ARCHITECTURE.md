# Experiment 41 — Diagnostic Analysis Specification

## Type

Analysis / diagnostic experiment. No model training. Analyzes validation predictions from an existing model checkpoint.

## Purpose

Determine why model entropy rises sharply with target distance. Two competing hypotheses: (A) more valid onsets in the window create genuine ambiguity, or (B) the cursor bottleneck degrades distant audio information.

## Model Analyzed

Exp 35-C checkpoint (eval 8). OnsetDetector (unified), 71.6% HIT.

## Data Analyzed

Validation set predictions from exp 35-C. For each sample, recorded:
- Target distance (bin offset from cursor)
- Number of future onsets in window
- Number of onsets skipped (between cursor and predicted position)
- Context length (number of past events)
- Density conditioning values
- Audio features at target position (mel energy, spectral flux)
- Prediction correctness (HIT/MISS)
- Top-1 confidence
- Entropy of the softmax distribution

## Method

1. Run exp 35-C checkpoint on the full validation set
2. For each sample, compute the features listed above
3. Compute Pearson correlation between entropy and each feature
4. Break down HIT rate and entropy by skip count (0, 1, 2, 3+, underprediction)
5. Break down HIT rate and entropy by target distance bins (0-15, 15-30, 30-60, 60-100, 100-200, 200-500)

## Scripts

| Script | Location | Purpose |
|---|---|---|
| analyze_entropy.py | `osu/taiko/analyze_entropy.py` | Runs val pass on a checkpoint, computes per-sample entropy, skip count, correlations, and breakdowns |

## Output Files

| File | Description |
|---|---|
| entropy_analysis.json | Full correlation table, skip-count breakdown, distance-bin breakdown |
| progression_results.json | Multi-checkpoint comparison data (used by exp 41-B) |

## Key Findings

- **Skip count is the primary predictor of failure**, not target distance
- Skip 0 (67% of samples): 93.7% HIT — model is excellent when it doesn't overshoot
- Skip 1 (11%): 0% HIT — model jumped to the 2nd onset, always wrong
- Target distance is nearly identical (~28-36 bins) across all skip counts
- Context length correlates negatively with entropy (r=-0.210) — more context helps
- n_future_onsets correlates negatively with entropy (r=-0.543) — dense sections are easier
- The failure mode is overprediction to a more salient distant onset, not inability to read distant audio

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
