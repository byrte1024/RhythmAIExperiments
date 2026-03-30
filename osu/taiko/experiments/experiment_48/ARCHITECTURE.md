# Experiment 48 — Diagnostic Analysis Specification

## Type

Analysis / diagnostic experiment. No model training. Cross-model failure analysis comparing four architectures on the same validation samples.

## Purpose

Determine whether models with different architectures fail on the same validation samples (structural failures inherent to the data) or on different samples (architecture-specific blind spots). If failures are shared, no architecture change will fix them. If failures are unique, each architecture has exploitable strengths.

## Models Analyzed

| Label | Experiment | Per-sample HIT | Architecture |
|---|---|---|---|
| exp14 | Exp 14 | 69.1% | OnsetDetector, no context (audio-only) |
| exp35c | Exp 35-C | 71.6% | OnsetDetector, mel ramp context |
| exp44 | Exp 44 | 73.5% | EventEmbeddingDetector, event embeddings |
| exp45 | Exp 45 | 72.2% | EventEmbeddingDetector, event embeddings + gap ratios |

## Data Analyzed

Full validation set (subsample 8). For each model, for each sample, recorded:
- Per-sample score [-1, +1]
- Predicted bin
- HIT/MISS classification

Cross-model analysis:
- Success heatmaps (512x512 image per model)
- Pairwise agreement rates (both HIT or both MISS)
- Failure overlap (shared vs model-specific failures)
- Shared failure characterization (direction, ratio, agreement on wrong bin)
- Model-specific unique failure counts
- Score correlation (Pearson r) between model pairs

## Method

1. Run each model on the validation set (subsample 8), record per-sample predictions
2. Generate 512x512 success heatmaps for visual comparison
3. Compute pairwise agreement: for each pair of models, percentage of samples where both HIT or both MISS
4. Compute failure overlap: how many failures are shared across all 4 models vs unique to one model
5. For shared failures: check if models predict the same wrong bin, same direction (over/undershoot), same ratio error
6. For model-specific failures: count and characterize samples where one model fails but all others succeed
7. Render video of worst shared failures with Griffin-Lim audio reconstruction

## Scripts

| Script | Location | Purpose |
|---|---|---|
| analyze_val_heatmap.py | `osu/taiko/analyze_val_heatmap.py` | Run a model on val set, produce 512x512 heatmap + raw scores .npy |
| analyze_cross_model.py | `osu/taiko/analyze_cross_model.py` | Load multiple .npy score files, produce overlap/agreement/failure analysis |

## Output Files

| File | Description |
|---|---|
| cross_model_analysis.json | Full analysis: pairwise agreement, overlap rates, shared failure stats |
| compare_heatmaps.png | Side-by-side success heatmaps for all 4 models |
| compare_good_core.png | Samples where all models succeed |
| compare_bad_core.png | Samples where all models fail |
| compare_full_animated.gif | Animated GIF cycling through model heatmaps |
| compare_good_animated.gif | Animated GIF of good core |
| compare_bad_animated.gif | Animated GIF of bad core |
| failures_exp44.mp4 | Video of 25 worst shared failures with audio and scrolling mel |

## Key Findings

### Failure Overlap
- **All models HIT: 55.4%** (41,026 samples) — easy, every architecture succeeds
- **All models MISS: 14.2%** (10,525 samples) — structurally unsolvable by any model built so far
- **Shared failure rate: 32.3%** — one-third of all failures are universal

### Pairwise Agreement
Models agree 80-85% of the time. ~15-20% of samples have model-specific outcomes. Context models (exp44/45) correlate more with each other than with exp14.

### Model-Specific Unique Failures
| Model | Unique failures | % of all |
|---|---|---|
| exp14 | 3,488 | 4.71% |
| exp35c | 2,102 | 2.84% |
| exp44 | 1,986 | 2.68% |
| exp45 | 1,647 | 2.22% |

Context resolves ~2.5pp of failures that audio alone cannot.

### Shared Failure Characterization
- 80-84% of shared failures predict the same wrong bin (within 5% tolerance)
- 60% overshoot, 40% undershoot, mean error +8-10 bins
- **Median error ratio: 1.89x** — consistent across ALL four models
- **42% of shared failures: 2x ratio** (predicting double the correct gap)
- **25% of shared failures: 0.5x ratio** (predicting half the correct gap)
- The octave/metric confusion (beat vs sub-beat) is universal and architecture-independent

### Score Correlation (Pearson r)
Moderate correlation (0.55-0.64). Models agree on easy/hard samples but have meaningful independence on medium-difficulty samples.

### Video Observation
The 25 worst shared failures are not difficult cases — clear audio, ordinary patterns. A human would identify the correct onset instantly. The model hears the rhythmic structure but picks the wrong level in the metric hierarchy (next measure instead of next beat, or next sub-beat instead of beat).

### Implications
- 14.2% of samples are structurally unsolvable by any current model — the 2x/0.5x metric hierarchy error
- Context does not fix the metric hierarchy problem
- The model needs meter awareness (understanding musical meter like "4/4 at 120 BPM")
- The 2.5s audio window + 2.5s of in-window events cannot capture a full musical phrase (8s at 120 BPM)
- This motivates the virtual token architecture for out-of-window context (25+ seconds of rhythm history)
