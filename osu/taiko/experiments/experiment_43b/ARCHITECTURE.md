# Experiment 43-B — Diagnostic Analysis Specification

## Type

Analysis / diagnostic experiment. No model training. Compares autoregressive resilience across three models with different context architectures.

## Purpose

Determine whether context-dependent models (exp 42, exp 35-C) are less AR-resilient than context-free models (exp 14). The human evaluation (exp 42-AR) found that exp 14 won despite lower per-sample HIT — this experiment tests whether the preference is explained by AR cascade degradation.

## Models Analyzed

| Label | Experiment | Per-sample HIT | Architecture |
|---|---|---|---|
| exp14 | Exp 14 | 68.9% | OnsetDetector, no context (audio-only) |
| exp35c | Exp 35-C | 71.6% | OnsetDetector, mel ramp context |
| exp42 | Exp 42 | 73.2% | EventEmbeddingDetector, event embeddings |

## Data Analyzed

1000 validation samples, 32 AR steps each. For each model, recorded:
- Light AR: per-step HIT rate (cascade degradation curve)
- Light AR: unique predictions per step (metronome detection)
- Light AR: prediction mean/std/range over steps (drift detection)
- Full AR: event HIT/MISS, prediction HIT/hallucination, density ratio
- Ablation benchmarks: metronome and no_events for reference

## Method

1. Select 1000 validation samples
2. For each sample, run 32 consecutive AR predictions with each model:
   - **Light AR**: compare pred[i] vs truth[i] directly — tracks how cascade errors misalign predictions
   - **Full AR**: match predicted event set against ground truth event set — measures overall chart quality
3. Track unique prediction count per step (metronome collapse indicator)
4. Compute prediction drift (mean/std/range of predictions over steps)
5. Compare all metrics across models

## Scripts

| Script | Location | Purpose |
|---|---|---|
| analyze_entropy.py | `osu/taiko/analyze_entropy.py` | Base analysis infrastructure with AR benchmark support |

## Output Files

| File | Description |
|---|---|
| ar_comparison.json | Full AR resilience comparison data across models |
| ar_hit_curves.png | Per-step HIT rate curves for all three models |
| ar_unique_preds.png | Unique predictions per step for all three models |
| ar_pred_drift.png | Prediction drift (mean/std) over AR steps |

## Key Findings

- **Exp 42 (deepest context) is the MOST AR-resilient, not the least** — beats exp 14 at every single AR step
- Light AR HIT at step 0: exp14=70.7%, exp35c=72.6%, exp42=74.2%
- Light AR HIT at step 8: exp14=9.9%, exp35c=11.9%, exp42=12.8%
- Full AR event HIT: exp14=32.0%, exp35c=31.1%, exp42=33.9%
- **Metronome collapse is universal**: all three models converge to ~15 unique predictions by step 10
- ~52% hallucination rate is consistent across models (all predict ~2x too many onsets)
- **The human evaluation preference for exp14 is NOT about AR resilience** — must be about output style/consistency
