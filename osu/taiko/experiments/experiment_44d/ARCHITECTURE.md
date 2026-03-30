# Experiment 44-D — Diagnostic Analysis Specification

## Type

Analysis / diagnostic experiment. No model training. Measures the per-sample cost of temperature sampling on Top-K and Top-U candidate sets.

## Purpose

Exp 44-C showed the model knows the right answer 91.8% of the time (Top-U 3 oracle) but only picks it 73.6% of the time (argmax). Temperature sampling could help break metronome patterns during AR generation by occasionally sampling non-dominant predictions, at the cost of per-sample accuracy. This experiment quantifies that cost.

## Model Analyzed

Exp 44 checkpoint (eval 20). EventEmbeddingDetector, 73.6% HIT. Reuses cached softmax probabilities from exp 44-C.

## Data Analyzed

74,075 non-STOP validation samples (subsample 8). For each sample, using Top-K (3, 5, 10, 20) and Top-U (3, 5, 10, 20) candidate sets:

1. Apply temperature to candidate confidences: `p_i^(1/T) / sum(p_j^(1/T))`
2. Sample from the reweighted distribution (5 random trials averaged per temperature for stability)
3. Measure HIT/GOOD/MISS at each of 50 temperature values from 0.01 to 100 (log-spaced)

At T approaching 0: collapses to argmax (always picks highest confidence). At T approaching infinity: uniform random across candidates.

## Scripts

| Script | Location | Purpose |
|---|---|---|
| analyze_temperature.py | `experiments/experiment_44d/analyze_temperature.py` | Runs temperature sweep on cached probabilities from exp 44-C |

## Output Files

| File | Description |
|---|---|
| temperature_results.json | Full temperature sweep results for all K/U combinations |
| temperature_graph.png | HIT rate vs temperature curves |
| temperature_comparison.png | Side-by-side Top-K vs Top-U temperature degradation |

## Key Findings

- **Temperature sampling is strictly worse on per-sample metrics.** Best HIT is always at T=0.01 (argmax) for every K and U. No sweet spot exists.
- **Top-K 3 is most resilient to temperature** — only loses 4.1pp at T=1.0 because the 3 candidates are near-duplicates. Sampling between them barely changes the prediction.
- **Top-U degrades faster but provides real diversity.** Top-U 5 at T=1.0 loses 13.6pp, but each sampled prediction is a genuinely different onset.
- **The value of temperature is in AR generation, not per-sample metrics.** Whether the per-sample cost buys better charts through metronome breaking requires a listening test.

### Temperature Impact at T=1.0

| Mode | T=0.01 (argmax) | T=1.0 | Loss |
|---|---|---|---|
| Top-K 3 | 73.6% | 69.5% | -4.1pp |
| Top-K 5 | 73.6% | 64.9% | -8.7pp |
| Top-K 10 | 73.6% | 58.8% | -14.8pp |
| Top-U 3 | 73.0% | 64.2% | -8.8pp |
| Top-U 5 | 73.0% | 59.4% | -13.6pp |
| Top-U 10 | 73.0% | 56.3% | -16.7pp |
