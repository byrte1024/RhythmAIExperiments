# Experiment 44-C — Diagnostic Analysis Specification

## Type

Analysis / diagnostic experiment. No model training. Analyzes prediction diversity using Top-K vs Top-U (Top Unique) oracle comparison.

## Purpose

Determine how many genuinely distinct onset candidates the model considers for each prediction. Raw Top-K candidates cluster tightly — multiple top-K predictions may be within a few bins of each other, representing the same "idea" at slightly different offsets. Top-U merges predictions within 5% of each other into clusters, giving a cleaner picture of the model's diversity of options.

## Model Analyzed

Exp 44 checkpoint (eval 20). EventEmbeddingDetector, 73.6% HIT.

## Data Analyzed

74,075 non-STOP validation samples (subsample 8). For each sample:
- Full 501-class softmax probabilities
- Top-K oracle: for K=1..10, pick top-K predictions by raw confidence, oracle selects closest to target
- Top-U oracle: cluster all 500 non-STOP classes by confidence (merge within 5% of cluster centroid), oracle selects from top-U clusters
- Threshold analysis: count unique clusters above each confidence threshold (1%, 2%, 5%, 10%, 15%, 20%, 30%, 50%)

## Method

### Top-K Oracle
For K=1..10, pick the top-K predictions by raw confidence. Oracle selects the one closest to the target. Measure HIT/GOOD/MISS.

### Top-U Oracle
1. Sort all 500 non-STOP classes by confidence descending
2. For each class: if within 5% of an existing cluster's centroid, merge (add confidence). Otherwise, create new cluster.
3. Normalize cluster confidences to sum to 100%
4. For U=1..10, oracle selects from top-U clusters the one closest to target. Measure HIT/GOOD/MISS.

### Threshold Analysis
For each confidence threshold T (1%, 2%, 5%, 10%, 15%, 20%, 30%, 50%), count how many unique clusters have normalized confidence >= T. Report mean/median/p10/p90 across all samples.

## Scripts

| Script | Location | Purpose |
|---|---|---|
| analyze_topk_topu.py | `experiments/experiment_44c/analyze_topk_topu.py` | Runs val pass, computes Top-K/Top-U oracles, threshold stats, generates graphs and JSON |

## Output Files

| File | Description |
|---|---|
| topk_topu_results.json | Full Top-K and Top-U oracle results at each K/U |
| topk_topu_graph.png | Top-K vs Top-U HIT rate comparison |
| topu_delta_graph.png | Delta between Top-U and Top-K |
| threshold_unique_graph.png | Number of unique clusters at each confidence threshold |

## Key Findings

- **The model chooses between 2-4 real options.** At 5% confidence threshold, median unique clusters is 4. At 10%, just 2. At 30%+, down to 1.
- **Top-U consistently outperforms Top-K.** Top-U 5 (95.0%) matches Top-K 7 (94.9%) — each unique option is worth ~1.4 raw top-K slots.
- **~40% of top-5 raw predictions are redundant near-duplicates.** Merging within 5% tolerance removes this redundancy.
- **Top-U 3 at 91.8% HIT and 7.7% MISS** — the model almost always has the right answer somewhere in its top 3 distinct ideas.
- **The gap between "model knows" and "model picks" is 18pp** (91.8% Top-U 3 oracle vs 73.6% argmax). Reranking strategies could exploit this.

### Top-K vs Top-U Oracle HIT Rate

| K/U | Top-K HIT | Top-U HIT |
|---|---|---|
| 1 | 73.6% | 73.0% |
| 2 | 84.2% | 88.5% |
| 3 | 90.8% | 91.8% |
| 5 | 93.7% | 95.0% |
| 10 | 96.0% | 98.0% |

### Threshold Analysis (unique clusters)

| Confidence threshold | Mean clusters | Median |
|---|---|---|
| 5% | 4.1 | 4 |
| 10% | 2.1 | 2 |
| 20% | 1.4 | 1 |
| 50% | 0.6 | 1 |
