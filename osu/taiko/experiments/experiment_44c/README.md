# Experiment 44-C - Top-K vs Top-U Oracle Analysis

## Hypothesis

From previous top-K scatter plots and prediction distribution analysis, we observed that the model's top predictions often cluster tightly — multiple top-K candidates are within a few bins of each other, representing the same "idea" at slightly different offsets. This means Top-K overstates the model's diversity of options.

**Top-U (Top Unique)** merges predictions within 5% of each other into clusters, summing their confidence. This gives a cleaner picture of how many *distinct* predictions the model is actually considering.

### Predictions

- **Top-K 10 oracle HIT ~96%, Top-K 5 ~93%** (supported by raw top-K data from exp 44 evals)
- **Top-U 3 oracle HIT ~95%** — because merging removes redundant near-duplicates, fewer unique options should reach the same accuracy. If Top-U 3 matches Top-K 5, it means ~40% of the top-5 raw predictions are redundant clusters of the same onset.
- **Threshold analysis** will show that most samples have only 2-3 unique clusters above 5% normalized confidence, with a long tail of low-confidence alternatives. This would confirm the model is fundamentally choosing between a small number of distinct options, not spreading probability across many.

### Method

Run exp 44 (eval 20 checkpoint) on the val set (subsample 8). For each sample:

**Top-K oracle:** For K=1..10, pick the top-K predictions by raw confidence. Oracle selects the one closest to target. Measure HIT/GOOD/MISS.

**Top-U oracle:** Cluster all 500 non-STOP classes by confidence:
1. Sort classes by confidence descending
2. For each class: if within 5% of an existing cluster's centroid, merge (add confidence). Otherwise, create new cluster.
3. Normalize cluster confidences to sum to 100%
4. For U=1..10, oracle selects from top-U clusters the one closest to target. Measure HIT/GOOD/MISS.

**Threshold analysis:** For each confidence threshold T (1%, 2%, 5%, 10%, 15%, 20%, 30%, 50%), count how many unique clusters have normalized confidence >= T. Report mean/median/p10/p90 across all samples.

### Scripts

- `analyze_topk_topu.py` — runs val pass, computes Top-K/Top-U oracles, threshold stats, generates graphs and JSON

## Result

*Pending*

## Lesson

*Pending*
