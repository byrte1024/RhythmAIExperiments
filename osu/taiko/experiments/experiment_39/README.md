# Experiment 39 - Overprediction Analysis (Diagnostic)

## Hypothesis

The 35-C model (71.6% HIT) has a persistent 2.0x error band — it frequently predicts double the correct bin offset. But what if these "overpredictions" aren't wrong — they're just predicting a REAL onset that's further ahead?

For example: cursor is at position X, next onset is at X+75 (target=75), but there's also an onset at X+150. The model predicts 150 — counted as a miss (2.0x error), but it's actually a valid onset, just not the nearest one.

**If most overpredictions match real future onsets, the model sees the full onset landscape but can't pick the nearest one.** This would mean the 2.0x error band is a ranking problem (which onsets to prefer), not a detection problem (where are the onsets).

Also check: for top-K predictions, how many match ANY future onset (not just the nearest)? This tells us if the model's candidate set contains valid onsets beyond the nearest one.

### Method

1. Run 35-C eval 8 checkpoint on val set
2. For each overprediction (pred > target), check if it matches any of the future onsets in the window
3. For top-K, check each candidate against all future onsets
4. Report: what % of overpredictions are valid future onsets, and how top-K changes when matching against all onsets

### Expected: ~80% of overpredictions match real future onsets

## Result

*Pending*

## Lesson

*Pending*
