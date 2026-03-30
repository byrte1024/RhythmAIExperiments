# Experiment 39 - Overprediction Analysis (Diagnostic)

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

The [35-C](../experiment_35c/README.md) model (71.6% HIT) has a persistent 2.0x error band — it frequently predicts double the correct bin offset. But what if these "overpredictions" aren't wrong — they're just predicting a REAL onset that's further ahead?

For example: cursor is at position X, next onset is at X+75 (target=75), but there's also an onset at X+150. The model predicts 150 — counted as a miss (2.0x error), but it's actually a valid onset, just not the nearest one.

**If most overpredictions match real future onsets, the model sees the full onset landscape but can't pick the nearest one.** This would mean the 2.0x error band is a ranking problem (which onsets to prefer), not a detection problem (where are the onsets).

Also check: for top-K predictions, how many match ANY future onset (not just the nearest)? This tells us if the model's candidate set contains valid onsets beyond the nearest one.

### Method

1. Run [35-C](../experiment_35c/README.md) eval 8 checkpoint on val set
2. For each overprediction (pred > target), check if it matches any of the future onsets in the window
3. For top-K, check each candidate against all future onsets
4. Report: what % of overpredictions are valid future onsets, and how top-K changes when matching against all onsets

### Expected: ~80% of overpredictions match real future onsets

## Result

**83.2% of overpredictions match real future onsets.** The model sees the onset landscape accurately — it just picks the wrong one.

```
Non-STOP samples: 74,075
HIT (nearest): 53,038 (71.6%)
MISS: 21,037 (28.4%)
  Overpredictions (pred > target): 13,269 (17.9%)
  Underpredictions (pred < target): 7,768 (10.5%)

Of 13,269 overpredictions:
  Matches ANY future onset: 11,041 (83.2%)
  Matches 2nd onset specifically: 9,042 (68.1%)
  Doesn't match any onset: 2,228 (16.8%)

Theoretical HIT if overpred→future counted:
  Current:  71.6%
  Adjusted: 86.5% (+14.9%)
```

**Top-K analysis:**

| K | Nearest match | Any future match | Gain |
|---|---|---|---|
| 1 | 71.6% | 86.5% | +14.9% |
| 2 | 138.4% | 169.3% | +30.9% |
| 3 | 187.9% | 235.2% | +47.3% |
| 5 | 241.7% | 315.6% | +74.0% |
| 10 | 285.8% | 422.8% | +137.0% |

(Cumulative: each K adds the new matches from that rank position across all samples)

**Key findings:**
- **83.2% of overpredictions are valid onsets** — the model detects real onsets but doesn't always pick the nearest one
- **68.1% specifically match the 2nd onset** — the classic "skipping one" error
- **Only 16.8% are truly wrong** — predicting a position with no onset
- **Top-K candidates are packed with valid future onsets** — the model's candidate set is much richer than single-nearest matching credits

## Lesson

- **The 2.0x error band is a selection problem, not a detection problem.** The model sees both the nearest and the next onset, but sometimes picks the further one.
- **Theoretical ceiling with perfect selection: 86.5%** — a +14.9pp improvement just by picking the right candidate from what the model already produces.
- **Inference-time reranking could recover most of this** — prefer closer predictions among confident candidates. No retraining needed.
- **The model's onset detection is far better than HIT rate suggests** — 86.5% of its top-1 predictions land on real onsets. The autoregressive framework (predict nearest only) undervalues what the model knows.
