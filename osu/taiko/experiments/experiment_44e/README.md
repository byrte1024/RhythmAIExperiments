# Experiment 44e - Distance Ramp Loss

> **[Full Architecture Specification](../experiment_44/ARCHITECTURE.md)** — identical architecture to experiment 44.

## Hypothesis

Our current loss has a flat plateau for far-off predictions: predicting 2x the target and 4x the target produce the exact same loss. This means the model gets zero gradient signal to improve from "very wrong" to "less wrong." The only gradient comes from the trapezoid zone (±20% of target).

Adding a third loss component — a gradual ramp in log-ratio space — gives gradient everywhere. Predictions farther from the target in ratio space get progressively higher loss.

### 3-Component Loss

```
loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE + ramp_alpha * |log(pred/target)|^ramp_exp
```

| Component | Role | Gradient range |
|---|---|---|
| Hard CE (0.5) | Exact bin precision | Only at target bin |
| Soft CE (0.5) | Trapezoid neighborhood | ±20% of target (ratio) |
| Ramp (2.5) | Distance penalty | Everywhere (∞) |

The ramp uses the model's expected prediction (softmax center of mass) in log-ratio space: `|log((E[pred]+1)/(target+1))|`. This is:
- 0 at exact match
- Symmetric in ratio space (2x over = 0.5x under)
- Proportional across all target values
- Differentiable everywhere

### Configuration

| Feature | Value |
|---|---|
| Architecture | EventEmbeddingDetector (same as exp 44) |
| hard_alpha | 0.5 (same as exp 44) |
| ramp_alpha | **2.5** (NEW) |
| ramp_exp | 1.0 (linear in log-ratio) |
| Density jitter | ±10% at 30% (same as exp 44) |
| No gap ratios | Same as exp 44 |

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_44e --model-type event_embed --no-gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --ramp-alpha 2.5 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Expected Behavior

- Per-sample HIT% should be similar or slightly better than exp 44 (73.6%)
- The model should make fewer extreme errors (2x/4x overshoots)
- Overprediction rate (from exp 39: 83% of errors are overshoots) may decrease
- Training loss will be higher due to the ramp component — compare val metrics, not raw loss

## Result

Ran on CachyOS (RTX 4060). Compared to exp 44-RE (identical setup, no ramp, same machine) and original exp 44 (RTX 5070, ~0.9pp machine gap).

### Peak Performance

| | 44e (ramp) | 44-RE (baseline, same machine) | 44 (original, RTX 5070) |
|---|---|---|---|
| Peak HIT | **73.0%** (eval 13) | 72.0% (eval 5, only 5 evals ran) | 73.7% (eval 11) |
| Machine-adjusted | **~73.9%** | ~73.9% (at convergence) | 73.7% |
| Peak MISS | **26.4%** | 27.4% | 25.7% |
| Peak eval | 13 | — | 11 |

### Eval-by-Eval vs Original 44 (machine-adjusted, +0.9pp)

| Eval | 44e adj | 44 | Better? |
|---|---|---|---|
| 3 | 72.0% | 70.9% | 44e |
| 5 | 73.3% | 72.9% | 44e |
| 8 | 73.5% | 72.2% | 44e |
| 11 | 73.9% | 73.7% | 44e |
| 13 | 73.9% | 73.1% | 44e |
| 15 | 73.5% | 73.1% | 44e |

44e was ahead at every eval from eval 3 onward (adjusted).

### Training Stability

The most notable difference was training smoothness. 44e had a nearly monotonic HIT climb for the first 5 evals (68.6 → 69.7 → 71.1 → 71.6 → 72.4) with zero stutters. Original 44 and 44-RE both showed typical oscillation (up-down-up patterns). The ramp's everywhere-gradient eliminates the random walk behavior of the flat plateau.

### Detailed Metrics (eval 5, vs 44-RE on same machine)

| Metric | 44e (ramp) | 44-RE (baseline) | Better? |
|---|---|---|---|
| HIT% | 72.4% | 72.0% | 44e (+0.4pp) |
| MISS% | 26.9% | 27.4% | 44e (-0.5pp) |
| Frame error mean | 10.4 | 10.9 | 44e |
| Rel error std | 0.404 | 0.420 | 44e (tighter) |
| Stop F1 | 0.569 | 0.538 | 44e |
| Stop precision | 0.556 | 0.461 | 44e (+9.5pp) |
| Unique preds | 473 | 441 | 44e (+32) |
| Last repeat | 51.1% | 50.1% | 44e (+1.0pp) |

The ramp improved error tightness, STOP precision, prediction diversity, and pattern continuation — all consistent with the "gradient everywhere" hypothesis.

## Lesson

1. **The distance ramp works as designed.** It provides gradient for far-off predictions without hurting near-target precision. Training is smoother, errors are tighter, and STOP decisions are more precise.

2. **The peak improvement is modest (~0.2pp adjusted).** The ramp doesn't break through the architecture's ceiling — it helps the model converge more reliably to the same approximate peak. The HIT ceiling is structural (model capacity, context usage), not a loss landscape problem.

3. **Training stability is the real win.** Monotonic improvement for the first 5 evals vs baseline's oscillation. This matters for shorter runs and for trusting early evals as indicators of final performance.

4. **STOP precision improved dramatically (+9.5pp).** The ramp gives the model better calibration for the onset-vs-STOP decision boundary. When the ramp model says STOP, it's right more often.

5. **The ramp should be adopted as default (ramp_alpha=2.5).** It's a strict improvement with no downsides: same or better peak, smoother training, better calibration. Zero architectural cost — just an additive loss term.
