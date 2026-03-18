# Experiment 36 - Multi-Target Training + Threshold Inference

## Hypothesis

The autoregressive single-prediction formulation has a catastrophic asymmetry:
- **Undershoot** (predict 35, true is 70): cursor lands at 35, model can predict again → **recoverable**
- **Overshoot** (predict 70, true is 35): cursor jumps past 35 forever → **unrecoverable**

The model has learned this — it rationally prefers conservative short predictions, explaining:
- The persistent **2.0x error band** (predicting double the correct value is safe; half is dangerous)
- Why **context disambiguation never worked** (the model ignores context when the safe choice is always "pick the shorter one")
- The **conservative bias** toward undershooting

21 experiments (14-35D) tried to fix this through architecture, loss, augmentation, and data. Exp 35-D proved focal loss can't fix it either — it's structural, not a loss or context problem.

**The fix: multi-target training.** Instead of predicting the single next onset, train the model to predict ALL onsets in the forward window. The model can say "I see beats at 35 AND 70" without choosing. At inference, take the earliest prediction above a confidence threshold.

### Changes from exp 35-C

1. **Multi-target soft labels**: Target is all onsets in the forward window, not just the nearest. Soft target = normalized sum of trapezoids (same log-ratio math). Empty windows get all mass on bin 500.
2. **MultiTargetOnsetLoss**: Hard CE against nearest target (30%) + soft CE against multi-target distribution (70%). `hard_alpha=0.3` (down from 0.5) because the soft target now carries more information.
3. **Threshold inference**: Instead of argmax, scan softmax from bin 0 upward, take first bin above threshold (0.05). If nothing above threshold → hop forward (STOP equivalent).
4. **Bidirectional metrics**: Event-side recall (did the model find each real onset?) and prediction-side precision (was each prediction real or hallucinated?). F1 combines both.

Everything else identical: same model (OnsetDetector/unified), exponential mel ramps, amplitude jitter 0.25-0.75.

### Architecture

Identical to exp 35-C. Model outputs (B, 501) logits — the interpretation changes, not the architecture.

### Expected outcomes

1. **Reduced 2.0x error band** — the model no longer needs to choose between 35 and 70. It predicts both, we take 35 first, then reach 70 on the next step.
2. **Higher event recall** — the model can express multiple candidates without penalty.
3. **Some hallucinations initially** — the model may fire predictions at non-onset positions. The threshold and training should calibrate this over time.
4. **Context delta maintained** — mel ramps still encode beat history. Context now helps calibrate density (how many peaks to fire) rather than disambiguate a single choice.

### Risk

- Loss magnitude change: multi-target soft CE has different scale than single-target. May need LR adjustment.
- Threshold sensitivity: too low = hallucinations, too high = missed events. The threshold sweep graph will find the sweet spot.
- The model was trained (via warm-start) to produce peaked single-prediction distributions. Multi-target training asks it to produce multi-modal distributions. The transition may be unstable initially.

### Why train from scratch (not warm-start)

The 35-C model was optimized to produce **peaked single-mode distributions** — all mass on one bin. Multi-target training wants **multi-modal distributions** with mass at multiple onset positions. Warm-starting from a single-mode optimum would fight against the new loss landscape:
- The output head's smoothing conv learned to sharpen peaks
- The fusion transformer's attention patterns route toward a single confident answer
- The entire model is a local minimum optimized for the wrong output shape

The AudioEncoder and GapEncoder representations should transfer (they encode audio/gap features, not output shape), but the fusion layers and output head would need to unlearn their single-mode bias. Training from scratch is cleaner — the model discovers multi-modal distributions naturally from step 1, with mel ramps providing context signal throughout.


## Result

*Pending*

## Lesson

*Pending*
