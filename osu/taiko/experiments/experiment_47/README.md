# Experiment 47 - Binary STOP Head

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

STOP prediction is broken: F1 ~0.52, precision 40% (60% of STOP predictions are hallucinated), and the model can't reliably stop on silence. The root cause is structural — STOP (class 500) competes with 500 onset bins in a single softmax. The "should I stop?" decision and the "where is the next onset?" decision are entangled.

**Fix: separate binary head.** Two independent outputs from the cursor token:
- **Gate head**: sigmoid → P(onset exists in window). Binary decision: onset or stop.
- **Onset head**: 500-class softmax → bin offset. Only meaningful when gate says "onset."

During training:
- Gate loss: BCE on all samples. Target=1 for onset samples, target=0 for STOP samples.
- Onset loss: CE on non-STOP samples only. The onset head never sees STOP — it only learns to locate onsets.
- Combined loss: `gate_loss * gate_weight + onset_loss`

During inference:
- If gate < threshold → STOP (hop forward)
- If gate >= threshold → use onset head argmax

### Why this should help

1. **STOP gets its own loss surface.** No more competing with 500 onset bins in softmax. The gate can learn "silence = no onset" independently.
2. **Onset head is cleaner.** It never has to waste capacity on STOP. All 500 classes are onsets — the distribution is more natural.
3. **Threshold is tunable at inference.** Can adjust gate threshold without retraining. Lower threshold = more events (denser charts). Higher = more conservative (sparser).
4. **no_audio behavior is explicit.** The gate can learn to output ~0 when audio is silent, regardless of context.

### Architecture

Same as exp [44](../experiment_44/README.md)/[45](../experiment_45/README.md) EventEmbeddingDetector base. Changes:
- Remove class 500 (STOP) from the output head
- Add gate head: cursor token → LayerNorm → Linear(d_model, 1) → sigmoid
- Onset head: cursor token → existing head → 500 logits (bins 0-499 only)
- gate_weight hyperparameter controls relative importance of gate vs onset loss

### Risks

- Gate and onset head may disagree — gate says "onset" but onset head is uncertain, or vice versa.
- BCE on the gate is imbalanced (99.7% onset, 0.3% STOP). Needs pos_weight or focal loss.
- The current soft target loss mixes STOP into the distribution. Need to handle this cleanly.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_47 --model-type event_embed --binary-stop --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Stopped at eval 1. Stop rate was 0% — model never predicted STOP.**

Root cause: `pos_weight=0.01` in BCE was backwards. The gate target had 1=onset, 0=stop, so `pos_weight` upweighted the already-dominant onset class 100x, making STOP invisible to the loss. The model learned to always output "onset."

## Lesson

- **BCE pos_weight scales the POSITIVE class, not the rare class.** With target 1=onset (99.7%), pos_weight=0.01 made onsets even less important — the opposite of intended.
- **Fixed in exp [47-B](../experiment_47b/README.md):** Flipped targets (1=stop, 0=onset) and switched to focal BCE (gamma=2) which naturally downweights easy negatives (confident onset predictions) and focuses on the rare positives (STOP boundaries).
