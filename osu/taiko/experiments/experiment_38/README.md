# Experiment 38 - Framewise Onset Detection with Causal Future Prediction

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Experiments [36](../experiment_36/README.md)-[37-C](../experiment_37c/README.md) proved that multi-target prediction from a single cursor token is fundamentally broken — one feature vector can't represent multiple onsets regardless of loss function (softmax, sigmoid BCE, or dice).

**Framewise detection** is a complete paradigm shift. Instead of extracting one cursor token and classifying into 501 bins, the model predicts onset probability at EVERY position in the future audio window. Each token independently answers "is there an onset here?" — naturally multi-target with no competition between positions.

### Architecture

```
Input: mel (80, 1000) with exponential ramps in past half (frames 0-499)
       + event_offsets/mask for ramp computation
       + conditioning (density stats)

Conv stem: mel → 250 tokens (d_model), 4x downsample
  Tokens 0-124:   past audio + ramps (bidirectional attention)
  Tokens 125-249: future audio (causal attention)

Self-attention (6 layers) with causal mask:
  Past tokens: see everything (full bidirectional)
  Future tokens: see all past + previous future (causal)
  Each future token receives onset feedback from previous token (teacher forcing)

Per-token onset head: future tokens → sigmoid P(onset)
Output: (B, 125) onset probabilities at ~20ms resolution
```

### Key design elements

**Causal masking**: Future token 130 can attend to all 125 past tokens + tokens 125-129. This gives it autoregressive behavior in a single forward pass — each prediction conditions on previous predictions.

**Onset feedback (teacher forcing)**: During training, a learned "onset embedding" is added to future tokens where the PREVIOUS position had a ground truth onset. The model learns "I just predicted an onset, don't predict another immediately." During inference, model's own predictions are used instead.

**Mel-embedded ramps (from exp [35-C](../experiment_35c/README.md))**: Past events are encoded as exponential decay spikes in the mel. The model sees past rhythm patterns directly through the audio pathway.

**Sliding window inference**: Slide the window by N frames, collect onset probabilities from each position, merge overlapping predictions via vote/max/avg, threshold to extract final onsets.

### What's different from all prior experiments

| | Exp [14](../experiment_14/README.md)-[35](../experiment_35/README.md) (single-target) | Exp [36](../experiment_36/README.md)-[37](../experiment_37/README.md) (multi-target) | **Exp 38 (framewise)** |
|---|---|---|---|
| Output | 501-class softmax | 501-class sigmoid/softmax | **125 independent sigmoids** |
| Extraction | Single cursor token | Single cursor token | **All future tokens** |
| Multi-onset | No (one prediction) | Attempted (failed) | **Natural (per-position)** |
| AR structure | Loop over predictions | Single pass | **Causal mask + onset feedback** |
| Resolution | ~5ms (bin level) | ~5ms | ~20ms (token level) |

### Training

- Dataset: multi-target (all future onsets) — reuses the existing `--multi-target` data pipeline
- Loss: **Focal dice + BCE** — dice loss (90%) measures set overlap between predicted and real onsets, BCE (10%) provides stable per-token gradients. Dice naturally handles the sparse class imbalance (~5 onsets in 125 tokens).
- Teacher forcing: ground truth onset positions fed back during training

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_38 --model-type framewise --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

### Risk

- ~20ms token resolution may be too coarse — onsets within 20ms of each other map to the same token
- BCE with sparse positives (~5 onsets in 125 tokens = 4%) may need class weighting
- Teacher forcing during training but self-feeding during inference creates train/test mismatch
- The causal mask prevents future tokens from seeing each other bidirectionally, which may limit the model's understanding of rhythmic structure within the prediction window

## Result

**Model predicts nothing — dice loss minimized by all-zero predictions.** Killed after eval 1.

| Metric | Value |
|--------|-------|
| HIT | 0.0% |
| Event recall | 0.0% |
| Preds/window | 0.0 |
| Train loss | 0.073 |

Dice with smooth=1.0 allows the model to minimize loss by predicting all zeros: `dice = smooth / (0 + target_sum + smooth)` is a small constant. The 0.1 BCE weight isn't strong enough to counter this.

## Lesson

- **Dice smooth term creates a degenerate minimum at all-zero predictions.** The smooth constant (needed to avoid NaN) makes "predict nothing" a valid low-loss strategy.
- **BCE should be the primary loss for framewise** — it directly penalizes each positive token that the model misses.
