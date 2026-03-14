# The Context Issue

The single biggest unsolved problem in this project. Across 13+ experiments (14-27B), the model consistently refuses to use chart context (gap history) for predictions, despite overwhelming evidence that context contains the information needed to break the ~70% HIT ceiling.

## What is context?

When predicting the next onset, the model receives two inputs:
1. **Audio** — a 5-second mel spectrogram window centered on the cursor
2. **Context** — the last 128 inter-onset gaps (time intervals between previous mapped notes)

Audio tells the model "what sounds are happening." Context tells the model "what rhythm pattern has been established."

## Why context matters

The model's errors are not random. They are systematic pattern disambiguation failures:

- The model predicts **rhythmically valid** gaps that are wrong by exact multiples — 150 instead of 75 (2x), 30 instead of 60 (0.5x)
- **Top-10 accuracy is 96%** — the correct answer is almost always in the model's candidate set, it just picks the wrong one
- **19 out of 20 misses have the target gap value in recent context** (exp 27-B)
- The model defaults to **conservative (longer) predictions** because undershooting cascades worse in autoregression

For a pattern like `31 16 16 32 | 31 16 16 32 | 31 16 16 31 | ?`, context unambiguously says the answer is 32 (continuing the `31 16 16 32` cycle). The model predicts 16 — halving the correct value, a rhythmically valid but wrong answer. This is a real example from exp 27-B.

## The evidence across 13 experiments

### Phase 1: Separate paths (exp 15-24)

10 experiments tried context as a separate system that overrides or nudges audio:
- **Reranking** (exp 15-23): context reranks audio's top predictions
- **Additive logits** (exp 24): context adds soft logit adjustments

Result: context in isolation reaches ~53% HIT. It can learn rhythm patterns but can't know when audio is already correct (70% of the time). Context's influence is random w.r.t. audio correctness → hurt ≈ helped → net negative.

**Lesson: context needs to see audio to know when to act.**

### Phase 2: Unified fusion (exp 25-27)

3 experiments put audio and context tokens in the same self-attention, so context CAN see audio:

| Experiment | Change | Best HIT | Context delta (final) |
|---|---|---|---|
| 25 | Unified fusion, light aug | 68.6% | 2.3% |
| 26 | + heavy audio augmentation | 68.8% | 1.7% |
| 27 | + full dataset (4x data) | 69.8% | 1.5% |

Context delta = accuracy gap between full model and no_events benchmark. A shrinking delta means the model is learning to ignore context.

In every experiment, context contribution starts high (~7-8%) and collapses to ~1.5% within a few epochs. The model converges to an audio-dominant solution regardless of:
- Augmentation strength (light vs heavy)
- Data volume (25% vs 100% of dataset)
- Training duration

**Lesson: making context available is not enough. The model must be forced to use it.**

### Phase 3: Diagnostic (exp 27-B)

Pattern analysis on the best model's (69.8% HIT) validation predictions:

- Strict pattern matching catches 22.5% of misses as solvable by context
- Manual inspection shows the true number is far higher (~95% of misses with context have the target value present)
- BUT blindly following patterns hurts (-3.9% net) because patterns can't tell when to break (transitions, new sections)

**Lesson: context has the answer, but needs audio to know when to apply it. This is exactly what the unified architecture should do but doesn't.**

## Why the model ignores context

The model learns to ignore context because **audio is a shortcut**:

1. **Audio is more immediately informative.** Local audio energy directly predicts nearby onsets. The gradient signal from audio is strong and immediate. Context requires multi-hop attention to extract patterns — a weaker, slower learning signal.

2. **Gradient competition.** In shared self-attention, audio tokens provide faster loss reduction. Context tokens' gradients are small in comparison. Over training, the model routes increasingly through audio and context pathways atrophy.

3. **Cursor bottleneck.** Single-token extraction at position 125 means the model's output is dominated by local audio features. Distant targets (where context would help most) require information to propagate through multiple attention layers.

4. **Context is noisy during training.** Augmentation adds jitter to event positions. The model learns that context is unreliable and defaults to audio. But in real (non-augmented) inference, context is clean and informative.

5. **The easy 70% dominates training.** Most samples have nearby, confident predictions where audio alone suffices. These easy samples dominate the gradient, and the model never learns to use context for the hard 30%.

## What we've tried that doesn't work

| Approach | Why it fails |
|---|---|
| Separate context path (exp 15-24) | Context can't see audio → doesn't know when to act |
| Unified self-attention (exp 25) | Audio drowns out context in attention |
| Heavier audio augmentation (exp 26) | Model becomes noise-robust, still ignores context |
| More training data (exp 27) | More data to memorize, same audio shortcut |
| Lighter augmentation (exp 25) | Overfits faster, context still ignored |

## What should work (untested)

Approaches that **structurally force** context usage rather than hoping the model discovers it:

1. **Focal loss** — downweight easy samples (confident, audio-solvable) and focus training on the hard disambiguation cases where context is necessary. Doesn't change architecture but redirects gradient signal.

2. **Auxiliary context loss** — second prediction head on gap tokens only, with its own loss term. Forces the gap encoder to learn useful representations regardless of what the fusion pathway does. The context pathway has no choice but to carry signal.

3. **Adversarial cursor masking** — randomly zero out the audio around the cursor during training (~10-15%). When audio is unavailable, the model must rely on context. Creates training samples where context is the only path to a correct prediction.

4. **Learnable cursor query** — replace fixed position-125 extraction with a learned query that cross-attends to all fused tokens. Removes the positional bottleneck that favors local audio.

5. **Context gating with minimum flow** — learned gate between audio and context features with a floor (e.g., context must contribute ≥10% of the final representation). Prevents complete context atrophy.

## The goal

Break past 70% HIT by making the model use context to disambiguate the ~20% of predictions where audio is ambiguous but context is clear. The ceiling with perfect context utilization is estimated at 90%+ based on top-10 accuracy and pattern analysis.
