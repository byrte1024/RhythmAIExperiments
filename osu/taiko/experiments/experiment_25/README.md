# Experiment 25 - Unified Audio + Gap Fusion

## Hypothesis

10 experiments (15-24) proved that separate audio and context paths cannot break even. Context without audio access caps at ~53% HIT — it can learn rhythm patterns but can't know when audio is already correct (70%). Whether reranking (discrete override) or additive (soft logit nudging), context's influence is random w.r.t. audio correctness, guaranteeing helped ~ hurt.

**The paradigm shift: unification.** No separate paths. Audio and gap features fused via self-attention in a single model.

The gap representation is proven useful (53% HIT standalone in exp 24, inter-onset intervals informative across exp 19-24). The audio encoder is proven (69.5% HIT). The problem was never the features — it was the interface. Now we make them attend to each other directly.

### Changes from exp 24

**1. GapEncoder replaces EventEncoder**

The old EventEncoder used sinusoidal encoding of absolute bin offsets — weak and ignored by the model (no_events benchmark ~ full accuracy at exp 14). The new GapEncoder computes inter-onset intervals, extracts ~50ms mel snippets at event positions, and processes through self-attention. Same proven representation from exp 19-24, but at d_model=384 (full dimension, no bottleneck).

**2. Fusion via self-attention, not cross-attention or addition**

Audio tokens (250) and gap tokens (C) are concatenated and fed through a shared fusion transformer. Every layer, audio attends to gaps and gaps attend to audio — bidirectional, deep interaction. The cursor at position 125 (center of audio window) naturally absorbs both modalities.

This is fundamentally different from:
- Exp 14's cross-attention (audio attends to events, but events don't attend to audio)
- Exp 24's addition (two independent predictions summed)

**3. Train from scratch**

No warm-start. The old AudioEncoder was trained for a different architecture (cross-attention to weak event tokens). Its learned attention patterns could interfere with the new fusion dynamics. Clean slate lets the model learn representations optimized for joint audio+gap reasoning from the start.

**4. Lighter augmentation**

Reduced augmentation to encourage context reliance. Previous experiments showed the model could achieve ~50% from audio alone — if augmentation is too aggressive, the model may learn to ignore gap features (since they're noisier under heavy augmentation). Better to overfit to context first and add augmentation back if needed.

**5. Single output, single loss**

No audio_logits, no context_logits, no combining. One forward pass → one set of 501 logits → one OnsetLoss. The model internally decides how to weight audio vs rhythm for each prediction.

### Architecture

```
mel (B, 80, 1000) → AudioEncoder (4 layers) → 250 audio tokens (d=384)
events (B, C)      → GapEncoder (2 layers)   → C gap tokens (d=384)
                              ↓
                   Concatenate → [250 + C] tokens
                              ↓
                   FusionTransformer (4 self-attention layers, FiLM)
                              ↓
                   Cursor at position 125 → output head → 501 logits
```

| Component | Params | Notes |
|-----------|--------|-------|
| AudioEncoder (conv + 4 transformer layers) | ~8.0M | Trains from scratch |
| GapEncoder (snippet enc + 2 transformer layers) | ~3.5M | New, proven gap repr |
| FusionTransformer (4 self-attention layers) | ~7.5M | New, deep fusion |
| Output head (norm + proj + smoothing) | ~0.2M | Same as before |
| cond_mlp | ~8K | Trains from scratch |
| **Total** | **~19M** | All trainable |

### Expected outcomes

1. **Early epochs: < 50% HIT** — training from scratch, no warm-start. The model needs time to learn mel features AND gap patterns simultaneously.
2. **Mid training: audio-level performance (65-70% HIT)** — once the audio encoder converges, should match exp 14's audio-only baseline since the same information is available.
3. **Late training: > 70% HIT** — if fusion works, the model should exceed audio-only by leveraging gap patterns to resolve ambiguous audio. This is the first experiment that could plausibly beat 69.5%.
4. **no_events benchmark < full accuracy** — the model should actually USE gap information, unlike exp 14 where events were ignored.
5. **Slower convergence** — training from scratch without warm-start means more epochs needed.

### Risk
- The fusion transformer sees 250 + C tokens (~378 total). Self-attention is O(n^2), so ~2.3x more compute per layer than the 250-token AudioEncoder. 4 fusion layers add significant training cost.
- Gap tokens may be drowned out by the 250 audio tokens in self-attention (7:1 ratio). The model might learn to ignore the ~128 gap tokens just like it ignored event tokens in exp 14.
- Without warm-start, the model has no head start. If it takes 20+ epochs to reach exp 14 parity, iteration speed drops significantly.
- Lighter augmentation may cause overfitting on training set — watch train/val loss divergence.

## Result

*Pending*

## Lesson

*Pending*
