# Experiment 30 - Cursor-Region Audio Masking

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

16 experiments (14-29B) show the model ignores context regardless of architecture, augmentation, data, loss reweighting, or auxiliary losses. The aux context head (exp 29, 29-B) failed because standalone 501-class prediction from gaps is too hard — the head can't learn, so it can't force the gap encoder to improve.

**The nuclear option: remove the audio.** For 20% of training samples, zero out the mel spectrogram around the cursor (100-300 frames centered on position 500 of the 1000-frame window). When the cursor region is blank, the model literally cannot predict from local audio. The only remaining signal is:
1. Gap tokens (context) — the rhythm pattern history
2. Audio far from the cursor — general song characteristics but not onset-specific
3. Density conditioning — chart-level statistics

This is structurally different from all prior approaches: instead of incentivizing context usage (aux loss, focal loss), we make it **impossible to succeed without context** for a significant fraction of training samples. The model must develop a context-reading pathway or fail on 20% of its training data.

### Architecture

**Identical to exp 27.** No aux head, no extra parameters. Same ~19M param unified fusion model.

### Augmentation change

Added to `_augment()`:
- **Cursor-region masking (20%)**: zero out 100-300 mel frames centered on the cursor (position 500). This covers the audio region the model relies on most for prediction.

All other augmentations unchanged (heavy audio aug from exp 26 + full dataset from exp 27).

### Expected outcomes

1. **Context delta stays high** — for 20% of samples, context is the only useful signal. The model must maintain a context pathway or lose on those samples.
2. **Possible HIT drop** — the model sees corrupted audio 20% of the time, which may lower overall performance. The question is whether the context pathway gained compensates.
3. **no_events benchmark should drop significantly** — if the model truly learns to use context, removing events should hurt more than before.
4. **Slower convergence** — harder training signal (20% of audio is missing).

### Risk

- The model may learn to **detect masking** (zeroed mel region is a distinct pattern) and switch to a "context mode" only when masking is detected, without using context on normal samples. Would show as: context delta high on masked samples, still ~0% on unmasked.
- 20% masking rate may be too aggressive, hurting main performance without enough context learning.
- The model may just learn to predict from audio edges (positions 0-400 and 600-1000) without ever using context.

## Result

**Killed early after 2 evals — pivoting to architectural rebalancing instead of augmentation tricks.** Context delta showed same collapse pattern.

| eval | epoch | HIT | Miss | Score | Acc | Val loss | no_events | Ctx Δ |
|------|-------|-----|------|-------|-----|----------|-----------|-------|
| 1 | 1.25 | 67.0% | 32.3% | 0.313 | 48.3% | 2.670 | 41.5% | 6.8% |
| 2 | 1.50 | 67.9% | 31.6% | 0.322 | 49.3% | 2.628 | 46.1% | 3.3% |

**Observations:**
- Context delta 6.8% → 3.3% — collapsing, same as every prior experiment. Higher than exp 27 at eval 2 (2.3%) but the trend is clear.
- HIT slightly ahead of exp 27 (67.9% vs 67.5% at eval 2) — masking works as regularization but not as context forcing.
- no_audio benchmark was 0.4% accuracy despite 20% of training having masked audio. The model learned "zeroed mel = use context" as a detectable mode, not "always consider context."
- Val loss lower than exp 27 (2.628 vs 2.635) — better generalization from harder training.

**Why it was stopped early:**
After 16 experiments (14-30) trying augmentation, loss, and training tricks, the conclusion is clear: **the problem is architectural, not training-related.** The model has 250 audio tokens vs ~128 gap tokens, 4 audio encoder layers vs 2 gap encoder layers. Audio dominates fusion by design. No training trick can overcome a 2:1 architectural advantage. The next step is rebalancing the architecture itself.

## Lesson

- **Cursor-region masking works as regularization** (lower val loss, slightly higher HIT) but doesn't force persistent context usage on unmasked samples.
- **Zero-masking is detectable** — the model switches to "context mode" when it sees zeroed mel, not when it genuinely needs context. Noise-based corruption would remove this shortcut but likely still won't overcome the architectural imbalance.
- **16 experiments confirm: training tricks cannot overcome architectural audio dominance.** Augmentation, focal loss, aux heads, audio masking — none change the ~1.5% context delta endpoint. The architecture must be rebalanced.
