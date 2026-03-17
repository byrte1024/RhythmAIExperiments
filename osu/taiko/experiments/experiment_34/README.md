# Experiment 34 - Context as FiLM Conditioning

## Hypothesis

Experiments 31-33 proved cross-attention between audio and gap tokens is fundamentally flawed for this task:
- **Exp 31**: Late cross-attention → context works (18.8% delta) but banding (gap activations overwhelm audio)
- **Exp 32**: Skip connection → banding fixed but context bypassed (-1.8% delta)
- **Exp 33**: Interleaved → cold start failure, can't even learn (19% HIT)

The root cause: cross-attention mixes fine-grained audio tokens (250, ±7 activation) with coarse gap tokens (128, ±20 activation) in the same attention operation. This either overwrites audio precision or gets bypassed entirely.

**Context as FiLM conditioning** avoids this entirely. Instead of putting gap tokens into attention alongside audio:
1. Gap tokens are summarized into a single conditioning vector via attention pooling
2. This context vector modulates audio features through FiLM (scale + shift) at each fusion layer
3. Audio self-attention operates over 250 audio tokens only — no gap tokens in the attention at all

Context doesn't compete with audio for attention bandwidth. It modulates *how audio features are interpreted* — "given this rhythm pattern, this audio should be read differently." This is the same mechanism as density conditioning (which is load-bearing at ~25pp), just learned from gap patterns instead of scalar statistics.

### Architecture

```
mel → AudioEncoder (4 layers, density FiLM) → 250 audio tokens
events → GapEncoder (2 layers) → C gap tokens → attention pooling → context vector (cond_dim)
density → MLP → density conditioning (cond_dim)

Audio Fusion (4 layers):
  for each layer:
    audio = self_attention(audio)           # 250 tokens only
    audio = density_FiLM(audio, density)    # density modulation
    audio = context_FiLM(audio, context)    # context modulation

cursor = audio[125] → output head → 501 logits
```

| Component | Params | Notes |
|-----------|--------|-------|
| AudioEncoder (conv + 4 transformer layers) | ~8.0M | Same as exp 25-27 |
| GapEncoder (snippet enc + 2 transformer layers) | ~3.5M | Same as exp 25-27 |
| Context pooling (attention + MLP) | ~0.5M | New: learned query → gap tokens |
| Audio fusion (4 self-attention layers) | ~7.5M | Audio-only attention (no gap tokens) |
| Context FiLM (4 layers) | ~0.2M | New: gap-derived FiLM at each fusion layer |
| Density FiLM (4 layers) | ~0.2M | Same as before |
| Output head | ~0.2M | Same |
| **Total** | **~20M** | Similar to exp 25-27 (~19M) |

### Key differences from prior architectures

- **vs Unified (exp 25-27)**: No gap tokens in fusion attention. Context enters only through FiLM modulation.
- **vs Cross-attention (exp 31-33)**: No cross-attention at all. No magnitude imbalance, no banding, no cold start.
- **vs Density conditioning**: Same FiLM mechanism but context is learned from gap tokens (d_model→cond_dim) instead of 3 scalar statistics.

### Expected outcomes

1. **No banding** — audio self-attention is pure audio, no coarse gap features injected.
2. **Fast convergence** — identical audio pathway to exp 27. Audio bootstraps normally.
3. **Context delta > 0** — FiLM conditioning is proven to work (density FiLM is load-bearing). Context FiLM should provide at least some signal.
4. **The question**: is FiLM expressive enough to carry pattern-level information? A cond_dim=64 vector must encode "the pattern is 150 150 75 75 and we're at position 3 in the cycle."

### Risk

- cond_dim=64 may not have enough capacity to encode complex rhythm patterns. Density FiLM works because density is 3 scalars; gap patterns may need more bandwidth.
- FiLM is global modulation (same scale+shift for all 250 audio tokens). It can't say "token 125 should predict 75 specifically" — only "shift all audio tokens toward shorter predictions."
- The model may learn to ignore context FiLM just like it ignored gap tokens in unified fusion — FiLM initialized to identity, gradient descent may keep it there.

## Result

*Pending*

## Lesson

*Pending*
