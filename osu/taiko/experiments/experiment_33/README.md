# Experiment 33 - Interleaved Self-Attention + Cross-Attention

## Hypothesis

Experiments 31-32 revealed a fundamental tension in dual-stream architectures:
- **Exp 31** (late cross-attention): 18.8% context delta but severe banding (53 unique predictions) — cross-attention overwrites fine-grained audio features
- **Exp 32** (+ audio skip connection): 363 unique predictions but -1.8% context delta — skip connection lets the model bypass cross-attention entirely

The problem: separating audio processing from context injection means the model must choose between fine-grained audio (pre-fusion) and context-aware audio (post-fusion). It can't have both.

**Interleaved architecture** solves this by weaving context into audio processing at every stage. Each block:
1. Audio self-attention — consolidate fine-grained temporal features
2. Gap self-attention — consolidate rhythm pattern features
3. Bidirectional cross-attention — exchange information

Audio never goes more than 1 layer without seeing context, AND it processes its own features between each cross-attention. No skip connection needed — audio self-attention naturally preserves fine-grained features while cross-attention adds context on top.

### Architecture

```
mel → Conv stem → 250 audio tokens
events → Gap feature extraction → C gap tokens

  Block 1:  audio self-attn → gap self-attn → audio↔gap cross-attn
  Block 2:  audio self-attn → gap self-attn → audio↔gap cross-attn
  Block 3:  audio self-attn → gap self-attn → audio↔gap cross-attn
  Block 4:  audio self-attn → gap self-attn → audio↔gap cross-attn

cursor = audio[125] → output head → 501 logits
```

Each block has: audio self-attention (1 layer) + gap self-attention (1 layer) + bidirectional cross-attention (1 layer), all with FiLM conditioning. 4 blocks = 4 audio self-attn + 4 gap self-attn + 4 cross-attn = 12 attention layers total.

Key difference from all prior architectures:
- **vs Unified (exp 25-27)**: audio and gap have dedicated self-attention, gap tokens can't be drowned out
- **vs Dual-stream (exp 31)**: cross-attention is interleaved, not bolted on at the end. Audio preserves fine-grained features between cross-attention injections
- **vs Skip connection (exp 32)**: no audio shortcut that bypasses context. Context is woven into every layer

### Expected outcomes

1. **Prediction diversity maintained** — audio self-attention between cross-attention layers preserves fine-grained temporal features. Should see 300+ unique predictions.
2. **Context delta stays high** — cross-attention at every stage means context can't be bypassed. Gap tokens influence audio features throughout, not just at the end.
3. **Both streams develop strong representations** — dedicated self-attention for each modality before information exchange.

### Risk

- 12 attention layers total is deeper than prior architectures (10 for unified/dual-stream). May be slower to train and harder to optimize.
- Cross-attention at every layer may still see gap activations overwhelming audio. The magnitude imbalance (gap ±20 vs audio ±7) from exp 31-B could recur.
- May need the activation clamping safety net from exp 31-B.

## Result

*Pending*

## Lesson

*Pending*
