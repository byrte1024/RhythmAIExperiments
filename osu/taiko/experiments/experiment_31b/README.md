# Experiment 31-B - Dual-Stream with 4 Cross-Attention Layers

## Hypothesis

Exp 31 proved that dual-stream architecture forces context dependence (18.8% context delta — highest ever). But 2 cross-attention layers bottleneck information flow, limiting the model to ~80 unique predictions (vs ~350 for the unified model). The model learns "which common gap value" but not "exactly which bin."

**Double the cross-attention layers (2→4)** to allow richer information exchange between audio and context streams while maintaining the stream separation that forces context dependence.

### Changes from exp 31

**cross_attn_layers: 2 → 4.** Everything else identical.

This adds ~5M params (total ~28M), making the model larger than exp 27 (~19M). If it works, we can optimize later.


## Result

*Pending*

## Lesson

*Pending*
