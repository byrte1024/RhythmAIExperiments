# Experiment 29 - Auxiliary Context Loss

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

14 experiments (14-28) confirm the model ignores context regardless of augmentation, data volume, or loss reweighting. The context delta collapses to ~0-1.5% in every run. The gap encoder's representations atrophy because the fusion pathway finds audio-only solutions — there is no gradient pressure forcing the gap encoder to carry useful signal.

**Auxiliary context loss** adds a second prediction head that operates on gap tokens only (before fusion with audio). This head has its own loss term, creating direct gradient pressure on the gap encoder to learn useful representations regardless of what the main fusion pathway does. The gap encoder has no choice but to carry signal.

This is structurally different from everything we've tried:
- Exp 15-24 (separate paths): context couldn't see audio → couldn't know when to act
- Exp 25-27 (unified fusion): context could see audio but was ignored
- Exp 28 (focal loss): redirected gradients to hard cases but model still solved them through audio
- **Exp 29: forces the gap encoder to independently predict, then lets fusion decide how to weight both signals**

See [THE_CONTEXT_ISSUE.md](../../THE_CONTEXT_ISSUE.md) for full background.

### Architecture change

Added to `OnsetDetector`:
- **Context head**: a learned query token cross-attends to gap tokens (after GapEncoder, before fusion), then projects to 501 logits
- During training: forward returns `(main_logits, ctx_logits)`. Loss = `main_loss + ctx_weight * ctx_loss`
- During inference: only main_logits used (no overhead)

The context head is lightweight (~0.3M params): one cross-attention layer + LayerNorm + Linear. It sees gap tokens only — no audio. This means its gradient flows exclusively through the gap encoder, forcing it to learn representations that are independently useful for onset prediction.

### Changes from exp 27

**Architecture**: + auxiliary context head (~0.3M params)
**Loss**: + `ctx_loss_weight * context_prediction_loss` (using 0.2 to start)
**Training**: same as exp 27 — full dataset (subsample=1), batch=48, evals-per-epoch=4, no focal loss

### Expected outcomes

1. **Context delta should stay high** — the gap encoder has direct gradient pressure from the aux head. Even if fusion ignores context, the gap encoder's representations stay useful.
2. **Context standalone accuracy > 0%** — the aux head should reach some level of standalone prediction ability (exp 24 showed context in isolation can reach ~53% HIT).
3. **Main HIT ≥ 69.8%** — the aux loss shouldn't hurt the main pathway (it only adds gradient to the gap encoder, doesn't remove gradient from audio).
4. **Main HIT > 69.8%** — if the gap encoder learns better representations AND fusion learns to use them, the main pathway benefits from context.

### Risk

- ctx_loss_weight too high could distort gap encoder representations toward standalone prediction at the expense of fusion-compatible features.
- The aux head might learn while the fusion pathway still ignores it — gap encoder carries signal but fusion doesn't read it.
- Training instability from two competing loss terms pulling the gap encoder in different directions.

## Result

**Aux weight 0.2 too low — context delta collapsed as usual, aux head barely learned.** Killed after eval 4 (~1 epoch).

| eval | epoch | HIT | Miss | Score | Acc | Val loss | Ctx loss | no_events | Ctx Δ |
|------|-------|-----|------|-------|-----|----------|----------|-----------|-------|
| 1 | 1.25 | 66.4% | 33.0% | 0.305 | 48.5% | 2.685 | 4.195 | 42.5% | 6.0% |
| 2 | 1.50 | 66.6% | 32.9% | 0.305 | 48.6% | 2.670 | 4.144 | 45.4% | 3.2% |
| 3 | 1.75 | 68.0% | 31.5% | 0.322 | 50.0% | 2.616 | 4.114 | 48.2% | 1.8% |
| 4 | 1.00 | 68.2% | 31.3% | 0.323 | 50.0% | 2.618 | 4.096 | 48.6% | 1.5% |

**What happened:**
- **Context delta collapsed identically** — 6.0% → 1.5% over 4 evals. Same pattern as exp 25-28.
- **Aux head barely learned** — ctx loss dropped from 4.195 to 4.096 over 4 evals (~2% reduction). Random 501-class loss is ~6.2, so 4.1 is barely above random. The gap encoder isn't producing useful standalone representations.
- **Main pathway unaffected** — HIT tracked exp 27 closely (68.2% vs 68.6% at eval 4). The aux loss didn't hurt but didn't help either.

**Why it failed:**
At weight 0.2, the aux gradient is ~20% of the gap encoder's total gradient. The other ~80% comes from the fusion pathway, which still tells the gap encoder "be quiet, let audio handle it." The aux head can't overcome fusion's dominance at this weight — it's a whisper in a shouting match.

## Lesson

- **0.2 aux weight is insufficient.** The gradient from the aux head is too small relative to fusion's gradient on the gap encoder. The gap encoder optimizes for fusion (which wants silence) and the aux head gets leftover capacity.
- **Ctx loss as a diagnostic is valuable.** Tracking the aux head's loss independently shows whether the gap encoder is learning useful representations. At 4.1 (barely above random), we know it isn't.
- **Need stronger forcing.** Next: increase weight to 1.0 so the aux gradient matches or exceeds fusion's gradient on the gap encoder. The gap encoder should feel equal pressure from both heads.
