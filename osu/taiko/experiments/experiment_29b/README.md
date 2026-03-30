# Experiment 29-B - Auxiliary Context Loss (Weight 1.0)

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp [29](../experiment_29/README.md) showed that ctx_loss_weight=0.2 is too weak — the aux head's gradient is ~20% of the gap encoder's total, dominated by fusion which tells it to be quiet. Context delta collapsed identically to every prior experiment. Ctx loss barely dropped (4.2→4.1, near random).

**Increase weight to 1.0** so the aux gradient matches the main loss gradient on the gap encoder. The gap encoder should feel equal pressure from both the standalone prediction head and the fusion pathway. This may finally break the audio-dominance equilibrium.

### Changes from exp [29](../experiment_29/README.md)

**ctx_loss_weight: 0.2 → 1.0.** Everything else identical.

### Risk

- Weight 1.0 means the total loss is `main_loss + 1.0 * ctx_loss`. With ctx_loss at ~4.1 and main_loss at ~3.3, the aux loss now dominates total gradient magnitude. This could distort the gap encoder toward standalone prediction at the expense of fusion-compatible features.
- Could destabilize training if the two loss terms pull the gap encoder in opposite directions.

## Result

**Higher weight didn't help — aux head still can't learn, context delta still collapses.** Killed after eval 6 (~2.5 epochs).

| eval | epoch | HIT | Miss | Score | Val loss | Ctx loss | no_events | Ctx Δ |
|------|-------|-----|------|-------|----------|----------|-----------|-------|
| 1 | 1.25 | 66.3% | 33.0% | 0.305 | 2.691 | 4.215 | 42.5% | 4.7% |
| 2 | 1.50 | 65.5% | 34.0% | 0.291 | 2.708 | 4.163 | 43.4% | 3.8% |
| 3 | 1.75 | 67.8% | 31.7% | 0.321 | 2.624 | 4.133 | 48.0% | 1.3% |
| 4 | 1.00 | 68.2% | 31.3% | 0.325 | 2.616 | 4.110 | 49.7% | 0.8% |
| 5 | 2.25 | **69.0%** | **30.5%** | **0.334** | **2.593** | 4.021 | 49.5% | 1.4% |
| 6 | 2.50 | 68.4% | 31.2% | 0.326 | 2.596 | 4.014 | 50.3% | 0.3% |

**What happened:**
- **Same collapse.** Context delta: 4.7% → 0.3% by eval 6. Identical pattern to every prior experiment.
- **Ctx loss still barely moved** — 4.215 → 4.014 over 6 evals. Slightly faster than exp [29](../experiment_29/README.md) (0.2 weight) but still near random (~6.2). The gap encoder can't produce representations useful for standalone 501-class prediction regardless of gradient strength.
- **HIT ~0.5pp behind exp [27](../experiment_27/README.md)** — the heavier aux loss is a drag on the main pathway without providing benefit.

**The fundamental issue:**
The aux head approach fails not because of gradient weight, but because **501-class prediction from gap tokens alone is too hard**. The aux head can't learn → can't reward the gap encoder → gap encoder never improves. The bottleneck is the task difficulty for the aux head, not gradient magnitude.

## Lesson

- **Aux context loss is the wrong forcing mechanism.** The standalone prediction task (501 classes from gaps alone) is nearly unsolvable with a lightweight head. No amount of gradient weight fixes this — you can't force a head to learn a task it can't solve.
- **Need to make audio absence impossible to ignore** — rather than adding an aux head, remove audio directly during training so the main pathway must use context. Cursor-region audio masking is the next approach.
