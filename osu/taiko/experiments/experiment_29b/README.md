# Experiment 29-B - Auxiliary Context Loss (Weight 1.0)

## Hypothesis

Exp 29 showed that ctx_loss_weight=0.2 is too weak — the aux head's gradient is ~20% of the gap encoder's total, dominated by fusion which tells it to be quiet. Context delta collapsed identically to every prior experiment. Ctx loss barely dropped (4.2→4.1, near random).

**Increase weight to 1.0** so the aux gradient matches the main loss gradient on the gap encoder. The gap encoder should feel equal pressure from both the standalone prediction head and the fusion pathway. This may finally break the audio-dominance equilibrium.

### Changes from exp 29

**ctx_loss_weight: 0.2 → 1.0.** Everything else identical.

### Risk

- Weight 1.0 means the total loss is `main_loss + 1.0 * ctx_loss`. With ctx_loss at ~4.1 and main_loss at ~3.3, the aux loss now dominates total gradient magnitude. This could distort the gap encoder toward standalone prediction at the expense of fusion-compatible features.
- Could destabilize training if the two loss terms pull the gap encoder in opposite directions.

## Result

*Pending*

## Lesson

*Pending*
