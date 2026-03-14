# Experiment 28 - Focal Loss

## Hypothesis

Exp 27 achieved 69.8% HIT with 96% top-10 accuracy — the model narrows to the correct answer almost every time but picks wrong on the hard cases. Exp 27-B confirmed that context contains the answer for ~95% of misses, but the model doesn't use it (context delta ~1.5%).

**The problem: easy samples dominate training.** ~70% of predictions are confident and correct from audio alone. These easy samples generate strong gradients that reinforce the audio pathway. The ~30% hard samples (where context would disambiguate) are drowned out. The model never faces enough pressure to learn context usage because it can reduce loss sufficiently through audio alone.

**Focal loss** directly addresses this by downweighting easy (high-confidence) samples and upweighting hard (low-confidence) ones. With gamma=2.0, a sample the model is 90% confident on gets its loss multiplied by 0.01, while a 50/50 sample keeps 0.25 of its loss. This redirects gradient signal toward the ambiguous cases — exactly the pattern disambiguation (75 vs 150) samples where context should matter.

See [THE_CONTEXT_ISSUE.md](../../THE_CONTEXT_ISSUE.md) for full background on why context utilization is the key bottleneck.

### Changes from exp 27

**Architecture: identical.** Same unified fusion model (~19M params), same heavy audio augmentation.

**Loss change: focal_gamma 0 → 2.0.** Everything else unchanged.

**Training: same as exp 27** — full dataset (subsample=1), batch=48, evals-per-epoch=4, train from scratch.

### Expected outcomes

1. **Slower early convergence** — focal loss downweights the easy 70%, so the model learns them slower. Initial HIT rate may lag behind exp 27.
2. **Better hard-case performance** — the ambiguous samples get proportionally more gradient. Should see improvement on medium/long gap predictions where context matters.
3. **Higher context delta** — if the hard cases are where context helps, focusing on them should increase context contribution. Watch for context delta staying above 2-3% instead of collapsing to 1.5%.
4. **Higher ceiling** — if the ~70% plateau was from gradient starvation on hard cases, focal loss should push past it.

### Risk

- gamma=2.0 might be too aggressive, making training unstable by focusing too much on outliers/noise.
- The hard cases might be hard for reasons other than context (e.g., genuinely ambiguous audio) — focal loss would amplify noise.
- Focal loss might improve accuracy on hard cases but reduce confidence on easy cases, leading to worse calibration.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_28 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --focal-gamma 2.0
```

## Result

*Pending*

## Lesson

*Pending*
