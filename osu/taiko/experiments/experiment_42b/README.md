# Experiment 42-B - Pure Hard CE (Frame Tolerance Only)

## Hypothesis

Exp 42's entropy profile is identical to 35-C despite +1.6pp HIT. The proportional soft target (good_pct=3%) creates wider distributions for distant targets — at target=200, 12 bins get full credit. This trains the model to be less confident at distance.

**Test: pure hard CE with ±3 frame tolerance.** No soft trapezoid at all (`hard_alpha=1.0`). The model gets credit only for hitting within 3 bins of the exact target, regardless of target distance. This is maximally sharpening — equal precision demanded at all distances.

Expected: worse HIT rate (hard CE is less forgiving), but significantly lower entropy. If entropy drops, the soft targets were the confidence bottleneck.

### Changes from exp 42

- **hard_alpha: 0.5 → 1.0** (pure hard CE, no soft trapezoid)
- **frame_tolerance: 2 → 3** (±3 bins acceptable)
- Short run (1-2 epochs) — diagnostic only

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_42b --model-type event_embed --hard-alpha 1.0 --frame-tolerance 3 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
