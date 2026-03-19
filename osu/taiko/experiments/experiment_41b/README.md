# Experiment 41-B - Entropy Progression Over Training (Diagnostic)

## Hypothesis

Exp 41 revealed the skip count is the primary failure mode (skip 0 = 85% HIT, skip 1 = 43%). But does this improve over training? If skip-1 HIT rate increases from eval 1→4→8, the model is gradually learning to prefer the nearest onset and we just need to train longer. If it's flat, the architecture can't solve this and we need a different approach.

Also tracks: does entropy decrease over training? Does confidence improve? Do distant predictions get better?

### Method

Run the skip/entropy analysis from exp 41 on three checkpoints from 35-C: eval 1, eval 4, eval 8. Compare all metrics side by side.

## Result

*Pending*

## Lesson

*Pending*
