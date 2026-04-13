# Experiment 65-S2v2 — Context Proposer (Per-Bin Output)

## Purpose

S2 (exp 65-S2) proved context alone carries 70.9% HIT signal, but its 251-class softmax output is incompatible with S1's per-bin sigmoid for fusion. S2 concentrates mass on 1-3 bins — great for top-1 accuracy, terrible for per-bin F1 (0.189 vs S1's 0.712).

S2-v2 retrains the context model with per-bin sigmoid output (250 independent binary detections), matching S1's output format exactly. This enables direct comparison, combination, and eventual S3 fusion.

## Architecture Change from S2

| | S2 (original) | S2-v2 (this) |
|---|---|---|
| Output | 251-class softmax (classification) | 250-bin sigmoid (detection) |
| Loss | OnsetLoss (hard+soft CE + ramp) | Focal BCE (pos_weight=5.0, gamma=2.0) |
| Output space | "which bin is the answer?" | "for each bin, is there an onset?" |
| Params | 4.9M | 13.0M |

Same GRU encoder (4 bidir layers, d=256), same input (gaps + ratios + density FiLM). The difference is the head: instead of `Linear(d_model, 251)`, it expands context to per-bin features then applies per-bin sigmoid.

## Configuration

| Param | Value |
|---|---|
| d_model | 256 |
| GRU layers | 4 (bidirectional) |
| B_PRED | 250 (per-bin output) |
| Loss | Focal BCE, pos_weight=5.0, gamma=2.0 |
| Batch size | 256 |
| Params | 13,043,329 |

## Launch

```bash
cd osu/taiko
python detection_s2v2_train.py taiko_v2 --run-name s2v2_experiment_65 --batch-size 256 --epochs 50 --evals-per-epoch 4 --workers 4
```

## Expected Behavior

- Per-bin F1 should be much higher than S2's converted 0.189
- Recall matters more than precision (proposer role — S3 selects)
- Direct comparison with S1 becomes meaningful
- Combined S1+S2v2 should show improved F1 over either alone

## Result

*Pending*

## Lesson

*Pending*
