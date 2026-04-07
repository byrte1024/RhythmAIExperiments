# Experiment 59-H - Extended Model Comparison

## Hypothesis

[Exp 59-G](../experiment_59g/README.md) validated the synthetic evaluator on the 53-AR models. Now test models that weren't in 53-AR to see where they rank. Includes the propose-select ATH (exp58) and notable experiments from the 50-series.

### Models

| Model | HIT% | Key trait | Role |
|---|---|---|---|
| exp44 | 73.7% | Gentle augmentation | Reference (53-AR volunteer winner) |
| exp53 | 72.1% | B_AUDIO/B_PRED split | Reference (53-AR 3rd place) |
| exp50b | ~73.2% | Anti-entropy loss (w=0.5) | New — bimodal entropy, untested AR |
| exp51 | 67.5% | Streak-ratio loss weighting | New — failed per-sample but untested AR |
| exp55 | 73.6% | Auxiliary ratio head | New — best val loss ever |
| exp58 | 74.6% | Propose-select two-stage | New — per-sample ATH |

exp44 and exp53 included as reference frame from 53-AR human evaluation.

### Launch

```bash
cd osu/taiko
python experiments/experiment_59h/extended_comparison.py
```

## Result

*Pending*

## Lesson

*Pending*
