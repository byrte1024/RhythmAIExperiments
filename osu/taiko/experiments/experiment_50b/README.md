# Experiment 50-B - Anti-Entropy Loss (Weight 0.5)

## Hypothesis

Exp 50 (weight 0.1) produced a sidegrade: better resilience, slightly lower HIT. The pressure was enough to improve robustness but too gentle to force better selection. At 0.1, entropy of ~2.3 nats adds ~0.23 to loss (~8%).

At weight 0.5, entropy adds ~1.15 to loss (~40% of onset loss). This is aggressive — the model will be strongly incentivized to produce peaked distributions. The question is whether this forces better discrimination between candidates or causes mode collapse.

### Changes from exp 50

- `--entropy-weight 0.5` (was 0.1)

### Risks

- At 40% of onset loss, the entropy penalty dominates training dynamics. The model may optimize for low entropy over correct predictions.
- Unique predictions could collapse — monitor closely at eval 1-2.
- Soft targets (which spread probability by design) fight directly against anti-entropy. This tension could destabilize training.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_50b --model-type event_embed --entropy-weight 0.5 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
