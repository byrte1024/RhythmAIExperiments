# Experiment 50 - Anti-Entropy Loss

## Hypothesis

From exp 44-C, the model typically considers 2-4 real options per prediction. It knows the answer is in a small set but can't commit. Entropy analysis (exp 41) showed a trimodal distribution: confident/correct (~1.4 nats), disambiguation zone (~2.3 nats), and uncertain (~3.0 nats). The disambiguation zone is where the model hedges between options instead of picking one.

**Idea:** Add an explicit entropy penalty to the loss. Force the model to produce sharper, more committed distributions. If the model is forced to pick one option confidently, it might learn to use context/audio features that currently get averaged out in the soft distribution.

### Changes from exp 44

- Same as exp 45 baseline (tight density jitter, gap ratios)
- Added `--entropy-weight W`: loss += W * mean_entropy_of_softmax
- Entropy is computed as `-sum(p * log(p))` over the softmax distribution

### Risk

- Too much anti-entropy could collapse the distribution to always predicting one bin (mode collapse)
- Might hurt soft target learning since soft targets intentionally spread probability
- Could worsen the 2x/0.5x error if the model commits to the wrong musical ratio more confidently

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_50 --model-type event_embed --entropy-weight 0.1 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
