# Experiment 50 - Anti-Entropy Loss

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

From exp [44-C](../experiment_44c/README.md), the model typically considers 2-4 real options per prediction. It knows the answer is in a small set but can't commit. Entropy analysis (exp [41](../experiment_41/README.md)) showed a trimodal distribution: confident/correct (~1.4 nats), disambiguation zone (~2.3 nats), and uncertain (~3.0 nats). The disambiguation zone is where the model hedges between options instead of picking one.

**Idea:** Add an explicit entropy penalty to the loss. Force the model to produce sharper, more committed distributions. If the model is forced to pick one option confidently, it might learn to use context/audio features that currently get averaged out in the soft distribution.

### Changes from exp [44](../experiment_44/README.md)

- Same as exp [45](../experiment_45/README.md) baseline (tight density jitter, gap ratios)
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

**Converged at eval 21. Sidegrade to exp [44](../experiment_44/README.md): slightly lower HIT, consistently better corruption resilience.**

### Progression

| Metric | Eval 8 | Eval 12 | Eval 15 | Eval 21 | Exp [44](../experiment_44/README.md) ATH (e19) |
|---|---|---|---|---|---|
| HIT | 72.3% | 72.7% | 73.1% | 72.9% | [**73.6%**](../experiment_44/README.md) |
| MISS | 27.1% | 26.8% | 26.4% | 26.5% | [**25.9%**](../experiment_44/README.md) |
| Exact | 53.7% | 53.8% | 54.2% | 54.1% | 54.7% |
| AR step0 | 73.0% | 76.0% | 73.2% | 73.7% | **76.7%** |
| AR step1 | 45.3% | 42.8% | 42.1% | 45.5% | **48.2%** |
| Metronome | 44.2% | **46.9%** | 49.5% | 45.5% | 42.9% |
| Adv metronome | — | — | 51.6% | **52.0%** | 50.1% |
| Time shifted | 47.3% | 48.4% | **49.2%** | 48.3% | 47.3% |
| stop_f1 | 0.548 | 0.514 | 0.531 | 0.534 | 0.520 |
| Unique preds | 462 | 435 | 429 | 442 | ~450 |

### Summary

- HIT plateaued at 72.9-73.2%, never reached exp [44](../experiment_44/README.md)'s 73.6%. The anti-entropy pressure caps peak accuracy slightly.
- Corruption resilience consistently better: metronome +2.6pp, adv metronome +1.9pp, time shifted +1.0pp over exp [44](../experiment_44/README.md) ATH.
- No mode collapse: unique predictions stable at 430-442 throughout training.
- Entropy weight 0.1 adds ~0.23 to loss (~8% of onset loss). Enough to sharpen without destroying diversity.

## Lesson

- **Anti-entropy makes a more robust model at the cost of peak accuracy.** The model commits more decisively, which helps against corruption (less swayed by fake context) but costs ~0.7pp on clean data.
- **No mode collapse risk at weight 0.1.** Unique predictions stayed above 420. The entropy penalty is gentle enough to sharpen without collapsing.
- **Peak HIT may need higher weight to break through.** 0.1 might be too gentle to force the selection improvement we hoped for. Or the anti-entropy is fighting the soft targets (which intentionally spread probability).
- **Next ([50-B](../experiment_50b/README.md)):** Try entropy_weight=0.5 — stronger pressure to commit.
