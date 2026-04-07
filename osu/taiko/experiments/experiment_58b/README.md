# Experiment 58-B - Propose-Select with Precision-Focused S1

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 58](../experiment_58/README.md) achieved 74.6% HIT (new ATH) with the propose-select architecture, but Stage 1's F1 plateaued at ~0.50 with 67 proposals per sample. Precision was only 37.7% — S2 had to sift through ~35 false positives per 30 real onsets.

**Theory: S1's high pos_weight (5.0) pushes too hard on recall at the expense of precision.** With fewer, higher-quality proposals, S2 would have a cleaner selection problem. Fewer candidates means less noise, potentially higher S2 accuracy.

### Changes from exp 58

| Param | Exp 58 | Exp 58-B |
|---|---|---|
| S1 pos_weight | 5.0 | **2.0** |
| S1 focal gamma | 2.0 | 2.0 (unchanged) |
| Everything else | — | Identical |

Lower pos_weight means false negatives (missing an onset) are penalized less relative to false positives. S1 should produce fewer proposals with higher confidence.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_58b --model-type event_embed_propose --a-bins 500 --b-bins 500 --b-pred 250 --gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --proposer-pos-weight 2.0 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

Stopped at eval 9 (epoch 3.2). **Peak at eval 8-9 (73.6% HIT).**

### Peak metrics:

| Metric | Value |
|---|---|
| HIT% | 73.6% |
| Accuracy | 55.1% |
| Val loss | 2.450 |
| S1 proposals | 19-22 |
| S1 precision | 0.578 |
| s2_picks_s1 | 43.9-48.3% |
| s2_override_acc | 52.9% |

### vs exp 58 (pw=5.0):

| Metric | 58 (pw=5) | 58-B (pw=2) |
|---|---|---|
| Peak HIT% | **74.6%** | 73.6% |
| S1 proposals | 67 | **19** |
| S1 precision | 0.377 | **0.578** |
| s2_picks_s1 | 82% | **44%** |
| s2_override_acc | 48.4% | **52.9%** |

## Lesson

Lower pos_weight produces fewer, higher-precision proposals and a more independent Stage 2 (overrides S1 majority of the time with 53% accuracy). But the HIT ceiling is 1pp lower than exp 58.

The tradeoff: more proposals (exp 58) gives S2 more correct positions to pick from, even though many are false positives. Fewer proposals (58-B) forces S2 independence but limits its ceiling. The recall-focused S1 (exp 58) produces the higher overall accuracy despite S2 being less independent.
