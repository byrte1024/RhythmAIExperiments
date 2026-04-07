# Experiment 62 - Multi-Onset Prediction

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 61](../experiment_61/README.md) revealed our models have high placement accuracy (DCHuman 90.8%) but low pattern diversity (Over.P-Space 10.1% vs human 11.7%). Our models predict one onset at a time, making greedy decisions that tend toward metronome repetition.

Inspired by TaikoNation (Halina & Guzdial, FDG 2021) which predicts 4 future timesteps simultaneously, **this experiment predicts 4 onsets at once**. By forcing the model to commit to a 4-note sequence per step, it must think in patterns rather than individual decisions. The overlapping predictions across AR steps should naturally produce more diverse rhythmic patterns.

### Key design

- Output: `(B, 4, 251)` — 4 independent 251-class predictions (0-249 offsets + STOP)
- All offsets are from the cursor (not from each other)
- **STOP cascade**: if onset_i = STOP, all onset_j (j > i) are forced to STOP for loss purposes
- Inference: place all non-STOP events, hop cursor to the last one
- Loss: average OnsetLoss across all 4 onset predictions

### Expected behavior

- **HIT% will likely decrease** — predicting 4 onsets is much harder than 1
- **Pattern diversity should increase** — the model learns 4-note patterns, not single decisions
- **onset_1 should be best**, onset_4 worst (further ahead = harder to predict)
- TaikoNation's P-Space advantage may close as our model learns pattern-level thinking

### Configuration

Same as [exp 58](../experiment_58/README.md) (ProposeSelectDetector) plus:

| Feature | Value |
|---|---|
| n_onsets | 4 |
| A_BINS / B_BINS | 500 / 500 |
| B_PRED | 250 (N_CLASSES=251) |
| Gap ratios | ON |
| Density jitter | ±10% at 30% (loose) |
| Proposer layers | 4 |
| Selector layers | 8 |
| Stage 2 freeze | First 2 evals |
| S1 pos_weight | 5.0 |

### Metrics tracked

**Per-onset step** (o1, o2, o3, o4): HIT%, MISS%, accuracy, score for each.

**Averaged** (oA): `multi_onset_avg_hit/miss/score` across all 4 onsets.

**Stage 1** (S1/P1): F1, precision, recall, avg proposals, confidence separation.

**Stage 2 interaction** (S2/P2): picks_s1%, agree accuracy, override accuracy, pick rank, naive baseline.

**Eval summary prints**: S1 stats, S2 stats, and per-onset HIT/MISS breakdown.

**Live training graph**: o1, o2, o3, o4 (green shades) and oA (orange) lines alongside HIT/MISS/Loss.

All existing benchmarks run on o1 for backward compatibility.

### Warm-start & no freeze

S1 (proposer) is identical to exp 58 — no reason to retrain it. Warm-start from exp 58's eval 2 checkpoint (end of S1-only freeze phase) to load trained S1 weights. S2's output head shape changed (251 → 1004), so it initializes fresh. No freeze needed since S1 is already trained.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_62 --model-type event_embed_propose --a-bins 500 --b-bins 500 --b-pred 250 --gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --n-onsets 4 --proposer-freeze-evals 0 --warm-start runs/detect_experiment_58/checkpoints/eval_002.pt --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
