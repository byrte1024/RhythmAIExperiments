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

All existing benchmarks run on o1 for backward compatibility, plus per-onset breakdown for o1-o4 and oA.

**Per-onset S1 agreement**: `onset_N_s2_picks_s1`, `onset_N_s2_agree_acc`, `onset_N_s2_override_acc` — tracks whether S2 relies on S1 proposals differently for near vs far onsets.

**Per-onset STOP distribution**: `onset_N_stop_pred_rate`, `onset_N_stop_target_rate` — how often each onset predicts/targets STOP. Expected: o4 targets STOP more than o1.

**Per-onset bin histograms**: `onset_N_pred_pct_0_10`, `onset_N_tgt_pct_0_10`, etc. — where predictions and targets land per onset step.

**TaikoNation patterning metrics on AR output**: `tn_over_pspace`, `tn_hi_pspace`, `tn_dc_human`, `tn_dc_rand` — computed from AR benchmark at 23ms binary resolution. Key comparison: does multi-onset improve pattern diversity (Over.P-Space)?

**Multi-onset structural metrics**:
- `strict_increasing` — % of samples where o1 < o2 < o3 < o4 (temporal ordering). Should be high if model understands sequence.
- `strict_stop_violation_rate` — % with onset-after-STOP (cascade violation). Should be ~0%.
- `all_stop_rate` — % where all 4 onsets are STOP (effective oA STOP).

**Eval graphs**: Full set generated per onset (o1, o2, o3, o4, oA) — heatmaps, scatter plots, distributions.

### Warm-start & no freeze

S1 (proposer) is identical to exp 58 — no reason to retrain it. Warm-start from exp 58's eval 2 checkpoint (end of S1-only freeze phase) to load trained S1 weights. S2's output head shape changed (251 → 1004), so it initializes fresh. No freeze needed since S1 is already trained.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_62 --model-type event_embed_propose --a-bins 500 --b-bins 500 --b-pred 250 --gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --n-onsets 4 --proposer-freeze-evals 0 --warm-start runs/detect_experiment_58/checkpoints/eval_002.pt --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

### Per-Sample (9 evals, best at eval 9 / epoch 3.25)

| Onset | HIT% | MISS% |
|---|---|---|
| o1 | **74.9%** | 24.6% |
| o2 | 58.3% | 38.9% |
| o3 | 49.7% | 40.0% |
| o4 | 43.0% | 36.6% |
| oAvg | 56.4% | 34.9% |

Val loss: 3.103 (still decreasing at eval 9, not fully converged).

**Progression**: o1 HIT climbed from 71.9% (eval 1) to 74.9% (eval 9), matching exp58's ATH (74.6%) despite the harder multi-onset task. Later onsets improved steadily but plateaued around eval 7-9.

**STOP rates**: Model over-predicts STOP on later onsets (o2: 7.6% pred vs 3.2% target, o4: 22.8% vs 13.9%). Too eager to stop the cascade.

**Structural metrics** (eval 9):
- strict_increasing: 69.9% — 70% of multi-onset predictions are in correct temporal order. Climbed from 54.7% at eval 1.
- stop_violation: 0.04% — STOP cascade works correctly.
- all_stop: 0.9% vs target 0.8% — close to expected.

### Autoregressive GT Matching (30 val songs)

| Regime | Close% | Far% | Hall% | d_ratio | err_med |
|---|---|---|---|---|---|
| song_density | 75.0% | 16.7% | 15.9% | 0.97 | 8ms |
| fixed_5.75 | 79.8% | 10.6% | 19.9% | 1.19 | 9ms |

### Comparison to exp58 (song_density regime)

| Metric | exp62 | exp58 | Delta | Better? |
|---|---|---|---|---|
| Close (<50ms) | 75.0% | 75.9% | -0.9pp | exp58 |
| Far (>100ms) | 16.7% | 16.6% | +0.1pp | ≈ |
| Hallucination | 15.9% | 15.6% | +0.3pp | ≈ |
| Density ratio | **0.97** | 0.92 | +0.05 | **exp62** |
| Error median | 8ms | 8ms | = | = |
| Over. P-Space | **12.0%** | 10.1% | **+1.9pp** | **exp62** |
| HI P-Space | **82.4%** | 81.1% | +1.3pp | **exp62** |
| DCHuman | 90.5% | 90.8% | -0.3pp | ≈ |

### TaikoNation Metrics (song_density)

| Model | Over. P-Space | HI P-Space | DCHuman | OCHuman |
|---|---|---|---|---|
| **exp62** | **12.0%** | **82.4%** | 90.5% | 92.9% |
| exp58 | 10.1% | 81.1% | 90.8% | — |
| Human GT | 11.7% | — | — | — |

**Exp62 surpasses human GT pattern diversity** (12.0% vs 11.7% Over.P-Space) — the first model to do so without sacrificing placement accuracy.

### Pattern Variety (song_density)

| Metric | exp62 | Meaning |
|---|---|---|
| gap_std | 174.9 | Higher = more varied timing |
| gap_cv | 0.712 | Coefficient of variation |
| dominant_gap_pct | 47.5% | Lower = less repetitive |
| max_metro_streak | 13.0 | Shorter = less metronomic |

## Lesson

1. **Multi-onset prediction improves pattern diversity without sacrificing timing.** P-Space jumped from 10.1% to 12.0%, surpassing human GT diversity (11.7%). Close rate dropped only 0.9pp (75.0% vs 75.9%) — negligible for the diversity gain.

2. **Density estimation improved.** d_ratio 0.97 vs exp58's 0.92. Multi-onset naturally places more events per AR step, reducing the systematic under-prediction problem. The 1.2x density inflation hack may no longer be needed.

3. **Per-sample metrics don't tell the full story.** oAvg HIT is only 56.4% — looks bad compared to exp58's 74.6%. But o1 alone matches exp58 (74.9%), and the AR inference uses all 4 onsets together, producing charts of equivalent quality with better diversity.

4. **Later onsets are fundamentally harder.** o1→o2 drops 16.7pp, o2→o3 drops 8.6pp, o3→o4 drops 6.7pp. The difficulty is front-loaded — predicting the very next onset after the first is the biggest jump.

5. **STOP over-prediction is the main weakness.** The model predicts STOP 1.6-2.4x more than the target across o2-o4, cutting sequences short. This may be why the diversity gain, while significant, isn't even larger. Future work: reduce STOP weight for later onsets, or use asymmetric loss.

6. **strict_increasing at 70% has room to grow.** 30% of predictions have temporal ordering violations, meaning the model sometimes predicts o3 before o2. This didn't fully converge — more training or explicit ordering constraints could help.
