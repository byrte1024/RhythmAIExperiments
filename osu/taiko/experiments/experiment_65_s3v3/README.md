# Experiment 65-S3v3 — Pure Proposal Fusion (No Audio Features)

## Purpose

S3v2 achieved 71.4% HIT per-sample but only 65.8% close rate in AR — 10pp below exp58. The agreement analysis showed S3v2 ignores both S1 and S2v2 proposals 57.7% of the time, predicting from its own audio features instead.

**Hypothesis**: The raw audio features give S3v2 enough capacity to bypass S1/S2v2. By removing ALL audio features, event embeddings, and density conditioning, we force the model to rely ONLY on S1 and S2v2 confidence signals.

If this works even moderately well, the proposal signals are sufficient. If it fails, the signals are too noisy/spread for single-onset prediction.

## Architecture: PureProposalFusion

Input: S1 confidence (250,) + S2v2 confidence (250,) only. Nothing else.

Treat the 250 prediction bins as a sequence. Each bin position has 2 features: [s1_conf, s2_conf]. Run a small transformer over this sequence so each bin can attend to all other bins' proposal confidences. Predict from cursor token.

```
s1_conf (B, 250) + s2_conf (B, 250)
→ stack → (B, 250, 2)
→ Linear(2, d_model) + positional encoding
→ N transformer layers (self-attention over 250 bins)
→ cursor token (bin 0, center of prediction range)
→ Linear(d_model, 251) → logits
```

No mel. No events. No density. No FiLM. Pure S1+S2v2 signals.

## Configuration

| Param | Value |
|---|---|
| d_model | 128 (small — only 2 input features per bin) |
| Layers | 4 |
| Heads | 4 |
| N_CLASSES | 251 |
| Input | S1 conf (250,) + S2v2 conf (250,) |
| No audio | By design |
| No events | By design |
| No density | By design |

**Estimated params: ~1-2M** (tiny)

## What This Tests

1. **Are S1/S2v2 signals sufficient for onset prediction?** If yes: the problem is how S3v2 integrates them, not the signals themselves.
2. **Does cross-bin attention help?** Each bin sees other bins' S1/S2v2 confidences — can it learn "if S1 peaks at bin 30 and S2v2 peaks at bin 33, pick 31"?
3. **What's the floor?** This is the minimum a pure-proposal model can achieve.

## Expected Behavior

- Per-sample HIT should be lower than S3v2's 71% (no audio for timing)
- But if proposal signals are useful, should beat random (~0.4%)
- If HIT > 50%, proposals contain real signal that S3v2 was wasting
- The agreement stats should show near 100% alignment with S1/S2v2 (forced by design)

## Launch

```bash
cd osu/taiko
python detection_train.py taiko_v2 --run-name s3v3_experiment_65 \
    --model-type pure_proposal_fusion \
    --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
    --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt \
    --b-pred 250 \
    --d-model 128 --enc-layers 2 --fusion-layers 2 --n-heads 4 \
    --ramp-alpha 2.5 \
    --epochs 50 --batch-size 256 --evals-per-epoch 4 --workers 3
```

Batch size 256 since the S3v3 model itself is tiny (~1-2M params). S1/S2v2 still run per sample (need mel + events) but their forward passes are fast.

Running on Windows (RTX 5070, 12GB).

## Result

### Per-Sample (4 evals, epoch 2.0)

| Eval | HIT | MISS | Acc | StpF1 | Val Loss |
|---|---|---|---|---|---|
| 1 | 64.0% | 35.5% | 46.7% | 0.436 | 3.676 |
| 2 | 65.6% | 34.0% | 48.4% | 0.391 | 3.579 |
| 3 | 66.9% | 32.7% | 48.9% | 0.426 | 3.417 |
| 4 | **67.8%** | **31.8%** | 49.7% | 0.454 | 3.335 |

67.8% HIT from pure proposals alone. Still climbing at epoch 2.0.

### Benchmarks: BOTH Signals Used

| Benchmark | Acc | Delta from normal |
|---|---|---|
| normal | 60.6% | — |
| no_s1 | 46.0% | **-14.6pp** (S1 critical) |
| no_s2 | 54.8% | **-5.8pp** (S2 contributes!) |
| random_s1 | 0.4% | -60.2pp |
| random_s2 | 47.8% | **-12.8pp** |
| no_s1s2 | 4.1% | -56.5pp |

Unlike S3v2 (which ignored S2v2 entirely), S3v3 genuinely uses both signals. Removing S2v2 costs 5.8pp, randomizing it costs 12.8pp. Without both, the model is helpless (4.1%).

### AR Evaluation (30 songs)

| Metric | S3v3 | S3v2 | exp58 |
|---|---|---|---|
| Close% | 60.0% | 65.8% | **75.9%** |
| Hall% | **15.4%** | 14.6% | 15.6% |
| d_ratio | 0.70 | 0.78 | **0.92** |
| Error med | 34ms | 35ms | **8ms** |
| P-Space | 9.4% | 9.9% | **10.1%** |

AR quality is poor — 60% close, heavily under-predicting (d_ratio 0.70), metronomic (P-Space 9.4%). Without audio features for precise timing, AR errors compound.

### Agreement (AR)

| | S3v3 | S3v2 |
|---|---|---|
| S1 agrees | 23.8% | 18.1% |
| S2 agrees | 37.3% | 33.3% |
| Both agree | 12.2% | 9.0% |
| Neither | 51.1% | 57.7% |

S3v3 follows proposals more than S3v2 (51% neither vs 58%) — forced to by design. But proposals aren't precise enough for AR on their own.

## Lesson

1. **Proposal signals ARE sufficient for per-sample prediction.** 67.8% HIT from S1+S2v2 confidences alone, no audio features. This proves the signals work — S3v2 chose to ignore them, not that they were useless.

2. **Both S1 and S2v2 contribute when the model can't bypass them.** S2v2 contributes 5.8-12.8pp when there's no audio fallback. The blackout augmentation + audio features in S3v2 taught it to ignore S2v2; removing audio forces genuine fusion.

3. **Proposals alone fail in AR.** 60% close rate, metronomic, under-predicting. Without audio for timing refinement, small errors compound. STOP behavior degrades (d_ratio 0.70).

4. **The integration problem remains unsolved.** S3v2 has audio but ignores proposals. S3v3 uses proposals but lacks audio for AR. The architecture must force the model to use BOTH audio and proposals without letting audio dominate. Ratio-based prediction or architectural separation may be needed.
