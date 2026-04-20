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

*Pending*

## Lesson

*Pending*
