# Experiment 65-S3v2 — Fusion Selector (Single-Onset Classification)

## Purpose

S3 (per-bin detection) achieved F1=0.771 but produces terrible AR results compared to exp58. The per-bin paradigm creates noisy, multi-placement outputs that compound errors in autoregressive inference. Key observations from S3 inference:

1. **Almost no white peaks** — S1 and S2v2 almost never agree on the exact same bin. S3 sides with S1 (magenta = S1+S3), suggesting it may be ignoring S2v2 for bin-level placement.
2. **S2v2 predictions are spread/faded** — broad low-confidence windows instead of sharp peaks. Per-bin sigmoid isn't the right output format for context.
3. **Threshold-based multi-placement** is wrong for AR — exp58's clean single-onset argmax produces much better charts.

**Solution**: Keep S3's rich input (audio tokens enriched with S1 + S2v2 confidence signals) but change output back to **single-onset 251-class classification** (like exp58). S3v2 = "exp58's selector with S2v2 context added."

## Architecture

Same as exp58's ProposeSelectDetector selector stage, but with an additional S2v2 confidence signal embedded into audio tokens:

```
Input tokens (250, d_model):
  audio_features (from S1 conv stem, projected)
  + S1_confidence_embedding (per-token, like current proposal_embed)  
  + S2v2_confidence_embedding (per-token, NEW)
  + event_embeddings (scatter-added to past tokens)
  + density FiLM conditioning

→ 8 transformer layers with FiLM
→ cursor token → Linear(d_model, 251) → logits
→ argmax → single onset prediction
```

### Key differences from S3 (per-bin)

| | S3 (per-bin) | S3v2 (single-onset) |
|---|---|---|
| Output | (B, 250) sigmoid | (B, 251) softmax |
| Loss | Focal BCE per bin | OnsetLoss (hard+soft CE+ramp) |
| AR sampling | threshold → multiple events | argmax → one event |
| Paradigm | Detection (framewise) | Classification (next-onset) |

### Key differences from exp58

| | exp58 | S3v2 |
|---|---|---|
| S1 signal | Proposal embedding | Same |
| S2 signal | None | **S2v2 confidence embedding (NEW)** |
| Context | Event embeddings only | Event embeddings + S2v2 pattern signal |

## Configuration

| Param | Value |
|---|---|
| d_model | 384 |
| Encoder layers | 8 (transformer with FiLM) |
| S1 embed | Linear(1, 384) → GELU → Linear(384, 384) |
| S2v2 embed | Linear(1, 384) → GELU → Linear(384, 384) (NEW) |
| Output | 251 classes (250 bins + STOP) |
| Loss | OnsetLoss (ha=0.5, ramp=2.5) |
| N_CLASSES | 251 |

## Expected Behavior

- AR quality should match or beat exp58 (74.6% HIT) — same output paradigm with richer input
- The S2v2 signal provides "which region has an onset" while audio provides "which exact bin"
- Single clean event per AR step — no threshold noise, no multi-placement
- Direct comparison to exp58: the ONLY difference is the added S2v2 confidence embedding

## Future Work Notes

- **S2 output format needs improvement**: S2v2's per-bin sigmoid produces spread/faded predictions. A scalar (single-onset classification) or peaked output might give S3v2 sharper context signal.
- **S2 as ratio predictor**: Instead of per-bin detection, S2 could predict a ratio distribution (1.0x, 0.5x, 2.0x) which maps to specific bin ranges. This would be naturally peaked.

## Implementation

Uses `detection_train.py` with `--model-type fusion_classifier`. No separate training script — inherits all eval infrastructure, benchmarks, graphs, live training display, AR benchmarks.

S1 and S2v2 run frozen in real-time per training sample. Augmentations flow through (audio aug → different S1 output, context jitter → different S2v2 output each epoch). Additional proposal augmentation: 5% gaussian shake on confidences, 5% blackout (33% S1 zero, 33% S2 zero, 33% both zero).

5 fusion-specific benchmarks added: no_s1, no_s2, no_s1s2, random_s1, random_s2.

## Launch

```bash
cd osu/taiko
python detection_train.py taiko_v2 --run-name s3v2_experiment_65 \
    --model-type fusion_classifier \
    --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
    --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt \
    --b-pred 250 \
    --gap-ratios \
    --ramp-alpha 2.5 \
    --epochs 50 --batch-size 48 --evals-per-epoch 4 --workers 3
```

Critical: `--b-pred 250` sets N_CLASSES=251, matching S1/S2v2 checkpoints which were trained with b_pred=250.

## Result

*Pending*

## Lesson

*Pending*
