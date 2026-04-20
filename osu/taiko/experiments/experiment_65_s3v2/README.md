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

### Per-Sample (3 evals, epoch 1.75)

| Eval | HIT | MISS | Acc | StpF1 | Val Loss |
|---|---|---|---|---|---|
| 1 (1.25) | 71.4% | 28.1% | 52.0% | 0.500 | 3.168 |
| 2 (1.50) | 71.4% | 28.2% | 52.5% | 0.513 | 3.158 |
| 3 (1.75) | 71.1% | 28.4% | 52.1% | 0.539 | 3.187 |

71.4% HIT at eval 1 — strong start, but plateaued immediately.

### Benchmarks: S3v2 Ignores S2v2

| Benchmark | Acc | Delta from normal |
|---|---|---|
| normal | 65.0% | — |
| no_s2 | 64.7% | **-0.3pp** |
| random_s2 | 65.2% | **+0.2pp** |
| no_s1 | 49.6% | -15.4pp |
| random_s1 | 3.6% | -61.4pp |
| no_audio | 2.8% | -62.2pp |
| no_events | 61.7% | -3.3pp |

**S2v2 has zero effect.** Zeroing or randomizing S2v2 doesn't change accuracy. S3v2 learned to rely entirely on S1 + audio features + events + density. The 5% blackout augmentation taught it S2 is unreliable.

### AR Evaluation (30 val songs)

| Metric | S3v2 | exp58 | Delta |
|---|---|---|---|
| Close% | 65.8% | **75.9%** | **-10.1pp** |
| Hall% | **14.6%** | 15.6% | -1.0pp |
| d_ratio | 0.78 | **0.92** | -0.14 |
| Error med | 35ms | **8ms** | +27ms |
| P-Space | 9.9% | 10.1% | -0.2pp |
| DCHuman | 90.1% | 90.8% | -0.7pp |

**10pp worse than exp58 on close rate.** Under-predicts density. Lower hallucination is the only win.

### Agreement Analysis: S3v2 Ignores Proposals

| Metric | Value |
|---|---|
| S1 agrees with S3v2 (±3 bins) | 18.1% |
| S2 agrees with S3v2 (±3 bins) | 33.3% |
| Both agree | 9.0% |
| **Neither agrees** | **57.7%** |
| S1 conf at S3v2's pick | 0.611 |
| S2 conf at S3v2's pick | 0.578 |

57.7% of the time, S3v2 picks a bin where neither S1 nor S2v2's argmax agrees. The model is predicting from its own audio features, not from proposals.

## Lesson

1. **S3v2 is a worse exp58.** Same architecture, but frozen S1's projected audio features are lossier than exp58's internal conv stem. 10pp close rate regression confirms the S1→project→S3v2 pipeline loses critical timing information.

2. **S2v2 is completely ignored.** Zero delta on no_s2/random_s2 benchmarks. The 5% blackout augmentation during training taught S3v2 that S2v2 is unreliable — it learned to never depend on it. Future iterations should reduce or remove S2 blackout.

3. **The model bypasses proposals.** 57.7% neither-agree rate shows S3v2 uses the raw audio features (from S1's conv stem) to independently detect onsets, ignoring both S1 and S2v2 confidence signals. The architecture gives it enough capacity to relearn onset detection from scratch.

4. **Single-onset classification works, but proposals don't help.** The 71.4% HIT proves the architecture can detect onsets. The problem is integration of S1/S2v2 signals, not the output format.
