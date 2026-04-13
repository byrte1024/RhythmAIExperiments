# Experiment 65-S2v2 — Context Proposer (Per-Bin Output)

## Purpose

S2 (exp 65-S2) proved context alone carries 70.9% HIT signal, but its 251-class softmax output is incompatible with S1's per-bin sigmoid for fusion. S2 concentrates mass on 1-3 bins — great for top-1 accuracy, terrible for per-bin F1 (0.189 vs S1's 0.712).

S2-v2 retrains the context model with per-bin sigmoid output (250 independent binary detections), matching S1's output format exactly. This enables direct comparison, combination, and eventual S3 fusion.

## Architecture Change from S2

| | S2 (original) | S2-v2 (this) |
|---|---|---|
| Output | 251-class softmax (classification) | 250-bin sigmoid (detection) |
| Loss | OnsetLoss (hard+soft CE + ramp) | Focal BCE (pos_weight=5.0, gamma=2.0) |
| Output space | "which bin is the answer?" | "for each bin, is there an onset?" |
| Params | 4.9M | 13.0M |

Same GRU encoder (4 bidir layers, d=256), same input (gaps + ratios + density FiLM). The difference is the head: instead of `Linear(d_model, 251)`, it expands context to per-bin features then applies per-bin sigmoid.

## Configuration

| Param | Value |
|---|---|
| d_model | 256 |
| GRU layers | 4 (bidirectional) |
| B_PRED | 250 (per-bin output) |
| Loss | Focal BCE, pos_weight=5.0, gamma=2.0 |
| Batch size | 256 |
| Params | 13,043,329 |

## Launch

```bash
cd osu/taiko
python detection_s2v2_train.py taiko_v2 --run-name s2v2_experiment_65 --batch-size 256 --epochs 50 --evals-per-epoch 4 --workers 4
```

## Expected Behavior

- Per-bin F1 should be much higher than S2's converted 0.189
- Recall matters more than precision (proposer role — S3 selects)
- Direct comparison with S1 becomes meaningful
- Combined S1+S2v2 should show improved F1 over either alone

## Result

### Per-Bin F1 (100K val samples × 250 bins = 25M bins total, 2.38M onset bins)

| Model | Best F1 | Threshold | Onset Conf | Non-Onset Conf | Separation |
|---|---|---|---|---|---|
| S1 (audio) | 0.712 | 0.50 | 0.623 | 0.091 | 0.531 |
| **S2v2 (context)** | **0.727** | 0.50 | 0.618 | 0.106 | 0.511 |

S2v2 beats S1 by +1.5pp F1 on per-bin detection. Both have nearly identical confidence profiles — context-only detection is as calibrated as audio detection.

### Agreement/Disagreement Analysis (both at threshold 0.5)

| Case | Bins | % of total | Onset rate | Meaning |
|---|---|---|---|---|
| **Both positive** | 1,827,161 | 7.3% | **82.8%** | Both agree → very reliable |
| S1 only positive | 628,039 | 2.5% | 33.5% | Audio fires, context doesn't → 1/3 real |
| S2v2 only positive | 687,502 | 2.8% | 38.9% | Context fires, audio doesn't → 2/5 real |
| Both negative | 21,857,298 | 87.4% | 1.8% | Both silent → mostly correct |

S2v2's solo detections are more trustworthy than S1's (38.9% vs 33.5% onset rate).

### Strong Disagreements

| Case (conf gap > 0.3) | Onset rate | Meaning |
|---|---|---|
| S2v2 >> S1 | 22.0% | Context-predicted onsets without audio evidence |
| S1 >> S2v2 | **14.5%** | Audio transients that don't fit rhythmic pattern — mostly false positives |

When S1 is very confident but S2v2 disagrees, S1 is right only 14.5% — these are noise transients that context correctly rejects.

### Missed by Both

391,245 onset bins (**16.4%** of all onsets) are missed by both models. Neither audio nor context detects them. These are the structurally hard cases.

**83.6% of all onsets are detected by at least one model** — the union coverage ceiling.

### Combined Models

| Combination | Best F1 | vs S1 | vs S2v2 |
|---|---|---|---|
| S1 only | 0.712 | — | — |
| S2v2 only | 0.727 | — | — |
| **Average** | **0.752** | **+4.0pp** | **+2.5pp** |
| Product sqrt(S1×S2v2) | 0.751 | +3.9pp | +2.4pp |
| S2v2 × S1^0.5 | **0.752** | +4.0pp | +2.5pp |
| Min(S1, S2v2) | 0.749 | +3.7pp | +2.2pp |
| Max(S1, S2v2) | 0.725 | +1.3pp | -0.2pp |
| 0.5×S1 + 0.5×S2v2 | 0.752 | +4.0pp | +2.5pp |

Simple averaging achieves the best combination (+4.0pp over audio alone). Max hurts because it takes either model's false positives. Min works well because it requires agreement, filtering noise.

## Lesson

1. **S2v2 per-bin detection works.** F1=0.727 from context alone, surpassing S1's 0.712 from audio. The per-bin format produces genuinely comparable and combinable confidence maps.

2. **Both-agree is gold (83% precision).** When S1 and S2v2 both fire, they're right 83% of the time. S3 should heavily trust these bins.

3. **S2v2 disagreements are more reliable than S1's.** When only S2v2 fires: 38.9% real. When only S1 fires: 33.5% real. Context catches real onsets audio misses, more often than vice versa.

4. **S1's high-confidence solo detections are mostly noise.** At confidence gap >0.3, S1-only detections have only 14.5% onset rate — audio transients that break the rhythmic pattern. S2v2 correctly suppresses these.

5. **Trivial combination gives +4.0pp.** Simple averaging of S1+S2v2 reaches F1=0.752. A learned S3 should do better. This is a floor, not a ceiling.

6. **16.4% of onsets are missed by both.** These are genuinely hard cases — no audio transient AND no rhythmic pattern predicts them. Likely overlaps with the ~14% structurally unsolvable samples from exp 48. Analysis pending on whether these concentrate in specific charts or spread uniformly.
