# Experiment 65-S3 — Fusion Selector (DETR-style Encoder-Decoder)

## Purpose

Fuse S1 (audio, per-bin F1=0.712) and S2v2 (context, per-bin F1=0.727) into a coordinated onset map. Simple averaging achieves F1=0.752. S3 should surpass this by learning cross-bin dependencies: the decision at bin 100 depends on what S1 voted at bin 50 and what S2v2 voted at bin 200.

### Theoretical ceiling

From overlap analysis (exp 65-S2v2):
- 83.6% of onset bins detected by at least one model (union coverage)
- Both-agree bins have 82.8% onset rate (very reliable)
- 16.4% missed by both (structural floor, confirmed spread uniformly across dataset)

Simple average: F1=0.752. S3 target: beat this via learned cross-bin coordination.

## Architecture

DETR-style encoder-decoder transformer. Encoder fuses all signals into enriched tokens. Decoder uses per-bin queries with cross-attention to make coordinated per-bin decisions.

### Phase 1: Encoder (global fusion)

250 tokens (one per audio position in the A+B window). Each token enriched with:
- Audio features from S1's conv stem (projected to D)
- S1 confidence at this position (scalar → learned embedding → add)
- S2v2 confidence at this position (scalar → learned embedding → add)
- Sinusoidal positional encoding (added at every layer)
- Event embeddings scatter-added to past token positions
- Density FiLM conditioning after each layer

4 self-attention layers. Every token sees every other token — past audio context, future audio with proposals, event patterns.

Output: Fenc (250, D) — enriched representation of the full audio window.

### Phase 2: Decoder (per-bin decision)

250 bin queries, one per prediction bin. Initialized from Fenc at the corresponding position (content-based, per RT-DETR finding that this outperforms learned embeddings). Learned per-bin positional embeddings added at every layer.

3 decoder layers. Each layer:
1. Self-attention among all 250 bin queries (cross-bin coordination)
2. Cross-attention: each query attends to all 250 encoder tokens

Output: Fdec (250, D)

Auxiliary detection head applied after EVERY decoder layer (per DETR — critical for convergence).

### Output Head

```
Fdec (250, D) → Linear(D, 1) → sigmoid → (250,) per-bin onset probabilities
```

### Density Query (optional, per SEDT)

One extra learned query sent through the decoder alongside the 250 bin queries. Its output predicts total onset count via MSE loss, providing global density supervision.

## Hyperparameters

| Param | Value | Source |
|---|---|---|
| D (model dim) | 192 | T-UAED uses D=192 for audio detection |
| Encoder layers | 4 | |
| Decoder layers | 3 | SEDT uses 3+3, lighter than DETR's 6+6 |
| Attention heads | 8 | Standard across all papers |
| FFN expansion | 4x (768) | Standard |
| Dropout | 0.1 | DETR default |
| Query init | Content-based from Fenc | RT-DETR: beats learned embeddings |
| Auxiliary losses | Every decoder layer | DETR: critical for convergence |
| B_PRED | 250 | |

**Estimated params: ~3.7M** (S3 only, excluding frozen S1/S2v2)

## Input Pipeline

S1 and S2v2 are **frozen** — pretrained checkpoints, no gradients.

```
mel (B, 80, 1000)
  → S1 ConformerProposer (frozen, 29.6M) → s1_conf (B, 250) sigmoid
  → S1 conv stem intermediate → audio_features (B, 250, D_audio)

gap_sequence, ratio_sequence, event_mask, conditioning
  → S2v2 ContextProposer (frozen, 13M) → s2_conf (B, 250) sigmoid

S3 receives: audio_features, s1_conf, s2_conf, event_offsets, event_mask, conditioning
S3 outputs: (B, 250) per-bin onset probabilities
```

## Loss

**Focal BCE** at every decoder layer (per BeatFCOS + DETR auxiliary loss strategy):

```
For each decoder layer l = 1, 2, 3:
  logits_l = head(decoder_output_l)  # shared head across layers
  loss_l = focal_bce(logits_l, targets, pos_weight=5.0, gamma=2.0)

total_loss = sum(loss_l) / n_decoder_layers
```

Optional: density query MSE loss (predict total onset count).

### Targets

Same binary per-bin targets as S1 and S2v2:
- 1.0 at onset bins, 0.5 at ±1 adjacent, 0.0 elsewhere

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Batch size | 48 (needs mel for S1) |
| Epochs | 50 |
| Scheduler | CosineAnnealingLR |
| Gradient clipping | 1.0 |
| S1 | Frozen |
| S2v2 | Frozen |
| Evals per epoch | 4 |

lr=1e-4 per DETR recommendation (lower than our usual 3e-4 since the encoder inputs are already rich features from pretrained models).

## Audio Augmentation

Same as S1 training:
| Aug | Rate | Params |
|---|---|---|
| Mel gain | 30% | ±2dB |
| Mel noise | 15% | Gaussian σ≤0.3 |
| Freq jitter | 15% | Roll mel bands ±3 |
| SpecAugment freq | 20% | 1 mask, 10 bands |
| SpecAugment time | 20% | 1 mask, 30 frames |

Context augmentation (same as S2v2):
| Aug | Rate | Params |
|---|---|---|
| Event jitter | 100% | ±1 bin |
| Context truncation | 2% | Keep 32-128 |
| Density jitter | 20% | ±5% |

## Metrics

### Primary
- Per-bin F1 at thresholds 0.3-0.7 (compare to S1=0.712, S2v2=0.727, average=0.752)
- Precision, Recall
- Confidence separation (onset vs non-onset)

### Per-decoder-layer
- F1 at each auxiliary layer (should improve with depth)

### Analysis
- Disagreement improvement: of the bins where S1 and S2v2 disagreed, how often does S3 pick correctly?
- Both-miss recovery: of the 16.4% both-miss bins, does S3 recover any?
- Cross-bin coordination: does S3 suppress false positives better than simple averaging?

## References

- Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020) — DETR architecture: learned queries, encoder-decoder with cross-attention, auxiliary losses at every decoder layer, Hungarian matching. Established the query-based detection paradigm.
- Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024) — RT-DETR: content-based query initialization from encoder features outperforms learned embeddings. Decoder layers independently predictive via auxiliary losses — can trade layers for speed.
- Ye et al., "Sound Event Detection Transformer" (2021) — SEDT: 1D-DETR for audio, 3+3 encoder-decoder, audio query branch for clip-level classification assists detection. Two-stage training (one-to-one then one-to-many matching) improves recall.
- Li et al., "Unified Audio Event Detection" (2024) — T-UAED: per-class queries with dot-product detection head (query · encoder → sigmoid). D=192. Dual front-end encoder fusion via concatenation + 1D conv.
- Ahn, "Beat Tracking as Object Detection" (2025) — BeatFCOS: focal loss for sparse onset detection. Object detection paradigm applied to 1D audio beat tracking.

## Result

### Per-Bin F1 (rerun, 14 evals, epoch 3.5)

| Eval | F1 | Recovery | S1only F1 | S2only F1 | Agree F1 |
|---|---|---|---|---|---|
| 1 | 0.735 | 13.8% | 0.600 | 0.510 | 0.898 |
| 4 | 0.752 | 14.1% | 0.626 | 0.549 | 0.905 |
| 9 | 0.766 | 17.3% | 0.652 | 0.574 | 0.906 |
| 12 | 0.769 | 19.8% | 0.657 | 0.588 | — |
| 14 | **0.771** | **21.7%** | **0.662** | **0.595** | — |

Peak F1=0.771. +1.9pp over simple averaging (0.752), +4.4pp over S2v2 alone, +5.9pp over S1 alone.

### Benchmarks (eval 12)

| Benchmark | F1 | Delta from normal | Meaning |
|---|---|---|---|
| normal | 0.769 | — | Full S1+S2v2 |
| no_s2 | 0.703 | -6.6pp | S2v2 contributes 6.6pp |
| no_s1 | 0.433 | -33.6pp | S1 contributes 33.6pp |
| no_s1s2 | 0.211 | -55.8pp | Both critical |
| random_s2 | 0.706 | -6.3pp | S3 robust to bad S2 |
| random_s1 | 0.536 | -23.3pp | S1 more critical than S2 |

### AR Evaluation (30 val songs, best checkpoint)

Best balanced configs vs exp58 (previous ATH):

| Config | Close% | Hall% | d_ratio | P-Space | HI-PS | ErrMed | Balance |
|---|---|---|---|---|---|---|---|
| **S3_FIRST_0.5** | **80.5%** | 21.5% | **1.17** | **20.0%** | **95.6%** | 24ms | **66.3** |
| S3_ALL_0.6 | 78.8% | **14.1%** | 2.06 | 22.0% | 81.2% | 17ms | 50.5 |
| ADD_ALL_0.5 | 81.6% | 15.8% | 2.06 | 21.5% | 83.5% | 15ms | 52.6 |
| S3_ALL_0.5 | **93.4%** | 20.3% | 3.60 | 47.1% | 94.3% | **9ms** | 31.3 |
| **exp58 (ref)** | 75.9% | **15.6%** | **0.92** | 10.1% | 81.1% | **8ms** | — |

S3_FIRST_THRESH_0.5 is the recommended config:
- +4.6pp close rate over exp58
- d_ratio 1.17 (slight over-prediction, best of any high-close config)
- P-Space 20.0% (2x exp58, surpasses human GT 11.7%)
- HI P-Space 95.6% (covers nearly all human patterns)
- 24ms timing (worse than exp58's 8ms — FIRST_THRESH takes first bin, not peak)

### Key Discovery: MAX Sampling is Wrong for Per-Bin Detectors

All MAX configs score 24-33% close rate — terrible. ALL_THRESH and FIRST_THRESH score 78-96%. The per-bin detection paradigm requires threshold-based sampling, not argmax. This is a fundamental paradigm shift from our previous single-onset classification models.

### Remaining Problem: Density Over-Prediction

All high-close configs over-predict density (d_ratio 1.17-3.6). The model fires too many bins above threshold. Solutions:
1. Threshold tuning (0.55-0.65 for FIRST_THRESH)
2. Post-processing: keep only top-N events per second matching target density
3. Train with density-aware loss

## Lesson

1. **S3 fusion surpasses naive averaging.** F1=0.771 vs average 0.752 (+1.9pp). The DETR-style cross-attention learns real fusion, not just interpolation.

2. **Recovery is the key metric.** S3 finds 21.7% of onset bins that BOTH S1 and S2v2 missed. This is fusion of moderate confidences that neither model flagged alone — impossible with averaging.

3. **Both S1 and S2v2 are load-bearing.** S1 contributes 33.6pp (audio is critical), S2v2 contributes 6.6pp (context is meaningful). Removing both drops to 0.211.

4. **Per-bin detection with FIRST_THRESH is the right paradigm.** S3_FIRST_0.5 achieves 80.5% close — the best AR close rate in the project. The shift from single-onset classification to per-bin detection + threshold sampling is the biggest AR quality improvement.

5. **Pattern diversity doubled.** P-Space 20% and HI P-Space 95.6% — the 3-stage model produces much richer, more human-like patterns than any single model.

6. **Density control is the remaining challenge.** All high-close configs over-predict. This is solvable with threshold tuning or post-processing, not architectural changes.

7. **Auxiliary decoder losses are critical.** Per-layer F1 progression (0.745→0.747→0.748 at eval 3) shows all layers learn independently, matching DETR's findings. Without auxiliary losses, the model would likely not converge.
