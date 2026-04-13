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

*Pending*

## Lesson

*Pending*
