# Experiment 66-1 — Pairwise Chart Quality Evaluator

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

Synthetic chart quality metrics (exp 59) achieve only 52% pairwise accuracy — barely above random. A learned evaluator trained on corruption detection + human ratings should do better. The model scores a (chart, audio) window as a scalar. Pairs with known ordering provide supervision via Bradley-Terry loss.

**Key risk:** The model learns song/audio quality instead of chart quality. osu! ratings conflate both — popular songs with good production get higher ratings regardless of chart quality.

**Mitigations:**
1. **16-bin mel bottleneck** — compress 80 mel bands → 16 via learned linear. Kills timbral/production detail, preserves onset/energy structure.
2. **Audio augmentation on rating pairs** — random gain, freq masking, time masking, noise. Makes audio quality unpredictable. Not applied to corruption pairs (audio identical there).
3. **Star rating conditioning** — model evaluates quality *for the intended difficulty*, not absolute.
4. **Corruption pairs dominate training (60%)** — audio identical within corruption pairs, so audio quality signal cancels completely.

## Architecture

```
Input: 10s window (2000 mel frames, variable events)

Mel (80, 2000) → Linear(80, 16) → [augment if rating pair]
  → ConvStem (4x down) → 500 audio tokens (d=256)
  → + star_rating_emb

Events in window → per-event:
  [ratio_before (sin d=256) | ratio_after (sin d=256) | gap_ms (sin d=256)]
  → Linear(768, 256) → GELU → Linear(256, 256)
  → scatter_add_ into audio tokens at temporal positions

→ Transformer (6 layers, 8 heads, d=256)
→ Attention pool (learnable query)
→ Linear(256, 1) → scalar score
```

### Configuration

| Feature | Value |
|---|---|
| Window | 10s (2000 mel frames) |
| Mel bins | 16 (compressed from 80) |
| d_model | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Event features | 3 (ratio_before, ratio_after, gap_ms) |
| Pooling | Attention (learnable query) |
| Star rating | Embedding(20, 256), 0.5* buckets |

### Training data

| Source | Proportion | Construction |
|---|---|---|
| Corruption pairs | 60% | 5 levels (CLEAN > LIGHT > MED > HIGH > GARBAGE), 10 pair types |
| Cross-set rating pairs | 40% | Different beatmapsets, star_rating ±0.5, rating gap ≥1.0 |

### Loss

Bradley-Terry with adaptive margin: `-log(σ(s_better - s_worse - α * level_gap))`

- Corruption pairs: level_gap ∈ {1, 2, 3, 4}
- Rating pairs: level_gap = |rating_a - rating_b|
- α ≈ 0.1 (tuned)

### Corruption levels

| Level | Jitter (per-event) | Jitter (all-event) | Insert center | Delete | Insert offset | Special |
|---|---|---|---|---|---|---|
| CLEAN | — | — | — | — | — | — |
| LIGHT | ±10ms | ±10ms | 1% | 1% | 1% | — |
| MED | ±30ms | ±30ms | 5% | 5% | 5% | — |
| HIGH | ±100ms | ±250ms | 25% | 15% | 10% | — |
| GARBAGE | — | — | — | — | — | Fully random gaps sampled from global distribution |

Post-corruption: sort events, merge within 2 bins, remove negative gaps, clamp to [0, mel_frames-1].

### Training recipe

| Phase | Epochs | Data | LR |
|---|---|---|---|
| Phase 1: corruption pretraining | 20 | 100% corruption pairs | 3e-4 |
| Phase 2: rating fine-tune | 10-15 | 60% corruption + 40% rating | 3e-5 |

## Result

*(experiment not yet run)*

## Success criteria

- Corruption pairs: >90% accuracy (easy, margin 3-4), >70% (hard, margin 1)
- Rating pairs: >60% pairwise accuracy (baseline: 50% random, 52% exp 59 formula)
- Generated charts score lower than real charts on average
- Scores correlate with human preference from exp 53-AR

## Data notes

- `rating`: 1-10 user vote average, **per-beatmapset** (all diffs share same rating)
- Distribution: median 9.29, p25 8.88, p75 9.62, heavily top-skewed
- 2,489 rated beatmapsets, 10,027 charts with ratings, 21 missing
- Within-set playcount varies but correlates with difficulty (spearman 0.63) — not usable as per-diff quality signal
- Global gap distribution: 6.9M gaps, 77.5% under 250ms, peaks at 80ms and 160ms
