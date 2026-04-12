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

### Phase 1: Corruption pretraining (8 evals, 4 epochs, stopped early)

Plateaued by eval 3-4. Corruption detection learned fast — 98% accuracy within half an epoch.

| Metric | Eval 1 | Eval 8 (final) |
|---|---|---|
| Pair accuracy | 97.6% | 98.4% |
| Val loss | 0.048 | 0.026 |
| Margin 1 accuracy | 96% | 98% |
| Margin 4 accuracy | 98% | 99% |
| Monotonic | Yes | Yes (all 8 evals) |

Score means at eval 8: CLEAN=+22.9, LIGHT=+11.9, MED=-2.4, HIGH=-11.0, GARBAGE=-19.7

Per-pair breakdown stable at: CLEAN vs GARBAGE = 100%, LIGHT vs MED = 96%, MED vs HIGH = 90-96% (hardest pair).

### Phase 2: Rating fine-tune (8 evals, 4 epochs)

Corruption performance held. No catastrophic forgetting.

| Metric | Eval 1 | Eval 8 (final) |
|---|---|---|
| Pair accuracy | 98.6% | 98.8% |
| Val loss | 0.019 | 0.016 |
| Monotonic | Yes | Yes (all 8 evals) |

### Rating correlation (eval on full dataset)

Scored all 10,027 rated charts (8 windows each). Compared model scores vs osu! user ratings.

| Metric | Value | Baseline |
|---|---|---|
| Spearman (all charts) | **+0.091** | 0.0 (random) |
| Spearman (per-beatmapset) | **+0.107** | 0.0 |
| Pairwise accuracy (rating pairs) | **55.9%** (735/1315) | 50% random, 52% exp 59 formula |

### Verdict

**Corruption detection: excellent.** 98.8% pairwise accuracy, perfect monotonicity, massive score separation (50+ point gap CLEAN to GARBAGE). The model reliably distinguishes structural chart quality.

**Human rating prediction: marginal.** Spearman 0.09 is statistically significant (p=1e-19) but very weak. 55.9% pairwise accuracy beats the exp 59 synthetic formula (52%) but falls short of the 60% target. The model is a strong corruption detector but a weak human preference predictor.

### Why it didn't work better

1. **Rating is per-beatmapset, not per-chart.** All diffs in a set share the same rating. Only 2,489 independent data points, not 10,027.
2. **Ratings are extremely compressed.** Median 9.29, IQR of 0.74 points (8.88-9.62). The model must learn very fine distinctions in a narrow band.
3. **Song quality confound.** Despite 16-bin mel bottleneck + audio augmentation, the rating signal is dominated by song popularity/quality, not chart quality.
4. **Corruption ≠ quality.** The model learned "structurally broken charts are bad" but this doesn't capture what humans actually care about (pattern creativity, musical interpretation, flow).

## Success criteria (reviewed)

| Criterion | Target | Achieved | Status |
|---|---|---|---|
| Corruption easy pairs (margin 3-4) | >90% | 98-100% | **Pass** |
| Corruption hard pairs (margin 1) | >70% | 90-98% | **Pass** |
| Rating pairwise accuracy | >60% | 55.9% | **Fail** |
| Score correlates with human pref | Spearman >0.2 | 0.091 | **Fail** |

## Data notes

- `rating`: 1-10 user vote average, **per-beatmapset** (all diffs share same rating)
- Distribution: median 9.29, p25 8.88, p75 9.62, heavily top-skewed
- 2,489 rated beatmapsets, 10,027 charts with ratings, 21 missing
- Within-set playcount varies but correlates with difficulty (spearman 0.63) — not usable as per-diff quality signal
- Global gap distribution: 6.9M gaps, 77.5% under 250ms, peaks at 80ms and 160ms
