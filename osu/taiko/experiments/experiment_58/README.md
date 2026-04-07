# Experiment 58 - Two-Stage Propose-Select Architecture

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 57](../experiment_57/README.md) revealed that future audio is the dominant signal (NA_B = 1.3% accuracy without it) and event context adds only ~3-4pp. The model predicts almost entirely from audio, with context as a minor correction.

**Idea: separate the two signals into two stages.**

- **Stage 1 (Proposer)**: Pure audio. Small transformer, no events, no density. Each token outputs sigmoid: "audio supports onset here." Trained recall-focused (don't miss anything). This explicitly learns audio onset detection.

- **Stage 2 (Selector)**: Full context. Standard EventEmbeddingDetector but audio tokens are enriched with Stage 1's proposal scores as additional embeddings. Event context + density + proposals → cursor → 251-class softmax. This learns chart-level selection from audio-supported candidates.

Stage 1 proposals are embedded INTO the audio tokens so Stage 2 sees "this position has onset confidence X%" alongside the raw audio features. Stage 2 can never blame audio for missing an onset — Stage 1 already flagged all audio-supported positions.

### Key design decisions

- **Stage 1 targets ALL onsets in the window**: Not just the next onset — every onset in the full A+B audio window (past events + all future onsets). Stage 1 learns pure audio onset detection across the entire visible range.
- **Stage 1 is recall-focused**: Focal loss with high pos_weight BCE. Missing a real onset is worse than false positives. Stage 1 says "here are all the options."
- **Stage 2 is precision-focused**: Standard OnsetLoss. Picks the right onset from Stage 1's proposals.
- **Stage 2 frozen initially**: Stage 2 weights frozen for first N evals while Stage 1 learns to produce useful proposals. Then unfreeze for joint training.
- **Proposal embedding**: Each audio token gets an additive embedding based on its Stage 1 confidence score.

### Architecture

```
Conv stem: mel (B, 80, 1000) → 250 audio tokens (B, 250, 384)

Stage 1 (Proposer):
  2-4 transformer layers (audio tokens only, no events, no density)
  Per-token: Linear(384, 1) → sigmoid → onset confidence
  Loss: Focal BCE (recall-focused)

Proposal embedding:
  confidence (B, 250, 1) → Linear(1, 384) → added to audio tokens

Stage 2 (Selector):
  Event embeddings scatter-added to enriched audio tokens
  FiLM density conditioning
  8 transformer layers
  Cursor → 251-class softmax
  Loss: OnsetLoss (standard)
```

### Configuration

| Feature | Value |
|---|---|
| A_BINS / B_BINS | 500 / 500 |
| B_PRED | 250 (N_CLASSES=251) |
| Gap ratios | ON |
| Density jitter | ±10% at 30% (loose) |
| Stage 1 layers | 4 |
| Stage 1 loss | Focal BCE (gamma=2, pos_weight=5) |
| Stage 2 freeze | First 2 evals |

### Stage 1 metrics to track

| Metric | Description |
|---|---|
| s1_recall | % of GT onsets with a proposal above threshold |
| s1_precision | % of proposals that match a GT onset |
| s1_f1 | Harmonic mean of precision/recall |
| s1_avg_proposals | Mean number of proposals per sample |
| s1_loss | Focal BCE loss |
| s1_onset_conf | Mean confidence at GT onset positions |
| s1_non_onset_conf | Mean confidence at non-onset positions |

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_58 --model-type event_embed_propose --a-bins 500 --b-bins 500 --b-pred 250 --gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

Stopped at eval 12 (epoch 3.0). **Peak at eval 9 (epoch 3.2). New all-time high: 74.6% HIT.**

### Peak (eval 9) metrics:

| Metric | Value |
|---|---|
| HIT% | **74.6%** (ATH, +0.9pp over previous) |
| Accuracy | 55.7% |
| Val loss | 2.427 |

### Stage 1 at peak:

| Metric | Value |
|---|---|
| S1 F1 | 0.502 |
| S1 Precision | 0.377 |
| S1 Recall | 0.740 |
| Avg proposals | 67 / 250 tokens |
| Confidence separation | 0.173 |

### Stage 2 interaction at peak:

| Metric | Value |
|---|---|
| S2 picks S1 | 82% |
| S2 agree accuracy | 57.4% |
| S2 override accuracy | 48.4% |
| S2 pick rank | 31.3 |
| S1 naive accuracy | 2.0% |

### Proposal benchmarks (eval 9):

| Benchmark | Accuracy | Delta from normal |
|---|---|---|
| normal | 57.1% | — |
| proposal_zero | 7.2% | **-49.9pp** |
| proposal_random | 19.0% | **-38.1pp** |

Stage 1 proposals are massively load-bearing. Zeroing them kills accuracy by 50pp.

### Progression:

| Eval | Epoch | HIT% | Delta | S1 F1 | S2 agree | S2 override |
|------|-------|------|-------|-------|----------|-------------|
| 3 | 1.7 | 69.6 | — | 0.484 | 53.4% | 42.3% |
| 5 | 2.2 | 72.6 | +0.8 | 0.508 | 55.7% | 47.4% |
| 7 | 2.7 | 73.6 | +0.7 | 0.478 | 56.5% | 48.9% |
| **9** | **3.2** | **74.6** | **+0.4** | **0.502** | **57.4%** | **48.4%** |
| 10 | 3.5 | 74.1 | -0.5 | 0.487 | 56.6% | 49.6% |
| 12 | 3.0 | 74.1 | +0.1 | 0.501 | 56.5% | 50.5% |

**7 consecutive improvements** (evals 3-9) before plateauing. Zero oscillations during climb — uniquely smooth compared to all prior experiments (which oscillate 67-80% of steps).

### vs all-time bests:

| Metric | Exp 58 | Exp 44 | Exp 55 | Exp 53-B |
|--------|--------|--------|--------|----------|
| HIT% | **74.6%** | 73.7% | 73.6% | 73.4% |
| Acc% | **55.7%** | 54.8% | 54.2% | 54.2% |
| Val loss | **2.427** | 2.480 | 2.463 | 2.479 |

## Lesson

The propose-select architecture **breaks the 73.7% HIT ceiling** that held across experiments 44-57. Key findings:

1. **Two-stage decoupling works.** Separating audio onset detection (Stage 1) from chart-level selection (Stage 2) produces smoother optimization and higher accuracy.

2. **Stage 1 proposals are essential.** Zeroing proposals drops accuracy by 50pp (57→7%). Stage 2 is not learning independently — it's genuinely selecting from S1's candidates.

3. **Remarkably linear convergence.** 7 consecutive improvements with zero oscillations — unique among all experiments. The two-stage architecture has a smoother loss landscape.

4. **S1 quality is the bottleneck.** F1=0.50 means S2 sifts through ~35 false positives per 30 real onsets per sample. Higher S1 precision would give S2 a cleaner selection problem.

5. **S2 learns genuine override ability.** Override accuracy reached 50.5% — when S2 disagrees with S1, it's right half the time. This is real selection, not rubber-stamping.

6. **S1 degrades under joint training.** F1 dropped from 0.56 (frozen) to 0.48 after unfreeze, then recovered to ~0.50. Joint gradients through the proposal embedding muddy S1's discrimination.

[Exp 58-B](../experiment_58b/README.md) tests lower recall weighting for S1 to improve precision.
