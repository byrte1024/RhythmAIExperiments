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

*Pending*

## Lesson

*Pending*
