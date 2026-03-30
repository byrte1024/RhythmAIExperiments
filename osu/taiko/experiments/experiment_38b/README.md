# Experiment 38-B - Framewise with Weighted BCE

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp [38](../experiment_38/README.md) with dice+BCE failed — the dice smooth term creates a degenerate minimum at all-zero predictions (0% recall, 0 preds/window).

**Weighted BCE** directly penalizes missing each positive token. With ~13% positive tokens, a 7x positive weight balances the classes. Unlike dice, BCE has no degenerate minimum — predicting 0 for a positive token always incurs high loss.

### Changes from exp [38](../experiment_38/README.md)

- **Loss**: dice+BCE → weighted BCE only (pos_weight=7.0 for onset tokens, 1.0 for non-onset)
- **Teacher forcing removed** — exp [38](../experiment_38/README.md)'s train HIT was 44% but val HIT was 0% because the model just echoed the teacher signal. No onset feedback during training; model must learn purely from audio + past ramps.
- Everything else identical (framewise model, causal mask, mel ramps)

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_38b --model-type framewise --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Model learns onset positions (24% recall) but overpredicts (46.7 preds/win, 65% hallucination).** Also fixed critical causal mask bug — past tokens were seeing future audio.

| Metric | Value |
|--------|-------|
| Event recall | 24.1% |
| Pred precision | 8.4% |
| F1 | 0.124 |
| Preds/window | 46.7 (real: 16.2) |
| Hallucination | 65.2% |
| Nearest HIT | 5.0% |

**Causal mask bug found and fixed:** Past tokens (0-124) could attend to future tokens (125-249), leaking future audio into past representations. Fixed by blocking past→future attention. This explained exp [38](../experiment_38/README.md)'s teacher forcing cheat — past tokens could see future onsets through the attention leak.

**pos_weight=7 causes overprediction.** The model finds onsets (24% recall, well above random 0.8%) but fires 3x too many predictions. The positive weighting pushes it to activate everywhere to avoid missing weighted positives.

## Lesson

- **The framewise architecture works** — 24% event recall with correct causal masking proves the model can learn onset positions from audio.
- **pos_weight=7 too aggressive** — same pattern as exp [37](../experiment_37/README.md)/[37-B](../experiment_37b/README.md). With 13% natural positive ratio, no upweighting needed.
- **Causal mask must block past→future** — critical for preventing information leakage. Past tokens should only see past.
