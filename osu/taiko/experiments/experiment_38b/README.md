# Experiment 38-B - Framewise with Weighted BCE

## Hypothesis

Exp 38 with dice+BCE failed — the dice smooth term creates a degenerate minimum at all-zero predictions (0% recall, 0 preds/window).

**Weighted BCE** directly penalizes missing each positive token. With ~13% positive tokens, a 7x positive weight balances the classes. Unlike dice, BCE has no degenerate minimum — predicting 0 for a positive token always incurs high loss.

### Changes from exp 38

- **Loss**: dice+BCE → weighted BCE only (pos_weight=7.0 for onset tokens, 1.0 for non-onset)
- **Teacher forcing removed** — exp 38's train HIT was 44% but val HIT was 0% because the model just echoed the teacher signal. No onset feedback during training; model must learn purely from audio + past ramps.
- Everything else identical (framewise model, causal mask, mel ramps)

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_38b --model-type framewise --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

*Pending*

## Lesson

*Pending*
