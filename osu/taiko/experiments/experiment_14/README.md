# Experiment 14 - Corrected Data Alignment

## Hypothesis

All previous experiments (05-13) trained on a dataset with a fundamental timing misalignment: `BIN_MS` was hardcoded to `5.0ms` but the actual mel frame duration is `HOP_LENGTH / SAMPLE_RATE * 1000 = 110 / 22050 * 1000 = 4.98866ms`. This 0.01134ms-per-frame error compounds over song duration:

| Song position | Drift between audio and event label |
|--------------|--------------------------------------|
| 30s | 68ms (13.6 frames) |
| 1 min | 136ms (27.3 frames) |
| 3 min | 408ms (81.8 frames) |
| 5 min | 680ms (136.4 frames) |

By 3 minutes, the model was seeing audio from **408ms before the labeled event**. This likely caused:
- The ~46% accuracy ceiling that no architecture or loss change could break
- Blurry scatter plots / diffuse heatmap diagonals (the model can't learn precise timing from misaligned labels)
- High GOOD-but-not-HIT entropy (the model spreads probability because the "correct" answer doesn't consistently align with audio features)
- Compounding autoregressive drift (each prediction inherits systematic alignment error)

The dataset has been regenerated with the exact `BIN_MS = HOP_LENGTH / SAMPLE_RATE * 1000`. Everything else is identical to experiment 13: same two-path architecture (~21M params), same loss (main + 0.2 audio aux), same AR augmentations (recency-scaled jitter, global shift, 8% insertions/deletions), same NaN guard.

**Expected outcomes:**
- Sharper scatter plots with a tighter diagonal — predictions should cluster close to the identity line instead of a diffuse band
- Miss rate dropping significantly, estimated **10-30%** (from exp 13's ~43%)
- Accuracy ceiling breaking well past 46% — potentially 60%+ since the model can now learn precise audio-event correspondence
- Cleaner heatmaps with less off-diagonal spread
- Better inference on real songs — the systematic drift that plagued all previous inference runs should be gone

This is the most impactful single change in the experiment series. Every model improvement from exp 05-13 was fighting against corrupted ground truth.

## Result

*Pending.*

## Lesson

*Pending.*
