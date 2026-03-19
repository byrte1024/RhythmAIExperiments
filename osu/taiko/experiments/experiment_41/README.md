# Experiment 41 - Deep Entropy Analysis (Diagnostic)

## Hypothesis

The model's entropy rises sharply with target distance. Exp 40 proved this isn't from undertraining — more exposure to distant bins didn't help. The question: **why is entropy higher?**

Two competing explanations:
- **A) More valid onsets in the window** — at target=200, there might be 5 onsets between cursor and target. The model correctly hedges across all of them. Entropy reflects genuine ambiguity.
- **B) The model can't read distant audio** — the cursor bottleneck means distant targets require multi-hop attention, degrading information quality. Entropy reflects model weakness.

If entropy correlates more strongly with `n_onsets_between_cursor_and_target` than with `target_distance` itself, it's explanation A (fundamental ambiguity). If entropy correlates with distance independent of onset count, it's explanation B (model limitation).

### Method

Measure correlations between entropy and:
- Target distance (bin offset)
- Number of future onsets in window
- Number of onsets SKIPPED (between cursor and the PREDICTED position, for overpredictions)
- Context length (past events)
- Density conditioning
- Audio features at target (mel energy, spectral flux)
- Prediction correctness (HIT/MISS)
- Top-1 confidence

Break down entropy by distance bins, by onset count, and by obstacle count.

## Result

*Pending*

## Lesson

*Pending*
