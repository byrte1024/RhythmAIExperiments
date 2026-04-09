# Experiment 64 - Delta-Encoded Multi-Onset Prediction

## Purpose

Exp 62 showed multi-onset prediction improves pattern diversity (P-Space 12.0% vs exp58's 10.1%) but had two problems:
1. **Duplicate predictions**: Multiple onsets (o1, o2, o3) frequently predicted the exact same absolute position from cursor, causing most of the 30% ordering violations.
2. **Difficulty scaling**: o1=74.9% HIT but o4=43.0% — predicting absolute offset 80 when o1 already told you "20" is redundant work.

**Solution**: Encode targets as **deltas from the previous onset** instead of absolute offsets from cursor. Each onset predicts the gap to the next event.

## Delta Encoding

### Example: cursor=0, next 4 onsets at bins [50, 100, 150, 200]

**Current (exp62)**: targets = [50, 100, 150, 200] — absolute offsets from cursor.
Model must independently discover 50, 100, 150, 200. A simple metronome at gap=50 looks like [50, 100, 150, 200] — increasingly hard.

**New (exp64)**: targets = [50, 50, 50, 50] — deltas from previous onset.
o1=50 (gap from cursor), o2=50 (gap from o1), o3=50 (gap from o2), o4=50 (gap from o3). Metronome = trivially [50, 50, 50, 50].

### Worked examples (B_PRED=250)

```
cursor, [onsets]           → deltas              → targets (STOP cascade)

0, [50, 100, 150, 200]    → [50, 50, 50, 50]    → [50, 50, 50, 50]        ✓ all in range
0, [1, 2, 3, 4]           → [1, 1, 1, 1]        → [1, 1, 1, 1]           ✓ fast stream
0, [100, 1000, 1200, 1400]→ [100, 900, 200, 200] → [100, STOP, STOP, STOP] 900 >= 250
1300, [1500, 1700, 1750, 1800] → [200, 200, 50, 50] → [200, 200, 50, 50]  ✓ all in range
500, [600, 700, 1700, 1833]→ [100, 100, 1000, 133]→ [100, 100, STOP, STOP] 1000 >= 250
100, [500, 600, 700, 800]  → [400, 100, 100, 100] → [STOP, STOP, STOP, STOP] 400 >= 250
```

**Rule**: Scan deltas left to right. If `delta[i] >= B_PRED`, set it and all following to STOP.

### Benefits

1. **Simpler patterns**: Metronome = (20, 20, 20, 20) not (20, 40, 60, 80). Model learns gaps directly.
2. **No ordering violations**: Any positive delta guarantees temporal order. strict_increasing ≈ 100% by construction.
3. **No duplicate positions**: delta=0 is degenerate (two notes at same time), extremely rare in real data.
4. **Natural STOP**: "No more events" = predict STOP. "Next event is too far" = delta >= B_PRED → STOP at training time.
5. **Consistent difficulty**: Each onset predicts a gap in the same [0, B_PRED) range, regardless of position. o4 isn't harder than o1.

## Audio Window Expansion

### The out-of-window problem

With deltas, worst case: all 4 onsets predict delta=249 (max). Total offset from cursor = 4 × 249 = 996. But current B_BINS=500 means the model only sees 500 frames of future audio. Events at frame 996 would have no audio coverage.

### Solution: B_BINS = (n_onsets + 1) × B_PRED

With n_onsets=4, B_PRED=250:
- B_BINS = 5 × 250 = **1250** frames (6.25s of future audio)
- Any combination of deltas lands within visible audio
- The +1 gives one B_PRED buffer past the last possible onset, helping STOP decisions
- Total audio window: A=500 + B=1250 = 1750 frames (8.7s) vs current 1000 (5s)

The proposer already attends to the full audio window, so it naturally covers the expanded range.

### Memory impact

1750 vs 1000 audio tokens after conv stem (÷4): 437 vs 250 tokens. ~75% more transformer tokens. May need batch size reduction (48 → 32-36).

## Configuration

Same as [exp 62](../experiment_62/README.md) plus:

| Feature | Value |
|---|---|
| delta_onsets | ON (new) |
| n_onsets | 4 |
| B_PRED | 250 (N_CLASSES=251) |
| B_BINS | auto: (4+1) × 250 = 1252 (aligned) |
| A_BINS | 500 |

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_64 --model-type event_embed_propose --a-bins 500 --b-pred 250 --gap-ratios --density-jitter-rate 0.30 --density-jitter-pct 0.10 --n-onsets 4 --delta-onsets --proposer-freeze-evals 2 --epochs 50 --batch-size 36 --subsample 1 --evals-per-epoch 4 --workers 3
```

Note: `--b-bins` is omitted — auto-computed from `--delta-onsets` as `(n_onsets + 1) * b_pred`, rounded to align with conv stride. Training from scratch (no warm-start) because the prediction landscape (deltas vs absolute) and audio window size (1752 vs 1000 tokens) are fundamentally different.

## Known Concerns

### 1. o1 cursor jitter

o1 still encodes as distance from cursor, not from the previous event. Because the cursor can land between events (STOP hop), a real pattern at bins (20, 40, 60, 80) with cursor at 11 looks like targets (9, 20, 20, 20) — o1 is noisy while o2-o4 are clean. This is the same situation exp58/62 already handle (o1 has always been "distance from cursor"), and o2-o4 become strictly easier, so net it's a win. But if o1 degrades, this is the likely cause.

### 2. Cascade blindness

At inference, o2 doesn't know what o1 predicted — it only sees audio and context. During training, o2's target is computed from the real o1 position, so it learns correct deltas. At inference, if o1 is slightly off, o2's delta is still approximately right (the gap pattern is preserved). This is actually more robust than absolute mode, where o2 must independently rediscover the correct absolute position even though o1 already provided the information. But if o1 is very wrong, o2 compounds the error. Monitor per-onset accuracy to check.

### 3. Far delta under-training

Most gaps in the dataset are small (dense charts). Large deltas (150-249) are rare, and with the STOP cascade rule, o2/o3/o4 targets are even more concentrated at small values (since if the previous delta was large enough to STOP, we never get to train o2). This means later onsets may be under-trained on large gaps. Our current balanced sampler weights by o1 distance — it may need adjustment to also boost samples where o2/o3/o4 have large deltas. Monitor per-onset STOP rates and bin histograms; if later onsets collapse to small predictions, consider reweighting.

## Result

*Pending*

## Lesson

*Pending*
