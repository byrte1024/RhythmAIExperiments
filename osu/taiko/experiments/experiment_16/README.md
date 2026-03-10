# Experiment 16 - Rank-Weighted Context Loss

## Hypothesis

Experiment 15 confirmed that standard CE aux loss (0.1) cannot break the context path's rubber-stamping local minimum. After 4 epochs, no_events accuracy never dropped below full accuracy — the context path contributes nothing despite direct gradient pressure.

**Why standard CE fails here:** The context path's optimal lazy strategy is "agree with audio's #1 choice." This is correct ~67% of the time (audio's hit rate). Standard CE treats all wrong answers equally — predicting bin 50 when the target is bin 48 (audio's #2) gets the same gradient as predicting bin 50 when the target is bin 300 (audio's #200). There's no signal saying "audio literally handed you the answer at rank 2 and you ignored it."

**The fix — rank-weighted context loss:** Weight each sample's context CE by how highly audio ranked the true target:

```
weight = clamp(5 / (rank + 4), min=0.1, max=1.0)
```

| Audio rank of target | Weight | Meaning |
|---------------------|--------|---------|
| 1 | 1.0 | Audio nailed it — context must agree |
| 2 | 0.83 | Very available, strong push |
| 3 | 0.71 | |
| 5 | 0.56 | |
| 10 | 0.36 | Still meaningful |
| 20 | 0.21 | Weak signal |
| 50+ | 0.10 | Not context's job, floor gradient |

This directly incentivizes "learn to select from audio's candidates" rather than "learn to predict independently." When audio has the right answer at #2 but context picks #1 (wrong), context gets 8.3x more gradient than when audio has it at #50.

### Why this avoids instability

- **Detached**: weights computed from `audio_logits` under `no_grad()` — no feedback loop through context
- **Bounded**: 10x max/min ratio (1.0 vs 0.1), not 500x
- **Smooth**: `1/(rank+4)` has no cliffs or discontinuities
- **Floor**: every sample gets at least 0.1x gradient — context still learns on hard samples
- **Safe for shared encoders**: mean weight across a batch ~0.3-0.5, same order as the old 0.1 flat weight

### Changes

**Loss:** `main + 0.2 * audio_aux + rank_weighted_context`
- Main loss: unchanged (OnsetLoss on combined logits)
- Audio aux: 0.2 * OnsetLoss (unchanged, not touched)
- Context: per-sample `weight * OnsetLoss(context_logits, target)` where weight = `clamp(5/(rank+4), 0.1, 1.0)`
- No outer multiplier on context — the weighting itself controls magnitude
- Reuses OnsetLoss internals (soft targets, hard_alpha mix, STOP weight) for consistency

Everything else identical to exp 14/15: same architecture (~21M params), same dataset (taiko_v2), same AR augmentations, same 10 ablation benchmarks.

### Expected outcomes

- **Context path activates**: no_events accuracy should drop below full accuracy (5-10pp gap)
- **Top-1 to top-3 gap narrows**: context learns to override audio's #1 when #2/#3 is correct
- **Accuracy breaks past 50%**: context contributes on top of audio's ~50% ceiling
- **Ray patterns reduce**: context uses event spacing to disambiguate harmonic multiples
- **Audio path unaffected**: audio aux still 0.2, no changes to audio gradient

### Risk

- Rank weighting could make context a *better* rubber-stamper (always agreeing with audio more confidently) rather than an independent selector. Watch for: no_events staying equal to full accuracy but with higher confidence.
- If mean context gradient is too high (batch mean weight > 0.5), training could destabilize. Watch for: val_loss oscillating or increasing vs exp 14/15.

## Result

*Pending.*

## Lesson

*Pending.*
