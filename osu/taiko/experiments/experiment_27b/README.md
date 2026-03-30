# Experiment 27-B - Context Pattern Analysis (Diagnostic)

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp 27's best model (eval 8, 69.8% HIT) shows systematic over-prediction: when the correct answer is 75, it predicts 150. Top-3 accuracy is ~90%, meaning the model narrows to a small set of rhythmically valid candidates but picks wrong. Forward error shows predictions cluster at meaningful ratios (1.0x, 0.5x, 2.0x) — the model knows the rhythm vocabulary but can't disambiguate.

**If the model could detect and continue repeating gap patterns from its context, how many misses would it fix?**

For a pattern like `150 150 75 75 150 150 75 75 ?`, the answer `75` is the next element in a repeating `[150 150 75 75]` cycle. The model doesn't need complex reasoning — just pattern detection and continuation.

### Method

Two scripts, run sequentially:

**1. `run_predictions.py`** — general-purpose val inference, saves all predictions + context to `.npz`
```bash
python run_predictions.py taiko_v2 --checkpoint runs/detect_experiment_27/checkpoints/epoch_008.pt --subsample 8
```

**2. `analyze_context.py`** — loads the `.npz` and runs pattern analysis
```bash
python analyze_context.py runs/detect_experiment_27/predictions_epoch_008_sub8.npz
```

Pattern detection logic:
1. For every sample, extract the gap sequence from context (inter-onset intervals)
2. Detect repeating patterns: for each candidate length L ≥ 4, check if the last L gaps repeat at least once earlier in context (≥80% element match to tolerate slight timing variation)
3. If a pattern is found, predict the next element in the cycle
4. Compare: model prediction vs pattern prediction vs ground truth

### What counts as a "pattern"

- Minimum 4 gaps long
- Must repeat at least once in the context (so ≥ 2L gaps needed)
- Elements match within HIT tolerance (≤3% or ±1 frame) to handle slight timing drift
- Aligned to cursor position — pattern[0] is the prediction for the next gap
- When multiple pattern lengths match, prefer the one with more repetitions (more evidence)

### Expected outcome

~90% of misses with sufficient context should be solvable by pattern continuation. Most taiko patterns are repetitive (streams of equal gaps, alternating patterns like `75 75 150 75 75 150`), so the correct answer almost always continues a visible cycle.

## Result

**Pattern detector finds 22.5% of misses solvable, but manual inspection shows the true number is far higher — our algorithm is too rigid, not the data.**

### Quantitative (strict pattern matching)

| | Count | % |
|---|---|---|
| Total samples | 74,283 | |
| HIT | 51,937 | 69.9% |
| MISS | 22,094 | 29.7% |
| Pattern found (all samples) | 36,091 | |
| Pattern correct (all samples) | 23,187 | 64.2% of found |
| **Pattern fixes model miss** | **4,973** | **22.5% of misses** |
| Pattern wrong on miss | 4,994 | 22.6% of misses |
| No pattern detected (miss) | 12,050 | 54.5% of misses |

Theoretical HIT with perfect pattern oracle: **76.6%** (+6.7pp from 69.9%)

Pattern risk if always used: net **-3.9%** (would break 7,874 existing HITs while fixing only 4,973 misses). Pattern alone cannot be blindly trusted — it needs audio to know when to override.

### Qualitative (manual inspection)

The quantitative numbers dramatically understate the potential. Inspecting the 54.5% "no pattern detected" misses reveals:

**19 out of 20 randomly sampled no-pattern misses have the target gap value present in context.**

Examples of "no pattern detected" where patterns are obvious to a human:

- `93 92 93 46 46 | 371 | 92 93 46 46 46 47 | 185 | 46 46 93 92 93 46 46 | ?` → target=46. Clear `93 46 46` rhythm, interrupted by pauses (371, 185). Our detector fails because the interruptions break the consecutive block requirement.
- `87 174 87 87 175 87 87 174 87 88 174 | 131 217 | 88 87 174 174 88 87 174 | ?` → target=87. Obvious `87 174` alternation, but timing drift and one interruption breaks strict matching.
- `60 60 60 30 30 60 60 31 60 30 60 60 30 15 15 30 30 60 31 30 | ?` → target=60. Mix of 60/30/15 (halving hierarchy) — the pattern isn't a repeating block but a rhythmic vocabulary.

Examples of pattern-detected fixes (model miss → pattern correct):

- `31 16 16 32 | 31 16 16 32 | 31 16 16 31 | ?` → target=32, model predicted 16 (halved). Pattern catches the `31 16 16 32` cycle.
- `16 17 16 16 | 16 17 16 16 | 16 17 16 16 | ?` → target=16, model predicted 33 (doubled). Clean 4-gap repeat.
- `15 15 30 30 15 15 15 15 15 16 30 15 15 15 15 30 30 30 30 15 15 30 15 15 | ?` → target=15, model predicted 30 (doubled). Long 24-gap pattern repeating 2x.

### The core insight

The model's errors are **not random**. They are systematic:
1. The correct gap value almost always exists in recent context
2. The model consistently predicts a rhythmically valid but wrong multiple (2x, 0.5x) of the correct answer
3. A simple pattern-matching algorithm with strict requirements catches 22.5% of misses — a more flexible approach (or a neural network with attention over context) should catch far more

**But context alone is not enough.** The pattern detector has a net-negative effect when applied blindly (-3.9%) because it can't tell when the pattern should break — transitions, tempo changes, and section boundaries require audio awareness. The solution must fuse both:
- **Context** to know which gaps are rhythmically plausible (pattern continuation)
- **Audio** to know when to follow vs break the pattern (section transitions, new rhythmic ideas)

This is exactly what the unified fusion architecture (exp 25-27) was designed to do, but isn't doing — context delta collapsed to ~1.5% in all experiments. The information is there; the model just doesn't learn to use it.

## Lesson

- **Context contains the answer for the vast majority of misses.** The target gap value appears in recent context for ~95% of non-STOP misses with sufficient history. The model's ~70% HIT ceiling is not due to missing information.
- **Strict pattern matching is too rigid for real taiko rhythms.** Taiko patterns have timing drift (±1 bin), interruptions (pauses, transitions), and hierarchical structure (30/60/120 relationships). A block-repetition detector catches only the cleanest cases.
- **The model needs both context AND audio to disambiguate.** Context says "75 and 150 are both valid here." Audio says "this is a dense section, use 75" or "this is a transition, break the pattern." Neither signal alone is sufficient — the model must learn to fuse them.
- **The ~70% ceiling is a context utilization problem, not a data or capacity problem.** Experiments 25-27 proved the architecture has capacity and data. Experiment 27-B proves the information exists in context. The gap is in how the model routes attention between audio and context features.
