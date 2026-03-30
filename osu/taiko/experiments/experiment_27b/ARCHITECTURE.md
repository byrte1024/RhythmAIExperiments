# Experiment 27-B — Full Architecture Specification

## Task

Diagnostic analysis: determine what fraction of exp 27's misses could be fixed by a simple pattern continuation algorithm applied to the event context (gap history). This is a purely analytical experiment — no model training, no architecture changes.

## Method

Two-stage offline analysis on exp 27's best model (eval 8, 69.8% HIT):

### Stage 1: Prediction Gathering (`run_predictions.py`)

```bash
python run_predictions.py taiko_v2 \
  --checkpoint runs/detect_experiment_27/checkpoints/epoch_008.pt \
  --subsample 8
```

Runs the exp 27 OnsetDetector in teacher-forced evaluation mode on 1/8 of validation data (74,283 samples). For each sample, saves:
- Model prediction (argmax of 501 logits)
- Ground truth target
- Full event context (past event offsets)
- Whether the prediction was a HIT or MISS

Output: `.npz` file with all predictions and context.

### Stage 2: Pattern Analysis (`analyze_context.py`)

```bash
python analyze_context.py runs/detect_experiment_27/predictions_epoch_008_sub8.npz
```

For each sample, computes gap sequence from context (inter-onset intervals), then applies pattern detection:

#### Pattern Detection Algorithm

1. Extract gap sequence from past events: `gaps[i] = event_offset[i+1] - event_offset[i]`
2. For each candidate pattern length L (starting from L=4 up to half the context length):
   a. Take the last L gaps as the candidate pattern
   b. Search earlier in the context for a matching block of L gaps
   c. Match criterion: each element within HIT tolerance (<=3% ratio or ±1 frame)
   d. Require at least one complete earlier repetition (so >=2L gaps needed)
3. If pattern found, predict: pattern[cursor_position_mod_L] = next gap
4. When multiple pattern lengths match, prefer the one with more repetitions

#### Match Criteria

- Minimum pattern length: 4 gaps
- Minimum repetitions: 2 (pattern appears at least twice in context)
- Element match tolerance: <=3% ratio OR ±1 frame (same as HIT criterion)
- Alignment: pattern[0] corresponds to the prediction for the next gap

### What Is Measured

For each sample, classify into one of:
- **HIT**: model already correct (no need for pattern)
- **MISS + pattern fixes**: model wrong, pattern prediction correct
- **MISS + pattern wrong**: model wrong, pattern prediction also wrong
- **MISS + no pattern**: model wrong, no repeating pattern detected

Theoretical oracle: if pattern prediction replaces model prediction whenever model is MISS and pattern is available and correct.

## Input Data

The analysis operates on exp 27's OnsetDetector model output. The model architecture is documented in experiment_27/ARCHITECTURE.md:
- OnsetDetector with unified fusion (~19M params)
- Trained on full dataset (subsample=1) with heavy augmentation
- Best checkpoint: eval 8 (epoch 2.0), 69.8% HIT

## Results

### Quantitative (strict pattern matching)

| Category | Count | % of total | % of misses |
|---|---|---|---|
| Total samples | 74,283 | 100% | |
| HIT | 51,937 | 69.9% | |
| MISS | 22,094 | 29.7% | 100% |
| Pattern found (all samples) | 36,091 | | |
| Pattern correct (all samples) | 23,187 | | |
| Pattern fixes model miss | 4,973 | | 22.5% |
| Pattern wrong on miss | 4,994 | | 22.6% |
| No pattern detected (miss) | 12,050 | | 54.5% |

Theoretical HIT with perfect pattern oracle: **76.6%** (+6.7pp from 69.9%).

Pattern risk if always applied blindly: net **-3.9%** (breaks 7,874 existing HITs while fixing only 4,973 misses).

### Qualitative (manual inspection of "no pattern detected" misses)

19 out of 20 randomly sampled no-pattern misses had the target gap value present in recent context. The strict block-repetition algorithm misses patterns with:
- Timing drift (±1 bin between repetitions)
- Interruptions (pauses, transitions)
- Hierarchical structure (30/60/120 relationships)
- Non-contiguous repeats

## Key Finding

The target gap value appears in recent context for ~95% of non-STOP misses with sufficient history. The model's ~70% HIT ceiling is not due to missing information — context contains the answer. The strict pattern matcher catches only the cleanest 22.5% of solvable cases. A more flexible approach (or a neural network with attention over context) should catch far more, but context alone applied blindly is net-negative (-3.9%). Both context AND audio are needed: context to know which gaps are rhythmically plausible, audio to know when to follow vs break the pattern.
