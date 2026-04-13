# Experiment 66-2 — Full Architecture Specification

## Task

Same as 66-1: score chart quality as a scalar via pairwise comparison. Key change: bidirectional corruption (random AND metronomic) to eliminate the regularity bias found in 66-1b.

## Model

Identical to 66-1. See `classifier_model.py` — ChartQualityEvaluator, ~5.2M params.

## Corruption recipes

Order of operations for all corruptions: apply corruption → sort events → merge within 2 bins → remove negatives → clamp to [0, mel_frames-1].

### CLEAN
No modification.

### Random corruption (same as 66-1)

#### LIGHT_RAND
```
per_event_jitter:  uniform(-2, +2) bins          # ±10ms
all_event_jitter:  uniform(-2, +2) bins           # ±10ms
insert_center:     1% per gap
delete:            1% per event
insert_offset:     1% per event, gap from global distribution
```

#### MED_RAND
```
per_event_jitter:  uniform(-6, +6) bins           # ±30ms
all_event_jitter:  uniform(-6, +6) bins           # ±30ms
insert_center:     5% per gap
delete:            5% per event
insert_offset:     5% per event
```

#### HIGH_RAND
```
per_event_jitter:  uniform(-20, +20) bins         # ±100ms
all_event_jitter:  uniform(-50, +50) bins          # ±250ms
insert_center:     25% per gap
delete:            15% per event
insert_offset:     10% per event
```

#### PURE_RAND
```
Fully random gap sequence sampled from global gap distribution.
Same event count as original. Cumulative placement from position 0.
```

### Metronomic corruption (NEW)

#### LIGHT_METRO
Subtle — barely noticeable.
```
grid_snap:
  - For 10% of gaps:
    local_median = median of surrounding 16 gaps
    grid = [local_median * r for r in (0.25, 0.5, 1.0, 2.0, 4.0)]
    gap = grid[argmin(|gap - grid|)]

const_gap_fill:
  - For each 5-second segment (non-overlapping):
    5% chance to replace all events with evenly spaced events
    at the segment's original event count

ratio_snap:
  - For 10% of consecutive gap pairs where ratio is not in {0.25, 0.5, 1.0, 2.0, 4.0} (±5%):
    snap gap_cur = gap_prev * nearest_allowed_ratio
```

#### MED_METRO
Clearly repetitive.
```
grid_snap:
  - 30% of gaps snapped to local grid

const_gap_fill:
  - 15% of 5s segments replaced with constant-gap fill

pattern_loop:
  - Select 10% of chart duration (random contiguous regions)
  - In each region: take first 4-8 gaps, tile to fill the region
  - Events repositioned by cumulative gap from region start

density_flatten:
  - For each 5s window: compute local event count
  - Pull 30% toward global mean:
    target = current + 0.30 * (global_mean - current)
    if current > target: delete random (current - target) events
    if current < target: insert (target - current) at even spacing
```

#### HIGH_METRO
Aggressively regular.
```
grid_snap:
  - 60% of gaps snapped to local grid

const_gap_fill:
  - 40% of 5s segments replaced

pattern_loop:
  - 30% of chart duration
  - Tile 2-4 gap micro-patterns

density_flatten:
  - 60% toward global mean

ratio_purge:
  - ALL gap ratios snapped to {0.25, 0.5, 1.0, 2.0, 4.0}
```

#### PURE_METRO
Maximum regularity. Randomly choose one of three sub-types (equal probability):

**Sub-type A: constant gap**
```
median_gap = median of all gaps in original chart
Replace ALL events: start at original start position,
  place events at exactly median_gap intervals.
Keep same total event count.
```

**Sub-type B: alternating pattern**
```
Find the 2 most frequent gap values (by 5% clustering).
Replace all gaps with alternating: [g1, g2, g1, g2, ...]
Rebuild events from original start position.
```

**Sub-type C: quantized grid**
```
Estimate BPM from median gap: bpm = 60000 / (median_gap_ms)
Beat interval (bins) = 60000 / bpm / BIN_MS
Snap ALL events to nearest 1/4 beat position.
Remove duplicates (events that land on same position).
```

## Pair construction

### Level encoding

| Level | Code | Type |
|---|---|---|
| CLEAN | 0 | — |
| LIGHT_RAND | 1 | random |
| MED_RAND | 2 | random |
| HIGH_RAND | 3 | random |
| PURE_RAND | 4 | random |
| LIGHT_METRO | 5 | metro |
| MED_METRO | 6 | metro |
| HIGH_METRO | 7 | metro |
| PURE_METRO | 8 | metro |

### Severity mapping

| Level | Severity |
|---|---|
| CLEAN | 0 |
| LIGHT_* | 1 |
| MED_* | 2 |
| HIGH_* | 3 |
| PURE_* | 4 |

### Pair types and margins

**Ordered pairs** (Bradley-Terry loss):
- CLEAN vs any corruption: margin = severity of corruption (1-4)
- Within random: margin = severity difference (1-3)
- Within metro: margin = severity difference (1-3)

**Tie pairs** (MSE loss):
- Cross-type same severity: LIGHT_RAND ≈ LIGHT_METRO, etc.

### Loss

```python
def loss(score_a, score_b, margin, is_tie, alpha=0.1):
    if is_tie:
        # push scores together
        return (score_a - score_b) ** 2
    else:
        # Bradley-Terry: a should be higher
        diff = score_a - score_b - alpha * margin
        return -log(sigmoid(diff))
```

### Batch composition

| Source | Proportion | Pair type |
|---|---|---|
| CLEAN vs random | 25% | ordered |
| CLEAN vs metro | 25% | ordered |
| Within-type random | 10% | ordered |
| Within-type metro | 10% | ordered |
| Cross-type ties | 15% | tie |
| Cross-set rating | 15% | ordered |

## Training

| Param | Value |
|---|---|
| Optimizer | AdamW |
| Batch size | 64 pairs |
| LR | 3e-4 |
| Epochs | 20 |
| Weight decay | 0.01 |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.1 |
| AMP | ON |
| Evals per epoch | 2 |

Single phase — rating pairs mixed in from the start.

## Validation

Same as 66-1 plus:
- Per-corruption-type accuracy (random pairs vs metro pairs)
- Tie pair score difference (should be near zero)
- Score by level (should be inverted U: CLEAN highest, both PURE types lowest)

## Evaluation (66-2b)

Run `classifier_eval_ar.py` on exp 14, 45, 58, 62 using `run_eval_66_1b.sh` (adapted for 66-2 checkpoint). Key metrics:
- Generator ranking should match GT metrics (exp 62 > 58 > 45 > 14)
- metro_streak should correlate NEGATIVELY with gen_score (opposite of 66-1b)

## Environment

Same as 66-1. CachyOS, RTX 4060 (8 GB), PyTorch 2.12.0+cu128.
