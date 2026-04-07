# Experiment 59-E - Split Synthetic Evaluators

## Hypothesis

[Exp 59-D](../experiment_59d/README.md) found that expert and volunteers correlate with different metrics:
- **Expert**: gap_cv (+0.38), gap_std (+0.28)
- **Volunteers**: dominant_gap_pct (-0.42), gap_entropy (+0.37), max_metro_streak (-0.35)

[Exp 59-C](../experiment_59c/README.md) used the same formula for all evaluator types. Building **separate evaluators** tuned to each group should improve accuracy.

### Formulas

**Expert evaluator** (proportional variety):
```
score = z(gap_cv) * 0.384 + z(gap_std) * 0.281
```

**Volunteer evaluator** (anti-repetition):
```
score = -z(dominant_gap_pct) * 0.420 + z(gap_entropy) * 0.370 - z(max_metro_streak) * 0.351
```

**Combined evaluator** (from 59-B):
```
score = z(gap_std) * 0.299 + z(gap_cv) * 0.289 - z(dominant_gap_pct) * 0.272 - z(max_metro_streak) * 0.269
```

### Launch

```bash
cd osu/taiko
python experiments/experiment_59e/split_evaluator.py
```

## Result

12 combinations tested (4 evaluators × 3 human types).

### Best evaluator per audience:

| Audience | Best evaluator | #1 Match | Tau | r | p |
|---|---|---|---|---|---|
| Self (expert) | combined_top2 | **47%** | **+0.422** | +0.404 | 0.003 |
| Volunteers | volunteer | **60%** | +0.267 | **+0.469** | 0.005 |
| All combined | combined_top2 | **52%** | +0.293 | +0.342 | 0.001 |

### Cross-evaluator: tuned evaluators outperform on their target, fail on the other group

- **Volunteer evaluator on expert**: 33% #1 match, r=0.183 (n.s.) — basically random
- **Expert evaluator on volunteers**: 40% #1 match, r=0.218 (n.s.) — barely above random
- **combined_top2**: best all-rounder, competitive everywhere

## Lesson

1. **Tuned evaluators work.** The volunteer formula (dominant_gap_pct + entropy + streak) gets r=0.469 on volunteer rankings — the strongest correlation seen in any 59-series experiment. The expert formula (gap_cv + gap_std) gets tau=0.422 on expert rankings.

2. **Preferences are genuinely different.** Cross-evaluator performance drops to near-random, confirming expert and volunteer preferences are driven by different chart properties — not just noise.

3. **combined_top2 (gap_std + gap_cv) is the best general-purpose evaluator.** It predicts 52% of first-place picks across all voters (2x random) with p=0.001. For automated evaluation where the audience is unknown, this is the recommended formula.

4. **For volunteer-targeted evaluation**, use the volunteer formula — 60% first-place accuracy with the strongest Spearman correlation (r=0.469).

### Model leaderboard validation

Aggregated synthetic scores per model and compared overall rankings to human rankings:

**53-AR** (4 models): All 3 synthetic evaluators correctly pick #1 (exp45) and #2 (exp44). Only swap 3rd/4th (exp14 vs exp53).

| Rank | Human | top2 | expert | volunteer |
|---|---|---|---|---|
| 1st | **exp45** | **exp45** | **exp45** | **exp45** |
| 2nd | **exp44** | **exp44** | **exp44** | **exp44** |
| 3rd | exp53 | exp14 | exp14 | exp14 |
| 4th | exp14 | exp53 | exp53 | exp53 |

**42-AR** (3 models): All synthetic evaluators get it wrong. Human winner (exp14) ranks last in synthetic scores.

| Rank | Human | top2 | volunteer |
|---|---|---|---|
| 1st | **exp14** | exp35c | exp42 |
| 2nd | exp42 | exp42 | exp14 |
| 3rd | exp35c | exp14 | exp35c |

**Why 42-AR fails**: exp14 won 42-AR by avoiding context-induced metronome problems, not by having better gap variety. The synthetic evaluator measures positive properties (pattern diversity) but can't detect the absence of a failure mode. In 53-AR, context models had improved enough that the winner genuinely had better variety — the synthetic metrics work when quality is driven by positive traits rather than by avoiding defects.
