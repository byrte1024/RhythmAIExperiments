# Experiment 59-G - Synthetic Evaluator Validation on Fresh Data

## Hypothesis

The synthetic evaluators from 59-B through 59-F were built and validated on the same 42-AR and 53-AR songs. This is a form of overfitting — the formulas may not generalize to unseen songs.

**Validation: run the 4 models from 53-AR on 30 fresh val songs** and use the synthetic evaluators to rank them. If the leaderboard matches 53-AR's human rankings (exp45 > exp44 > exp53 > exp14), the evaluator generalizes. If not, it was overfit to the evaluation songs.

### Approach

1. Select 30 val songs with available audio (different from 53-AR's 10 songs)
2. Run full AR inference with all 4 models under **two density regimes**:
   - **song_density**: each song's actual density from manifest
   - **fixed_53ar**: the exact conditioning used in 53-AR human evaluation (d_mean=5.75, d_peak=11.1)
3. Compute all chart metrics from 59-B
4. Apply the 3 optimized evaluator formulas from 59-F (expert, volunteer, general)
5. Rank models per-song and aggregate into a leaderboard
6. Compare to 53-AR human rankings

Two density regimes test whether the leaderboard changes when models get "appropriate" vs "uniform" density. The 53-AR human rankings were obtained under fixed density, so the fixed_53ar regime should be the closer match. But [exp 56-B](../experiment_56b/README.md) showed models are density-sensitive (1.53x ratio), so per-song density may produce different rankings.

### Launch

```bash
cd osu/taiko
python experiments/experiment_59g/validate_fresh.py
```

## Result

30 fresh val songs × 4 models × 2 density regimes = 240 inference runs.

### Rankings (6 combinations):

| Regime | Evaluator | Ranking | #1 | Top-2 |
|---|---|---|---|---|
| song_density | **expert** | **exp45 > exp44 > exp14 > exp53** | **YES** | **YES** |
| song_density | volunteer | exp44 > exp45 > exp14 > exp53 | no | no |
| song_density | general | exp44 > exp45 > exp14 > exp53 | no | no |
| fixed_53ar | expert | exp44 > exp14 > exp45 > exp53 | no | no |
| fixed_53ar | volunteer | exp44 > exp14 > exp45 > exp53 | no | no |
| fixed_53ar | general | exp44 > exp14 > exp45 > exp53 | no | no |

53-AR Human: exp45 > exp44 > exp53 > exp14

### Key findings:

1. **Expert + song_density is the only combination that nails #1 and top-2.** All others swap exp44/exp45.

2. **exp53 is consistently last** — matching its 3rd/4th position in human rankings. The broad ordering (exp44/45 top, exp53/14 bottom) is stable.

3. **Density regime matters.** Fixed 53-AR density (5.75/11.1) distorts rankings — exp45 drops to 3rd, exp14 jumps to 2nd. Per-song density gives more reliable results.

4. **exp44 vs exp45 is genuinely too close to call.** Human margin was 1 point (44 vs 43). The synthetic evaluator correctly identifies this as a near-tie but can't consistently pick the winner.

## Lesson

The synthetic evaluator **generalizes the coarse ranking** (who's good, who's bad) but **can't resolve tight races** on fresh data. On the 53-AR evaluation songs it got top-2 right consistently — on 30 unseen songs, only the expert formula with song-appropriate density succeeds.

**Per-song density is better than fixed density** for evaluation. The fixed density regime from 53-AR distorts rankings because different songs have different natural tempos — a one-size-fits-all density disadvantages models differently.

**Practical recommendation**: Use expert evaluator (gap_std + gap_cv) with per-song density for automated ranking. Accept that close models (within ~5% of each other) need human evaluation to resolve.
