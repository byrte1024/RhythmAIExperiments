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

*Pending*

## Lesson

*Pending*
