# Experiment 56-B — Full Architecture Specification

## Task

Test whether varying density conditioning (0.8x, 1.0x, 1.2x) actually changes model output. Same model and songs as experiment 56, but each song is run 3 times with different density scaling.

## Method

For each of 50 val songs, run AR inference 3 times:
1. density_mean * 0.8, density_peak * 0.8, density_std * 0.8
2. density_mean * 1.0, density_peak * 1.0, density_std * 1.0 (baseline)
3. density_mean * 1.2, density_peak * 1.2, density_std * 1.2

Compare predicted event counts and metrics across the 3 runs per song.

## Model Under Test

[Exp 45](../experiment_45/README.md) — same as exp 56.

## Inference Settings

| Param | Value |
|---|---|
| hop_ms | 75 |
| Sampling | Argmax (no temperature) |
| Density scales | 0.8x, 1.0x, 1.2x of chart's actual density |

## Metrics

Same as exp 56 (matched/close/far at 25ms/50ms/100ms thresholds), plus:
- **Sensitivity**: ratio of predicted events at 1.2x vs 0.8x density
- **Per-song delta**: how much does event count change with density scaling

## Output

- `results/density_sweep_results.json` — per-song, per-scale metrics
- `results/csvs/` — predicted and GT event CSVs (3 predicted per song)
- `results/density_sweep.png` — comparison graphs
