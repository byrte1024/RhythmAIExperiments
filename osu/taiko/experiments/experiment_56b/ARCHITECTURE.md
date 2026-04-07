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

Exp 45 — EventEmbeddingDetector with gap ratios and tight density jitter (±2%/10%). Config: A_BINS=500, B_BINS=500, N_CLASSES=501, gap ratios ON. 72.1% HIT at peak. Selected for strongest density adherence.

## Inference Settings

| Param | Value |
|---|---|
| hop_ms | 75 |
| Sampling | Argmax (no temperature) |
| Density scales | 0.8x, 1.0x, 1.2x of chart's actual density |

## Metrics

Event matching from GT perspective: **Matched** (closest prediction within 25ms), **Close** (within 50ms), **Far** (beyond 100ms). Prediction perspective: **Hallucination** (predicted events with no GT event within 100ms). Additionally:
- **Sensitivity**: ratio of predicted events at 1.2x vs 0.8x density
- **Per-song delta**: how much does event count change with density scaling

## Output

- `results/density_sweep_results.json` — per-song, per-scale metrics
- `results/csvs/` — predicted and GT event CSVs (3 predicted per song)
- `results/density_sweep.png` — comparison graphs
