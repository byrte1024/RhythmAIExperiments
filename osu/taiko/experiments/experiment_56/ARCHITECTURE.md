# Experiment 56 — Full Architecture Specification

## Task

Analyze the relationship between density conditioning and AR (autoregressive) generation quality. This is an analysis experiment, not a training experiment — no model is trained.

## Method

Run full AR inference on 50 val songs using each song's actual chart density from the manifest. Compare predicted onsets to ground truth onsets to measure how well the model generates charts when given "correct" density information.

## Model Under Test

[Exp 45](../experiment_45/README.md) — EventEmbeddingDetector with gap ratios and tight density jitter (±2%/10%). Selected because it ranked highest on expert self-evaluation in [53-AR](../experiment_53ar/README.md) and has the strongest density adherence of any model.

### Model config (from exp 45):
- A_BINS=500, B_BINS=500, N_CLASSES=501
- Gap ratios: ON
- Density jitter: ±2% at 10% rate
- 72.1% HIT at peak (eval 5)

## Song Selection

50 songs from the val set (10% split, seed 42, by beatmapset_id). For each song, the chart with median density_mean is selected. Songs are spread evenly across the density range to ensure coverage from sparse (~1-2 events/sec) to dense (~7+ events/sec).

## Inference Settings

| Param | Value |
|---|---|
| hop_ms | 75 |
| Sampling | Argmax (no temperature) |
| Density | Per-chart actual (density_mean, density_peak, density_std from manifest) |

## Metrics

### Event matching (GT perspective)
For each ground truth event, find the closest predicted event:
- **Matched**: closest prediction within 25ms
- **Close**: closest prediction within 50ms
- **Far**: closest prediction beyond 100ms (effectively missed)

### Prediction matching (Pred perspective)
For each predicted event, find the closest GT event:
- **Hallucination**: predicted events with no GT event within 100ms

### Density
- **Conditioned density**: density_mean fed to model
- **GT density**: actual events/sec in ground truth
- **Predicted density**: actual events/sec in model output
- **Density ratio**: predicted / GT (1.0 = perfect match)

## Output

- `results/ar_results.json` — per-song metrics
- `results/csvs/` — predicted and GT event CSVs for each song
- `results/ar_analysis.png` — per-song bar charts (catch rate, hallucination, density comparison)
- `results/error_distributions.png` — error histograms per song
- `results/density_correlation.png` — scatter plots: density vs catch rate, hallucination, adherence

## Audio Source

Audio files from `osu/taiko/audio/`, named `{beatmapset_id} {artist} - {title}.{ext}`. Ground truth events from `datasets/taiko_v2/events/`. Events are stored as mel frame indices (int32), converted to ms via `bin * 4.9887`.
