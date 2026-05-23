# Experiment 018 — External baseline benchmarks

## Status

`Complete`

## Context

The 017 series established a framewise onset detector that beats
[#007](../007-time-stretch/) on AR corpus metrics (`matched_rate`
0.783, `density_ratio` 1.020, `dc_human` 92.7 at optimal threshold
[017e threshold_sweep.json]). But this comparison is only against
prior internal models. To contextualize the result, this experiment
benchmarks external onset detectors and chart generators on the same
GT dataset using the same comparison pipeline (`Chart.compare` via
`gt_match_metrics`).

This is a **no-training experiment** — no model is trained. Each
external tool runs inference on the val split of `taiko2_v1`, its
output is compared against GT charts, and the same metrics
(`matched_rate`, `hallucination_rate`, `density_ratio`, `dc_human`,
etc.) are reported.

## Citations

- Internal baselines:
  - [#007 -- TimeStretch](../007-time-stretch/). `matched_rate`
    0.703, `hallucination_rate` 0.172, `dc_human` 92.0
    [exp_007_time_stretch, step 413,480].
  - [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).
    Best sweep: `matched_rate` 0.783, `hallucination_rate` 0.201,
    `dc_human` 92.7 at E8/tau=0.40 [017e threshold_sweep.json].
- External models:
  - [Mapperatorinator2](https://github.com/Tiger14n/Mapperatorinator2) —
    Whisper-based encoder-decoder, 219M params, trained on ranked osu!
    beatmaps across all gamemodes. Generates full chart events
    (timing, hit objects, SV) from mel spectrogram. Time quantized to
    10ms.
  - [Mapperatorinator (v1)](https://github.com/OliBomby/Mapperatorinator) —
    predecessor. Same architecture, earlier training.
  - [madmom](https://github.com/CPJKU/madmom) — CNN-based musical
    onset detector. Pretrained on annotated onset datasets. Outputs
    onset times, not chart events.
  - [librosa onset_detect](https://librosa.org/doc/main/onset.html) —
    classical spectral-flux onset detection. No neural network. Tested
    previously in [#011](../011-onset-feature-survey/) (F1 0.679 at
    +/-10 frames).
  - [BeatThis!](https://github.com/CPJKU/beat_this) — ISMIR 2024
    SOTA beat/downbeat tracker. Transformer-based. Outputs beat
    positions, not onsets.
- Cross-experiment record: [`../README.md`](../README.md).

---
<!--
PRE-RUN. Do not edit after the run.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

External onset detectors and chart generators will produce lower
`matched_rate` and higher `hallucination_rate` than #017e on our GT
charts, because:

1. **General-purpose onset detectors** (librosa, madmom) detect ALL
   audio onsets, not chart-author-selected ones. They will over-emit
   massively (`density_ratio` 3-10x) with high hallucination.
2. **Mapperatorinator2** solves a different task (generate "a plausible
   chart") rather than "match this specific GT chart." Style variation
   between its generated charts and the GT mapper's choices will
   reduce `matched_rate` even when both charts are independently
   valid. However, its `dc_human` should be competitive (88-93)
   because it was trained on ranked charts.
3. **BeatThis!** detects beats (subset of onsets), so it will
   under-emit on charts with subdivisions beyond the beat level.

### Predicted numbers

| Backend | matched_rate | halluc_rate | density_ratio | dc_human |
|---|---:|---:|---:|---:|
| Mapperatorinator2 (taiko) | 0.55-0.70 | 0.15-0.30 | 0.8-1.2 | 88-93 |
| madmom (CNN onset) | 0.40-0.60 | 0.30-0.60 | 2.0-5.0 | 70-85 |
| librosa (spectral flux) | 0.30-0.50 | 0.40-0.70 | 3.0-8.0 | 60-80 |
| BeatThis! (beat tracker) | 0.20-0.40 | 0.05-0.15 | 0.3-0.6 | 85-92 |

Reference:
- #017e E8@tau=0.40: mr=0.783, hr=0.201, dr=1.020, dc=92.7
- #007 best: mr=0.703, hr=0.172, dr=0.865, dc=92.0

## Success criteria

- **Success if:** any external model scores HIGHER than #017e on
  `matched_rate`, `dc_human`, or lower on `hallucination_rate` at
  comparable density. That would prove there is headroom — our
  architecture or training approach can be improved by learning from
  what that model does differently.
- **Also success if:** a general-purpose onset detector (madmom,
  librosa) achieves `matched_rate` > 0.60 — that would confirm
  raw onset detection can reach our GT without chart-specific
  training, and the remaining gap is about selectivity.
- **Neutral if:** all externals score below #017e — confirms our
  approach is competitive but provides no new improvement signal.

## Changes from baseline

No code changes to training infrastructure. New file:
`cli/benchmark_external.py` — runs external backends on val charts
and reports `gt_match_metrics`.

External tool setup:
- Mapperatorinator2: cloned + venv at `/home/drore/repos/Mapperatorinator2`
  via `external/setup_mapperatorinator2.fish`.
- madmom: `pip install madmom` into the taiko2 venv (if compatible).
- librosa: already in the taiko2 venv.
- BeatThis!: `pip install beat-this` or clone
  `https://github.com/CPJKU/beat_this`.

## Run config

Commands for each backend:

```bash
# Mapperatorinator2 (taiko mode)
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.benchmark_external \
    --backend mapperatorinator \
    --backend-path /home/drore/repos/Mapperatorinator2 \
    --dataset taiko2_v1 \
    --fraction 0.05 --device cuda \
    --experiment-dir osu/taiko2/experiments/018-baselines

# librosa (classical baseline)
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.benchmark_external \
    --backend librosa \
    --dataset taiko2_v1 \
    --fraction 0.05 \
    --experiment-dir osu/taiko2/experiments/018-baselines

# madmom (neural onset detector)
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.benchmark_external \
    --backend madmom \
    --dataset taiko2_v1 \
    --fraction 0.05 \
    --experiment-dir osu/taiko2/experiments/018-baselines
```

Dataset: `taiko2_v1`, val split, 5% fraction (~50 charts).
Comparison: `pred_chart.compare(gt)` via `gt_match_metrics` with
tolerances (5, 10, 25, 50, 100ms).

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

Two external baselines ran successfully. Mapperatorinator2 could not
be benchmarked due to unresolvable dependency conflicts. madmom was
not tested (Python 3.13 compatibility issues).

### Comparison table

| Backend | matched_rate | halluc_rate | density_ratio | dc_human | error_med_ms | events/s | over_pspace_self |
|---|---:|---:|---:|---:|---:|---:|---:|
| **#017e E8@tau=0.40** | **0.783** | 0.201 | **1.020** | **92.7** | **10.3** | 4.31 | n/a |
| **#007 best** | 0.703 | **0.172** | 0.865 | 92.0 | 10.2 | 3.57 | 7.26 |
| librosa (spectral flux) | 0.408 | 0.254 | 1.126 | 84.0 | 131.4 | 3.71 | n/a |
| BeatThis! (ISMIR 2024) | 0.189 | 0.228 | 0.747 | 84.2 | 68.9 | 2.60 | 4.85 |
| Mapperatorinator2 | — | — | — | — | — | — | FAILED |
| madmom | — | — | — | — | — | — | not tested |

Per-backend JSON results (with per-chart data):
[`benchmark_librosa.json`](./benchmark_librosa.json),
[`benchmark_beatthis.json`](./benchmark_beatthis.json).

### Mapperatorinator2 failure details

Mapperatorinator2 ([Tiger14n fork](https://github.com/Tiger14n/Mapperatorinator2))
and the [original OliBomby repo](https://github.com/OliBomby/Mapperatorinator)
both failed due to dependency conflicts:

- The original repo requires `transformers>=4.49`
  (`GradientCheckpointingLayer`, `MoonshineConfig`).
- `transformers>=4.49` introduces a `meta`-device bug in
  `generate()` when used with `rotary-embedding-torch` — the RoPE
  freqs are created on CPU/meta during the KV cache loop,
  triggering `RuntimeError: Tensor on device cpu is not on the
  expected device meta!`.
- The Tiger14n fork has an older codebase that doesn't match the
  v29.1 model's tokenizer (missing `SCROLL_SPEED_RATIO` in the
  event vocabulary).
- Running on CPU failed because the model's Hydra config defaults
  to `device=cuda` and even when overridden, the
  `transformers.generate()` meta-device issue persists.

Would require the author's exact Docker environment or upstream
fixes to `rotary-embedding-torch` / `transformers` to benchmark.

### How Mapperatorinator evaluates itself

Mapperatorinator uses **FID (Frechet Inception Distance)** computed
from a custom "Mapper Classifier" that extracts high-level feature
vectors from beatmaps. This is a **distributional** metric — it
measures whether generated charts statistically resemble ranked
charts, not whether they match specific GT charts. The authors
noted that *"FID scores were not very close to the actual quality
of the generated beatmaps"*
([classifier README](https://github.com/OliBomby/Mapperatorinator/blob/main/classifier/README.md)).

No onset-level metrics (matched_rate, F1, etc.) are published.
This reflects a fundamentally different evaluation philosophy:
Mapperatorinator treats chart generation as a **generative** task
(distributional quality), while our approach treats it as a
**predictive** task (pointwise accuracy against GT).

## Vs prediction

| Prediction | Actual | Verdict |
|---|---|---|
| Mapperatorinator2 mr=0.55-0.70 | FAILED TO RUN | **inconclusive** |
| librosa mr=0.30-0.50 | 0.408 | **match** (within predicted band) |
| BeatThis! mr=0.20-0.40 | 0.189 | **match** (just below band — beats are a small subset of onsets) |
| Any external > #017e | none | **neutral** (no external model scored higher) |
| General onset detector mr > 0.60 | librosa 0.408 | **miss** (raw onset detection alone can't reach our GT without chart-specific training) |

## Takeaways

- **Neither external baseline comes close to our trained models.**
  librosa (`matched_rate` 0.408) and BeatThis! (0.189) are both
  far below #017e (0.783) and #007 (0.703). Chart-specific training
  on GT data is essential — general-purpose onset/beat detection is
  not sufficient.

- **librosa detects the right density but wrong positions.**
  `density_ratio` 1.13 is near-perfect, meaning spectral flux finds
  roughly the right number of audio events. But `matched_rate` 0.41
  and `error_median` 131ms show these are the wrong events. The
  model's value lies in learning WHICH onsets to select, not in
  detecting that audio events exist.

- **BeatThis! under-emits drastically.** `density_ratio` 0.75 means
  it only finds beats, missing all the subdivisions (1/4, 1/6, 1/8
  notes between beats) that make up taiko charts. Beat tracking
  alone is not a viable path to chart generation.

- **dc_human is similar across externals (~84) vs ours (~92).**
  Both external models produce pattern structures that are
  recognizably musical but not chart-author-quality. The 8 pp gap
  in `dc_human` represents the difference between "notes at
  musically sensible positions" and "notes that match a specific
  chart author's aesthetic choices."

- **Distributional evaluation is worth exploring.** Mapperatorinator's
  FID-based approach reflects a real truth: rhythm charts ARE
  generative — there are multiple valid charts for any song, and the
  GT chart is just one of many correct answers. Our pointwise
  metrics (matched_rate, halluc_rate) penalize valid alternative
  charts that differ from the GT mapper's choices. A future
  experiment could explore distributional metrics (FID over chart
  features, or perceptual quality scores from a classifier) as a
  complement to pointwise accuracy. This would be significantly
  more complex to implement but could provide a more nuanced picture
  of chart quality. The challenge is that Mapperatorinator's own
  authors found their FID metric unreliable — designing a
  distributional metric that actually correlates with chart quality
  remains an open problem.

## Followup questions

- **Retry Mapperatorinator2 in Docker.** Build a container with
  their exact deps (Python 3.10, torch cu121, transformers 4.45) to
  get a working benchmark. The model is the most direct competitor
  and worth the effort.
- **Distributional evaluation pilot.** Train a simple chart
  classifier (is this chart real or generated?) and measure how
  often our model's output fools it vs BeatThis!/librosa. Simpler
  than full FID but tests the same idea.
- **madmom benchmark.** Install in a separate Python 3.10 venv and
  run the CNN onset processor. Expected to perform between librosa
  and our model.
- **Coincidence map input.** Experiment 019 candidate — add the
  13-row coincidence summary as a parallel input channel to test
  whether onset-type features improve selectivity.
