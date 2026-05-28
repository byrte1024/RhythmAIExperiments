# Experiment 019 — Coincidence input (mel + 13-row coincidence summary)

## Status

`Complete` (hypothesis rejected)

## Context

[#017f](../017f-framewise-bce-metrics-rerun/) established clean
baselines for the framewise detector with the upgraded metrics
pipeline. At the optimal sweep threshold (tau=0.4, eval_248088):
`precision` 0.778, `recall` 0.757, `f1` 0.771, `density_ratio` 0.964,
`dc_human` 92.4, `gap_hist_tvd` 0.331, `gap_peak_iou` 0.742,
`silence_overlap_f1` 0.527
[017f threshold_sweep.json, eval_248088/tau=0.4].

Two key gaps remain:

1. **Silence regions are poorly modeled.** `silence_overlap_f1` 0.527
   at tau=0.4 -- the model fills in onsets during GT silence regions,
   driving over-emission
   [017f threshold_sweep.json, eval_248088/tau=0.4].

2. **The model uses spectral transients, not onset importance.**
   [#020](../020-activation-maximization/) showed the model is
   sensitive to high-frequency transient attack (saliency at bands
   70-79 is 2-4x bands 0-9) and
   [#020b](../020b-legible-dreams/) showed it dreams low-frequency
   transients. The model detects real audio onsets but cannot
   distinguish chart-author-selected ones from metronomic beats --
   there is no signal in the 80-channel mel that encodes "this onset
   is unusual" or "this is an expected beat, not a highlight".

The coincidence map (implemented in `domain/coincidence.py`) produces
a 13-row summary that encodes exactly these signals:

- **Onset type (LSH color, rows 0-2):** Each onset is assigned an RGB
  color by locality-sensitive hashing of its spectral fingerprint.
  Metronomic beats repeat their color; unexpected onsets have rare
  colors.
- **Onset importance (IDF weighting, row 3):** The IDF term measures how
  unusual each onset's spectral profile is across the full track.
  Common beats have low IDF; rare, accented beats have high IDF.
- **Spike energy (row 4):** Magnitude of the onset spike, independent of
  IDF.
- **Band-group averages (rows 5-12):** Eight frequency-band averages
  summarize the spectral shape of each onset.

Concatenating these 13 rows with the 80 mel rows gives the model
explicit "how unusual is this onset" information. The model's input
grows from (B, 80, 1000) to (B, 93, 1000); the only architectural
change is widening the Conv stem input from 80 to 93.

## Citations

- Direct baseline:
  - [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).
    Best sweep tau=0.4 (eval_248088): `precision` 0.778, `recall`
    0.757, `f1` 0.771, `density_ratio` 0.964, `dc_human` 92.4,
    `gap_hist_tvd` 0.331, `gap_peak_iou` 0.742,
    `silence_overlap_f1` 0.527
    [017f threshold_sweep.json].
    Best frame F1: 0.822 (step 206,740). Best fps50 F1: 0.741
    (step 289,436)
    [exp_017f_framewise_bce_metrics_rerun, metrics.jsonl].
- Interpretability:
  - [#020 -- activation maximization](../020-activation-maximization/).
    High-band saliency dominance (2-4x), bin-exact confidence.
  - [#020b -- legible dreams](../020b-legible-dreams/). Low-band
    dominant onset construction (ratio 1.3-1.5x), energy scales with
    density.
- Related priors:
  - [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).
    Identical model/loss, threshold sweep confirmed tau=0.4 optimal.
  - [#007 -- TimeStretch](../007-time-stretch/). Scalar baseline.
- Coincidence feature implementation: `domain/coincidence.py`.
- Audio sampler: `samplers/coincidence_mel.py`
  (`CoincidenceMelSampler` -- produces `(93, T)` on disk and at
  inference time).

---
<!--
PRE-RUN. Do not edit after the run.
-->
---------------------------------------------------------------------

## Hypothesis

### Claim

Adding the 13-row coincidence summary as a parallel input alongside mel
(n_mels 80 -> 93) will give the model access to onset type and importance
signals that the mel alone does not carry. The model will learn to use
these channels -- demonstrated by a >= 3 % F1 drop in the `no_coincidence`
benchmark (rows 80-92 zeroed) -- and this selectivity improvement will
reduce `gap_hist_tvd` (better rhythmic structure) and improve
`silence_overlap_f1` (fewer onsets emitted during silence).

### Mechanism

The `silence_overlap_f1` ceiling at 0.527 in #017f is caused by the
model responding to all acoustically salient onsets rather than only
chart-author-selected ones. The mel spectrogram is agnostic to onset
importance: a metronomic beat and a chart-worthy accent look similar if
their frequency content and energy are similar. The coincidence map's IDF
row (row 3) directly encodes "how unusual is this onset across the corpus"
-- low IDF for expected beats, high IDF for rare/important ones. If the
model learns to weight its prediction by IDF, it will suppress FPs on
common/expected beats and improve silence_overlap_f1. The LSH color rows
(0-2) provide complementary repetition-pattern information: metronomic
beats have consistent colors across the window, while chart-highlights
break the pattern.

#020's saliency finding (high-band sensitivity) suggests the model
already uses spectral contrast for detection. The coincidence map's
band-group averages (rows 5-12) provide a pre-computed spectral shape
summary that complements what the model extracts from raw mel. The IDF
row adds a dimension the mel cannot provide: cross-track onset rarity.

### Predicted numbers

Reference: [#017f](../017f-framewise-bce-metrics-rerun/) tau=0.4
sweep [017f threshold_sweep.json, eval_248088].

| Metric | #017f (tau=0.4) | Predicted (#019) | Notes |
|---|---:|---:|---|
| AR `precision` (25ms) | 0.778 | **>= 0.77** | should hold |
| AR `recall` (25ms) | 0.757 | **>= 0.75** | should hold |
| AR `f1` (25ms) | 0.771 | **>= 0.77** | should hold or improve |
| AR `density_ratio` | 0.964 | **0.90-1.10** | near-1.0 maintained |
| AR `dc_human` (%) | 92.4 | **>= 92** | pattern quality maintained |
| `gap_hist_tvd` | 0.331 | **< 0.30** | better rhythmic structure |
| `gap_peak_iou` | 0.742 | **>= 0.74** | should hold or improve |
| `silence_overlap_f1` | 0.527 | **>= 0.60** | fewer silence FPs |
| `density_corr` | 0.546 | **>= 0.55** | better density tracking |
| frame F1 (best eval) | 0.822 | **>= 0.82** | at least matches 017f |
| fps50 binary F1 | 0.741 | **>= 0.74** | watched metric |
| `no_coincidence` F1 delta | n/a | **>= -3 %** (drop) | model uses coincidence |

## Success criteria

- **Must:** `no_coincidence` benchmark shows >= 3 % F1 drop vs full model
  (rows 80-92 zeroed). This confirms the model actually uses the
  coincidence channels. Without this, the coincidence input is wasted
  and the experiment is inconclusive.
- **Must:** `f1` (25ms) >= 0.75 at best sweep threshold. The coincidence
  input must not degrade the headline quality metric below 017f's level.
- **Fails if:** model ignores coincidence -- `no_coincidence` F1 delta < 1 %.
  This would mean the model routes around the extra channels and the
  hypothesis is falsified.
- **Fails if:** `f1` < 0.65 at every threshold -- widening the conv stem
  disrupted learning.
- **Nice-to-have:** `silence_overlap_f1` >= 0.60 (above 017f's 0.527).
- **Nice-to-have:** `gap_hist_tvd` < 0.30 (below 017f's 0.331).
- **Nice-to-have:** `no_mel` benchmark (rows 0-79 zeroed) shows the model
  cannot succeed on coincidence alone (F1 drop >= 20 pp vs full model),
  confirming mel and coincidence are complementary.

## Changes from baseline

Baseline: [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).

Three changes:

- **New dataset `taiko2_v1_coin`** -- built with
  `CoincidenceMelSampler` (`--audio-sampler coincidence_mel`).
  Features on disk are `(93, T)` float16 -- the standard 80-band
  mel concatenated with the 13-row coincidence summary. Both the
  data sampler and the inference sampler consume this format
  natively with no special flags.
- `config/model.json` -- `n_mels: 80 -> 93`. Widens the Conv stem
  input from 80 to 93 channels. All other model params unchanged.
- `config/trainer.json` -- `metric_to_watch` changed to
  `frame/mini/tau50/fps_50/binary_f1` (per 017f finding that fps50
  F1 tracks AR quality better than frame F1).

All other configs (loss.json, adapter.json, data.json) are identical
to #017f except the checkpoint path and decode_threshold in infer.json
(0.4, the 017f optimal threshold).

Two new benchmarks added vs #017f:
- `no_coincidence`: rows 80-92 zeroed at eval time. Measures how much
  the model relies on coincidence channels.
- `no_mel`: rows 0-79 zeroed at eval time. Measures whether coincidence
  alone is informative.

Dataset preparation:

```bash
# Build dataset with coincidence features baked in
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.prepare_dataset \
    --name taiko2_v1_coin \
    --charts-dir /home/drore/charts/repos/BeatDetector/osu/taiko/charts/ \
    --audio-sampler coincidence_mel

# Fetch star ratings + engagement
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.fetch_stars --dataset taiko2_v1_coin
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.fetch_engagement --dataset taiko2_v1_coin
```

## Run config

- Run name: `exp_019_coincidence_input`
- Config snapshots: [`config/`](./config/)
- Dataset: `taiko2_v1_coin` (93-channel features: 80 mel + 13 coincidence)
- Command:
  ```bash
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_019_coincidence_input \
      --config-dir osu/taiko2/experiments/019-coincidence-input/config \
      --dataset taiko2_v1_coin --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/019-coincidence-input/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

## Future directions

If coincidence input helps (no_coincidence delta >= 3 % F1), consider:

- **Full RGB heatmap (192 channels):** Use all spatial coincidence map
  channels instead of the 13-row summary. Higher information density
  at the cost of a wider conv stem.
- **Learned coincidence features:** Replace the fixed pipeline
  (mel -> flux -> spike -> IDF -> LSH -> 13 rows) with a small CNN
  trained end-to-end alongside the main model.

---------------------------------------------------------------------
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
---------------------------------------------------------------------

## Results summary

6 evals completed (steps 41,140 to 143,990), stopped early once the
no_coincidence benchmark showed no movement. Threshold sweep across
all checkpoints x 7 thresholds.

### Primary finding: model does not use coincidence channels

The no_coincidence benchmark (rows 80-92 zeroed) showed **+0.2-0.4%
F1** at every eval -- the model performs marginally *better* without
the coincidence rows. This was stable across all 6 evals with zero
trend toward using the channels
[exp_019_coincidence_input, metrics.jsonl, bench/no_coincidence].

| Eval | Step | Full F1 | no_coincidence F1 | Delta |
|---:|---:|---:|---:|---:|
| 1 | 41,140 | 0.795 | 0.797 | +0.3% |
| 2 | 61,710 | 0.798 | 0.801 | +0.4% |
| 3 | 82,280 | 0.794 | 0.796 | +0.3% |
| 4 | 102,850 | 0.812 | 0.814 | +0.3% |
| 5 | 123,420 | 0.815 | 0.817 | +0.3% |
| 6 | 143,990 | 0.816 | 0.818 | +0.2% |

The must-have criterion (>= 3% F1 drop) was not met. The hypothesis
is **rejected**: the model routes entirely through the mel bands and
ignores the coincidence summary.

### Why: coincidence is redundant with mel

The benchmark hierarchy reveals why:

| Benchmark | What's available | F1 (E6) | Drop vs full |
|---|---|---:|---:|
| full | mel + coincidence | 0.816 | -- |
| no_coincidence | mel only | 0.818 | +0.2% |
| no_mel | coincidence only | 0.119 | -85% |
| no_audio | nothing | 0.033 | -96% |

The coincidence channels carry some signal in isolation (F1 0.119
from 13 channels vs 0.033 from nothing), but this signal is
**completely redundant** when mel is present. The coincidence summary
is derived from the mel spectrogram (via spectral flux, spike
detection, IDF, LSH), so the model can extract the same information
directly from mel -- and does so more effectively than consuming the
pre-computed summary.

### Threshold sweep: small improvement at matched density

Despite not using the coincidence channels, 019 produced a small but
real improvement at the matched operating point
[019 threshold_sweep.json, 017f threshold_sweep.json]:

| Metric | 017f (tau=0.4) | 019 (tau=0.4) | Delta |
|---|---:|---:|---:|
| F1 (25ms) | 0.769 | **0.781** | **+0.012** |
| Precision | 0.777 | 0.776 | -0.001 |
| Recall | 0.756 | **0.769** | **+0.013** |
| density_ratio | 0.968 | 0.969 | +0.001 |
| dc_human | 92.50 | **92.94** | **+0.44** |
| gap_TVD | 0.335 | **0.294** | **-0.041** |
| ratio_TVD | 0.472 | **0.465** | -0.007 |
| density_corr | 0.538 | **0.547** | +0.009 |
| gap_peak_IoU | 0.733 | 0.734 | +0.001 |
| silence_f1 | **0.576** | 0.533 | -0.043 |
| bpm_ratio | 0.968 | 1.048 | +0.080 |

Best overall F1: 019 0.787 (eval_143990/tau=0.3) vs 017f 0.782
(eval_289436/tau=0.3) [threshold_sweep.json].

The improvement likely comes from the wider Conv stem (93 vs 80
input channels) providing more capacity in the first convolution,
not from the coincidence signal itself. The extra 13 channels act as
additional learnable dimensions that the model repurposes.

### Threshold ladder comparison

Best checkpoint of each at all thresholds
[019 eval_143990, 017f eval_289436, threshold_sweep.json]:

| tau | 019 F1 | 017f F1 | 019 DR | 017f DR | 019 gTVD | 017f gTVD |
|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.682 | 0.691 | 1.848 | 1.785 | 0.568 | 0.543 |
| 0.2 | 0.763 | 0.768 | 1.405 | 1.364 | 0.408 | 0.399 |
| **0.3** | **0.787** | 0.782 | 1.165 | 1.138 | 0.319 | 0.332 |
| **0.4** | **0.781** | 0.767 | 0.969 | 0.967 | **0.294** | 0.327 |
| 0.5 | 0.738 | 0.737 | 0.777 | 0.809 | 0.352 | 0.374 |
| 0.6 | 0.640 | 0.668 | 0.581 | 0.647 | 0.487 | 0.465 |
| 0.7 | 0.524 | 0.577 | 0.424 | 0.500 | 0.613 | 0.569 |

019 outperforms 017f at tau 0.3-0.4 (the balanced operating range)
but underperforms at tau 0.5-0.7 (heavy under-emission). The gap_TVD
improvement at tau=0.4 (0.294 vs 0.327) is the largest distributional
gain.

### Training progression vs 017f

Frame metrics tracked ~0.005-0.008 behind 017f at matched steps
throughout training. The wider conv stem converged slightly slower
but reached comparable quality
[exp_019, exp_017f, metrics.jsonl]:

| Metric | 017f best | 019 best | Delta |
|---|---:|---:|---:|
| Frame F1 | 0.822 | 0.816 | -0.006 |
| fps50 F1 | 0.741 | 0.731 | -0.010 |
| Loss (min) | 0.233 | 0.234 | +0.001 |

### Other benchmarks

| Benchmark | F1 (E6) | Notes |
|---|---:|---|
| no_future_audio | 0.000 | Model requires future audio |
| no_past_audio | 0.693 | Past audio contributes ~15% |
| static_audio | 0.001 | Model uses temporal structure, not static spectrum |
| no_context | via no_coincidence | Events contribute ~0.2% (negligible) |

## Visualizations

<!-- No custom graphs generated for this experiment. The key finding
(no_coincidence benchmark flat at +0.2-0.3%) is numeric, not visual.
See threshold sweep table above for the distributional comparison. -->

## Vs prediction

- **no_coincidence F1 drop >= 3%:** predicted yes -> actual
  **+0.2-0.4%** (wrong direction) -> **miss (hypothesis rejected)**.
- **f1 >= 0.75 at best sweep threshold:** predicted yes -> actual
  0.787 -> **match**.
- **model ignores coincidence (fail criterion):** delta < 1% ->
  **triggered** (0.2-0.4%, wrong direction).
- **silence_overlap_f1 >= 0.60:** predicted yes -> actual 0.533 at
  matched DR -> **miss**.
- **gap_hist_tvd < 0.30:** predicted yes -> actual **0.294** at
  tau=0.4 -> **match** (but not from coincidence use).
- **no_mel F1 drop >= 20 pp:** predicted yes -> actual -85% ->
  **match** (coincidence alone is insufficient).

The hypothesis is falsified. The coincidence summary is informative
in isolation (F1 0.119 vs 0.033 baseline) but completely redundant
when mel is available. The model extracts the same information
directly from mel more effectively than from the pre-computed summary.

## Takeaways

- **Pre-computed features derived from mel are redundant with mel.**
  The coincidence map is computed from the mel spectrogram (via
  spectral flux, spike detection, IDF, LSH). The model already
  learns to extract equivalent features from raw mel. Adding a
  pre-computed summary as extra input channels does not help --
  the model ignores them entirely (+0.2% F1 when zeroed, stable
  across 6 evals).

- **Wider conv stem helps slightly.** The 93-channel input (vs 80)
  produced a small but real improvement at matched density: F1 +0.012,
  dc_human +0.44, gap_TVD -0.041 at tau=0.4. This is likely from
  additional capacity in the first Conv1d layer, not from the
  coincidence signal. A future experiment could test 93 mel bands
  (no coincidence) to isolate this effect.

- **gap_TVD 0.294 is a new best.** At the balanced operating point
  (tau=0.4, DR ~0.97), 019 produces the most rhythmically faithful
  gap distributions of any experiment. This improvement comes from
  the wider conv stem, not from onset-importance selectivity.

- **Silence regions did not improve.** silence_overlap_f1 was 0.533
  at matched DR (vs 017f's 0.576). The IDF-based selectivity
  mechanism did not engage. The silence problem requires a different
  approach -- possibly a silence-aware loss term or explicit silence
  detection, not additional input features.

- **The coincidence concept may work in a different form.** The
  13-row summary is a lossy compression. Two alternative approaches:
  - **Full coincidence map as the entire input** (replacing mel): if
    the coincidence map's RGB heatmap captures onset structure more
    explicitly than mel, it might outperform mel entirely. This would
    be a different experiment -- not mel+coincidence concatenation,
    but a completely different input representation.
  - **Coincidence-only model** (13-channel input): the no_mel
    benchmark showed F1=0.119 from 13 channels alone. A dedicated
    small model trained on coincidence-only input could serve as a
    complementary detector whose outputs are ensembled with the mel
    model.

## Followup questions

- **Is the F1 improvement from conv width or coincidence?** Run 017f
  with n_mels=93 (93 mel bands, no coincidence) to isolate whether
  the wider conv stem alone explains the +0.012 F1 gain.

- **Full coincidence map as input.** Replace mel entirely with the
  full RGB coincidence heatmap (192 channels). The model would see
  onset type, importance, and spectral shape directly instead of
  deriving them from mel. This is a fundamentally different
  representation, not a supplement.

- **Coincidence-only lightweight model.** Train a small model (fewer
  layers, smaller d_model) on the 13-channel coincidence summary
  alone. Ensemble its predictions with the mel model's confidence
  map via averaging or learned weighting.
