# Experiment 019 — Coincidence input (mel + 13-row coincidence summary)

## Status

`Planned`

## Context

[#017e](../017e-framewise-bce-regularized/) is the strongest framewise
model to date. At the optimal threshold (E8/tau=0.40) it achieves
`matched_rate` 0.783, `density_ratio` 1.020, `dc_human` 92.7, and
`error_median_ms` 10.3
[exp_017e_framewise_bce_regularized, threshold_sweep.json].

The remaining gap to ideal is `hallucination_rate` 0.201. Post-run
analysis in [#017e](../017e-framewise-bce-regularized/) identified the
root cause: the model detects real audio onsets but cannot distinguish
chart-author-selected ones from metronomic beats. The model's input —
an 80-band log-mel spectrogram — carries energy and frequency content,
but not onset *type* or onset *importance*. A regular snare hit and a
chart-worthy accent produce similar mel activations if their energy is
similar. There is no signal in the 80-channel input that encodes "this
onset is unusual" or "this is the expected beat, not a highlight".

The coincidence map (implemented in `domain/coincidence.py`) produces
a 13-row summary that encodes exactly these signals:

- **Onset type (LSH color, rows 0-2):** Each onset is assigned an RGB
  color by locality-sensitive hashing of its spectral fingerprint.
  Metronomic beats repeat their color; unexpected onsets have rare
  colors. A model that sees the color sequence can learn which colors
  are "on-grid" vs "off-grid".
- **Onset importance (IDF weighting, row 3):** The IDF term measures how
  unusual each onset's spectral profile is across the full track corpus.
  Common, expected beats have low IDF (low importance); rare, accented
  beats have high IDF (high importance). This is the most direct signal
  for distinguishing chart-worthy onsets from filler.
- **Spike energy (row 4):** Magnitude of the onset spike, independent of
  IDF. Complements IDF by separating loud-but-common from
  loud-but-rare.
- **Band-group averages (rows 5-12):** Eight frequency-band averages
  summarize the spectral shape of each onset. Encodes whether the
  energy is bass-heavy, treble-heavy, or broadband.

Concatenating these 13 rows with the 80 mel rows gives the model
explicit "how unusual is this onset" information that the mel alone
cannot provide. The model's input grows from (B, 80, 1000) to
(B, 93, 1000); the only architectural change is widening the Conv stem
input from 80 to 93.

## Citations

- Direct baseline:
  - [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).
    Best sweep: E8 (step 165,392), tau=0.40, `matched_rate` 0.783,
    `hallucination_rate` 0.201, `density_ratio` 1.020, `dc_human` 92.7,
    `error_median_ms` 10.3
    [exp_017e_framewise_bce_regularized, threshold_sweep.json].
    Best F1: E11 (step 227,414), F1 0.827, Precision 0.882, Recall 0.779.
- Related priors:
  - [#017d -- framewise BCE noweight](../017d-framewise-bce-noweight/).
    `matched_rate` 0.742 at E9/tau=0.3
    [exp_017d_framewise_bce_noweight, threshold_sweep.json].
  - [#007 -- TimeStretch](../007-time-stretch/). `matched_rate` 0.703,
    `hallucination_rate` 0.172, `density_ratio` 0.865, `dc_human` 92.0
    [exp_007_time_stretch, step 413,480].
- Coincidence feature implementation: `domain/coincidence.py`.
- Audio sampler: `samplers/coincidence_mel.py`
  (`CoincidenceMelSampler` — produces ``(93, T)`` on disk and at
  inference time).

---
<!--
PRE-RUN. Do not edit after the run.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

Adding the 13-row coincidence summary as a parallel input alongside mel
(n_mels 80 -> 93) will give the model access to onset type and importance
signals that the mel alone does not carry. The model will learn to use
these channels — demonstrated by a >= 3 % F1 drop in the `no_coincidence`
benchmark (rows 80-92 zeroed) — and this selectivity improvement will
push `hallucination_rate` below 0.18.

### Mechanism

The halluc_rate ceiling at ~0.20 in the 017 family is caused by the
model responding to all acoustically salient onsets rather than only
chart-author-selected ones. The mel spectrogram is agnostic to onset
importance: a metronomic beat and a chart-worthy accent look similar if
their frequency content and energy are similar. The coincidence map's IDF
row (row 3) directly encodes "how unusual is this onset across the corpus"
— low IDF for expected beats, high IDF for rare/important ones. If the
model learns to weight its prediction by IDF, it will suppress FPs on
common/expected beats and reduce hallucination. The LSH color rows (0-2)
provide complementary repetition-pattern information: metronomic beats
have consistent colors across the window, while chart-highlights break
the pattern.

### Predicted numbers

Reference: [#017e](../017e-framewise-bce-regularized/) E8/tau=0.40 sweep
[exp_017e_framewise_bce_regularized, threshold_sweep.json] and
[#007](../007-time-stretch/) step 413,480.

| Metric | #017e sweep | #007 | Predicted (#019) | Notes |
|---|---:|---:|---:|---|
| AR `matched_rate` | 0.783 | 0.703 | **>= 0.75** | should hold or improve |
| AR `hallucination_rate` | 0.201 | 0.172 | **< 0.18** | IDF selectivity helps |
| AR `density_ratio` | 1.020 | 0.865 | **0.90-1.10** | near-1.0 maintained |
| AR `dc_human` (%) | 92.7 | 92.0 | **>= 92** | pattern quality maintained |
| frame F1 (best eval) | 0.827 | n/a | **>= 0.82** | at least matches 017e |
| `no_coincidence` F1 delta | n/a | n/a | **>= -3 %** (drop) | model uses coincidence |

## Success criteria

- **Must:** `no_coincidence` benchmark shows >= 3 % F1 drop vs full model
  (rows 80-92 zeroed). This confirms the model actually uses the
  coincidence channels. Without this, the coincidence input is wasted
  and the experiment is inconclusive.
- **Must:** `matched_rate` >= 0.75 at best sweep threshold. The coincidence
  input must not degrade the headline quality metric below 017e's level.
- **Fails if:** model ignores coincidence — `no_coincidence` F1 delta < 1 %.
  This would mean the model routes around the extra channels and the
  hypothesis is falsified.
- **Fails if:** `matched_rate` < 0.65 at every threshold — widening the
  conv stem disrupted learning.
- **Nice-to-have:** `hallucination_rate` < 0.18 (below #007's 0.172
  [exp_007_time_stretch, step 413,480]).
- **Nice-to-have:** `no_mel` benchmark (rows 0-79 zeroed) shows the model
  cannot succeed on coincidence alone (F1 drop >= 20 pp vs full model),
  confirming mel and coincidence are complementary.

## Changes from baseline

Baseline: [#017e -- framewise BCE regularized](../017e-framewise-bce-regularized/).

Three changes:

- **New dataset `taiko2_v1_coin`** — built with
  `CoincidenceMelSampler` (``--audio-sampler coincidence_mel``).
  Features on disk are ``(93, T)`` float16 — the standard 80-band
  mel concatenated with the 13-row coincidence summary. Both the
  data sampler and the inference sampler consume this format
  natively with no special flags.
- `config/model.json` — `n_mels: 80 -> 93`. Widens the Conv stem
  input from 80 to 93 channels. All other model params unchanged.
- `config/infer.json` — `audio_sampler` changed from `MelSampler`
  to `CoincidenceMelSampler` so live inference also produces
  ``(93, T)`` features from audio.

All other configs (loss.json, trainer.json, adapter.json, data.json)
are identical to #017e except the checkpoint path and
decode_threshold in infer.json (0.4, the #017e optimal threshold).

Two new benchmarks added vs #017e:
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

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

<!-- TODO: fill after run -->

## Visualizations

<!-- TODO: fill after run -->

## Vs prediction

<!-- TODO: fill after run -->

## Takeaways

<!-- TODO: fill after run -->
