# Experiment 021 — Wider conv stem

## Status

`Planned`

## Context

[#019](../019-coincidence-input/) showed the model ignores coincidence
channels but produced a small F1 improvement at matched density:
F1 +0.012, gap_TVD -0.041 at tau=0.4
[019 threshold_sweep.json, eval_143990/tau=0.4]. The most likely
explanation is the wider Conv stem (93 vs 80 input channels) providing
more capacity in the first convolution, not the coincidence signal.

This experiment tests that hypothesis directly: widen the Conv stem
intermediate channels from 192 to 256 while keeping the input at 80
mel bands. If the improvement reproduces, the gain is from stem
capacity. If not, 019's gain was specific to the 93-channel input
layout (e.g., the coincidence rows acting as additional learnable
dimensions even though the model "ignored" them per the benchmark).

The conv stem is currently:
```
Conv1d(80, 192, k=7, s=2) -> GroupNorm -> Conv1d(192, 384, k=7, s=2)
```
This experiment changes it to:
```
Conv1d(80, 256, k=7, s=2) -> GroupNorm -> Conv1d(256, 384, k=7, s=2)
```

Cost: +208K parameters (22.1M vs 21.9M, +0.9%).

## Citations

- Direct baseline:
  - [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).
    Best sweep tau=0.4 (eval_248088): `f1` 0.771, `precision` 0.778,
    `recall` 0.757, `density_ratio` 0.964, `dc_human` 92.4,
    `gap_hist_tvd` 0.331, `silence_overlap_f1` 0.527
    [017f threshold_sweep.json].
- Motivation:
  - [#019 -- coincidence input](../019-coincidence-input/).
    At matched DR (tau=0.4): F1 0.781 (+0.012 vs 017f), gap_TVD 0.294
    (-0.041). Model ignored coincidence channels (no_coincidence
    benchmark +0.2%). Gain attributed to wider conv stem (93 vs 80
    input channels) [019 threshold_sweep.json].
- Interpretability:
  - [#020 -- activation maximization](../020-activation-maximization/).
    High-band saliency dominance suggests the conv stem is where
    onset-relevant spectral features are extracted.
- Implementation: `models/common.py:AudioConvStem`, new `stem_width`
  config field on `EventEmbeddingConfig`.

---
<!--
PRE-RUN. Do not edit after the run.
-->
---------------------------------------------------------------------

## Hypothesis

### Claim

Widening the Conv stem intermediate channels from 192 to 256
(`stem_width: 256`) will reproduce the F1 and gap_TVD improvements
seen in #019 (F1 +0.012, gap_TVD -0.041 at matched density), because
019's gain came from conv capacity, not from the coincidence signal.

### Mechanism

The first Conv1d layer mixes mel bands into spectral features. With
192 intermediate channels, it can learn 192 spectral combinations of
the 80 mel bands. Widening to 256 adds 64 more combinations (+33%),
giving the model more expressive power to capture the high-frequency
transient patterns that #020's saliency showed are important for onset
detection. #019 accidentally tested a similar widening (93 input
channels -> 192 intermediate channels meant 93 spectral mixtures in
the first conv; the wider input effectively increased the first
layer's capacity).

### Predicted numbers

Reference: [#017f](../017f-framewise-bce-metrics-rerun/) tau=0.4
sweep [017f threshold_sweep.json, eval_248088] and
[#019](../019-coincidence-input/) tau=0.4
[019 threshold_sweep.json, eval_143990].

| Metric | #017f (tau=0.4) | #019 (tau=0.4) | Predicted (#021) | Notes |
|---|---:|---:|---:|---|
| AR `f1` (25ms) | 0.771 | 0.781 | **>= 0.775** | Between 017f and 019 |
| AR `precision` | 0.778 | 0.776 | **~0.78** | Should hold |
| AR `recall` | 0.757 | 0.769 | **~0.76** | Should hold |
| `density_ratio` | 0.964 | 0.969 | **0.93-1.03** | Near 1.0 at tau=0.4 |
| `dc_human` | 92.4 | 92.9 | **>= 92.4** | Should hold or improve |
| `gap_hist_tvd` | 0.331 | 0.294 | **< 0.32** | Better rhythmic structure |
| `silence_overlap_f1` | 0.527 | 0.533 | **~0.53** | Not expected to change |
| frame F1 | 0.822 | 0.816 | **>= 0.82** | Should match 017f |
| fps50 F1 | 0.741 | 0.731 | **>= 0.73** | Should match 017f |

## Success criteria

- **Must:** frame F1 >= 0.81. The wider stem must not degrade frame
  quality.
- **Must:** AR F1 at tau=0.4 >= 0.77 (matches or exceeds 017f).
- **Confirms hypothesis if:** gap_TVD < 0.32 at matched density,
  reproducing 019's improvement range.
- **Fails if:** frame F1 < 0.80 or AR F1 < 0.75 -- the wider stem
  hurt rather than helped.
- **Rejects hypothesis if:** gap_TVD >= 0.33 (no improvement over
  017f) -- would mean 019's gain was not from conv width.
- **Nice-to-have:** F1 improvement >= 0.01 over 017f at tau=0.4.

## Changes from baseline

Baseline: [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).

One change:

- `config/model.json` -- `stem_width: 256` (was 0, defaulting to
  d_model // 2 = 192). Adds 208K params (+0.9%).

All other configs identical to #017f. decode_threshold set to 0.4
in infer.json (017f's sweep-optimal threshold) for direct AR
comparison during training.

Code change:
- `models/common.py:AudioConvStem` -- added `stem_width` parameter
  (default 0 = d_model // 2, backward compatible).
- `models/event_embedding.py:EventEmbeddingConfig` -- added
  `stem_width: int = 0` field.

## Run config

- Run name: `exp_021_wider_stem`
- Config snapshots: [`config/`](./config/)
- Dataset: `taiko2_v1`, split `train` / `val`
- Total params: ~22.10 M (vs 017f's 21.89 M)
- Command:
  ```bash
  set -e CUDA_VISIBLE_DEVICES && ulimit -n 65536 && \
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_021_wider_stem \
      --config-dir osu/taiko2/experiments/021-wider-stem/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/021-wider-stem/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

---------------------------------------------------------------------
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
---------------------------------------------------------------------

## Results summary

<!-- TODO: fill after run -->

## Visualizations

<!-- TODO: fill after run -->

## Vs prediction

<!-- TODO: fill after run -->

## Takeaways

<!-- TODO: fill after run -->
