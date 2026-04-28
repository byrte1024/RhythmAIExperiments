# Experiment 009b — MDN with capped sigma (max_sigma=3.0)

## Status

`Planned`

## Context

[#009](../009-mdn/) confirmed the MDN split IS real: components 0
and 2 specialized (sigma 1.7 and 2.0, pi 0.28 and 0.55) while
component 1 inflated to sigma=48 and became a "junk drawer"
absorbing uncertainty. Headline metrics suffered (miss +11 pp vs
#007) primarily because of the inflated component.

This experiment caps sigma at 3.0 bins so no component can inflate.
All three must place mu precisely — either specializing into
different prediction regions or collapsing together. The question:
with sigma capped, does component 1 learn something useful (e.g.
octave predictions) instead of going wide?

## Citations

- Direct parent: [#009 — MDN](../009-mdn/). Component analysis
  showed 2 of 3 components specializing, 1 inflating.
- Baseline for metrics: [#007 — time-stretch](../007-time-stretch/).

---

## Hypothesis

### Claim

With `max_sigma=3.0` all three MDN components will produce sharp
peaks. At least two components will show distinct specialization in
the per-component heatmaps (e.g. one on the diagonal, one on an
octave line). Coverage_2bin will exceed 0.85. Headline miss will
be within 5 pp of #007 (≤ 0.29).

### Predicted numbers

| Metric | #009 @ E3 | Predicted (#009b) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.3450 | ≤ 0.29 | sigma cap → sharper mu |
| val/single/onset/exact | 0.2779 | ≥ 0.35 | sharp sigma → better bin precision |
| mdn/coverage_2bin | 0.7948 | ≥ 0.85 | components forced to be precise |
| mdn/n_active_components | 2.29 | ≥ 2.0 | should stay multi-modal |
| mdn/mean_sigma | 17.3 | ≤ 3.0 | by construction |
| comp heatmap specialization | 2 of 3 | 3 of 3 | no junk drawer |

## Success criteria

- **Must have:** all 3 per-component heatmaps show distinct structure
  (no component is a featureless blob).
- **Must have:** `mdn/n_active_components` ≥ 1.5 (not collapsed).
- **Must have:** `mdn/coverage_2bin` ≥ 0.80.
- **Nice-to-have:** miss ≤ 0.27 (within 3 pp of #007).
- **Nice-to-have:** per-component ratio_error heatmaps show one
  "clean diagonal" component and at least one "octave band"
  component — mapping the ambiguity structure.
- **Fails if:** all 3 components collapse to identical positions
  (n_active ≈ 1).
- **Fails if:** miss > 0.35 (worse than uncapped #009).

## Changes from baseline

Baseline: [#009](../009-mdn/).

- `config/loss.json` — add `max_sigma: 3.0` (was uncapped).
  Sigma is clamped to `[1.0, 3.0]` after softplus. Everything
  else identical: same model head (K=3, output dim 10), same
  augmentations, same dataset.

## Run config

- Run name: `exp_009b_mdn_capped`.
- Config snapshots: [`config/`](./config/).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_009b_mdn_capped \
      --config-dir osu/taiko2/experiments/009b-mdn-capped-sigma/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/009b-mdn-capped-sigma/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

---
<!-- Post-run below -->
---

## Results summary

_(To fill post-run.)_

## Visualizations

_(Post-run.)_

## Vs prediction

_(Post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
