# Experiment 017c — Framewise BCE with low pos_weight

## Status

`Planned`

## Context

[#017](../017-framewise-bce/) proved the framewise framing matches
[#007](../007-time-stretch/) on pattern quality (`dc_human` 91.0 vs
92.0) but over-emits ~44% extra notes (`density_ratio` 1.44,
`hallucination_rate` 0.32). [#017b](../017b-framewise-focal/) showed
focal loss is not the solution — it compressed confidences without
improving selectivity.

Analysis of the dataset reveals the pos_weight mechanism as the likely
cause. With `pos_weight_clamp = [10, 200]`, the typical training
window (11 GT onsets / 500 bins) gives `pos_weight = 44.5` — each
missed positive costs **44x** a false positive. The model rationally
responds by firing at every plausible audio onset to avoid the
asymmetric penalty.

The scalar model ([#007](../007-time-stretch/)) uses softmax-CE where
all 501 bins compete in a single distribution — no explicit positive
weighting, and the competition naturally enforces selectivity. The
framewise model's per-bin BCE has no such competition mechanism, but
lowering pos_weight dramatically reduces the incentive to over-emit.

Dataset statistics (sampled 89,753 windows from taiko2_v1):
- GT=1 bins: 2.54 % (38.4:1 negative-to-positive ratio)
- GT onsets per window: median 11, p5=3, p95=27
- At median (11 onsets), raw pos_weight = 44.5

## Citations

- Direct baseline:
  - [#017 -- framewise BCE](../017-framewise-bce/). `pos_weight_clamp
    = [10, 200]`. Best `density_ratio` 1.44, `dc_human` 91.0,
    `hallucination_rate` 0.32 [exp_017_framewise_bce, step 82,696].
  - [#017b -- framewise focal](../017b-framewise-focal/). Showed
    focal loss compresses confidences without fixing selectivity.
    `density_ratio` stuck at 2.35 [exp_017b_framewise_focal, step
    82,696].
  - [#007 -- TimeStretch](../007-time-stretch/). `density_ratio`
    0.87, `dc_human` 92.0, `hallucination_rate` 0.17
    [exp_007_time_stretch, step 413,480].
- Cross-experiment record: [`../README.md`](../README.md).

---
<!--
PRE-RUN. Do not edit after the run.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

If `pos_weight_clamp` is reduced from [10, 200] to [3, 8] on the
otherwise-identical #017 architecture, then **AR `density_ratio`
will reach 0.80-1.15** and **`hallucination_rate` will drop below
0.20**, because the model will no longer have a 44x incentive to
fire at every audio onset and will instead learn to be selective
about which beats receive notes.

### Mechanism

With `pos_weight_clamp = [3, 8]`, the typical window (11 GT onsets)
gets pos_weight = 8 (clamped from raw 44.5). A missed onset now
costs only 8 FPs instead of 44 — the model can tolerate a few
missed onsets in exchange for much cleaner output. At the p5 window
(3 onsets, raw 165), the weight clamps to 8 instead of 165 — sparse
charts no longer get runaway recall bias.

The floor of 3 ensures even dense charts (27+ onsets, raw weight
< 18) still get moderate recall pressure, preventing collapse to
all-zeros.

### Predicted numbers

Reference: [#017](../017-framewise-bce/) best (E4) and
[#007](../007-time-stretch/) best (E18).

| Metric | #017 (E4) | #007 | Predicted (#017c) | Notes |
|---|---:|---:|---:|---|
| AR `density_ratio` | 1.44 | 0.87 | **0.80-1.15** must | primary target |
| AR `hallucination_rate` | 0.32 | 0.17 | **<= 0.20** must | consequence of density fix |
| AR `dc_human` | 91.0 | 92.0 | **>= 88** must | may regress if recall drops too much |
| AR `matched_rate` | 0.90 | 0.70 | **0.65-0.80** | expected to drop as over-emission drops |
| AR `error_median_ms` | 6.1 | 10.2 | **<= 12** | timing should hold |
| Recall | 0.990 | n/a | **>= 0.90** | may trade some recall for precision |
| Precision | 0.641 | n/a | **>= 0.70** | should improve with fewer FPs |
| frame F1 | 0.778 | n/a | **>= 0.78** | precision gain + recall loss should net ≈ neutral |
| `conf_fp_median` | 0.793 | n/a | **<= 0.60** | FPs should be less confident with lower reward |
| mini tau50 density_ratio | 4.70 | n/a | **<= 2.5** | raw per-window over-emission |

## Success criteria

- **Must have:** AR `density_ratio` in [0.60, 1.30] at the best eval
  -- over-emission materially reduced vs #017's 1.44.
- **Must have:** AR `dc_human` >= 85 -- pattern quality does not
  regress catastrophically.
- **Must have:** Recall >= 0.85 at any post-warmup eval -- the model
  did not collapse to all-zeros under weak positive pressure.
- **Nice-to-have:** AR `hallucination_rate` <= 0.15 -- below #007.
- **Nice-to-have:** AR `density_ratio` in [0.85, 1.05] -- near-perfect
  density matching.
- **Fails if:** AR `density_ratio` > 1.40 at every eval -- low
  pos_weight did not reduce over-emission (the problem is
  architectural, not loss-weight-driven).
- **Fails if:** Recall < 0.70 at every eval -- pos_weight too low,
  model collapsed.
- **Fails if:** AR `dc_human` < 75 -- catastrophic pattern regression.

## Changes from baseline

Baseline: [#017 -- framewise BCE](../017-framewise-bce/).

**Single change:** `loss.json` `pos_weight_clamp` from [10, 200] to
[3, 8]. All other configs byte-identical.

No code changes — the existing `FramewiseBCELoss` already supports
arbitrary clamp values.

Config snapshots ([`config/`](./config/)):

- `config/model.json` -- byte-identical to #017.
- `config/loss.json` -- `pos_weight_clamp_min: 3.0`,
  `pos_weight_clamp_max: 8.0`.
- `config/adapter.json` -- byte-identical to #017.
- `config/data.json` -- byte-identical to #017.
- `config/trainer.json` -- byte-identical to #017.
- `config/infer.json` -- byte-identical to #017 except checkpoint.

## Run config

- Run name: `exp_017c_framewise_bce_lowweight`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`.
- Command:
  ```bash
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_017c_framewise_bce_lowweight \
      --config-dir osu/taiko2/experiments/017c-framewise-bce-lowweight/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/017c-framewise-bce-lowweight/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

Future note: a potential architectural approach would make bins
compete directly (e.g. softmax-over-windows, energy-based output,
or a learned NMS layer). Deferred to a later experiment.

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

### Final vs baseline

| Metric | Baseline (exp N) | This run (final) | Delta | Direction |
|---|---:|---:|---:|:---:|
| — | — | — | — | — |

### Per-eval progression

Machine-readable copies: [`metrics.json`](./metrics.json).

## Visualizations

![Training loss](graphs/01_train_loss.png)
*Training loss over steps (log-y).*

## Vs prediction

## Takeaways

## Followup questions
