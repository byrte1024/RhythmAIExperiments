# Experiment 010c — Ratio decomposition with 128 bins (half resolution)

## Status

`Planned`

## Context

[#010](../010-ratio-decomposition/) and
[#010b](../010b-ratio-smooth-k3/) both plateaued at miss ≈ 0.33 and
rgood ≈ 0.66 with 255 ratio bins. Conv1d smoothing kernel size (k=5
vs k=3) didn't change the ceiling. The hypothesis: 255 bins is too
fine-grained — each bin is ~1.65% in log-ratio space, the model
spreads predictions across many near-identical bins instead of
committing to specific musical ratios. Halving to 128 bins doubles
the per-bin training signal and makes each bin ~3.3% wide — still
finer than the 10% RGOOD tolerance but coarser enough that the model
should commit more confidently.

## Citations

- Direct parent: [#010](../010-ratio-decomposition/) and
  [#010b](../010b-ratio-smooth-k3/) — both plateaued at same ceiling.
- Baseline for metrics: [#007 — time-stretch](../007-time-stretch/).

---

## Hypothesis

### Claim

Halving ratio bins from 255 to 128 (same 0.125×–8.0× range) will
improve ratio precision (rgood ≥ 0.70, rhit ≥ 0.55) and lower
derived-bin miss below 0.31, breaking the plateau #010/#010b hit.

### Mechanism

At 255 bins, each bin spans ~1.65% in log-ratio space — much finer
than the model's natural prediction precision. The model distributes
mass across 3–5 adjacent bins for each prediction, and the Conv1d
smoothing further correlates them. The result: the argmax bounces
between near-identical bins, adding noise to the ratio × divisor
multiplication.

At 128 bins, each bin spans ~3.3% — close to the RHIT tolerance
(3%) and well within RGOOD (10%). The model can commit to ONE bin
per musical ratio instead of spreading across several. The argmax
is more stable, the multiplication is less noisy, and the derived
bin is more precise.

Additionally, 128 bins means 2× more training samples per bin for
the same dataset. Rare ratios (0.25×, 3.0×, etc.) that barely got
gradient at 255 bins may now have enough signal to learn.

### Predicted numbers

| Metric | #010 best | Predicted (#010c) | Notes |
|---|---:|---:|---|
| miss | 0.329 | ≤ 0.31 | break the 0.33 plateau |
| r_rgood | 0.662 | ≥ 0.70 | sharper per-bin commits |
| r_rhit | 0.498 | ≥ 0.55 | more concentrated peaks |
| div_acc | 0.717 | ≥ 0.70 | should be unaffected |

## Success criteria

- **Must have:** r_rgood ≥ 0.68 (improves on #010's 0.662).
- **Must have:** derived-bin miss ≤ 0.32 (below #010's plateau).
- **Nice-to-have:** miss ≤ 0.30.
- **Nice-to-have:** ratio floor at bin ~30 (= former bin 60 at 255)
  is reduced — more low-ratio predictions appear.
- **Fails if:** miss > 0.35 (worse than #010).
- **Fails if:** rgood < 0.60 (coarser bins hurt instead of help).

## Changes from baseline

Baseline: [#010](../010-ratio-decomposition/).

- `config/model.json` — `ratio_bins: 255 → 128`. Same 0.125×–8.0×
  range, half the bins. Each bin now ~3.3% wide (21 bins per octave
  vs 42). Conv1d smoothing stays at k=5, 8ch (#010's default —
  #010b showed changing it doesn't help).
- `config/loss.json` — `ratio_bins: 255 → 128`. Ratio head output
  = 128 + 1 (STOP) = 129 classes.
- `config/infer.json` — decoder `ratio_bins: 255 → 128`.
- Output tensor width: 500 + 100 + 129 = **729** (was 856).
- Everything else identical: backbone, divisor/offset heads,
  augmentations, schedule, seed.

## Run config

- Run name: `exp_010c_ratio_128`.
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_010c_ratio_128 \
      --config-dir osu/taiko2/experiments/010c-ratio-128bins/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --cursor-shift-prob 0.3 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/010c-ratio-128bins/config/infer.json \
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
