# Experiment 010b — Ratio decomposition with reduced smoothing (k=3, 4ch)

## Status

`Planned`

## Context

[#010](../010-ratio-decomposition/) proved the ratio decomposition
works structurally (div_acc 72%, ratio_rgood 66%, hi_pspace 93.5%)
but plateaued at derived-bin miss 0.33, 8 pp behind #007. Two
issues identified: (1) Conv1d smoothing (k=5, 8ch) over-spreads
ratio predictions into non-musical values, and (2) a floor at ratio
bin ~60 (≈ 0.33×) prevents low-ratio predictions.

This experiment reduces the smoothing to k=3, 4 channels — halving
both the kernel receptive field (from ~8% to ~5% log-ratio range)
and the channel capacity. The prediction should be sharper, more
concentrated at musical-ratio peaks, while still preventing the
single-bin collapse taiko1 exp 67 observed without any smoothing.

## Citations

- Direct parent: [#010 — ratio decomposition](../010-ratio-decomposition/).
- Baseline for metrics: [#007 — time-stretch](../007-time-stretch/).

---

## Hypothesis

### Claim

Reducing Conv1d smoothing from k=5/8ch to k=3/4ch will tighten the
ratio error distribution (less smear between musical-ratio peaks)
and improve derived-bin miss by at least 2 pp vs #010's 0.329
(targeting ≤ 0.31) without causing ratio collapse.

### Mechanism

The Conv1d smoothing on the ratio head correlates neighboring output
bins via a residual `ratio_logits += smooth(ratio_logits)`. With
k=5, each output bin is influenced by its 4 nearest neighbors —
spanning ~8.2% in log-ratio space, nearly the full RGOOD tolerance
(10%). This smears the softmax peak across multiple bins, placing
mass at non-musical ratio values.

Reducing to k=3 halves the receptive field to ~5% log-ratio (~2
neighbors each side). This is tight enough that a peak at 1.0×
doesn't bleed into 1.05× or 0.95×, but wide enough that isolated
single-bin spikes (the taiko1 exp 67 collapse pathology) are still
suppressed. Halving channels from 8 to 4 further reduces the
smoothing capacity — less room for the Conv1d to learn a wide
spread pattern.

The expected outcome: ratio error distribution concentrates more
tightly at musical-ratio peaks (0, ±log 2, ±log 3) with less mass
between them. Each correctly-placed ratio peak is sharper →
multiplicative derivation `divisor × ratio` produces a more precise
bin → derived-bin miss improves.

### Predicted numbers

| Metric | #010 @ E7 | Predicted (#010b) | Notes |
|---|---:|---:|---|
| val miss | 0.329 | ≤ 0.31 | sharper ratios → better precision |
| ratio/rgood | 0.662 | ≥ 0.70 | tighter peaks → more within ±10% |
| ratio/rhit | 0.498 | ≥ 0.55 | sharper → more within ±3% |
| ratio collapse | no | still no | k=3 should still prevent spikes |

## Success criteria

- **Must have:** ratio/rgood ≥ 0.68 (improves on #010's 0.662).
- **Must have:** ratio error distribution shows sharper peaks at
  musical ratios (0, ±log 2, ±log 3) with less mass between them.
- **Must have:** no ratio collapse (<10 unique values).
- **Nice-to-have:** val miss ≤ 0.31.
- **Fails if:** ratio collapse to <10 values (smoothing too weak).
- **Fails if:** val miss > 0.35 (worse than #010).

## Changes from baseline

Baseline: [#010](../010-ratio-decomposition/).

- `config/model.json` — `ratio_smooth_kernel: 5 → 3`,
  `ratio_smooth_channels: 8 → 4`. Everything else identical.

## Run config

- Run name: `exp_010b_ratio_k3`.
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_010b_ratio_k3 \
      --config-dir osu/taiko2/experiments/010b-ratio-smooth-k3/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --cursor-shift-prob 0.3 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/010b-ratio-smooth-k3/config/infer.json \
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
