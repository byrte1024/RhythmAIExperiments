# Performance

Headline numbers across taiko2 experiments. Detailed per-eval
progression, predictions vs actuals, and graphs live under each
experiment's own folder.

## Best per metric

| Metric | Value | Experiment | Step / eval |
|---|---:|---|---|
| val/single/onset/miss (per-step) | **0.241** | [#007](experiments/007-time-stretch/) | E18 / 372,132 |
| val/single/onset/hit | 0.733 | [#007](experiments/007-time-stretch/) | E18 / 372,132 |
| val/single/onset/exact | 0.557 | [#007](experiments/007-time-stretch/) | E18 / 372,132 |
| val/single/onset/frame_err_p90 | 30 | [#007](experiments/007-time-stretch/) | E18 / 372,132 |
| val/single/onset/stop_f1 | 0.583 | [#007](experiments/007-time-stretch/) | E18 / 372,132 |
| AR `matched_rate` (gt_cond, median) | **0.612** | [#010e](experiments/010e-aux-frozen/) | E13 / 268,762 |
| AR `error_median_ms` | **12** | [#010e](experiments/010e-aux-frozen/) | E28 / 578,872 |
| AR `dc_human_pct` | 92.1 | [#010e](experiments/010e-aux-frozen/) | E28 / 578,872 |
| Per-step `ratio/rhit` (within ±3 % log-ratio) | 0.533 | [#010e](experiments/010e-aux-frozen/) | E23 |
| Per-step `ratio/rgood` (within ±10 %) | 0.678 | [#010e](experiments/010e-aux-frozen/) | E23 |
| `ratio/div_acc` (divisor exact) | 0.766 | [#010e](experiments/010e-aux-frozen/) | E3 (warmup) |
| `ratio/off_acc` (offset exact) | 0.951 | [#010](experiments/010-ratio-decomposition/) | E3 |

The val miss (0.241) and AR generation (`matched_rate` 0.612,
`error_median` 12 ms) winners are different experiments. They
trade off — direct-bin prediction (#007) wins per-step accuracy;
ratio decomposition with frozen aux heads (#010e) wins cumulative
AR quality. See each experiment's takeaways for the mechanism.

## Plateaus and ceilings (so far)

- **Direct-bin val miss plateaus at ≈ 0.24** across #007, #008,
  #009. The `±log(2)` / `±log(3)` ratio-banding ridges are
  present in every direct-bin run — the per-sample loss can't
  discriminate octave / triplet confusions.
- **Ratio-decomposition val miss plateaus at ≈ 0.33** across #010
  / #010b / #010c / #010e. The bottleneck is divisor accuracy
  (capped at ~0.78 across all variants); the multiplicative
  reconstruction `bin = div × ratio − offset` makes any divisor
  error a near-guaranteed bin miss.
- **Strict-frame onset detection from MIR algorithms collapses
  below ±2 frames.** [#011](experiments/011-onset-feature-survey/)
  showed every classical ODF (energy, SF, log-SF, HFC, SuperFlux,
  sub-band SF) hits F1 < 0.09 at ±0 frames against GT, climbing
  to F1 ≈ 0.68 at ±10 frames. Implication: any channel-input
  augmentation should target the ±5 / ±10 frame regime.

## Cross-experiment context

Per-eval comparison tables and side-by-side graphs are scattered
across the post-run sections of #010 → #010e and #007 vs #008.
The most cross-cutting analyses live in:

- [#011 — ODF survey vs GT](experiments/011-onset-feature-survey/)
- [#011b — pairwise ODF disagreement](experiments/011b-onset-disagreement/)

For the up-to-date status of every experiment (planned, running,
complete, abandoned), see
[`experiments/README.md`](experiments/README.md).
