# Performance

Headline numbers across taiko2 experiments. Detailed per-eval
progression, predictions vs actuals, and graphs live under each
experiment's own folder.

## Best per metric

All values cited from `runs/exp_*/metrics.jsonl` and
`runs/exp_*/infer_corpus/eval_*/{gt_cond,fixed_cond}/comparisons_summary.json`.

| Metric | Value | Experiment | Source |
|---|---:|---|---|
| val/single/onset/miss (per-step) | **0.2406** | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 372,132, val/single/onset/miss] |
| val/single/onset/hit | 0.7512 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 372,132, val/single/onset/hit] |
| val/single/onset/exact | 0.5748 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 372,132, val/single/onset/exact] |
| val/single/onset/frame_err_p90 | 30 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 372,132, val/single/onset/frame_err_p90] |
| val/single/onset/stop_f1 (run-best) | **0.6152** | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 392,806, val/single/onset/stop_f1] |
| AR `matched_rate` (gt_cond, median) | **0.7061** | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 413,480, infer_corpus/eval_413480/gt_cond/comparisons_summary.json:fields.matched_rate.median] |
| AR `error_median_ms` (gt_cond, median) | **8** | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 248,088 onwards (multiple evals tied), infer_corpus/eval_*/gt_cond/comparisons_summary.json:fields.error_median_ms.median] |
| AR `dc_human` (gt_cond, median) | 92.79 | [#007](experiments/007-time-stretch/) | [exp_007_time_stretch, step 330,784, infer_corpus/eval_330784/gt_cond/comparisons_summary.json:fields.dc_human.median] |
| Per-step `ratio/rhit` (within ±3 % log-ratio) | 0.5332 | [#010e](experiments/010e-aux-frozen/) | [exp_010e_aux_frozen, step 475,502 (E23), val/single/ratio/rhit] |
| Per-step `ratio/rgood` (within ±10 %) | 0.6781 | [#010e](experiments/010e-aux-frozen/) | [exp_010e_aux_frozen, step 475,502 (E23), val/single/ratio/rgood] |
| `ratio/div_acc` (divisor exact, run-best) | 0.7723 | [#010e](experiments/010e-aux-frozen/) | [exp_010e_aux_frozen, step 103,370 (E5, late warmup), val/single/ratio/div_acc] |
| `ratio/off_acc` (offset exact, run-best) | 0.9547 | [#010](experiments/010-ratio-decomposition/) | [exp_010_ratio, step 186,066 (E9), val/single/ratio/off_acc] |

#007 wins per-step (`miss` / `hit` / `exact` / `stop_f1`) **and**
AR generation (`matched_rate` 0.7061 / `error_median` 8 ms /
`dc_human` 92.79). The ratio-decomposition family (#010 → #010e)
introduces ratio-space metrics (`rhit`, `rgood`, `div_acc`,
`off_acc`) that don't apply to direct-bin runs, so #010e's wins
in those rows are the only structural ones — they're not
metrics that exist on #007. **There is no taiko2 metric on
which a non-#007 run beats #007.**

## Plateaus and ceilings (so far)

- **Direct-bin val miss plateaus in the 0.24–0.26 band** across
  the headline direct-bin runs:
  - #007 best miss 0.2406 [exp_007_time_stretch, step 372,132,
    val/single/onset/miss].
  - #008 best miss 0.2572 [exp_008_log_emd, step 248,088,
    val/single/onset/miss].
  #009 (MDN) is **not** in the plateau — best miss 0.3450
  [exp_009_mdn, step 62,022, val/single/onset/miss], roughly 10 pp
  above #007/#008. The `±log(2)` / `±log(3)` ratio-banding
  ridges are present in every direct-bin run — the per-sample
  loss can't discriminate octave / triplet confusions.
- **Ratio-decomposition val miss plateaus in the 0.31–0.33 band**:
  - #010 best 0.3285 [exp_010_ratio, step 144,718].
  - #010b best 0.3255 [exp_010b_ratio_k3, step 248,088].
  - #010c best 0.3260 [exp_010c_ratio_128, step 186,066].
  - #010e best 0.3114 [exp_010e_aux_frozen, step 475,502].
  All `val/single/onset/miss`. The bottleneck is divisor accuracy
  (run-best `val/single/ratio/div_acc` ranges from 0.7235
  [exp_010_ratio, step 186,066] to 0.7723 [exp_010e_aux_frozen,
  step 103,370] across the family — call it the **0.72–0.77
  band**); the multiplicative reconstruction
  `bin = div × ratio − offset` makes any divisor error a
  near-guaranteed bin miss.

- **Strict-frame onset detection from MIR algorithms collapses
  below ±2 frames.** [#011](experiments/011-onset-feature-survey/)
  showed every classical ODF (energy, SF, log-SF, HFC, SuperFlux,
  sub-band SF) hits F1 ≤ 0.086 at ±0 frames against GT (best is
  log_filtered_flux at 0.086) [011-onset-feature-survey/results/summary.json:by_algo.log_filtered_flux.by_tolerance.0.best_f1],
  climbing to F1 = 0.679 at ±10 frames (best is spectral_flux)
  [011-onset-feature-survey/results/summary.json:by_algo.spectral_flux.by_tolerance.10.best_f1].
  Implication: any channel-input augmentation should target the
  ±5 / ±10 frame regime.

## Cross-experiment context

Per-eval comparison tables and side-by-side graphs are scattered
across the post-run sections of #010 → #010e and #007 vs #008.
The most cross-cutting analyses live in:

- [#011 — ODF survey vs GT](experiments/011-onset-feature-survey/)
- [#011b — pairwise ODF disagreement](experiments/011b-onset-disagreement/)

For the up-to-date status of every experiment (planned, running,
complete, abandoned), see
[`experiments/README.md`](experiments/README.md).
