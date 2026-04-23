# Experiment 005 — Gaussian soft-target CE with binary STOP

## Status

`Planned`

## Context

[#002](../002-exp45-full/) ported taiko1 exp 45's mixed hard/trapezoid-
soft CE verbatim. It reproduces the baseline numbers (HIT 72.96 % @ E8,
MISS 26.06 % @ E11, both inside the ±1.5 pp tolerance band) but leaves
two persistent failure modes visible in the run artifacts:

- **Ratio-error banding** at `±log 2` and `±log 3` (graph 08) — the
  classic octave / triplet confusions. The trapezoid target has a
  hard plateau in log-ratio space with a 2-frame floor: it does not
  distinguish "3 frames off" from "50 frames off" in the tail.
- **Frame-error tail flat** at p90 = 30–33 bins across every eval,
  while median frame error stayed at 0.

The trapezoid mixes two forgiveness geometries (log-ratio plateau +
linear frame floor) and stacks them with a hard-CE term and a STOP
per-sample reweighting. A Gaussian soft target over bins is a
drastically simpler alternative that falls off smoothly with frame
distance. This experiment isolates that substitution — same everything
else — to measure the loss's independent contribution.

## Citations

- Baseline: [#002 — exp 45 full recreation](../002-exp45-full/).
  Final watched metric `val/single/onset/miss = 0.2606` at eval 11
  (step 227,414).
- Loss-family precedent: Gaussian soft targets are standard in onset-
  detection MIR papers (e.g. Schlüter & Böck 2014 onset networks use
  a ~3-frame Gaussian target window around annotated onsets) for
  exactly the smoothness property we want here.

---

## Hypothesis

### Claim

If we replace #002's mixed hard/trapezoid-soft CE with a single-knob
Gaussian soft-target CE (`sigma_bins = 2.0`) and treat STOP as a
separate binary task at the loss level (binary BCE on the STOP logit +
softmax Gaussian CE over the 500 non-STOP bins), keeping everything
else identical, the watched metric `val/single/onset/miss` will land
**within ±2 pp of #002's 0.2606** at its best eval, because the
Gaussian's smooth falloff preserves the near-target partial credit the
trapezoid plateau provides (σ=2 puts ~68 % mass within ±2 bins, ~95 %
within ±4) while removing four tunable hyperparameters (`hard_alpha`,
`good_pct`, `fail_pct`, `frame_tolerance`) that #002 never ablated.

### Mechanism

Two independent effects:

1. **Loss-shape change (bins).** Trapezoid plateau `[0, 2]` gives
   every bin within ±2 frames full credit; ramp in log-ratio space
   gives linearly decaying credit; past the ramp cutoff the target
   is exactly 0. Gaussian (σ=2) has a single peak at the target,
   decays as `exp(-0.5 · (d/2)²)`, and never hits exactly 0 — so the
   tail of very-wrong predictions still contributes gradient. This
   should pull the frame-error p90 tail inwards without sacrificing
   bin-precision near the target.
2. **STOP decoupling.** In #002, STOP shares a 501-way softmax with
   the 500 bin classes and carries a `stop_weight = 1.5` per-sample
   multiplier. In #005, STOP is routed through a separate sigmoid
   BCE on logit[500] while the 500 bin logits go through softmax +
   Gaussian soft target. STOP never steals or donates soft-mass
   to/from nearby bins. The STOP-calibration volatility #002 showed
   at evals 8 and 10 (precision/recall flipping) should either
   improve or at minimum not regress.

Everything else (model, data, augmentations, schedule, optimizer,
seed) is identical to #002. Any delta is attributable to the loss
change.

### Predicted numbers

Reference: #002 @ its best eval (E11, step 227,414):

| Metric | #002 @ E11 | Predicted (this run, best eval) | Notes |
|---|---:|---:|---|
| val/single/onset/miss           | 0.2606 | 0.241–0.281 | watched metric, ±2 pp |
| val/single/onset/hit            | 0.7292 | 0.710–0.750 | paired, ±2 pp |
| val/single/onset/exact          | 0.5485 | ≥ 0.52      | near-target precision should survive |
| val/single/onset/frame_err_p90  | 31     | ≤ 31        | tail should tighten, not widen |
| val/single/onset/pred_stop_rate | 0.0019 | ≤ 0.01      | binary STOP BCE should not blow up false STOPs |
| val/single/onset/stop_f1        | 0.599  | ≥ 0.55      | decoupled head should match, ideally exceed |

Not predicted: the ratio-error banding ridges at `±log 2` / `±log 3`
in graph 08. Observational — the Gaussian alone may or may not touch
them; the banding is partly a pattern-level phenomenon orthogonal to
per-sample loss shape.

## Success criteria

- **Must have:** final `val/single/onset/miss` within ±2 pp of #002's
  0.2606 (i.e. 0.241–0.281).
- **Must have:** training runs to completion without NaN / Inf / OOM;
  all artifacts (heatmap, distributions, ratio_error, error_hist,
  ratio_hit, metronome, stop-derived curves) write every eval.
- **Must have:** `pred_stop_rate` stays under 0.01.
- **Nice-to-have:** `frame_err_p90` improves (lower) vs #002's 31.
- **Nice-to-have:** matches #002's HIT at its best eval.
- **Fails if:** final miss above 0.30 (loss change actively hurt).
- **Fails if:** `pred_stop_rate` above 0.05 (STOP decoupling broke
  STOP calibration).

## Changes from baseline

Baseline: [#002](../002-exp45-full/).

- `config/loss.json` — swap `OnsetLossConfig` (`hard_alpha=0.5`,
  `good_pct=0.03`, `fail_pct=0.20`, `frame_tolerance=2`,
  `stop_weight=1.5`) → `GaussianCELossConfig` (`sigma_bins=2.0`).
- `training/losses.py` — new `GaussianCELoss`. Routes STOP through
  BCE-with-logits on logit[-1]; routes bins through softmax Gaussian
  CE over logits[:-1]; `loss = stop_bce + bin_ce` (STOP BCE averaged
  over all B samples; bin CE averaged over non-STOP samples only, or
  0 if the batch is STOP-only).
- `cli/train.py` — loss instantiation dispatches on config type
  (`GaussianCELossConfig → GaussianCELoss`, else `OnsetLoss`).

Nothing else changes: model, data sampler, augmentations, adapter,
trainer schedule, seed, dataset split — all identical to #002.

## Run config

- Run name: `exp_005_gaussian_ce`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42,
  song-grouped).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_005_gaussian_ce \
      --config-dir osu/taiko2/experiments/005-gaussian-ce/config \
      --dataset taiko2_v1 \
      --device cuda
  ```

---
<!-- Everything below written after the run. Do not pre-populate. -->
---

## Results summary

_(To fill post-run.)_

### Final vs baseline

_(Table.)_

### Per-eval progression

_(Table generated from `runs/exp_005_gaussian_ce/metrics.jsonl`.)_

## Visualizations

_(Graphs post-run.)_

## Vs prediction

_(One line per predicted metric post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
