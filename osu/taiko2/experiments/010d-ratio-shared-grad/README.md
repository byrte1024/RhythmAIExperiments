# Experiment 010d — Ratio decomposition with shared gradients

## Status

`Planned`

## Context

[#010](../010-ratio-decomposition/), [#010b](../010b-ratio-smooth-k3/),
and [#010c](../010c-ratio-128bins/) all converged to the same
plateau (miss ≈ 0.33, rgood ≈ 0.66). Bin count and Conv1d kernel
size don't change the ceiling. The ratio error histograms in all
three runs show the same systematic "non-musical-ratio prediction"
blur — predictions continuously distributed between musical-ratio
peaks instead of concentrated at them.

The remaining unexplored knob: the **stop-gradient** between the
auxiliary heads (divisor, offset) and the ratio head. taiko1 exp 67
and taiko2 #010 both detach the soft expectations `div_val` and
`off_val` before they enter the ratio head — so the ratio loss
gradient never flows back into the divisor/offset heads. The aux
heads only train via their own auxiliary CE losses.

This experiment removes that detach. The ratio loss can now shape
divisor and offset predictions too, providing a richer training
signal that might break the systematic blur — or destabilize
training, if the ratio loss's gradient pollutes the aux heads.

## Citations

- Direct parent: [#010](../010-ratio-decomposition/). Same
  configuration, only difference is `aux_stop_gradient: false`.
- Plateau context: [#010b](../010b-ratio-smooth-k3/),
  [#010c](../010c-ratio-128bins/) — both confirmed the 0.33
  ceiling is structural to the ratio decomposition.
- Original design: taiko1 exp 67 used stop-gradient by default
  ("Loss A: divisor_CE + offset_CE [stop gradient from Loss B]").
  This experiment tests the alternative.

---

## Hypothesis

### Claim

Removing the gradient stop between the divisor/offset heads and
the ratio head will let the ratio loss reshape the aux predictions
toward configurations the ratio head can predict cleanly. Result:
the systematic ratio error blur reduces, and derived-bin miss drops
below 0.31, breaking the plateau #010/#010b/#010c hit.

### Mechanism

In #010, gradient flow is one-directional:
```
ratio_loss → ratio head only
div_ce    → divisor head only
off_ce    → offset head only
```

The ratio head sees `div_val + off_val` as INPUTS but cannot
influence them. If the ratio head can't predict cleanly given a
particular `(div_val, off_val)` configuration, it has no way to
suggest a different one. The aux heads optimize their own losses
in isolation — they have no idea what would help the ratio head.

In #010d:
```
ratio_loss → ratio head AND (via soft expectation backprop) div+off heads
div_ce    → divisor head
off_ce    → offset head
```

The ratio head can now backprop "your divisor estimate is making
my prediction harder" through `div_val` into the divisor head.
The divisor head's training is the union of:
- div_ce gradient (match GT divisor)
- ratio_loss gradient (produce a divisor that helps the ratio head)

If these are aligned, the system finds a better joint optimum than
either head could find alone. If they conflict, the divisor head
might drift away from GT to give the ratio head an easier target —
which would show up as div_acc dropping while ratio metrics
improve. Either outcome is informative.

The systematic blur in #010 may stem from the ratio head being
forced to predict against div/off predictions it can't influence.
With shared gradients, the heads can co-optimize toward
configurations that produce cleaner ratio predictions.

### Predicted numbers

| Metric | #010 best | Predicted (#010d) | Notes |
|---|---:|---:|---|
| miss | 0.329 | ≤ 0.30 | break the plateau |
| r_rgood | 0.662 | ≥ 0.72 | tighter ratio peaks |
| r_rhit | 0.498 | ≥ 0.55 | sharper musical-ratio commits |
| ratio error blur | continuous | concentrated at musical ratios | the qualitative test |
| div_acc | 0.717 | ≥ 0.65 | may drop slightly as divisor co-adapts |
| off_acc | 0.947 | ≥ 0.92 | offset more stable, less noisy gradient |

## Success criteria

- **Must have:** miss ≤ 0.31 (break the 0.33 plateau).
- **Must have:** r_rgood ≥ 0.68 (improves on #010's 0.662).
- **Nice-to-have:** ratio error histogram visibly more concentrated
  at musical-ratio peaks (less blur between).
- **Nice-to-have:** miss ≤ 0.29.
- **Fails if:** miss > 0.35 (shared gradients destabilized training).
- **Fails if:** div_acc < 0.55 (divisor head drifted too far from GT).

## Changes from baseline

Baseline: [#010](../010-ratio-decomposition/).

- `config/model.json` — `aux_stop_gradient: false` (was true / not set).
  The model's `_apply_head` will skip `.detach()` on `div_val` and
  `off_val` before they enter the ratio head's input. Ratio loss
  gradients flow back through the soft expectations into the
  divisor/offset head MLPs.
- Everything else identical to #010: same backbone, same losses,
  same augmentations, same schedule, same ratio_bins=255, same
  Conv1d (k=5, 8ch).

## Run config

- Run name: `exp_010d_shared_grad`.
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_010d_shared_grad \
      --config-dir osu/taiko2/experiments/010d-ratio-shared-grad/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --cursor-shift-prob 0.3 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/010d-ratio-shared-grad/config/infer.json \
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
