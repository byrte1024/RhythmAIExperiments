# Experiment 010e — Frozen aux heads after extended warmup

## Status

`Planned`

## Context

[#010d](../010d-ratio-shared-grad/) confirmed that letting the ratio
loss flow back into the divisor and offset heads breaks the
decomposition: the ratio head learns to invert divisor noise back
into a correct bin, leaving div_acc at 0.39 and ratio outputs as a
trivial diagonal in raw bin space rather than musical-ratio
structure. The user reading the divisor heatmap saw the
pre-decomposition direct-detector "fanning rays" pattern reappear —
the divisor head had stopped predicting tempo entirely.

The opposite intervention is the question this experiment tests:
what happens if the divisor and offset heads are **prevented from
moving at all** during ratio training? With aux heads literally
frozen, the ratio head cannot drag them off-target (the #010d
failure), and it cannot rely on aux noise to bypass the
decomposition (no degree of freedom on the aux side).

A diagnostic done across [#010](../010-ratio-decomposition/),
[#010b](../010b-ratio-smooth-k3/), and
[#010c](../010c-ratio-128bins/) showed that **div_acc is essentially
saturated by the first eval** (~20k steps) — it oscillates between
0.66 and 0.74 for the rest of every run. off_acc has a brief climb
through E2–E3 then also saturates. So extending the warmup window
gets us a stable freeze point, not a higher-accuracy one. The
hypothesis below is therefore **not** "frozen high-accuracy aux
helps ratio", but the more modest "frozen #010-quality aux prevents
the collapse failure mode and lets the ratio head find better
musical-ratio structure."

## Citations

- Direct parent: [#010](../010-ratio-decomposition/). Same
  decomposition design and configs except for the warmup window and
  the freeze-aux-after-warmup flag.
- Failure mode being avoided: [#010d](../010d-ratio-shared-grad/).
  Removed the stop-gradient between aux and ratio heads → div_acc
  cratered, ratio head turned into bin-space diagonal (cheating).
  This experiment is the structural opposite intervention.
- Plateau context: [#010b](../010b-ratio-smooth-k3/),
  [#010c](../010c-ratio-128bins/) — both confirmed the 0.33 ceiling
  on the standard ratio decomposition.
- Aux saturation diagnostic: see "Per-run div/off head curves"
  numbers in #010d's followup notes — div_acc plateaus at E1 across
  every ratio run we have.

---

## Hypothesis

### Claim

Freezing the divisor and offset heads after an 8-eval warmup,
combined with skipping the ratio head's forward/backward during
those warmup evals (so all warmup compute goes into div/off), will
produce a ratio head that:
1. Cannot collapse the decomposition (aux side has no gradient
   path to corrupt) — the failure mode of #010d is structurally
   impossible.
2. Finds sharper musical-ratio predictions than #010 because every
   ratio gradient step optimizes ratio quality alone, not a noisy
   joint with two aux losses.

Result: derived-bin miss drops below 0.31, breaking the 0.33
plateau, and the ratio error histogram shows tighter peaks at
musical-ratio positions (less continuous blur between).

### Mechanism

In #010 (default), the joint optimization runs for the full schedule:
```
each step: ratio_loss(ratio head) + aux_weight × (div_ce + off_ce)
gradients: ratio_head ← ratio_loss
           div_head   ← div_ce
           off_head   ← off_ce
```
The aux losses keep nudging div/off after they've saturated, which
is wasted signal at best and slow drift at worst (#010 div_acc
moved 0.716 → 0.719 over 10 evals — nine evals of noise around the
asymptote).

In #010d (no stop-gradient), the ratio loss can also nudge the aux
heads toward easier-to-predict configurations, which collapsed
into the bypass solution.

In #010e:
```
warmup (steps 0..N): forward = backbone + div_head + off_head only
                     backward = div_ce + off_ce only
                     ratio head: not touched (forward + backward skipped)
boundary (step N):   div_head, off_head, val_emb params frozen
                     (requires_grad=False)
post-warmup:         forward = full graph
                     backward = ratio_loss → ratio_head only
                     (div/off frozen, contribute no gradient)
```
Three behaviors fall out:
1. **Compute saving in warmup**: the ratio MLP and Conv1d are not
   computed and not backpropped through during warmup. Per-batch
   compute drops by ~the size of those layers (small but nonzero).
2. **Stable freeze point**: by E8 the aux heads have had ~165k
   steps of training with no ratio interference, longer than any
   prior run had pure aux training. Aux quality is at its
   asymptote.
3. **Ratio head trains against fixed targets**: because the
   dynamic ratio target is `(target_bin + off_pred) / div_pred`,
   freezing div/off makes that target deterministic per-sample for
   the rest of training. The ratio head sees a stationary target
   distribution rather than one that shifts every step.

### Predicted numbers

Reference: [#010](../010-ratio-decomposition/) E7 (best for #010).

| Metric | #010 @ E7 | Predicted (#010e mature eval) | Notes |
|---|---:|---:|---|
| miss | 0.329 | ≤ 0.30 | break the plateau |
| rgood | 0.662 | ≥ 0.70 | tighter ratio prediction |
| rhit | 0.498 | ≥ 0.55 | sharper musical-ratio commits |
| div_acc | 0.717 | 0.71–0.74 | frozen at warmup-end value |
| off_acc | 0.947 | 0.94–0.96 | frozen at warmup-end value |
| ratio_ce | 2.63 | ≤ 2.50 | sharper softmax over ratio bins |

Observational (not gated):
- The divisor heatmap should look like #010's at E7 and stay that
  way (frozen).
- The ratio heatmap should show **stronger musical-ratio structure**
  than #010 — the ratio bins corresponding to 1.0× / 0.5× / 2.0× /
  1.5× / 0.33× should be more populated relative to the continuous
  fill between them.
- The ratio error histogram peaks at 0 / ±log(2) / ±log(3) should
  be sharper than #010's.

## Success criteria

- **Must have:** miss ≤ 0.31 by E12 (break the plateau).
- **Must have:** rgood ≥ 0.68 by E12 (improves on #010's 0.662).
- **Must have:** divisor heatmap at any post-warmup eval matches
  the warmup-end frozen state (sanity check the freeze took effect).
- **Nice-to-have:** miss ≤ 0.29.
- **Nice-to-have:** ratio error histogram visibly more peaked at
  musical ratios (less continuous blur).
- **Fails if:** miss > 0.35 (post-freeze ratio training never beat
  the #010 plateau).
- **Fails if:** ratio_ce > 3.5 at E12 (ratio head can't fit even
  with frozen aux — would suggest the multiplicative precision
  floor is the real ceiling).

## Changes from baseline

Baseline: [#010](../010-ratio-decomposition/).

- `config/loss.json`:
  - `ratio_freeze_evals: 1 → 8`. Eight warmup evals before the
    ratio head starts training.
  - New flag `freeze_aux_after_warmup: true`. At the warmup
    boundary, the divisor and offset heads (and their soft-
    expectation embeddings) are frozen via `requires_grad=False`.
- `models/ratio_detector.py` gains warmup awareness: while the
  model is in training mode and `_fwd_step < warmup_step_limit`,
  the ratio MLP and Conv1d are skipped (zero ratio block returned),
  and div/off heads are not yet frozen. On the first training-mode
  forward past the boundary, the freeze is applied lazily.
- `training/ratio_loss.py` already gated `ratio_ce = 0` during
  warmup (#010 design). Extended to also skip the ratio
  HIT/GOOD/MISS metric computation during warmup, since
  argmax over zero-filled logits would produce noise.
- `cli/train.py` passes the warmup step count and the freeze flag
  to the model alongside the existing `loss.set_freeze_limit` call.
- Everything else identical to #010: backbone, augmentations
  (TimeStretch 0.3, CursorShift 0.3), schedule, ratio_bins=255,
  Conv1d (k=5, 8ch), seed.

### Run-length consideration

#010 ran 10 evals total. With 8 warmup evals here, only 2–3 evals
of ratio training fit in a 10-eval run, which is unlikely to be
enough for the ratio head to converge. We extend the planned run
length to **at least 16 evals** (8 warmup + 8 ratio training),
matching #010's ratio-training budget. The trainer doesn't enforce
a hard cap; we'll let the run go until rmiss plateaus or rises.

## Run config

- Run name: `exp_010e_aux_frozen`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_010e_aux_frozen \
      --config-dir osu/taiko2/experiments/010e-aux-frozen/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --cursor-shift-prob 0.3 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/010e-aux-frozen/config/infer.json \
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
