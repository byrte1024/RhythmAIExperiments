# Experiment 001 — exp 45 port, subsample-16 smoke test

## Status

`Planned`

## Context

First experiment against the taiko2 framework. Ports the architecture
and training recipe from taiko1's exp 45 (event-embedding detector +
trapezoid-soft-target loss + 13-aug set + class-balanced sampling)
and runs it on 1/16th of the `taiko2_v1` training set. The goal is
**not** to reproduce exp 45's HIT numbers — the goal is to **prove
the plumbing works end-to-end** before committing GPU-hours to a full
50-epoch run.

## Citations

- Baseline architecture & recipe: [taiko1 exp 45](../../../taiko/experiments/experiment_45/)
- Taiko1 exp 44 (prior ATH that exp 45 forked from): [taiko1 exp 44](../../../taiko/experiments/experiment_44/)
- Taiko2 infra it exercises: this run is the first end-to-end user of
  `training/loop.py`, `training/augmentations.py`,
  `training/metrics_onset.py`, `training/artifacts.py`,
  `persistence/checkpoint.py` — all under `osu/taiko2/`.

---

## Hypothesis

### Claim

If we train exp 45's ported architecture + loss + augmentation set on
a subsample-16 slice of `taiko2_v1` for 2 epochs, we'll see
`val/single/onset/bad` drop below **1.0** and `val/single/onset/good`
rise above **0.05** by the final eval — proving gradients flow,
metrics compute, checkpoints write, and the model is learning
something non-trivial despite the data cut.

### Mechanism

Subsample-16 on ~388 k overlap-filtered train samples leaves ~24 k.
At batch 32 that's ~760 steps/epoch; two epochs ≈ 1520 steps.
Trapezoid-soft-target loss plus ±2-frame floor means even a
marginally-informed model lands *some* predictions inside the
frame-tolerance window. The 50/50 hard/soft mix keeps optimization
well-behaved from step 1. The only way this run doesn't drop below
100 % miss is if something is wired wrong.

### Predicted numbers

| Metric | Current | Predicted | Notes |
|---|---:|---:|---|
| val/single/onset/bad | — | ≤ 0.95 by eval 8 | must drop from 1.0 |
| val/single/onset/good | — | ≥ 0.05 by eval 8 | nice-to-have |
| val/single/loss | — | ≤ 4.0 by eval 8 | trapezoid CE baseline |
| (all metrics) no NaN / Inf | — | true | pipeline integrity |

## Success criteria

- **Must have:** `val/single/onset/bad < 1.0` at any point during training.
- **Must have:** loss goes down monotonically (allowing small eval-to-eval wobble) and never NaNs.
- **Must have:** all four eval artifacts write to `runs/{run}/eval_{step}/` on every eval.
- **Must have:** `metrics.jsonl` contains lines for `step`, `eval`, and `train_end` events.
- **Nice-to-have:** `val/single/onset/good >= 0.05` at final eval.
- **Nice-to-have:** `val/single/onset/hit >= 0.02`.
- **Fails if:** any of the must-haves miss, or training crashes.

Passing the must-haves means the pipeline is correct; the nice-to-haves
are about whether 1/16 of the data is enough signal for exp 45 to
learn meaningfully in 2 epochs — an honest question, not a framework
bug.

## Changes from baseline

Baseline: [taiko1 exp 45](../../../taiko/experiments/experiment_45/).

Differences, all intentional for a **smoke run** (not for
reproduction):

- `data.subsample: 16` (taiko1 exp 45 used 1 = full dataset).
- `trainer.epochs: 2` (taiko1: 50).
- `trainer.batch_size: 32` (taiko1: 48 — smaller so a 12 GB card handles d_model=384 comfortably even under AMP-off).
- `trainer.evals_per_epoch: 4` (same as taiko1).
- `data.min_cursor_bin: 0` (taiko1: 6000 — turned off so even short charts contribute; with subsample-16 we want every sample we can get for the smoke).
- Dataset: `taiko2_v1` (10 048 charts, audio sr=22 000 / hop=110 → 5.000 ms/frame, 200 bins/s — the taiko2 defaults; matches exp 45's bin rate to within rounding).
- Audio and event augs match exp 45 exactly: `build_exp45_post_augs()` from `training/augmentations.py`.

Everything else (architecture: d_model=384 / 8 layers / gap_ratios;
loss: hard_alpha=0.5 / good_pct=0.03 / fail_pct=0.20 / stop_weight=1.5;
class-balanced sampling with `power=0.5`; AdamW 3e-4 / wd 0.01 /
CosineAnnealingLR / grad_clip 1.0) is identical to taiko1 exp 45.

## Run config

- Run name: `exp_001_exp45_smoke`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42, song-grouped).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_001_exp45_smoke \
      --config-dir osu/taiko2/experiments/001-exp45-smoke/config \
      --dataset taiko2_v1
  ```

─────────────────────────────────────────────────────────────────────
<!-- Everything below written after the run. Do not pre-populate. -->
─────────────────────────────────────────────────────────────────────

## Results summary

### Final vs baseline

| Metric | Baseline (none) | This run (final) | Δ | Direction |
|---|---:|---:|---:|:---:|
| val/single/onset/bad | — | — | — | — |
| val/single/onset/good | — | — | — | — |
| val/single/loss | — | — | — | — |

Final eval: eval step `—`, wall time `—`, epochs `—`.

### Per-eval progression

| Eval | Step | val/single/onset/bad | val/single/onset/good | val/single/onset/hit | val/single/onset/fhit | val/single/onset/rhit | val/single/onset/exact | val/single/onset/ihit | val/single/loss | train/batch/loss | lr | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   |  |  |  |  |  |  |  |  |  |  |  |  |

Machine-readable copy: [`metrics.json`](./metrics.json).

## Visualizations

![](graphs/01_train_loss.png)
*Training loss over steps, log-y.*

![](graphs/02_val_progression.png)
*Watched val metric (`onset/bad`) across evals.*

![](graphs/03_scatter.png)
*Target-vs-predicted heatmap at the final eval.*

![](graphs/04_distributions.png)
*Target and predicted bin-offset distributions at the final eval.*

![](graphs/05_ratio_error.png)
*Log-log ratio error scatter at the final eval.*

## Vs prediction

- `val/single/onset/bad`: predicted `≤ 0.95` → actual `—` → **{match / beat / miss / wrong direction}**
- `val/single/onset/good`: predicted `≥ 0.05` → actual `—` → **{…}**
- `val/single/loss`: predicted `≤ 4.0` → actual `—` → **{…}**

## Takeaways

- {One concrete bullet after the run.}

## Followup questions

- {Open questions.} — {suggested next experiment}
