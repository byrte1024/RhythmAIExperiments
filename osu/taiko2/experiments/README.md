# taiko2 experiments

One directory per experiment. Fresh sequential numbering — taiko1's 67
experiments live at `osu/taiko/experiments/` and are referenced via
cross-repo links, not duplicated here.

## Index

| # | Name | Status | Key result |
|---|------|--------|------------|
| [001](001-exp45-smoke/) | exp 45 port, subsample-16 smoke | `Complete` | Pipeline works end-to-end; miss 0.72 → 0.55, hit 0.13 → 0.28 in 2 epochs on 1/16 data. Still descending at end. |
| [002](002-exp45-full/)  | exp 45 full recreation       | `Complete` | Port confirmed: HIT 72.96 % @ E8 [exp_002_exp45_full, step 165,392, val/single/onset/hit] (+1.06 pp over taiko1 exp 45's 71.9 %), MISS 26.08 % (−1.42 pp). All 6 pre-run predictions passed. AR corpus: `dc_human` 92.81 % [exp_002_exp45_full, step 227,414, infer_corpus/eval_227414/gt_cond/comparisons_summary.json:fields.dc_human.median], `error_median_ms` 9.0 [same file:fields.error_median_ms.median]; ratio-error banding at ±log 2 / ±log 3 flagged for followup. |
| [003](003-gap-ratio-corpus/) | Gap / ratio shape corpus reference | `Complete` | Ratio mass: 1.0x = 54.8 %, 0.5x = 20.1 %, 2.0x = 20.0 %. Three canonical ratios hold 94.85 % of corpus mass. Triplets beat quarters by mass (unexpected). Median gap + ratio peak count both = 4. |
| [004](004-engagement-corpus/) | Engagement × chart-metric corpus | `Complete` | Pass_rate strongly predictable (top 15 pairs all pass_rate × difficulty, max \|r\| = 0.656). `favourite_count` has NO correlate (max \|r\| = 0.044). Playcount ≠ favourites (r = +0.012). Higher-rated charts are **less** chaotic, not more. |
| [005](005-gaussian-ce/) | Gaussian soft-target CE + binary STOP | `Complete` | Loss change near-neutral on miss (best 0.2664 vs #002's 0.2606). First run with `train_noaug` diagnostic: gap grew −1.0 → −2.9 pp by E6 — **first clean overfitting signal in taiko2**. STOP head collapsed at decode time (recall 0.06 vs #002's 0.77) due to sigmoid/softmax scale mismatch. Loss-side fix didn't compress the `±log 2` ridges. |
| [006](006-cursor-overlap/) | Cursor-overlap filtering ablation | `Complete` (hypothesis rejected) | `allowed_overlap = 0 → 500` dropped sample count ~13.6×. Overfitting got *worse*, not better: gap grew +7.7 pp across 5 evals (vs #005's +2.5 across 8). Best miss 0.358 (worse than #002 by 7 pp at matched step). **Cursor overlap is providing useful per-target variety, not redundancy.** Don't adopt overlap-filtering. |
| [007](007-time-stretch/) | Time-stretch augmentation | `Complete` | **Largest single-experiment delta in the series.** Best miss 0.2406 (E18) [exp_007_time_stretch, step 372,132, val/single/onset/miss] vs #002's 0.2606 [exp_002_exp45_full, step 227,414, val/single/onset/miss]. +1.13 pp at matched step, +2.0 pp with 64 % more compute. First run to move `frame_err_p90` (31 → 30). All bin-precision metrics +1.6–2.6 pp. Train_noaug gap stabilized at ~−2.5 pp (vs #005's runaway). **Adopted as new baseline.** Banding ridges still present on both val and train_noaug — capability problem confirmed. |
| [008](008-log-emd/) | Log-ratio EMD loss | `Complete` (hypothesis rejected) | Entropy-floor escape confirmed (`log_emd` dropped 17 % vs #007's `soft_ce` 4.5 %), but headline metrics regressed 1.5 pp. Critical finding: **side-by-side ratio_error heatmaps at the same step show identical ridges** — three loss families now tested (trapezoid, Gaussian, log-EMD), all produce the same `±log 2` ridges. Loss-side approaches to ridges are exhausted. |
| [009](009-mdn/) | Mixture Density Network | `Stopped early` | K=3 MDN. Components specialized (n_active=2.2, comp 0 short-gap σ=1.7, comp 2 dominant σ=2.0) — confirmed model has multi-modal internal state. But comp 1 inflated to σ=48 ("junk drawer" pathology). Headline regressed 11 pp on miss. |
| [009b](009b-mdn-capped-sigma/) | MDN with `max_sigma=3` | `Complete` (hypothesis rejected) | Hard sigma cap blocked mu exploration: coverage_2bin 0.77 → 0.29 catastrophic. NLL 2.84 → 7.78 (2.7× worse). Model can't learn mu placement when sigma is forced tight from step 0. Sigma annealing flagged as the principled fix. |
| [010](010-ratio-decomposition/) | Ratio-decomposed onset prediction | `Complete` (hypothesis rejected) | Structural decomposition works (div_acc 72 %, ratio_rgood 66 %, AR `hi_pspace` 93.5 % at E2 — highest of any experiment), but derived-bin miss plateaus at 0.33, 8 pp behind #007. Multiplicative `div × ratio − offset` compounds errors; Conv1d smoothing over-spreads; ratio bin floor at ≈ 0.33×. |
| [010b](010b-ratio-smooth-k3/) | Reduced Conv1d smoothing (k=3, 4ch) | `Complete` (hypothesis rejected) | Same plateau as #010 (miss 0.326), 5 evals later, 3.8 pp behind on rhit. Conv1d kernel size is not the bottleneck. |
| [010c](010c-ratio-128bins/) | Ratio with 128 bins (half resolution) | `Complete` (hypothesis rejected) | Miss 0.326, converges 2× faster (E5 advantage 5.4 pp), same ceiling. Bin count not the bottleneck either — 255 / 255-smooth / 128 all land at miss ≈ 0.33. |
| [010d](010d-ratio-shared-grad/) | Remove stop-gradient between aux heads and ratio head | `Complete` (hypothesis rejected) | Decomposition collapsed: div_acc 0.39 (fanning-ray heatmap), ratio head turned into bin-space diagonal — inverse-noise shortcut. Reproduces the failure stop-gradient was introduced to prevent. |
| [010e](010e-aux-frozen/) | 8-eval warmup then freeze divisor + offset heads | `Complete` | Best AR within the ratio-decomposition family — matched_rate 0.612 [exp_010e_aux_frozen, step 268,762, infer_corpus/eval_268762/gt_cond/comparisons_summary.json:fields.matched_rate.median] (+0.041 abs / +7.3 % rel over #010), error_median_ms 12 vs #010's 15. Still does NOT beat #007. Plateau is div_acc-bound (0.72–0.77 across the family). |
| [011](011-onset-feature-survey/) | Survey classical mel-domain ODFs against `taiko2_v1` GT | `Complete` | `spectral_flux` hits F1 = 0.679 at ±10 frames [011-onset-feature-survey/results/summary.json:by_algo.spectral_flux.by_tolerance.10], well above the pre-run F1 ≈ 0.40 prediction. Every ODF collapses below ±2 frames (peaks 2–3 frames late); right encoding for downstream use is bucket-pooled at ±5/±10. |
| [011b](011b-onset-disagreement/) | Pairwise ODF complementarity matrix | `Complete` (hypothesis rejected) | Best 2-channel union (`hfc_mel + spectral_flux`) hits recall 0.9052 [011b-onset-disagreement/results/summary.json:pairwise.pairs.hfc_mel+spectral_flux.recall_union], +12.5 pp over the best single channel (`energy` 0.7802). Cross-group pairs win; sub-band DON/KA specialization not observed — abandon sub-band-as-channel design. |
| [012](012-onset-channels/) | Sub-band spectral-flux input channels | `Complete` | First taiko2 run to break #007's per-step ceiling. Best miss 0.2331 [exp_012_onset_channels, step 349,690, val/single/onset/miss] (−0.75 pp vs #007). Best AR `matched_rate` 0.7080 [step 308,550, infer_corpus/eval_308550/gt_cond/comparisons_summary.json:fields.matched_rate.median] (+0.19 pp vs #007's 0.7061). Channel input acts as regularizer (train_noaug gap −0.7 to −1.0 pp narrower). |
| [013](013-conformer/) | Conformer trunk | `Complete` (hypothesis rejected) | Best miss 0.2536 [exp_013_conformer, step 227,414, val/single/onset/miss], +0.43 pp behind #007 despite +80 % params (29.47 M vs 16.35 M). Regressed on 9 of 10 input-distortion benchmarks. Capacity is not the bottleneck. |
| [014](014-diffusion/) | Diffusion output head | `Complete` (hypothesis rejected) | Best gt_cond AR `matched_rate` 0.640 [014-diffusion/ablations/ddim_16_e0_n4/gt_cond/comparisons_summary.json:fields.matched_rate.median] vs #007's 0.7028 (−9 pp), but `error_median_ms` 12 within 2 ms and `hi_pspace` 100 % (+9 pp). `n_samples=4` mean-of-softmax lifted matched_rate +9.6 pp at fixed checkpoint. All `eta > 0` variants collapsed to ≈0.06 (`loss/per_t_q3` 22× above q0). Three structural blockers: decode soft-margin ceiling, stop_weight bias, q3 undertrain. |
| [015](015-diffusion-patched/) | Diffusion head + 5 literature patches | `Complete` (hypothesis rejected) | Best gt `matched_rate` 0.6468 [015-diffusion-patched/ablations/ddim_4_e0_n4_off1/gt_cond/comparisons_summary.json:fields.matched_rate.median] (+0.7 pp over #014, −5.6 pp below #007). `stop_f1` 0.7663 new ATH [exp_015_diffusion_patched, step 186,066, val/single/onset/stop_f1] (+15.1 pp over #007). `n_samples=4` lift collapsed from #014's +9.6 pp to +2.6 pp (confirms soft-margin ceiling). Stochastic samplers still broken. Ceiling is in the diffusion design itself, not the loss/sampler config. |
| [016](016-framewise-diffusion/) | Framewise activation-map diffusion | `Complete` (hypothesis rejected) | AR `matched_rate` 0.973 [exp_016_framewise_diffusion, step 62,022, val/single/corpus/gt_cond_cmp/matched_rate_mean] (+0.27 abs vs #007) is a tolerance illusion — `density_ratio` 10.92, `hallucination_rate` 0.357 (model over-emits 11×). DDIM sampler regresses across its 16 steps and the regression widens with training (`rollout/final_vs_best_delta` −0.083 → −0.092). Root cause: `training/framewise_diffusion_loss.py:137` applies the ε-prediction Min-SNR formula `min(snr,γ)/snr` to an x0-param model (correct x0 form per Hang 2023 §4.2 is `min(snr,γ)`); kills low-t gradient → sampler saturates broad blobs instead of sharpening them. Halted at step 103,370 (1 epoch of 5). |
| [016b-mini](016b-mini/) | Min-SNR sign probe (`snr_x0_mode=true`, subsample 8) | `Complete` (hypothesis rejected) | Single eval at step 2,584 (1/8 data). `rollout/final_vs_best_delta` improved from #016 eval 1's −0.083 to −0.063 [exp_016b_mini, step 2,584, rollout_final_vs_best_delta.npy] (~24 % less negative; missed the ≥ −0.03 gate). Per-t-quartile `q3/q0` ratio swung past balanced to opposite-direction asymmetry (21.9× → 35.9×) — x0-form starves high-t instead of ε-form starving low-t. **k=15 peak saturation unchanged**: mean local-max value 0.997 (vs #016's 0.997), frac > 0.95 = 98.7 % (vs 99.4 %). Sign was a real bug but not load-bearing; saturation traces to the σ=2 Gaussian-smoothed target shape, not the loss weighting. Probe stopped at eval 1; the saturation signal would not change with more training. |
| [017](017-framewise-bce/) | Framewise BCE (non-diffusion control) | `Complete` | Matches #007's pattern quality (`dc_human` 91.0 vs #007's 92.0 [exp_017_framewise_bce, step 82,696, val/single/corpus/gt_cond_cmp/dc_human_mean]) with better timing (`error_median_ms` 6.1 vs 10.2). Over-emits ~44% (`density_ratio` 1.44 [exp_017_framewise_bce, step 82,696, val/single/corpus/gt_cond_cmp/density_ratio_mean], `hallucination_rate` 0.32 vs #007's 0.17). FP confidence indistinguishable from TP (`conf_fp_median` 0.80 vs `conf_tp_median` 0.93) — no threshold separates them. Early metronomic collapse (E1-E2) resolved spontaneously at E3. Overfitting from E4 onward (val loss 0.29 → 0.48; train_noaug 0.23 → 0.16). Proves framewise framing works at #007-class quality; remaining gap is selectivity, not detection. |
| [017b](017b-framewise-focal/) | Framewise focal loss | `Complete` (hypothesis rejected) | Focal (gamma=2) compressed the confidence range (`conf_tp_median` 0.74 vs #017's 0.94 [exp_017b_framewise_focal, step 82,696, val/single/frame/conf_tp_median]) without improving selectivity. `density_ratio` stuck at 2.35 [exp_017b_framewise_focal, step 82,696, val/single/corpus/gt_cond_cmp/density_ratio_mean] (vs #017's 1.44). `dc_human` 83.4 (vs #017's 91.0). Metronomic collapse never resolved; `over_pspace_self` increased to 53 [exp_017b_framewise_focal, step 103,370, val/single/corpus/gt_cond_cmp/over_pspace_self_mean]. Focal suppresses TP gradient (`focal_weight_pos` = 0.074) preventing the confidence commitment needed for the selectivity phase transition. |
| [017c](017c-framewise-bce-lowweight/) | Framewise BCE low pos_weight [3, 8] | `Complete` (partial success) | Best E1 of any framewise run: F1 0.846, precision 0.757 [exp_017c_framewise_bce_lowweight, step 20,674, val/single/frame/f1_τ_50_tol_2]. Eliminated metronomic collapse (`over_pspace_self` 13.7 at E1 vs #017's 37.1). But precision plateaued at ~0.74 across 7 evals while recall climbed +0.10 — the 8x asymmetry routes all gradient to recall. AR `density_ratio` stuck at 1.51-1.60, `dc_human` peaked at 90.0 [exp_017c_framewise_bce_lowweight, step 144,718, val/single/corpus/gt_cond_cmp/dc_human_mean] — never matched #017's post-transition 91.0. First-note hit rate 0.77 at tau=70 matches #007's 0.75. Confirms pos_weight is the recall-precision balance knob; any asymmetry >1 routes gradient to recall. |
| [017d](017d-framewise-bce-noweight/) | Framewise BCE no pos_weight (symmetric) | `Complete` | Symmetric BCE produces precision-first training (precision 0.92 at E1, recall climbs +0.13 across 10 evals). Best per-eval AR corpus: `matched_rate` 0.675, `halluc_rate` 0.151, `dc_human` 92.9 [exp_017d_framewise_bce_noweight, step 165,392]. Model is 99% audio-driven (no_future_audio F1=0.000, no_context drops only 5%). Confidence range compressed (TP median ~0.74) but well-calibrated (ECE 0.004). Loss-optimal checkpoint (E2) is worst for AR — `metric_to_watch` should not be loss. Overfitting from E3 limits ceiling. **Amendment: threshold_sweep.json numbers are INVALID (comparison order was flipped); per-eval AR corpus metrics are correct.** |
| [017e](017e-framewise-bce-regularized/) | Framewise BCE regularized (label smoothing + dropout + F1 watch) | `Complete` | **Best framewise model.** Label smoothing 0.05 + head dropout 0.2 + metric_to_watch=frame/f1 push overfitting wall 40-50% further than #017d. Threshold sweep found optimal at E8 (step 165,392) tau=0.40: `matched_rate` 0.783, `density_ratio` 1.020, `dc_human` 92.7, `error_median_ms` 10.3 [threshold_sweep.json]. Beats #007 on matched_rate (+8 pp), density (1.02 vs 0.87), dc_human (+0.7 pp). Density starts near 1.0 from E1 (label smoothing prevents early conservatism). Halluc_rate 0.201 remains above #007's 0.172 — the selectivity ceiling. Model more context-dependent than 017d (no_past_audio drops 13.5% vs 8.0%). max_notes_per_step sweep shows no effect. |
| [018](018-baselines/) | External baseline benchmarks | `Complete` | librosa `matched_rate` 0.408, BeatThis! 0.189 — both far below #017e's 0.783 and #007's 0.703. Chart-specific training is essential; general-purpose onset/beat detection is not sufficient. Mapperatorinator2 failed to run (dependency conflicts). Neither external model uses pointwise GT comparison — Mapperatorinator uses distributional FID which they found unreliable. Distributional evaluation is worth exploring as a complement to pointwise metrics. |

---

## How to start an experiment

1. **Pick the next ID**, zero-padded to 3 digits. Append a short
   kebab-case slug. Example: `001-exp45-port`,
   `014-conditioning-dropout`.

2. **Copy the template:**
   ```bash
   cp -r osu/taiko2/experiments/_template  osu/taiko2/experiments/NNN-slug
   ```

3. **Fill in the pre-run sections of README.md.** Don't touch anything
   below the horizontal-rule separator — that stays empty until the
   run finishes. If you can't fill the pre-run sections truthfully
   (no hypothesis, no predicted numbers), you don't have an
   experiment; you have a training run.

4. **Commit the pre-run README.** This is the proof of what you
   predicted before you saw the numbers.

5. **Populate `config/*.json`** with the exact configs you'll pass to
   the trainer. These are snapshots — edit them in the experiment
   folder, not in-place in `configs/`.

6. **Run.** `metrics.jsonl` and checkpoints land under the matching
   `osu/taiko2/runs/{run_name}/` — the experiment folder cross-
   references that run_name.

7. **After the run:** fill the post-run sections. Numbers and at least
   two graphs are mandatory. See the [format rules](#format-rules).

8. **Update this index.** Add one row to the table above with
   status + one-sentence key result.

---

## Format rules

### Strict section order

```
# Experiment NNN — {title}

## Status
## Context
## Citations
          ── pre-run ──
## Hypothesis
## Success criteria
## Changes from baseline
## Run config
────────────────────────────────
      (post-run below)
────────────────────────────────
## Results summary
## Visualizations
## Vs prediction
## Takeaways
## Followup questions
```

### Status values

| Status          | Meaning |
|-----------------|---------|
| `Planned`       | Pre-run written, not yet running. |
| `Running`       | Training in progress. |
| `Complete`      | Post-run filled, done. |
| `Abandoned`     | Stopped before completion. Post-run explains why. |
| `Superseded by [#NNN](...)` | Replaced by a later experiment. See [Superseded rule](#superseded-rule). |
| `No hypothesis` | Ran without a real pre-run prediction — quarantined from the index's "key result" column. |

### Superseded rule

When experiment M is superseded by experiment N:

1. **Experiment N** must cite experiment M explicitly in its
   `## Citations` section under a "Supersedes" sub-list.
2. **Experiment M**'s `## Status` line is the one exception to the
   "never edit the pre-run" rule: change it to
   `Superseded by [#N](../N-slug/)` when N lands. Add one line below
   the status explaining what changed. Leave everything else untouched.

### Citations (mandatory)

Inline markdown links; three canonical forms:

- **Sibling taiko2 experiment** — `[#042](../042-{slug}/)`
- **taiko1 experiment** — `[taiko1 exp 45](../../../taiko/experiments/experiment_45/)`
- **External source** — `[TaikoNation paper](https://...)`

Every claim about prior work needs a citation. "This was shown to
work" without a link → flagged.

### Hypothesis

- **Claim** — one sentence, *if-then-because* form.
- **Mechanism** — 2-4 sentences of reasoning.
- **Predicted numbers** — a table, ≥ 3 rows:
  - a must-move metric (what you expect to change)
  - a should-stay-stable metric (guard against obvious regressions)
  - the watched eval metric (whatever the trainer is tracking as best)

Rough ranges OK (`HIT +1-2pp`). **Predictions must exist before the
run.**

### Success criteria

Three explicit buckets:

```markdown
- **Must have:** {}
- **Nice-to-have:** {}
- **Fails if:** {}
```

If a criterion can only be written after seeing results, the
experiment is not hypothesis-driven — mark `No hypothesis`.

### Changes from baseline

- Point at code diffs (`models/event_embedding.py:L210-L245`) or
  config diffs (`model.json: n_layers 8 → 16`).
- Link the baseline experiment this forks from.

### Results summary (post-run, required)

**Two tables. Both mandatory.**

**(a) Final-eval comparison vs baseline**, exact shape:

```markdown
| Metric | Baseline (exp N) | This run (final) | Δ | Direction |
|---|---:|---:|---:|:---:|
| val/single/hit_e1 | 71.9% | 73.1% | +1.2pp | ↑ good |
| val/single/miss   | 27.5% | 26.0% | −1.5pp | ↓ good |
| train/overall/loss| 2.41  | 2.32  | −0.09  | ↓ good |
```

**(b) Per-eval progression across the whole run.** One row per eval
step (eval 1, eval 2, …). Include **every metric the trainer
reported**, even ones you don't care about — reviewers need to see the
full picture, not the curated subset. Long runs can abbreviate headers
but never hide rows.

```markdown
| Eval | Step  | val/single/hit_e1 | val/single/miss | val/single/loss | train/running/loss | train/overall/loss | lr     | wall_time |
|-----:|------:|------------------:|----------------:|----------------:|-------------------:|-------------------:|-------:|----------:|
|   1  |  2000 |             0.412 |           0.580 |           3.881 |              4.230 |              4.230 | 3.0e-4 |    00:12  |
|   2  |  4000 |             0.554 |           0.430 |           3.241 |              3.110 |              3.675 | 3.0e-4 |    00:24  |
| …    |       |                   |                 |                 |                    |                    |        |           |
| 48   | 96000 |             0.731 |           0.260 |           2.317 |              2.285 |              2.410 | 1.2e-5 |    11:34  |
```

Generate the per-eval table directly from `runs/{run_name}/metrics.jsonl`
— don't hand-curate. If a metric was reported in any eval, it must
appear as a column.

Both tables' numbers are also in [`metrics.json`](./metrics.json) for
machine-queryable crunching.

### Visualizations (post-run, required)

**At least two PNGs.** Mandatory:

- Training loss over steps (log-y).
- Validation metric progression over evals — same x-axis as the
  per-eval progression table in **Results summary (b)**.

**Custom graphs are encouraged.** Add anything the experiment makes
more visible: overfit curves, per-star-rating breakdowns, prediction
distributions, AR density adherence, entropy-over-time, per-kind
confusion, model-internal attention plots — whatever lets a reviewer
see what changed. Don't strip a graph because "it didn't tell us
anything new"; that's information about the experiment too.

Numbering convention:
- `01_train_loss.png`, `02_val_progression.png` — the two mandatory.
- `03_*.png` onward — custom. Name descriptively (`03_hit_by_star.png`,
  `04_prediction_distribution.png`). Reference in the README with a
  one-sentence caption each:

```markdown
![train loss](graphs/01_train_loss.png)
*Training loss, log-y. Convergence plateau around step 40k.*

![hit by star rating](graphs/03_hit_by_star.png)
*Per-star-rating HIT E1 on val. This run improves most at 3-5★;
7★+ tail unchanged.*
```

### Custom data (optional — encouraged)

Any data that doesn't fit the standard tables or graphs goes in
`custom/` alongside the README. Rules:

- One directory per kind of artifact: `custom/attention_maps/`,
  `custom/ar_density_curves/`, `custom/confusion_per_kind/`.
- Each directory has its own `README.md` with a one-paragraph
  explanation of what it contains, how it was computed, and what the
  main takeaway is.
- Prefer CSV/JSON for data, PNG for graphs, `.npz` for tensors.
  Nothing so large it needs to be gitignored — summaries only.
- Reference anything important from the main README under a
  **Custom analyses** section (post-run, optional):

```markdown
## Custom analyses

- [Attention maps](custom/attention_maps/) — Layer 6 attention on 10
  hand-picked windows; shows the model attending strongly to the most
  recent 2 past events under short-IOI regimes.
- [AR density curves](custom/ar_density_curves/) — Per-chart density
  sweep across conditioning values.
```

### Vs prediction (post-run)

One line per predicted metric: **match / beat / miss / wrong
direction**. Then a one-paragraph takeaway. If the hypothesis was
falsified, say so here, move the explanation to Takeaways.

### Takeaways

- Bullet list, one concrete sentence each.
- **No retrofitting.** If a result surprised you, write "unexpected:
  …". Don't claim you predicted something you didn't.

### Followup questions

- Open questions this experiment raises, even unrelated to its result.
- Format: `{question} — {suggested next exp or dataset}`.

---

## The amendment rule (pre-run honesty)

**Pre-run text is a historical record. Never edit it after the run.**

When post-run evidence contradicts a pre-run statement, add an
amendment immediately below it:

```markdown
**Predicted:** HIT will increase by 1-2 pp.
> *Amendment (post-run, eval 18): HIT decreased by 0.4 pp.
> Hypothesis rejected — see Takeaways.*
```

Keeps "what we thought" vs "what happened" visible in the same
document. No silent edits.

---

## What lives where

| Thing | Lives in | Tracked? |
|---|---|---|
| README.md | `experiments/NNN-slug/` | yes |
| ARCHITECTURE.md | `experiments/NNN-slug/` | yes (required — see rules below) |
| Config snapshots | `experiments/NNN-slug/config/` | yes |
| Aggregate metrics | `experiments/NNN-slug/metrics.json` | yes |
| Graphs (PNG) | `experiments/NNN-slug/graphs/` | yes |
| AR sample outputs | `experiments/NNN-slug/ar_samples/` | yes (small; optional) |
| Per-step `metrics.jsonl` | `runs/{run_name}/metrics.jsonl` | no |
| Checkpoints (`.pt`) | `runs/{run_name}/checkpoints/` | no |
| Dataset features | `datasets/{name}/features/` | no |

Bottom line: the experiment folder is the **summary and story**, the
run folder is the **raw artifacts**. They cross-reference each other
by `run_name`.

---

## ARCHITECTURE.md — required, reference-free

Every experiment folder must contain `ARCHITECTURE.md`. This is the
**sole document someone could open with zero context and, from it
alone, reproduce the experiment.**

### Rules

- **Self-contained.** No cross-references, no "see exp 45 for the
  mel params", no "inherits from the default adapter". Everything
  the experiment uses is written out in full, inline.
- **No links.** Not to other experiments, not to papers, not to the
  taiko1 repo. If an idea came from a paper, paraphrase the mechanism
  in full here; cite the paper in the README's `Citations` section
  only.
- **Not just the model.** Audio preprocessing, event encoding, sample
  construction, augmentations (with rates + ranges), loss math, loss
  hyperparameters, training schedule, optimizer, scheduler,
  inference procedure, dataset, split rule, environment versions.
  Enough that someone could write the code from scratch by reading
  this file.
- **Tables, shapes, numbers.** Every layer's input/output shapes.
  Every hyperparameter with its value. Every augmentation with its
  rate and parameter ranges. No prose hand-waves.
- **Updated in lockstep with the experiment.** If a run deviates from
  the pre-run ARCHITECTURE.md (e.g. you bumped batch size mid-run),
  add a dated addendum at the end of ARCHITECTURE.md describing what
  changed and when. Never silent-edit.

### Why reference-free

Cross-referenced architecture docs rot. A link to taiko1's exp 45
works today; it may point to a moved file next year. A link to
"the default adapter" assumes the reader knows which adapter was
default at the time. Future readers reproducing an experiment don't
have the working context we had — ARCHITECTURE.md must stand alone.

The README.md is where links and cross-references live. Two
documents, different jobs:

| Doc | Purpose | Links allowed? |
|---|---|---|
| README.md | Narrative: hypothesis, results, takeaways, citations | Yes, heavily |
| ARCHITECTURE.md | Complete spec to reproduce the experiment | **No** |
