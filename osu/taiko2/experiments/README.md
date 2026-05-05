# taiko2 experiments

One directory per experiment. Fresh sequential numbering — taiko1's 67
experiments live at `osu/taiko/experiments/` and are referenced via
cross-repo links, not duplicated here.

## Index

| # | Name | Status | Key result |
|---|------|--------|------------|
| [001](001-exp45-smoke/) | exp 45 port, subsample-16 smoke | `Complete` | Pipeline works end-to-end; miss 0.72 → 0.55, hit 0.13 → 0.28 in 2 epochs on 1/16 data. Still descending at end. |
| [002](002-exp45-full/)  | exp 45 full recreation       | `Complete` | Port confirmed: HIT 72.96 % @ E8 (+1.06 pp over taiko1 exp 45's 71.9 %), MISS 26.08 % (−1.42 pp). All 6 pre-run predictions passed. AR corpus: `dc_human` 91.7 %, median error 12 ms, ratio-error banding at ±log 2 / ±log 3 flagged for followup. |
| [003](003-gap-ratio-corpus/) | Gap / ratio shape corpus reference | `Complete` | Ratio mass: 1.0x = 54.8 %, 0.5x = 20.1 %, 2.0x = 20.0 %. Three canonical ratios hold 94.85 % of corpus mass. Triplets beat quarters by mass (unexpected). Median gap + ratio peak count both = 4. |
| [004](004-engagement-corpus/) | Engagement × chart-metric corpus | `Complete` | Pass_rate strongly predictable (top 15 pairs all pass_rate × difficulty, max \|r\| = 0.656). `favourite_count` has NO correlate (max \|r\| = 0.044). Playcount ≠ favourites (r = +0.012). Higher-rated charts are **less** chaotic, not more. |
| [005](005-gaussian-ce/) | Gaussian soft-target CE + binary STOP | `Complete` | Loss change near-neutral on miss (best 0.2664 vs #002's 0.2606). First run with `train_noaug` diagnostic: gap grew −1.0 → −2.9 pp by E6 — **first clean overfitting signal in taiko2**. STOP head collapsed at decode time (recall 0.06 vs #002's 0.77) due to sigmoid/softmax scale mismatch. Loss-side fix didn't compress the `±log 2` ridges. |
| [006](006-cursor-overlap/) | Cursor-overlap filtering ablation | `Complete` (hypothesis rejected) | `allowed_overlap = 0 → 500` dropped sample count ~13.6×. Overfitting got *worse*, not better: gap grew +7.7 pp across 5 evals (vs #005's +2.5 across 8). Best miss 0.358 (worse than #002 by 7 pp at matched step). **Cursor overlap is providing useful per-target variety, not redundancy.** Don't adopt overlap-filtering. |
| [007](007-time-stretch/) | Time-stretch augmentation | `Complete` | **Largest single-experiment delta in the series.** Best miss 0.2406 (E18) vs #002's 0.2606. +1.13 pp at matched step, +2.0 pp with 64 % more compute. First run to move `frame_err_p90` (31 → 30). All bin-precision metrics +1.6–2.6 pp. Train_noaug gap stabilized at ~−2.5 pp (vs #005's runaway). **Adopted as new baseline.** Banding ridges still present on both val and train_noaug — capability problem confirmed. |
| [008](008-log-emd/) | Log-ratio EMD loss | `Complete` (hypothesis rejected) | Entropy-floor escape confirmed (`log_emd` dropped 17 % vs #007's `soft_ce` 4.5 %), but headline metrics regressed 1.5 pp. Critical finding: **side-by-side ratio_error heatmaps at the same step show identical ridges** — three loss families now tested (trapezoid, Gaussian, log-EMD), all produce the same `±log 2` ridges. Loss-side approaches to ridges are exhausted. |
| [009](009-mdn/) | Mixture Density Network | `Stopped early` | K=3 MDN. Components specialized (n_active=2.2, comp 0 short-gap σ=1.7, comp 2 dominant σ=2.0) — confirmed model has multi-modal internal state. But comp 1 inflated to σ=48 ("junk drawer" pathology). Headline regressed 11 pp on miss. |
| [009b](009b-mdn-capped-sigma/) | MDN with `max_sigma=3` | `Complete` (hypothesis rejected) | Hard sigma cap blocked mu exploration: coverage_2bin 0.77 → 0.29 catastrophic. NLL 2.84 → 7.78 (2.7× worse). Model can't learn mu placement when sigma is forced tight from step 0. Sigma annealing flagged as the principled fix. |
| [010](010-ratio-decomposition/) | Ratio-decomposed onset prediction | `Complete` | Decomposes into divisor + offset + ratio (taiko1 exp 67 design on taiko2 backbone). Structural success: div_acc 72 %, ratio_rgood 66 %, AR `hi_pspace` 93.5 % at E2 (highest of any experiment). But derived-bin miss plateaued at 0.33 — 8 pp behind #007. Multiplicative precision (`div × ratio`) compounds errors. Two issues identified: Conv1d smoothing over-spreads, ratio bin floor at ≈ 0.33×. |
| [010b](010b-ratio-smooth-k3/) | Reduced Conv1d smoothing (k=3, 4ch) | `Complete` | Same plateau as #010 (best miss 0.326), reached 5 evals later, 3.8 pp behind on rhit. **Conv1d kernel size is not the bottleneck.** |
| [010c](010c-ratio-128bins/) | Ratio with 128 bins (half resolution) | `Complete` | Best miss 0.326. Converges 2× faster (E5 advantage 5.4 pp), same ceiling. **Bin count is not the bottleneck either** — 255 / 255-with-weak-smoothing / 128 all land at miss ≈ 0.33. The systematic ratio-prediction blur appears in all three. |
| [010d](010d-ratio-shared-grad/) | Remove stop-gradient between aux heads and ratio head | `Complete` (hypothesis rejected) | Decomposition collapsed: divisor head dropped to 0.39 acc with fanning-ray heatmap (looks like pre-decomposition direct detector); ratio head turned into a near-perfect bin-space diagonal — the inverse-noise shortcut. Product still recovers approximate bin (miss 0.36) but neither head encodes musical structure. **Reproduces the failure stop-gradient was originally introduced to prevent.** |
| [010e](010e-aux-frozen/) | 8-eval warmup, then freeze divisor + offset heads; ratio MLP skipped during warmup | `Complete` | Val miss matched #010 at fair step count (plateau is divisor-accuracy-bound at ~0.78 across every ratio variation; multiplicative `bin = div × ratio − offset` makes any div error a guaranteed bin miss). **But AR-generation quality is the best of any ratio run**: matched_rate 0.612 (vs #010's 0.527, +16 % rel.), error_median 12 ms (vs #010's 19 ms). Per-step sharpness (rhit, rce) compounds across AR steps. |
| [011](011-onset-feature-survey/) | Survey classical onset detection algorithms (mel-domain) against `taiko2_v1` GT, frame-wise P / R / F1 across tolerances | `Complete` | Single `spectral_flux` channel hits **F1 = 0.679 / R = 0.742 / P = 0.625 at ±10 frames** — far higher than the pre-run prediction of F1 ≈ 0.40 (chart-author GT tracks audio onsets much more tightly than expected). **Every ODF collapses below ±2 frames** (peaks 2-3 frames late on the attack), so for downstream channel use the right encoding is **bucket-pooled at ±5 or ±10 frames**, not raw 5 ms grid. Sub-band variants tie broadband SF when collapsed; their per-band benefit needs different evaluation. |
| [011b](011b-onset-disagreement/) | Pairwise disagreement / complementarity matrix + per-kind / per-density / per-star sub-analyses on the same algorithms as #011 | `Planned` | Tests whether single-algorithm misses are correlated or complementary (50%+50% disjoint = 100% union), where the F1 ceiling concentrates by content cut, and which sub-band channels specialize on DON vs KA. Informs whether channel-stacking is worth the extra parameters in #012. |

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
