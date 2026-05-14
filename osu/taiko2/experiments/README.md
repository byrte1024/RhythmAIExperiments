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
| [010](010-ratio-decomposition/) | Ratio-decomposed onset prediction | `Complete` | Decomposes into divisor + offset + ratio (taiko1 exp 67 design on taiko2 backbone). Structural success: div_acc 72 %, ratio_rgood 66 %, AR `hi_pspace` 93.5 % at E2 (highest of any experiment). But derived-bin miss plateaued at 0.33 — 8 pp behind #007. Multiplicative precision (`div × ratio`) compounds errors. Two issues identified: Conv1d smoothing over-spreads, ratio bin floor at ≈ 0.33×. |
| [010b](010b-ratio-smooth-k3/) | Reduced Conv1d smoothing (k=3, 4ch) | `Complete` | Same plateau as #010 (best miss 0.326), reached 5 evals later, 3.8 pp behind on rhit. **Conv1d kernel size is not the bottleneck.** |
| [010c](010c-ratio-128bins/) | Ratio with 128 bins (half resolution) | `Complete` | Best miss 0.326. Converges 2× faster (E5 advantage 5.4 pp), same ceiling. **Bin count is not the bottleneck either** — 255 / 255-with-weak-smoothing / 128 all land at miss ≈ 0.33. The systematic ratio-prediction blur appears in all three. |
| [010d](010d-ratio-shared-grad/) | Remove stop-gradient between aux heads and ratio head | `Complete` (hypothesis rejected) | Decomposition collapsed: divisor head dropped to 0.39 acc with fanning-ray heatmap (looks like pre-decomposition direct detector); ratio head turned into a near-perfect bin-space diagonal — the inverse-noise shortcut. Product still recovers approximate bin (miss 0.36) but neither head encodes musical structure. **Reproduces the failure stop-gradient was originally introduced to prevent.** |
| [010e](010e-aux-frozen/) | 8-eval warmup, then freeze divisor + offset heads; ratio MLP skipped during warmup | `Complete` | Val miss matched #010 at fair step count (plateau is divisor-accuracy-bound at run-best `val/single/ratio/div_acc` 0.72–0.77 across the family — 0.7235 [exp_010_ratio, step 186,066] to 0.7723 [exp_010e_aux_frozen, step 103,370]; multiplicative `bin = div × ratio − offset` makes any div error a guaranteed bin miss). **Best AR generation within the ratio-decomposition family** (does NOT beat #007 — see PERFORMANCE.md): matched_rate 0.612 [exp_010e_aux_frozen, step 268,762, infer_corpus/eval_268762/gt_cond/comparisons_summary.json:fields.matched_rate.median] (vs #010's best 0.5706 [exp_010_ratio, step 186,066, infer_corpus/eval_186066/gt_cond/comparisons_summary.json:fields.matched_rate.median] = **+0.041 absolute / +7.3 % rel.**), error_median_ms 12 [exp_010e_aux_frozen, step 578,872, infer_corpus/eval_578872/gt_cond/comparisons_summary.json:fields.error_median_ms.median] vs #010's best 15 [exp_010_ratio, step 186,066, infer_corpus/eval_186066/gt_cond/comparisons_summary.json:fields.error_median_ms.median]. Per-step sharpness (rhit, rce) compounds across AR steps. |
| [011](011-onset-feature-survey/) | Survey classical onset detection algorithms (mel-domain) against `taiko2_v1` GT, frame-wise P / R / F1 across tolerances | `Complete` | Single `spectral_flux` channel hits **F1 = 0.679 / R = 0.742 / P = 0.625 at ±10 frames** [011-onset-feature-survey/results/summary.json:by_algo.spectral_flux.by_tolerance.10] — far higher than the pre-run prediction of F1 ≈ 0.40 (chart-author GT tracks audio onsets much more tightly than expected). **Every ODF collapses below ±2 frames** (peaks 2-3 frames late on the attack), so for downstream channel use the right encoding is **bucket-pooled at ±5 or ±10 frames**, not raw 5 ms grid. Sub-band variants tie broadband SF when collapsed; their per-band benefit needs different evaluation. |
| [011b](011b-onset-disagreement/) | Pairwise disagreement / complementarity matrix + per-kind / per-density / per-star sub-analyses on the same algorithms as #011 | `Complete` | **Pre-run "redundancy dominates" hypothesis rejected.** Best 2-channel union (`hfc_mel + spectral_flux`) hits **recall 0.9052** [011b-onset-disagreement/results/summary.json:pairwise.pairs.hfc_mel+spectral_flux.recall_union], **+12.5 pp over the best single channel** (`energy` at 0.7802 [same file:pairwise]). Cross-group pairs (envelope-based + difference-based) consistently top the table (J 0.57-0.69, marg 11-13 pp); within-group pairs near-redundant. F1 still peaks at K=1; recall keeps climbing through K=3 (0.932). **Sub-band DON/KA specialization not observed** — abandon the sub-band-as-channel design. Per-density and per-star F1 *inverse* the prediction: sparse / easy charts have low precision (many unmapped audio onsets), dense / insane have high. Recommended channel set for #012: SF + HFC (or + energy). |
| [012](012-onset-channels/) | Append 4-band sub-band spectral flux as extra mel rows; everything else identical to #007 | `Complete` | **First taiko2 run to break #007's per-step ceiling.** Best miss **0.2331** [exp_012_onset_channels, step 349,690 (E17), val/single/onset/miss] vs #007's 0.2406 (−0.75 pp at 94 % of #007's best-step compute). Best AR `matched_rate` **0.7080** [exp_012_onset_channels, step 308,550 (E15), infer_corpus/eval_308550/gt_cond/comparisons_summary.json:fields.matched_rate.median] vs #007's 0.7061 (+0.19 pp at 75 % of the steps), `error_median_ms` 8 tied. Train_noaug gap 0.7-1.0 pp narrower than #007 at every matched eval — channel input acts as regularizer. `dc_human` regression (−0.99 pp vs #007) flagged as the only metric where #007 still leads. |
| [013](013-conformer/) | Replace 8-layer post-norm Transformer trunk with 8-layer Conformer (macaron FFN + MHSA + depthwise-conv module). Everything else identical to #007 (taiko2_v1 dataset, 80-row mel) | `Complete` (hypothesis rejected) | **First trunk-architecture variation in 143 experiments; result: null on per-step miss, regression on robustness.** Best miss **0.2536** [exp_013_conformer, step 227,414 (E11), val/single/onset/miss], **+0.43 pp behind #007** at matched compute despite +80 % params (29.47 M vs 16.35 M). **Systematic brittleness regression**: regressed on 9 of 10 input-distortion benchmarks (`no_audio` +9.3 pp, `no_context` +4.1 pp, `context_time_shifted` +4.0 pp). AR shows heavy-tail bimodality — `error_median_ms` 16.6 (slightly better than #007's 17.6) but `error_mean_ms` 1100 vs 742 (+48 %). Conformer specialized on clean training distribution rather than building robustness. **+80 % capacity is not the bottleneck** — strengthens the case that the bottleneck is data distribution or audio representation, not trunk architecture. |
| [014](014-diffusion/) | Diffusion output head — replace `Linear(384, 501) + Conv1d_smooth` softmax head with a `(CosineSchedule T=64, GaussianContinuousProcess x0-param, MLPDenoiser)` triple; AR decode runs a 16-step DDIM sampler against the cursor token. Trunk identical to #007 (taiko2_v1, 80-row mel) | `Complete` (hypothesis rejected on headline, design viable with follow-up) | **First diffusion-output-head experiment in the series; headline below #007, mechanism findings strong.** Manually stopped at E12/step 248,088. Best gt_cond AR `matched_rate` 0.640 [014-diffusion/ablations/ddim_16_e0_n4/gt_cond/comparisons_summary.json:fields.matched_rate.median] vs #007's 0.7028 [exp_007_time_stretch, step 413,480, val/single/corpus/gt_cond_cmp/matched_rate_mean] — **−9 pp on the headline**, **but `error_median_ms = 12 ms` is within 2 ms of #007's 10.2 ms** [same comparison, error_median_ms_mean fields], and `hi_pspace = 100 %` beats #007's 91 % by +9 pp. **`n_samples = 4` mean-of-softmax marginalization lifted matched_rate +9.6 pp** (0.544 → 0.640) at fixed checkpoint — sampler-variance was a real contributor to the eval-to-eval AR bistability seen during training. **All stochastic-sampler variants (`eta > 0`) collapsed to matched_rate ≈ 0.06** because `loss/per_t_q3` plateaued 22× above q0 (undertrained schedule end). Three structural blockers identified: decode_to_logits soft-margin ceiling (top softmax prob capped at ~0.005), `stop_weight = 1.5` biasing unconditional output toward STOP (AR over-emits STOP → density_ratio 0.71 vs #007's 0.87), q3 undertrain (Min-SNR weighting is the textbook fix, already wired as config flag). Headline failure mode: missing > hallucinating by ~2.6× (vs #007's 1.5×) — model under-emits in sparse audio. Diffusion produces structurally more diverse output (gap_peak_count 2× #007). |
| [015](015-diffusion-patched/) | Diffusion head + five literature-informed patches stacked on #014's design: `snr_weighting=true` (Min-SNR γ=5, Hang 2023) + `stop_weight=1.0` (was 1.5) + `logit_scale=5` in `decode_to_logits` (CARD-style sharpening, Han 2022) + Self-Conditioning + Asymmetric Time Intervals (both Chen, Zhang, Hinton 2022 Analog Bits). Trunk + dataset + augmentations identical to #014. Denoiser +0.77 M params from self-cond's wider first Linear (24.47 M total) | `Complete` (hypothesis rejected on headline, mechanism wins partial) | **Patches achieved loss-level effects but matched_rate ceiling did not lift.** 18 evals, 41.4 h. Best post-ablation gt `matched_rate` **0.6468** [015-diffusion-patched/ablations/ddim_4_e0_n4_off1/gt_cond/comparisons_summary.json:fields.matched_rate.median] — **+0.7 pp over #014's 0.6398** but **−5.6 pp below #007's 0.7028** [exp_007_time_stretch, step 413,480, val/single/corpus/gt_cond_cmp/matched_rate_mean]. `stop_f1 = 0.7663` at E9 [exp_015_diffusion_patched, step 186,066, val/single/onset/stop_f1] — **new taiko2 ATH**, +5.1 pp over #014's 0.7213 and +15.1 pp over #007's 0.6152. `error_median_ms = 11.0 ms` at best variant — within 0.8 ms of #007's 10.2 ms. **Bistability structurally suppressed** (one dip at E13 vs #014's three deep collapses); train_noaug → val gap +0.35 pp (essentially zero). **`n_samples=4` lift collapsed from #014's +9.6 pp to +2.6 pp** — confirming the soft-margin ceiling hypothesis (sharper per-sample softmaxes → averaging gain shrinks). **Stochastic samplers still broken**: `ddim_16_e1_n1 = 0.156`, `ddpm_64_e1_n1 = 0.012` (even worse than #014). Min-SNR did not lift absolute q3 quality (0.0046 → 0.0047) despite rebalancing q3/q0 ratio from 23× → 15×. New diagnosis: **matched_rate ceiling is in the diffusion design itself**, not the loss/sampler config — every config-level #014 failure mode is addressed yet ceiling sits at ~0.65. Next move: CARD-style anchored forward process (#015b) where diffusion learns residuals on top of an existing classifier's softmax. |
| [016](016-framewise-diffusion/) | Reframes the task entirely: replace #014/#015's next-bin diffusion (single-onset prediction over 501-class simplex) with **framewise activation-map diffusion** — output is `M ∈ [0, 1]^B_PRED` (B_PRED=500) over the future window, where M[b]=1 if a GT onset exists at bin offset b. New 1D Conv denoiser (~5.91 M params; smaller than #015's MLP) with FiLM modulation and per-bin audio-feature input channel (future-half audio tokens linearly upsampled from 125 → 500 along time axis). Target encoded as Gaussian σ=2-frame smoothed activation map. Decoder thresholds at τ=0.5 and emits all positive bins per AR step (multi-emit; cursor advances to last emitted; empty positive set → STOP_HOP=20). All #015's loss-side patches kept (Min-SNR γ=5, self-cond, asymmetric time). Total 22.27 M params (16.35 M trunk unchanged + 5.91 M Conv denoiser). 5-pass eval cadence introduced: EVAL_1 + NOAUG_1 (sampled-t leaky, full val 5%); EVAL_K + NOAUG_K (full T_inf-step rollout on 32 charts × 5 windows, with per-step convergence metrics, mini-chart at step K, and rendered GIFs); AR-corpus pass extended to 5 tolerances (5/10/20/40/100 ms). Drops HIT/GOOD/MISS/STOP_F1 (next-bin semantics). Adds 101-threshold sweep + tolerance grid stored as `eval_{step}/curves.npz`, per-sample per-step M_k tensors as `rollout_maps.npz`, plus convergence curves + best/p75/p50/p25/worst GIF renderings. | `Planned` | n/a (pre-run) |

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
