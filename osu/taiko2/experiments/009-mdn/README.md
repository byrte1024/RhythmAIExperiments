# Experiment 009 — Mixture Density Network (embrace multi-modality)

## Status

`Stopped early` — sigma inflation. Components did NOT collapse
(n_active=2.2, the split is real) but learned sigma=18–27 bins,
making predictions imprecise. Headline miss +11 pp worse than #007.
Continued as [#009b](../009b-mdn-capped-sigma/) with capped sigma.

## Context

Experiments #005, #007, and #008 converged on one conclusive
finding: the `±log 2` / `±log 3` ratio-banding ridges are a
**capability** problem, not a loss-shape problem. The model's hidden
states genuinely cannot distinguish `t` from `2t` in some cases, and
no loss-side change (Gaussian CE, log-ratio EMD, time-stretch
augmentation) resolved the ridges. However, taiko1 experiments (39,
44-C) showed that the **correct answer IS in the model's top-K
90%+ of the time** — the model sees the right onset, it just can't
commit to it.

Every loss we've tried forces the model to produce a SINGLE peaked
distribution. That fights the natural multi-modality: when the model
genuinely sees `t` and `2t` as plausible, a single-peak loss forces
it to hedge (bimodal softmax) or collapse to one (losing the other).
The ridges are the visible consequence of that hedging.

**Idea: stop fighting multi-modality, embrace it.** Replace the
501-class softmax head with a Mixture Density Network (K=3 Gaussian
components). The model explicitly outputs K candidate peaks, each
with a position (mu), width (sigma), and weight (pi). The loss only
requires that SOME component covers the target — the other
components can freely sit at `2t`, `t/2`, `3t`, etc. without any
penalty. The model is now ALLOWED to say "I think it's either `t` or
`2t`" — and the per-component heatmaps will show us exactly where
the ambiguity lives.

This is a **diagnostic experiment first, accuracy experiment second.**
We expect headline metrics to be slightly worse (continuous mu
rounding loses bin precision vs discrete softmax), but the
per-component heatmaps are the real deliverable: if the components
specialize into "correct tempo", "octave up", "octave down", we've
mapped the ambiguity structure for the first time — and that's the
information needed to design a targeted fix.

## Citations

- Baseline: [#007 — time-stretch](../007-time-stretch/). Best val
  miss 0.2406 at step 372,132. Same augmentation pipeline as #009.
- Loss-side exhaustion evidence:
  [#005](../005-gaussian-ce/), [#008](../008-log-emd/). Three loss
  families tested, ridges unchanged.
- Top-K oracle evidence: taiko1 exp 39 (83.2 % of overpredictions
  match real future onsets), exp 44-C (top-U 3 oracle = 91.8 % HIT).
- MDN foundational: Bishop 1994, "Mixture Density Networks."
- Multi-hypothesis precedent: Rupprecht et al. 2017, "Learning in
  an Uncertain World" (winner-takes-all for ambiguous targets).

---

## Hypothesis

### Claim

If we replace the 501-class softmax head with a K=3 Gaussian MDN
(+ sigmoid STOP gate) and keep everything else identical to #007,
the per-component heatmaps will show **component specialization**:
at least one component will sit near the target while others sit
near octave / triplet multiples. The headline watched metric (val
miss) may regress by up to 3 pp relative to #007, but the
**diagnostic value** of seeing where each component lives is the
primary goal.

### Mechanism

The MDN mixture likelihood `-log(Σ_k π_k N(t | μ_k, σ_k))` only
requires SOME component to cover the target for low loss. Other
components are free to predict anything. If the model's hidden states
carry information about multiple plausible onset positions (which
taiko1's top-K oracle data strongly suggests), the K=3 MDN gives
them a place to go — each component can represent one of the model's
"hypotheses" about where the next onset is.

The per-component heatmaps (`mdn/comp{k}_heatmap.png` and
`mdn/comp{k}_ratio_error.png`) will show whether the components
specialize or collapse. Collapse (all K components at the same
position) means the model doesn't have multi-modal internal state
to express. Specialization (components at different ratio-multiples)
means the model DOES see multiple candidates and the MDN lets it
express them.

### Predicted numbers

Reference: #007 @ best (E18, step 372,132).

| Metric | #007 @ E18 | Predicted (#009, best eval) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.2406 | 0.24–0.27 | may regress — continuous mu loses bin precision |
| val/single/onset/exact | 0.5748 | ≥ 0.45 | expected lower — rounding from continuous mu |
| mdn/coverage_2bin | n/a | ≥ 0.85 | oracle: some component within ±2 bins of target |
| mdn/n_active_components | n/a | ≥ 1.5 | multiple components used, not collapsed to 1 |
| comp{k}_heatmap | n/a | visible specialization | the headline qualitative prediction |

## Success criteria

- **Must have:** per-component heatmaps show visible specialization
  — at least one component's heatmap has a clean diagonal (correct
  predictions) while at least one other has mass along octave lines
  (`p = 2t` or `p = t/2`).
- **Must have:** `mdn/coverage_2bin` ≥ 0.80 by best eval (the
  model's mixture covers the target most of the time).
- **Must have:** `mdn/n_active_components` ≥ 1.5 averaged across
  bin-target samples (components are not collapsing to one).
- **Must have:** training runs to completion without NaN.
- **Nice-to-have:** val miss ≤ 0.27 (within 3 pp of #007).
- **Nice-to-have:** the combined ratio_error heatmap has WEAKER
  ridges than #007's (multi-modal expression resolves some hedging).
- **Fails if:** all 3 components collapse to the same position
  (n_active_components ≈ 1, heatmaps identical).
- **Fails if:** val miss > 0.35 (MDN fundamentally can't learn).

## Changes from baseline

Baseline: [#007](../007-time-stretch/).

- `config/model.json` — `n_mdn_components: 3` (was 0). Output head
  changes from `Linear(384, 501) + Conv1d_smooth` to
  `Linear(384, 10)`. Output tensor is `(B, 10)`: 1 STOP gate +
  3 × (μ_raw, log_σ, log_π).
- `config/loss.json` — swap `OnsetLossConfig` → `MdnLossConfig`
  (`n_components=3, b_pred=500, stop_weight=1.5`). Mixture NLL +
  sigmoid-gated STOP BCE.
- `config/infer.json` — swap `ArgmaxDecoder` → `MdnDecoder`
  (`n_components=3, b_pred=500, stop_threshold=0.5`). Picks
  highest-π component's rounded μ as the bin prediction.
- `training/metrics_onset.py` + `training/artifacts.py` — MDN-aware
  prediction decoding (detects output shape, parses MDN params
  instead of argmax).
- New `MdnComponentArtifact` — auto-included when
  `n_mdn_components > 0`. Saves per-component heatmaps + ratio-
  error + combined to `{eval_dir}/mdn/`.
- Everything else identical to #007: augmentations (including
  TimeStretch), dataset, optimizer, schedule, seed.

## Run config

- Run name: `exp_009_mdn`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1`, splits `train` / `val` (90 / 10, seed 42).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_009_mdn \
      --config-dir osu/taiko2/experiments/009-mdn/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/009-mdn/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

---
<!-- Everything below written after the run. Do not pre-populate. -->
---

## Results summary

Run stopped at **eval 5 / step 103,370** after sigma inflation was
identified at E1 and confirmed persistent through E5. Best val miss
was **eval 3 (0.3450 @ step 62,022)**, +7.8 pp worse than #007 at
the same step. The MDN DID express multi-modality (n_active=2.2,
components 0 and 2 specialized) but component 1 inflated to sigma=48
and degraded overall precision. Continued as
[#009b](../009b-mdn-capped-sigma/) with `max_sigma=3.0`.

### Final vs baseline

At matched step (E5, step 103,370):

| Metric | #007 @ E5 | #009 @ E5 | Δ |
|---|---:|---:|---:|
| val/single/onset/miss | 0.2665 | 0.3800 | **+11.4 pp** |
| val/single/onset/hit  | 0.7243 | 0.5824 | −14.2 pp |
| val/single/onset/exact | 0.5500 | 0.2495 | **−30.1 pp** |
| val/single/onset/rhit | 0.6232 | 0.3658 | −25.7 pp |
| val/single/onset/frame_err_p90 | 32 | 50 | +18 |
| val/single/onset/stop_f1 | 0.5448 | 0.4088 | −13.6 pp |

### Per-eval progression

| E | Step | miss | hit | exact | coverage_2 | coverage_5 | n_active | dominant_w | correct_w | mean_sigma | mixture_nll | stop_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,674 | 0.3832 | 0.5055 | 0.1606 | 0.632 | 0.788 | 2.17 | 0.672 | 0.582 | **26.9** | 3.17 | 0.058 |
| 2 | 41,348 | 0.3606 | 0.5694 | 0.1958 | 0.716 | 0.852 | 2.37 | 0.645 | 0.549 | **20.9** | 2.96 | 0.262 |
| **3** | **62,022** | **0.3450** | 0.6155 | 0.2779 | 0.795 | 0.870 | 2.29 | 0.665 | 0.563 | **17.3** | 2.65 | 0.379 |
| 4 | 82,696 | 0.3583 | 0.5966 | 0.2389 | 0.780 | 0.880 | 2.27 | 0.656 | 0.557 | **19.5** | 2.73 | 0.235 |
| 5 | 103,370 | 0.3800 | 0.5824 | 0.2495 | 0.765 | 0.832 | 2.21 | 0.679 | 0.567 | **18.2** | 2.84 | 0.409 |

Mean sigma stayed in the 17–27 bin range across all evals,
fluctuating but never approaching the 1–3 bin range needed for
bin-level precision. `coverage_2bin` peaked at 0.795 (E3) — below
the 0.85 target. `n_active_components` stayed in the 2.2–2.4
range, confirming the split IS real.

### Per-component analysis (E3, step 62,022)

| Component | mean pi | mean sigma | Specialization |
|---|---:|---:|---|
| **comp 2** | **0.546** | **2.0** | **Dominant predictor** — strong diagonal in heatmap, tight sigma, carries >50 % of mass. The model's primary answer. |
| comp 0 | 0.283 | 1.7 | **Short-gap specialist** — mass at low targets (0–100 bins). Tight sigma. Genuine sub-task learning. |
| comp 1 | 0.171 | **47.9** | **Junk drawer** — sigma inflated to ~48 bins. No diagonal structure, diffuse vertical streaks. Absorbs "I don't know" probability. Classic MDN pathology. |

Components 0 and 2 learned useful, sharp specializations. Component
1 exploited the uncapped sigma to become a catch-all wide Gaussian
that covers the target cheaply without placing mu precisely.

## Visualizations

Per-component heatmaps saved at `runs/exp_009_mdn/eval_62022/mdn/`:

- `comp0_heatmap.png` — short-gap specialist. Mass concentrated at
  targets 0–100, diagonal visible but limited to the low range.
- `comp1_heatmap.png` — **the pathological component.** Diffuse
  vertical streaks across all mu values, no diagonal structure.
  sigma=48 bins means each Gaussian is a flat blob ~100 bins wide.
  This is what "sigma inflation" looks like.
- `comp2_heatmap.png` — dominant predictor. Strong diagonal
  throughout the full target range. Octave bands (2x, 0.5x) faintly
  visible alongside the diagonal, matching the ridge pattern from
  #002/#007/#008.
- `combined_heatmap.png`, `combined_ratio_error.png` — argmax-pi
  prediction, dominated by comp 2 since it carries 55 % of the
  weight.

## Vs prediction

- val miss ≤ 0.27: actual **0.3450** → **MISS** by 7.5 pp.
- val exact ≥ 0.45: actual **0.2779** → **MISS** by 17 pp.
- `mdn/coverage_2bin` ≥ 0.85: actual **0.795** → **MISS**.
- `mdn/n_active_components` ≥ 1.5: actual **2.29** → **MET**.
- Per-component specialization: **partially met** — 2 of 3
  components specialized, 1 inflated.

## Takeaways

- **The MDN split is real.** Components 0 and 2 genuinely
  specialized into different sub-tasks with tight sigma (1.7 and
  2.0). The model DOES have multi-modal internal state that the MDN
  lets it express. This is the first time we've been able to SEE
  the model's internal hypothesis structure — prior experiments only
  saw the collapsed single-peak output.
- **Sigma inflation is the dominant failure mode.** Component 1's
  sigma=48 is 24× the useful range. The MDN loss rewards covering
  the target with a wide Gaussian over precisely placing mu — a
  known MDN pathology. This is NOT a fundamental problem with the
  MDN approach; it's a hyperparameter problem. Capping sigma at
  3 bins should force all components to be precise.
- **Bin-precision metrics are not meaningful with uncapped sigma.**
  The −30 pp on `exact` and −26 pp on `rhit` are artifacts of the
  inflated component, not reflections of the model's actual
  knowledge. Coverage_2bin at 0.80 says 80 % of samples have a
  component within 2 bins — the model KNOWS the answer, the output
  format just makes it imprecise.
- **Train_noaug gap near zero (−0.16 pp).** The MDN is not
  overfitting — the head has fewer params than the softmax head and
  the diffuse predictions don't memorize the training distribution.
  A secondary benefit of the MDN approach.

## Followup questions

- **Does capping sigma fix the precision without killing the split?**
  → [#009b](../009b-mdn-capped-sigma/) with `max_sigma=3.0`.
- **If 009b works: what does the per-component ratio_error show?**
  With sigma=3 max, each component's heatmap should clearly show
  whether it's on the diagonal (correct) or on an octave line
  (alternative). This is the diagnostic goal of the MDN series.
- **Could the K=3 MDN eventually match #007's headline metrics?**
  Only if mu placement becomes as precise as softmax argmax, which
  requires sigma in the 1–2 bin range. `max_sigma=3` is a step;
  if precision is still low, try `max_sigma=2` or fixed sigma.
