# Experiment 009 — Mixture Density Network (embrace multi-modality)

## Status

`Planned`

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

_(To fill post-run.)_

### Final vs baseline

_(Table.)_

### Per-eval progression

_(Table.)_

## Visualizations

_(Graphs post-run.)_

## Vs prediction

_(One line per predicted metric post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
