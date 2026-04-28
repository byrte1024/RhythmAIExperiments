# Experiment 009b — MDN with capped sigma (max_sigma=3.0)

## Status

`Complete` — hypothesis **rejected**. Capping sigma at 3.0 from step
0 made the model unable to learn mu placement. Coverage_2bin dropped
from 0.77 (#009) to 0.29; hit dropped 28 pp; STOP broke entirely.
The model needs wide sigma early as an exploration mechanism, then
tightening — a hard cap from the start is too restrictive.

## Context

[#009](../009-mdn/) confirmed the MDN split IS real: components 0
and 2 specialized (sigma 1.7 and 2.0, pi 0.28 and 0.55) while
component 1 inflated to sigma=48 and became a "junk drawer"
absorbing uncertainty. Headline metrics suffered (miss +11 pp vs
#007) primarily because of the inflated component.

This experiment caps sigma at 3.0 bins so no component can inflate.
All three must place mu precisely — either specializing into
different prediction regions or collapsing together. The question:
with sigma capped, does component 1 learn something useful (e.g.
octave predictions) instead of going wide?

## Citations

- Direct parent: [#009 — MDN](../009-mdn/). Component analysis
  showed 2 of 3 components specializing, 1 inflating.
- Baseline for metrics: [#007 — time-stretch](../007-time-stretch/).

---

## Hypothesis

### Claim

With `max_sigma=3.0` all three MDN components will produce sharp
peaks. At least two components will show distinct specialization in
the per-component heatmaps (e.g. one on the diagonal, one on an
octave line). Coverage_2bin will exceed 0.85. Headline miss will
be within 5 pp of #007 (≤ 0.29).

### Predicted numbers

| Metric | #009 @ E3 | Predicted (#009b) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.3450 | ≤ 0.29 | sigma cap → sharper mu |
| val/single/onset/exact | 0.2779 | ≥ 0.35 | sharp sigma → better bin precision |
| mdn/coverage_2bin | 0.7948 | ≥ 0.85 | components forced to be precise |
| mdn/n_active_components | 2.29 | ≥ 2.0 | should stay multi-modal |
| mdn/mean_sigma | 17.3 | ≤ 3.0 | by construction |
| comp heatmap specialization | 2 of 3 | 3 of 3 | no junk drawer |

## Success criteria

- **Must have:** all 3 per-component heatmaps show distinct structure
  (no component is a featureless blob).
- **Must have:** `mdn/n_active_components` ≥ 1.5 (not collapsed).
- **Must have:** `mdn/coverage_2bin` ≥ 0.80.
- **Nice-to-have:** miss ≤ 0.27 (within 3 pp of #007).
- **Nice-to-have:** per-component ratio_error heatmaps show one
  "clean diagonal" component and at least one "octave band"
  component — mapping the ambiguity structure.
- **Fails if:** all 3 components collapse to identical positions
  (n_active ≈ 1).
- **Fails if:** miss > 0.35 (worse than uncapped #009).

## Changes from baseline

Baseline: [#009](../009-mdn/).

- `config/loss.json` — add `max_sigma: 3.0` (was uncapped).
  Sigma is clamped to `[1.0, 3.0]` after softplus. Everything
  else identical: same model head (K=3, output dim 10), same
  augmentations, same dataset.

## Run config

- Run name: `exp_009b_mdn_capped`.
- Config snapshots: [`config/`](./config/).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_009b_mdn_capped \
      --config-dir osu/taiko2/experiments/009b-mdn-capped-sigma/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/009b-mdn-capped-sigma/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

---
<!-- Post-run below -->
---

## Results summary

Run stopped at **eval 5 / step 103,370**. Best val miss was **E5
(0.3757)** — comparable to #009's 0.380 at the same step, but all
other metrics collapsed: hit −28 pp, exact −19 pp, coverage_2bin
−47 pp vs #009. STOP completely broken (f1=0.004).

### Final vs #009 at E5

| Metric | #009 @ E5 | #009b @ E5 | Δ |
|---|---:|---:|---:|
| miss | 0.3800 | 0.3757 | −0.4 pp |
| hit | 0.5824 | **0.3016** | **−28.1 pp** |
| exact | 0.2495 | **0.0641** | **−18.5 pp** |
| coverage_2bin | 0.7652 | **0.2925** | **−47.3 pp** |
| coverage_5bin | 0.8322 | 0.5901 | −24.2 pp |
| n_active | 2.21 | 1.75 | −0.46 |
| mean_sigma | 18.2 | **3.0** | capped |
| mixture_nll | 2.84 | **7.78** | **2.7× higher** |
| stop_f1 | 0.4088 | **0.0039** | broken |

### Per-eval progression

| E | Step | miss | hit | exact | cov_2 | cov_5 | n_active | dom_w | sigma | mix_nll | stop_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,674 | 0.4921 | 0.2088 | 0.0441 | 0.188 | 0.410 | 1.86 | 0.777 | 3.0 | 10.44 | 0.000 |
| 2 | 41,348 | 0.4657 | 0.2316 | 0.0485 | 0.203 | 0.442 | 1.76 | 0.798 | 3.0 | 10.05 | 0.000 |
| 3 | 62,022 | 0.4259 | 0.2613 | 0.0548 | 0.239 | 0.502 | 1.93 | 0.732 | 3.0 | 8.62 | 0.000 |
| 4 | 82,696 | 0.4075 | 0.2734 | 0.0600 | 0.259 | 0.538 | 1.79 | 0.774 | 3.0 | 8.30 | 0.000 |
| 5 | 103,370 | 0.3757 | 0.3016 | 0.0641 | 0.293 | 0.590 | 1.75 | 0.786 | 3.0 | 7.78 | 0.004 |

Miss is slowly improving (0.49 → 0.38) and coverage is climbing
(0.19 → 0.29) — the model IS learning mu placement, just very slowly
because the tight sigma makes every wrong mu catastrophically
expensive. At this rate it would take 30+ evals to reach #009's
coverage, let alone #007's headline metrics.

## Vs prediction

- miss ≤ 0.29: actual **0.3757** → **MISS**.
- exact ≥ 0.35: actual **0.0641** → **MISS** by 29 pp.
- coverage_2bin ≥ 0.85: actual **0.2925** → **MISS** by 56 pp.
- n_active ≥ 2.0: actual **1.75** → **MISS** marginally.
- mean_sigma ≤ 3.0: actual **3.0** → **MET** (by construction).
- 3 of 3 component specialization: not evaluable — model hasn't
  converged enough for heatmaps to be interpretable.
- miss > 0.35 (fails-if): **0.3757 at E5** → **triggered at E1–E4**,
  barely escaped at E5.

**All gated predictions missed except the trivial sigma cap.**

## Takeaways

- **Hard sigma cap from step 0 is too restrictive.** With σ=3, a mu
  that's 10 bins off target gets NLL penalty ~5.6 per component
  (vs ~0.13 at σ=20). The loss landscape becomes a field of steep
  narrow wells around each possible target, with vast flat deserts
  between — the optimizer can't navigate from a random mu to the
  right well. This is the fundamental problem: **the model needs
  wide sigma to EXPLORE, then tight sigma to EXPLOIT.**
- **The model IS slowly learning despite the handicap.** Miss
  dropped from 0.49 to 0.38 across 5 evals; coverage climbed from
  0.19 to 0.29. The gradient signal exists, it's just weak relative
  to the NLL magnitudes. More training might eventually work, but
  at ~10× the cost of #009 — not efficient.
- **STOP is collateral damage.** The mixture NLL dominates the total
  loss (7.78 vs stop_bce ~0.7), so the STOP gate's gradient is
  overwhelmed. STOP can't learn until the mixture NLL comes down to
  a comparable magnitude. A solution: scale stop_bce up, or use a
  separate optimizer for the STOP gate.
- **Next direction: sigma annealing or warm-start from #009.** Two
  paths to get "wide sigma for exploration → tight sigma for
  precision":
  1. **Anneal max_sigma** from 50 → 3 over training. Principled,
     requires a scheduler.
  2. **Warm-start from #009's E3 checkpoint** (comp 0 and 2 already
     at σ≈2) and continue with max_sigma=3. Cheapest — comp 0 and
     2 survive; comp 1 is forced to tighten from its inflated
     state.

## Followup questions

- **Does warm-starting from #009 with max_sigma=3 work?** Load
  #009's best checkpoint (where two components already specialized
  at σ≈2), resume training with the cap. Component 1 (σ=48) would
  be forced to collapse or specialize. Zero new code; test the idea
  in a few evals.
- **Would sigma annealing produce a better trajectory?** Linear
  anneal from max_sigma=50 (eval 1) to max_sigma=3 (eval 10) lets
  the model explore with wide sigma early and tighten gradually.
  More principled than warm-start but requires a scheduler hook.
- **Is K=3 the right number?** #009 showed comp 1 became a junk
  drawer while 0 and 2 specialized. With K=2, there's no spare
  component to inflate — both must be useful. Cheaper to train,
  cleaner diagnostic.
