# Experiment 015 — Diffusion head with literature-informed patches

## Status

`Planned`

## Context

[#014 — diffusion output head](../014-diffusion/) was the first
diffusion-based output head in the taiko2 series. The hypothesis
("iterative head + multi-sample marginalization lifts `matched_rate`
≥ 1 pp above #007's 0.7061") was **rejected on the headline** —
best gt_cond `matched_rate = 0.640` [014-diffusion/ablations/ddim_16_e0_n4/gt_cond/comparisons_summary.json:fields.matched_rate.median],
**−6.8 pp below #012's all-time best of 0.7080** [exp_012_onset_channels,
step 308,550, infer_corpus/eval_308550/gt_cond/comparisons_summary.json:fields.matched_rate.median]
— but the post-run analysis identified **three concrete structural
blockers** plus **two unused improvements from the diffusion literature**
that together justify a follow-up rather than abandoning the design.

The three blockers (per [#014's Takeaways](../014-diffusion/README.md#takeaways)):

1. **`decode_to_logits` soft-margin ceiling.** With
   `logits = x_0_hat / x0_scale` and `x0_scale = 2.0`, a perfectly
   predicted scaled one-hot produces a +1.0 logit margin over the 500
   non-positive bins. Softmax top-1 prob is capped at
   `exp(1) / (exp(1) + 500) ≈ 0.0054` (observed directly in #014's
   inference traces). The mean-of-softmax aggregation across
   `n_samples=4` therefore averages already-near-uniform distributions;
   the +9.6 pp lift `n_samples=4` produced on #014 is much smaller than
   it could be with sharper per-sample softmaxes.
2. **`stop_weight = 1.5` unconditional bias.** The default was
   inherited from `OnsetLoss` (#002, softmax-CE) where it counters
   class imbalance. Under MSE-on-scaled-one-hot the multiplier has
   the **opposite effect** — it tells the denoiser the STOP target's
   reconstruction error matters 1.5× more, which biases the
   denoiser's unconditional (high-t) output toward STOP. Manifested
   as #014's AR `density_ratio = 0.71` vs #007's 0.87 [exp_007_time_stretch,
   step 413,480, infer_corpus/eval_413480/gt_cond/comparisons_summary.json:fields.density_ratio.median].
3. **`loss/per_t_q3` plateau (q3 = high-t / near-prior regime).**
   At the final eval, `loss/per_t_q3 = 0.00465`, **22× larger than
   `loss/per_t_q0 = 0.00019`** and barely moved from E5 onward
   [exp_014_diffusion, step 248088, val/single/diff/loss/per_t_q*].
   Stochastic samplers (`eta > 0`) inject fresh noise at each
   reverse step, repeatedly landing the trajectory back in the
   undertrained q3 regime — collapsing all three eta=1 ablation
   variants to `matched_rate ≈ 0.06` (DDPM-64, the supposed
   "upper bound full-schedule reverse process", was the **worst**
   variant of the eight).

A literature pass over the modern diffusion stack (see Citations
below) identified that all three blockers have textbook fixes —
**Min-SNR weighting** (Hang et al. 2023) for (3) was already wired
as a config flag in #014's `DiffusionLossConfig` but never enabled;
**Brier-temperature-style logit decoding** (closest analog: CARD,
Han et al. 2022) for (1) is the literature's name for what the
post-run analysis called the "soft-margin ceiling"; **stop-weight
sign** is a one-line config flip. Two additional improvements
absent from #014 — **Self-Conditioning** and **Asymmetric Time
Intervals** (Chen, Zhang, Hinton 2022 — *Analog Bits*) — are
architecturally compatible with the current stack at low cost and
address per-step denoiser quality (which compounds across the
16-step DDIM sampler) and few-step DDIM truncation error
respectively.

This experiment stacks all five interventions in a single run.

## Citations

- Direct baseline (this run's parent): [#014 — diffusion output head](../014-diffusion/).
  Best gt_cond `matched_rate = 0.640`, `error_median_ms = 12.0 ms`,
  best fixed_cond `matched_rate = 0.720` [014-diffusion/ablations/ddim_16_e0_n4/{gt_cond,fixed_cond}/comparisons_summary.json:fields.*.median].
  Identical trunk, schedule, dataset, augmentation pipeline.
- Direct baseline for headline comparison (the taiko2 all-time best
  before #014): [#012 — onset-feature channels](../012-onset-channels/).
  Best gt_cond `matched_rate = 0.7080` [exp_012_onset_channels, step
  308,550, infer_corpus/eval_308550/gt_cond/comparisons_summary.json:fields.matched_rate.median];
  best val `miss = 0.2331` [step 349,690, val/single/onset/miss].
- The patches:
  - [Efficient Diffusion Training via Min-SNR Weighting Strategy (Hang et al., ICCV 2023)](https://arxiv.org/abs/2303.09556) —
    `w_t = min(SNR_t, γ)` on `‖x_0 − x̂_0‖²` for x0-prediction.
    γ=5 robust across {1, 5, 10, 20} (their Table 2). For x0+const
    weighting (our exact #014 setup), Min-SNR resolves the per-t
    gradient conflict that drives the q3 plateau (their Figure 6,
    [800, 900)-bin loss panel: const+x0 has the worst high-t loss
    of the four weightings tested).
  - [CARD: Classification and Regression Diffusion Models (Han, Zheng, Zhou, NeurIPS 2022)](https://arxiv.org/abs/2206.07275) —
    closest published analog for diffusing a categorical target.
    Provides the **Brier-temperature decoding** scheme
    (`Pr(y=k) ∝ exp(−(y_0 − 1_C)²_k / τ)`) that bypasses the
    soft-margin ceiling without changing the training-time process.
    Our `logit_scale = 5` is the linear-logit analog of CARD's
    temperature.
  - [Analog Bits: Generating Discrete Data Using Diffusion Models with Self-Conditioning (Chen, Zhang, Hinton, ICLR 2023)](https://arxiv.org/abs/2208.04202) —
    most architecturally similar reference (continuous diffusion on
    a continuous representation of categorical data — same family
    as our scaled-one-hot setup). Source for:
    - **Self-Conditioning**: at each reverse step the denoiser
      consumes the previous step's predicted `x_0` as extra input.
      Training uses a two-pass recipe (p=0.5: zeros; p=0.5:
      stop-grad output of a first denoising pass). Reported
      "significant" sample-quality lift.
    - **Asymmetric Time Intervals**: at sampling time, the denoiser
      is called with `t + ξ` while the reverse process still
      transitions to `t`. Improves few-step DDIM sample quality
      with no training-side change. Their Figure 3 shows visible
      few-pixel error reduction at large reverse steps.
- Other literature reviewed but not used in this run (saved for
  follow-up if this design lands below 0.72):
  - [Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., NeurIPS 2022)](https://arxiv.org/abs/2206.00364) —
    EDM preconditioning + log-normal σ sampling. More principled
    than DDPM but a substantial refactor (continuous-σ paradigm).
    Deferred to potential #015c.
  - [Progressive Distillation / v-parameterization (Salimans & Ho, ICLR 2022)](https://arxiv.org/abs/2202.00512) —
    `v = α_t·ε − σ_t·x_0`. Registered as a parameterization in
    `GaussianContinuousProcess` but unused. If Min-SNR alone
    doesn't fix q3, v-parameterization is the next axis.
  - [Perception Prioritized Training (P2 Weighting; Choi et al., CVPR 2022)](https://arxiv.org/abs/2204.00227) —
    `λ'_t = λ_t / (k + SNR)^γ`. Alternative to Min-SNR, similar
    spirit. Not stacked here to keep one weighting scheme at a
    time.
  - [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD; Lou, Meng, Ermon, ICML 2024)](https://arxiv.org/abs/2310.16834) —
    modern score-entropy approach to true discrete diffusion.
    Different framework entirely; reserved for a hypothetical
    #015c (D3PM-style follow-up) if continuous Gaussian on
    one-hot proves fundamentally limited.
  - [Simple and Effective Masked Diffusion Language Models (MDLM; Sahoo et al., NeurIPS 2024)](https://arxiv.org/abs/2406.07524) —
    masked / absorbing-state discrete diffusion. Doesn't apply
    to a single-token classification target (collapses to
    standard masked CE).
- Cross-experiment record: [`../README.md`](../README.md).

---
<!--
Everything above this divider may be written freely.
Everything between the two dividers is PRE-RUN and must be filled
BEFORE the run. Do not edit it afterwards — use the amendment rule.
-->
─────────────────────────────────────────────────────────────────────

## Hypothesis

### Claim

If we apply the following five literature-informed patches on top
of [#014's](../014-diffusion/) frozen architecture — everything else
identical (same trunk, dataset, augmentations, optimizer schedule,
seed) — then AR `matched_rate` (gt_cond, median across val charts)
will reach a best value **≥ 0.706** (matches #007's training-time
best) and **ideally ≥ 0.720** (new taiko2 best, +1.2 pp over #012's
all-time 0.7080):

1. **Min-SNR weighting** (`loss.snr_weighting = true`, `γ = 5`).
2. **Stop-weight to 1.0** (`loss.stop_weight = 1.0`, down from 1.5).
3. **Logit-scale = 5** (`process.logit_scale = 5.0`, sharpens
   decoded logits without changing argmax).
4. **Self-Conditioning** (`denoiser.self_cond = true`, +n_bins
   input channel; training uses the Analog-Bits two-pass recipe).
5. **Asymmetric time intervals** at sampling time
   (`sampler.time_offset` swept in the ablation matrix; default
   `0.0` for the per-eval AR-corpus hook).

### Mechanism

Each patch attacks a specific failure mode:

1. **Min-SNR-5 on x0-prediction**, `w_t = min(SNR_t, γ)`, balances
   per-t gradient pressure across the schedule. Hang et al. 2023's
   Figure 6 is direct evidence — with constant+x0 (our #014 setup)
   the [800, 900)-step bin has the worst loss of the four weightings
   tested. Their analysis maps cleanly onto #014's observed
   `q3 = 22 × q0` plateau. Min-SNR-5 rebalances away from low-t
   over-emphasis, indirectly improving q3 by reducing the multi-task
   gradient interference that was suppressing it.
2. **`stop_weight = 1.0`** removes the unconditional STOP bias from
   the denoiser's training signal. STOP samples (≈ 0.3 % of training
   under our class-balanced sampler) no longer get their MSE
   reconstruction error multiplied by 1.5 — eliminating the
   "denoiser prefers STOP-class output when uncertain" effect that
   manifested as #014's low `density_ratio = 0.71`.
3. **`logit_scale = 5`** removes the soft-margin ceiling.
   `decode_to_logits(x_0_hat) = (x_0_hat / x0_scale) * logit_scale`.
   A perfect prediction of the scaled one-hot now produces a +2.5
   logit margin (vs +1.0 before); softmax top-1 prob jumps from
   ~0.005 → ~0.94. **argmax decoding is unaffected** (scale
   invariance) — the gain is in (a) `n_samples > 1` mean-of-softmax
   aggregation, where averaging sharp per-sample softmaxes is
   strictly better than averaging near-uniform ones; (b) the
   `extras.top_k_prob` reporting; (c) any downstream confidence
   reading.
4. **Self-Conditioning** gives the denoiser access to its previous
   step's predicted `x_0` at every reverse step from i=1 onward.
   Each denoiser call therefore conditions on (cursor_token, x_t,
   t, prev_x0_hat) instead of just (cursor_token, x_t, t).
   Mechanism per Chen, Zhang, Hinton 2022: the model can use its
   prior estimate as a "stable backbone" while x_t carries the
   stochastic component, reducing the variance of the implied
   trajectory. Their reported quality lift is "significant" with
   ~25 % training-time overhead (the two-pass logic). Concretely,
   the MLP denoiser's first Linear grows from in_features 1141
   to 1642 (+501), totalling 24.47 M params vs #014's 23.70 M
   (+3.2 %).
5. **Asymmetric time intervals** is sampler-only and free at
   training time. With `time_offset = ξ ≥ 0`, the denoiser is
   called with timestep `min(t + ξ, T − 1)` while the reverse
   transition still goes `t → t_prev`. The denoiser "sees" the
   input as if it were slightly noisier than it actually is,
   compensating for the optimistic step-size assumption that
   accumulates DDIM truncation error in few-step sampling. The
   ablation matrix tests `time_offset ∈ {0.0, 1.0}` at 4-step
   DDIM (where Analog Bits's Figure 3 shows the biggest gains).

### Predicted numbers

Reference: [#007](../007-time-stretch/) (the headline pre-diffusion
baseline) and [#014](../014-diffusion/) (the failure mode this
experiment fixes). gt_cond medians from each run's
`infer_corpus/eval_{best}/gt_cond/comparisons_summary.json:fields.*.median`.

| Metric | #007 best | #014 best | Predicted (#015, best ablation variant) | Notes |
|---|---:|---:|---:|---|
| AR `matched_rate` (gt_cond, median) | 0.7061 | 0.640 | **≥ 0.706** must / **0.72–0.74** nice | (1)+(2)+(3) lift STOP-collapse-prone cursors; (4) tightens per-step prediction |
| AR `error_median_ms` | 10.17 | 12.0 | 9–12 | (1) sharpens denoiser; should hold close to #014's already-near-#007 number |
| AR `density_ratio` | 0.865 | 0.71 | **0.85–0.92** | (2) removes the STOP bias directly |
| AR `hallucination_rate` | 0.144 | 0.156 | 0.13–0.17 | sharper peaks → fewer spurious emissions; small effect either direction |
| AR `hi_pspace` | 0.909 | 1.000 | ≥ 0.95 | #014 already wins here; should stay there |
| AR `dc_human` | 0.928 | 0.905 | 0.91–0.93 | recover most of #014's 1-pp regression vs #007 |
| `loss/per_t_q3` at final eval | n/a | 0.00465 | **< 0.0025** | (1) Min-SNR breaks the multi-task interference |
| `loss/per_t_q3 / loss/per_t_q0` ratio | n/a | 22× | < 10× | direct test of the rebalance |
| Ablation: `ddim_16_e1_n1` `matched_rate` | n/a | 0.063 | **> 0.50** | stochastic samplers unblocked once q3 trains |
| Ablation: `ddpm_64_e1_n1` `matched_rate` | n/a | 0.062 | > 0.55 | DDPM-64 stops being the worst variant |
| val/single/onset/miss (at-sampled-t, leaky) | 0.241 (clean) | 0.134 (leaky) | leaky; not gated | the leaky metric is not comparable across loss families |
| train_noaug → val gap @ best | −3.50 pp | +0.49 pp | −2 to +1 pp | logit_scale doesn't change overfit shape |
| Wall time / eval | 2.18 h | 2.04 h | 2.3–2.6 h | self-cond two-pass training adds ~25 % per step; AR-corpus hook now also pays one extra denoiser call per reverse step for self-cond |
| Total params | 16.35 M (#007) | 23.70 M | **24.47 M** | self-cond adds 769 k (+3.2 % over #014). Comparable confound to #014. |

Observational (not gated):

- **`n_samples` curve.** #014's `n=1 → n=4` lift was +9.6 pp on
  gt matched_rate (0.544 → 0.640). With logit_scale=5, per-sample
  softmaxes are already sharp; the mean-of-softmax aggregation
  should still help, but the marginal gain may shrink. If
  `n=4 − n=1` is < 3 pp in this run, logit_scale is doing what
  it should (and `n=1` is approaching the sharper-softmax ceiling).
- **Self-cond ablation.** Cannot disentangle within this run —
  filed for #015a-vs-#015b future work if needed.
- **`±log(2)` ratio-banding ridges.** Per #014, these are
  capability-bound (present on both val and train_noaug). None of
  the five patches addresses them structurally. Expectation: ridges
  visible at #014's intensity in `ratio_error.png`.

## Success criteria

- **Must have:** AR `matched_rate` ≥ 0.706 at the best AR-corpus
  eval across any ablation variant (matches #007's training-time
  best, the floor for "diffusion is competitive").
- **Must have:** training stable, no NaN / Inf; loss curve descends
  through E5; `loss/per_t_q3` drops below 0.0025 by E10 (direct
  test of the Min-SNR rebalance).
- **Must have:** at least one `eta > 0` variant in the post-run
  ablation reaches `matched_rate > 0.50` (direct test that q3 is
  no longer catastrophic).
- **Must have:** AR `density_ratio` ≥ 0.82 at the best variant
  (direct test of the stop-weight fix).
- **Nice-to-have:** AR `matched_rate` ≥ 0.720 — new taiko2 SOTA,
  beats #012's 0.7080.
- **Nice-to-have:** AR `matched_rate` ≥ 0.730 — clear architectural
  win that justifies the diffusion design as the new baseline.
- **Nice-to-have:** `time_offset = 1.0` lifts `ddim_4_e0_*` by
  ≥ 1 pp over `time_offset = 0.0` at matched n_samples (Analog
  Bits's Figure-3-equivalent in our domain).
- **Fails if:** AR `matched_rate` < 0.66 at every eval across every
  variant — the patches are insufficient; CARD-style anchored
  diffusion (#015b) becomes the next move.
- **Fails if:** `loss/per_t_q3` stays > 0.004 after E10 — Min-SNR
  failed to break the multi-task interference; would suggest q3 is
  intrinsically hard given the current trunk capacity, not just
  underweighted.
- **Fails if:** training diverges or loss curve plateaus before E5
  — the patches combined are introducing instability (e.g.
  self-cond two-pass requires a higher-variance training signal
  than no-self-cond).

## Changes from baseline

Baseline: [#014 — diffusion output head](../014-diffusion/).

Code changes:

- **`osu/taiko2/domain/diffusion.py`**:
  - `DenoiserConfig` gained `self_cond: bool = False` (backward-
    compatible default — old configs / checkpoints load unchanged).
  - `DenoiserHead.forward` gained an optional `prev_x0_hat: torch.Tensor | None = None`
    kwarg. Concrete denoisers may ignore it (the default behavior
    when `self_cond = False`). Backward-compatible — no existing
    concrete impl breaks.
- **`osu/taiko2/diffusion/processes.py`**:
  - `GaussianContinuousProcessConfig` gained `logit_scale: float = 1.0`.
    Default preserves #014's `decode_to_logits` output exactly.
  - `decode_to_logits` formula changed from `x_0_hat / x0_scale`
    to `x_0_hat * (logit_scale / x0_scale)` — identical at
    `logit_scale = 1.0`.
- **`osu/taiko2/diffusion/denoisers.py`**:
  - `MLPDenoiser.__init__` expands the first `Linear`'s `in_features`
    by `n_bins` when `self_cond` is enabled.
  - `MLPDenoiser.forward` accepts `prev_x0_hat` (defaults to zeros
    when `None` under self-cond mode).
- **`osu/taiko2/diffusion/samplers.py`**:
  - `DDIMSamplerConfig` gained `time_offset: float = 0.0`. Default
    matches #014's behavior.
  - Both `DDPMSampler.sample` and `DDIMSampler.sample` thread
    `prev_x0_hat` through the reverse loop when the bound
    denoiser's config sets `self_cond = True`.
  - `DDIMSampler.sample` honors `time_offset`: the denoiser is
    called with `min(t + offset, T − 1)` while the reverse
    transition still uses `t`.
- **`osu/taiko2/models/diffusion_detector.py`**:
  - `forward_diffusion` gained a `self_cond_prob: float = 0.5`
    kwarg and implements the Analog-Bits two-pass training recipe
    when the denoiser opts in: with prob `self_cond_prob`, do a
    no-grad first denoise pass and use its (stop-grad) predicted
    `x_0` as `prev_x0_hat` for the loss pass; otherwise zeros.
- **`osu/taiko2/training/diffusion_loss.py`**:
  - The inline diffusion forward (the branch that handles inference-
    shape outputs from the train loop's `model.predict`) mirrors
    the model's two-pass logic for self-cond.
- **Tests**: 15 new tests across `tests/test_diffusion.py` (logit
  scale, self-cond input dim / forward signature, asymmetric time
  intervals) and `tests/test_diffusion_detector.py` (two-pass
  training, inline loss path with self-cond, integration
  logit_scale). Full suite at 489/489 passing.

Config changes (this experiment's `config/*.json`):

- `config/loss.json`: `snr_weighting: false → true`,
  `stop_weight: 1.5 → 1.0`. Everything else identical to #014.
- `config/model.json`: `process_config.logit_scale: 1.0 → 5.0`
  (new field — default preserves old behavior), `denoiser_config.self_cond: false → true`.
  Everything else (n_steps=64, parameterization="x0", x0_scale=2.0,
  hidden_dim=1536, n_layers=3) identical to #014.
- `config/infer.json`: decoder's sampler config gains
  `time_offset: 0.0` (new field, default). Ablation matrix sweeps
  it.
- `config/{trainer,data,adapter}.json`: byte-identical to #014.
- `config/ablation_matrix.json`: 10 variants (vs #014's 8). New
  axes: `ddim_4_e0_n1_off1` (asymmetric time at 4 steps),
  `ddim_4_e0_n4` (combined Pareto: 4 steps × 4 samples = same
  compute as #014's n=1 reference), `ddim_4_e0_n4_off1` (the
  candidate inference config). Ordered most-decisive-first per
  the #014 convention.

## Run config

- Run name: `exp_015_diffusion_patched`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1` (80-row mel; same as #007 / #013 / #014).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_015_diffusion_patched \
      --config-dir osu/taiko2/experiments/015-diffusion-patched/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --benchmarks all --benchmark-fraction 0.05 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/015-diffusion-patched/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```
- Post-run ablation:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.diffusion_sampler_ablation \
      --base-spec osu/taiko2/experiments/015-diffusion-patched/config/infer.json \
      --matrix osu/taiko2/experiments/015-diffusion-patched/config/ablation_matrix.json \
      --dataset taiko2_v1 \
      --out-dir osu/taiko2/experiments/015-diffusion-patched/ablations
  ```

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

_(filled post-run)_

### Final vs baseline

| Metric | Baseline (#014) | This run (final) | Δ | Direction |
|---|---:|---:|---:|:---:|
| AR `matched_rate` (gt_cond, median) | 0.640 | — | — | — |
| AR `error_median_ms` | 12.0 | — | — | — |
| AR `density_ratio` | 0.71 | — | — | — |
| `loss/per_t_q3` (final) | 0.00465 | — | — | — |

### Per-eval progression

_(generated post-run from `runs/exp_015_diffusion_patched/metrics.jsonl`)_

Machine-readable copies (both tables): [`metrics.json`](./metrics.json).

## Visualizations

_(post-run)_

## Custom analyses

- [Sampler ablation matrix](ablations/) — output of
  `cli.diffusion_sampler_ablation`. 10 variants covering steps ∈
  {4, 8, 16, 32, 64}, eta ∈ {0, 1}, n_samples ∈ {1, 4},
  time_offset ∈ {0, 1}.

## Vs prediction

_(post-run)_

## Takeaways

_(post-run)_

## Followup questions

_(post-run; expected candidates given hypothesis-level analysis:
#015b — CARD-style anchored forward process at #007's softmax;
#015c — full EDM preconditioning refactor; #016 — D3PM/SEDD-style
discrete diffusion.)_
