# Experiment 016b-mini — Min-SNR sign probe

## Status

`Planned`

## Context

[#016](../016-framewise-diffusion/) trained the framewise activation-
map diffusion design to step 103,370 (1 epoch of 5) and was halted
when its rollout artefacts showed the DDIM sampler **regressing
monotonically** across its 16 steps and **the regression growing
with training** (`rollout/final_vs_best_delta` −0.083 → −0.092 across
the 5 evals; `rollout/monotone_fraction` dropping to 0.425). Per-step
peaks bleached from mean value 0.911 at k=0 (graded confidences) to
0.997 at k=15 (saturated to ≈1.0 — threshold knob destroyed).

Root cause was traced to `training/framewise_diffusion_loss.py:137`:
the loss applies the **ε-prediction** form of Min-SNR
(`w = min(snr, γ) / snr`) to an **x0-parameterized** model. Per
Hang et al. 2023 §4.2, the canonical x0-mode form is
`w = min(snr, γ)` — no division by SNR. The applied (wrong-signed)
form multiplies the low-t (refinement-regime) gradient by `γ/snr → 0`,
training only the high-t "predict from pure noise + audio
conditioning" regime. The full per-t-quartile imbalance
`loss/per_t_q3 / loss/per_t_q0` grew 21.9× → 35.1× across the 5
evals — the model got progressively better at the audio-conditioned
first guess but never learned to refine, which is exactly the
sampler regression observed.

This experiment is a **minimum-scope probe** of that diagnosis. It
runs the same code path as #016 with **exactly one change**:
`snr_x0_mode: true` in `loss.json` (the new config flag added to
`FramewiseDiffusionLossConfig`). To keep the probe fast it also
trains on `subsample=8` (1/8 of training samples). Everything else
— trunk, denoiser, adapter, schedule, eval cadence, rollout hook —
is byte-identical to #016.

Goal: 1–2 evals' worth of evidence on whether the formula switch
flips the per-step sampler trajectory from regressing to converging.
If it does, #016b proper (full dataset, longer schedule, optionally
the BeatThis + ALIKE loss patches) is justified. If it doesn't, the
continuous-Gaussian path is wrong for sparse activation-map outputs
and #016c (D3PM absorbing-state) is justified instead.

## Citations

- [#016 — framewise activation-map diffusion](../016-framewise-diffusion/).
  Direct parent. Best `rollout/final_vs_best_delta` at eval 1:
  −0.083 [exp_016_framewise_diffusion, eval_20674/rollout_final_vs_best_delta.npy,
  mean over 160 samples]. Per-t-quartile ratio at eval 1: 21.9×
  [exp_016_framewise_diffusion, eval_20674/eval.json:metrics.loss/per_t_q3
  divided by loss/per_t_q0]. These are the numbers this probe needs
  to beat (or at least not match).
- [Efficient Diffusion Training via Min-SNR Weighting Strategy
  (Hang et al., ICCV 2023)](https://arxiv.org/abs/2303.09556) §4.2.
  Source of the x0-mode formula `w = min(snr, γ)`. Their Table 1
  verifies the strategy improves FID on both ε- and x0-prediction
  networks when each uses its own per-parameterization form.
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

Replacing the Min-SNR formula with its x0-parameterization form
(`min(snr, γ)` — no division) on otherwise-identical #016 code
**flips the per-step DDIM sampler trajectory from monotonically
regressing to non-regressing** within 1–2 evals on 1/8 dataset. The
**absolute headline numbers** (frame F1, AR `matched_rate`) are
expected to be **worse** than #016's eval 1 because the probe sees
8× fewer training samples, but the **per-step convergence signal**
must improve.

### Mechanism

The Min-SNR formula sets per-t loss weight. With the ε-form applied
to x0-prediction:

- low-t (refinement, near-clean x_t): `w = γ/snr → 0` ⇒ no gradient
  ⇒ model never learns to refine.
- high-t (near-noise x_t): `w = 1` ⇒ full gradient ⇒ model learns
  the audio-conditioned initial guess.

With the x0-form:

- low-t: `w = γ = 5` ⇒ refinement gets the full clipped weight.
- high-t: `w = snr → 0` ⇒ the initial-guess regime is naturally
  downweighted (its loss is already large by construction so this
  cancels out).

Under the x0-form, the model should produce gradient signal at every
t-quartile, and the DDIM chain at inference should monotonically
refine instead of regressing.

### Predicted numbers

Reference: #016's eval 1 (step 20,674) is the matched comparison
point — it represents 1/5 of an epoch with ε-form Min-SNR. The mini
probe trains on 1/8 the data per epoch and the predictions below
assume 1 eval at roughly step 2,584 (~1/4 epoch × 1/8 data).

| Metric | #016 eval 1 | Predicted (#016b-mini eval 1) | Direction the prediction tests |
|---|---:|---:|---|
| `loss/per_t_q3 / loss/per_t_q0` | 21.9× | **≤ 5×** | imbalance collapses ⇒ refinement regime gets gradient |
| `loss/per_t_q0` | 0.0033 | **≥ 0.01** | low-t loss is now load-bearing (instead of pre-trained-to-zero by the kill-weight) |
| `rollout/final_vs_best_delta` (mean) | **−0.083** | **≥ −0.03** (ideally ≥ 0) | sampler stops regressing — chain refines or holds |
| `rollout/monotone_fraction` (mean) | 0.429 | **≥ 0.55** | majority of samples improve through the chain |
| `rollout/best_k_step` (median) | 3.0 | **≥ 6** | best F1 moves later in the chain (model uses refinement budget) |
| frame F1 (τ=0.5, ±2 frames) | 0.869 | 0.60–0.78 | worse than #016 in absolute terms — expected, 1/8 data |
| frame `pos_rate_pred_50` | 0.176 | 0.05–0.15 | model emits fewer hot bins as it learns to be sparser |
| AR `matched_rate` (gt_cond) | 0.969 | 0.70–0.92 | worse than #016 (less training); not gated on |
| Peak value at k=15, mean | 0.997 | **≤ 0.95** | peaks remain graded, not saturated to 1.0 |
| Peak value at k=15, frac > 0.95 | 99.4 % | **≤ 80 %** | saturation collapse is averted |

The probe is gated on the convergence-signal metrics (rows 1–5 and
the last two). The headline F1 / matched_rate rows are expected to
*regress*; that's a normal consequence of training on 1/8 data and
does not invalidate the probe.

## Success criteria

- **Must have:** `rollout/final_vs_best_delta ≥ −0.03` at eval 1 or
  eval 2 — the per-step sampler trajectory is no longer in the
  monotonically-regressing regime that defined #016.
- **Must have:** `loss/per_t_q3 / loss/per_t_q0 ≤ 5×` — the per-t
  quartile imbalance has collapsed from #016's 21.9× starting
  point, confirming low-t gradient flow is restored.
- **Must have:** training stable, no NaN, ran past eval 1 without
  divergence; `frame/separation > 0.3` (model is producing
  distinguishable activations).
- **Nice-to-have:** `rollout/monotone_fraction ≥ 0.55` — majority
  of rollout samples are non-decreasing in F1 across the 16 sampler
  steps.
- **Nice-to-have:** k=15 peak mean ≤ 0.95 — peaks remain graded.
- **Fails if:** `rollout/final_vs_best_delta ≤ −0.06` at every eval
  — the per-step regression persists at #016-magnitude, meaning
  Min-SNR sign was not the dominant cause. In this case the
  continuous-Gaussian path is structurally wrong for sparse maps
  and #016c (D3PM absorbing-state) opens.
- **Fails if:** training diverges (loss NaN, `frame/pos_rate_pred_50`
  collapses to 0 or > 0.5).

Headline numbers (F1, AR matched_rate) are **not gated** — they
are expected to be lower than #016 because the probe trains on 1/8
the data, and the probe's purpose is the convergence-signal pivot.

## Changes from baseline

Baseline: [#016 — framewise activation-map diffusion](../016-framewise-diffusion/).

Code change (single file):

- **`osu/taiko2/training/framewise_diffusion_loss.py`** —
  `FramewiseDiffusionLossConfig` gains a `snr_x0_mode: bool = False`
  field (default preserves #016 behaviour for backcompat).
  `_snr_weights()` branches on it: when `True`, returns
  `min(snr, γ)` (x0-mode, Hang 2023 §4.2); when `False`, returns
  `min(snr, γ) / snr` (ε-mode, the buggy default that #016 ran on).
  Tested in `tests/test_framewise_pipeline.py::TestLoss::test_snr_x0_mode_flips_weight_direction`
  — asserts the two formulas weight low-t and high-t in opposite
  directions, and produce numerically-different end-to-end losses.

Config changes (this experiment's `config/`):

- **`loss.json`** — `snr_x0_mode: true` (the change being probed).
  Everything else byte-identical to #016/config/loss.json.
- **`data.json`** — `subsample: 8` (1/8 training samples; 8× fewer
  steps per epoch). Probe-only.
- **`trainer.json`** — `epochs: 2` (cap; the run is expected to be
  stopped manually at eval 1 or eval 2).
- **`model.json`, `adapter.json`, `infer.json`** — byte-identical
  to #016 modulo the run-dir reference in `infer.json:checkpoint`.
- **`ablation_matrix.json`** — removed. The probe doesn't run
  ablations; the full matrix can be re-introduced in #016b proper
  if the probe passes.

## Run config

- Run name: `exp_016b_mini`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1` (80-row mel; same as #007/#013/#014/#015/#016).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_016b_mini \
      --config-dir osu/taiko2/experiments/016b-mini/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/016b-mini/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```
  Same CLI as #016; the only behavioural difference comes through
  the config-dir contents (`loss.json:snr_x0_mode` + `data.json:subsample`).
- Expected: first eval fires at ~step 2,584 (1/4 epoch × 1/8 data).
  Wall time per eval expected ≈ 30–45 min (corpus pass + rollout
  pass are not subsampled).

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

_(filled post-run)_

## Visualizations

_(post-run)_

## Vs prediction

_(post-run)_

## Takeaways

_(post-run)_

## Followup questions

_(post-run; expected candidates depending on outcome: #016b proper
with full data + remaining literature patches (binary target,
BeatThis max-pool BCE, ALIKE dispersity peak) if the probe passes;
#016c with D3PM absorbing-state if it doesn't.)_
