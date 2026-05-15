# Experiment 016b-mini — Min-SNR sign probe

## Status

`Complete (hypothesis rejected). Stopped after eval 1 (step 2,584).`

The formula switch produced a measurable but small improvement on
the regression delta (−0.083 → −0.063, ~24 % less negative) and
flipped the per-t-quartile imbalance from ε-form to x0-form
direction, but did **not** fix the dominant pathology — the peak
saturation at k=15 is unchanged. Two of three must-have criteria
failed. Probe stopped at eval 1 rather than running eval 2; the
signal was unambiguous and eval 2 would not change the verdict.

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

The probe ran exactly one eval at **step 2,584** (~1/4 epoch on the
1/8-subsampled training set) before being stopped. The single eval
is enough to decide the hypothesis because the rollout-convergence
signal — the metric this probe is gated on — is computed directly
from the per-step `M_k` tensors saved by the rollout hook and does
not need additional training to stabilise.

### Final vs baseline

The fair comparison is #016 eval 1 (step 20,674), the
matched-position eval in #016 also at one eval into training but at
**8× more samples seen**. The probe's absolute headline metrics are
expected to be weaker than #016 eval 1 (less training); the
convergence-signal metrics are what matter.

| Metric | #016 eval 1 (step 20,674) | #016b-mini eval 1 (step 2,584) | Δ | Verdict |
|---|---:|---:|---:|---|
| `loss/per_t_q0` | 0.0033 [exp_016_framewise_diffusion, step 20,674, val/single/loss/per_t_q0] | 0.0044 [exp_016b_mini, step 2,584, val/single/loss/per_t_q0] | +33 % | low-t loss slightly larger but still ≪ 0.01 prediction |
| `loss/per_t_q3` | 0.0724 | 0.1591 | **+120 %** | high-t loss 2.2× larger — x0-mode starves the noise-pred regime as expected |
| `q3 / q0` ratio | 21.9× | **35.87×** | +64 % | predicted ≤ 5×; got worse — both formulas asymmetric, just in opposite directions |
| `rollout/final_vs_best_delta` (mean) | −0.083 [exp_016_framewise_diffusion, step 20,674, rollout_final_vs_best_delta.npy] | **−0.063** [exp_016b_mini, step 2,584, rollout_final_vs_best_delta.npy] | +0.020 | predicted ≥ −0.03; got −0.063 — improved but not enough |
| `rollout/best_k_step` (median) | 3.0 | **1.0** | −2.0 | predicted ≥ 6; got 1.0 — best F1 now sits even earlier in the chain |
| `rollout/monotone_fraction` (mean) | 0.429 | 0.473 | +0.044 | predicted ≥ 0.55; modest improvement, well below target |
| `frame/precision_τ_50_tol_2` | 0.7801 | 0.6674 | −14 % | expected: model has 8× less training |
| `frame/recall_τ_50_tol_2` | 0.9901 | 0.9705 | −2 % | expected |
| `frame/f1_τ_50_tol_2` | 0.8694 | 0.7881 | −9 % | inside the predicted 0.60–0.78 band (just at top edge) |
| `frame/auc_pr` | 0.6010 | 0.4455 | −26 % | expected |
| `frame/auc_roc` | 0.9792 | 0.9525 | −3 % | expected |
| `frame/separation` (mean_pos − mean_neg) | 0.7804 | 0.6986 | −10 % | meets must-have (> 0.3) |
| `frame/pos_rate_pred_50` | 0.1759 | 0.2030 | +15 % | model still over-emits |
| Peak value at k=15, mean (NMS=3, > 0.5) | 0.997 | 0.997 | 0.000 | predicted ≤ 0.95 — **unchanged** |
| Peak value at k=15, frac > 0.95 | 99.4 % | 98.7 % | −0.7 pp | predicted ≤ 80 % — **essentially unchanged** |
| Peak value at k=0, mean | 0.911 | 0.890 | −2 % | k=0 graded as in #016 |
| Peak value at k=0, frac > 0.95 | 59.7 % | 53.7 % | −6 pp | k=0 slightly less saturated |
| AR `matched_rate` (gt_cond) | 0.9693 | 0.9618 | −0.8 pp | not gated; cluster-spam regime intact |
| AR `hallucination_rate` (gt_cond) | 0.3660 | 0.3950 | +2.9 pp | worse — cluster spam unchanged |
| AR `density_ratio` (gt_cond) | 14.42 | 16.15 | +12 % | model over-emits 16× the GT density |
| AR `error_median_ms` (gt_cond) | 1.71 | 1.94 | +0.23 ms | both well below the ≤ 12 ms gate |

Citations omitted on rows after the first four are from the same
two `eval.json` files (`runs/exp_016_framewise_diffusion/eval_20674/eval.json`
and `runs/exp_016b_mini/eval_2584/eval.json`) under the matching
metric path.

### Per-eval progression

Only one eval was run, so no progression table applies. Single-row
machine-readable copy at [`metrics.json`](./metrics.json).

### K-step F1 trajectory at this eval

Sampler step → mean F1 over 160 eval samples, taken from
`runs/exp_016b_mini/eval_2584/rollout_maps.npz:f1`:

```
k:    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
f1:  .629 .625 .616 .617 .619 .617 .610 .604 .598 .594 .588 .584 .581 .579 .579 .579
```

Drop from k=0 to k=15: 0.050. #016 eval 1's drop was 0.066 (from
0.761 → 0.695). The probe's chain has a 24 % smaller drop magnitude
and starts ~0.13 F1 lower at k=0 (model has 8× less training; the
high-t regime that produces the k=0 guess is now under-weighted by
the x0-form formula).

## Visualizations

Trainer-emitted artifacts:

- [`graphs/loss.png`](graphs/loss.png) — training loss curve.
- [`graphs/convergence_curves_step_2584.png`](graphs/convergence_curves_step_2584.png)
  — F1 / MSE / mass-at-target across the 16 sampler steps at eval 1,
  mean ± p10/p25/p75/p90 bands over 160 samples.
- [`graphs/framewise_heatmap_step_2584.png`](graphs/framewise_heatmap_step_2584.png)
  — predicted activation map vs target for 64 representative windows.
- [`graphs/framewise_distribution_step_2584.png`](graphs/framewise_distribution_step_2584.png)
  — distribution of predicted activations at GT-positive vs
  GT-negative bins.
- [`graphs/summary_histogram_step_2584.gif`](graphs/summary_histogram_step_2584.gif)
  — population-summary GIF of the K-step refinement.

## Custom analyses

- [`custom/min_snr_sign_delta/`](custom/min_snr_sign_delta/) —
  three direct head-to-head plots vs #016 eval 1: K-step F1
  trajectory, per-t-quartile loss bars, and peak-value histograms
  at k=0 and k=15.

## Vs prediction

| Metric | Predicted | Actual | Verdict |
|---|---:|---:|---|
| `loss/per_t_q3 / loss/per_t_q0` | ≤ 5× | 35.87× | **FAIL** — flipped to opposite direction, didn't balance |
| `loss/per_t_q0` | ≥ 0.01 | 0.0044 | **FAIL** — low-t loss is larger than #016's 0.0033 but still small |
| `rollout/final_vs_best_delta` (mean) | ≥ −0.03 | −0.063 | **FAIL** — improved 24 % but missed gate |
| `rollout/monotone_fraction` (mean) | ≥ 0.55 | 0.473 | **FAIL** — modest improvement from 0.429 |
| `rollout/best_k_step` (median) | ≥ 6 | 1.0 | **FAIL** — best F1 sits earlier in the chain, not later |
| frame F1 (τ=0.5, ±2) | 0.60–0.78 | 0.788 | **PASS** (just at upper edge of expected band) |
| frame `pos_rate_pred_50` | 0.05–0.15 | 0.203 | **FAIL** — model still over-emits |
| AR `matched_rate` (gt_cond) | 0.70–0.92 | 0.962 | **PASS** (above band — tolerance illusion intact) |
| Peak value at k=15, mean | ≤ 0.95 | 0.997 | **FAIL** — unchanged from #016 |
| Peak value at k=15, frac > 0.95 | ≤ 80 % | 98.7 % | **FAIL** — saturation unchanged |

Must-have criteria audit (pre-run):

- **`rollout/final_vs_best_delta ≥ −0.03`**: **FAIL** (actual −0.063).
  Sampler is still in the regressing regime, just less aggressively.
- **`loss/per_t_q3 / loss/per_t_q0 ≤ 5×`**: **FAIL** (actual 35.87×).
  Imbalance flipped direction; the formula asymmetry doesn't go
  away by changing which half is starved.
- **Training stable + `frame/separation > 0.3`**: **PASS** (0.699,
  no NaN, eval ran cleanly).

Nice-to-haves: `rollout/monotone_fraction ≥ 0.55`: FAIL (0.473).
Peak at k=15 ≤ 0.95: FAIL (0.997).

Fail-criteria: not triggered. `rollout/final_vs_best_delta` is
−0.063, which is **better** than the fail threshold of ≤ −0.06 (by
0.003) — narrowly avoids the formal "continuous-Gaussian path is
structurally wrong" conclusion. Training did not diverge.

**Summary**: 1 of 3 must-haves PASSED, 2 FAILED. The formula switch
produced real but small effects on the convergence-signal metrics
and **no effect** on the saturation behaviour at k=15. Hypothesis
(formula switch flips per-step trajectory from regressing to
non-regressing) is rejected.

## Takeaways

1. **The Min-SNR sign was a real bug, not the dominant one.** The
   regression delta improved 24 % (−0.083 → −0.063) and the per-t
   q3/q0 ratio swung past balanced to the opposite-direction extreme
   (21.9× → 35.87×). Both effects confirm the formula was being
   applied wrong-signed for x0-prediction, exactly as Hang 2023 §4.2
   describes. But the chain still regresses monotonically and the
   k=15 saturation pathology is unchanged.

2. **The formula choice is a starvation knob, not a balancing knob.**
   ε-form starves low-t (refinement). x0-form starves high-t
   (initial guess). Neither formula produces balanced per-t
   gradient flow. The q0 value moved from 0.0033 to 0.0044 (only
   +33 %) while q3 moved from 0.0724 to 0.1591 (+120 %) — the
   absolute magnitudes confirm the per-t loss-weight asymmetry is
   built into the Min-SNR family. Achieving balanced per-t loss
   would require either `snr_weighting=false` (uniform-t MSE) or
   abandoning the Min-SNR scheme for this parameterisation entirely.

3. **k=15 peak saturation is independent of Min-SNR.** Mean peak
   value at k=15 was 0.997 in #016 and 0.997 in this probe (NMS=3,
   raw > 0.5). Frac > 0.95 was 99.4 % vs 98.7 %. Both probes
   produce a chain whose final step bleaches all detected peaks to
   ≈1.0, regardless of which half of the t schedule the gradient
   flows through. The saturation is a consequence of the **target
   shape**, not the loss weighting: the σ=2 Gaussian-smoothed target
   has value 1.0 at GT bins and ~0.61 at ±2 bins, so a model that
   correctly predicts this target produces saturated peaks by
   construction. No amount of t-weight tuning fixes this.

4. **k=0 behaviour confirms the high-t starvation.** The probe's
   k=0 F1 is 0.629 vs #016's 0.761 (−0.132). k=0 maps to the
   highest-t sampler step (most-noise input) — exactly where the
   x0-form weight `min(snr, γ)` is dominated by `snr → 0`. The
   model is undertrained on the noise-conditioned initial guess
   precisely because that's what the new formula downweights. This
   is the asymmetry from takeaway 2 made concrete in the rollout.

5. **AR-corpus metrics confirm cluster spam unchanged.** `matched_rate`
   0.9618, `hallucination_rate` 0.3950, `density_ratio` 16.15 —
   essentially identical to #016 eval 1's 0.9693 / 0.3660 / 14.42.
   The model still over-emits ≈16× the GT density, just with a
   slightly weaker initial guess and a slightly less-regressing
   chain.

6. **Probe conclusion: Min-SNR sign correction is keepable but not
   load-bearing.** It costs ~5 LOC, has a real direction-of-correct
   effect on the per-t balance, and modestly improves the regression
   delta. But it does not unblock the framewise design on its own.
   The dominant pathology — the smoothed Gaussian target encoding
   1.0 at the centre and producing saturated-and-spread peaks at the
   final sampler step — has to be addressed at the target level,
   not the loss weighting.

## Followup questions

- **Target-shape redesign is now required.** Three candidates,
  ordered by closeness to existing code: (a) binary target at all t
  (smallest delta to #016; loses the σ=2 smoothing benefit at
  training time); (b) variable-σ target whose smoothness depends on
  t (large σ at high t, σ→0 at low t — implements coarse-to-fine
  refinement); (c) pure-blur Cold-Diffusion-style forward process
  with Algorithm 2 sampler (no Gaussian noise at all). Decision
  pending; planned for #016b proper.
- **Once the target is redesigned, re-evaluate Min-SNR.** A binary
  or near-binary target at low-t makes the low-t loss genuinely
  load-bearing (the model has to predict an exact spike, not
  identity-pass a near-clean input), at which point uniform-t
  weighting may be sufficient and Min-SNR can be turned off.
- **Why is the k=0 trajectory `final_f1 < best_f1` even though k=0
  is the only step trained well under x0-form?** Open. The chain
  produces best F1 at k=1 (median), not k=0, despite k=0 being the
  only step the model has gradient on. Probably reflects the
  reverse-step formula carrying some self-consistency benefit that
  isn't in the raw model output. Worth a deeper look only if the
  target redesign doesn't resolve it.
