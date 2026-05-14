# Experiment 016 — Framewise activation-map diffusion

## Status

`Planned`

## Context

[#014](../014-diffusion/) and [#015](../015-diffusion-patched/)
explored diffusion as a replacement output head for the
[#007](../007-time-stretch/) softmax-CE next-bin classifier. Both
runs share the same task framing: at each AR cursor advance, the
model predicts **one** future event's bin offset over a 501-class
distribution (500 bin offsets + 1 STOP). Diffusion ran over that
501-dim simplex; the AR loop emitted one event per step.

Both diffusion runs hit the same ceiling — best gt `matched_rate`
≈ 0.65 [015-diffusion-patched/ablations/ddim_4_e0_n4_off1/gt_cond/comparisons_summary.json:fields.matched_rate.median = 0.6468,
014-diffusion/ablations/ddim_16_e0_n4/gt_cond/comparisons_summary.json:fields.matched_rate.median = 0.6398],
**−5.6 pp below #007's 0.7028** [exp_007_time_stretch, step
413,480, val/single/corpus/gt_cond_cmp/matched_rate_mean]. #015's
post-run analysis identified the ceiling as **structural to the
diffusion design** (not the loss / sampler / decoder config) —
every named config-level failure mode from #014 was addressed and
the headline did not lift.

This experiment changes the task framing rather than patching the
head further. Instead of "diffuse over the next event's bin
offset," the model **diffuses over a per-frame activation map**:
`M ∈ [0, 1]^B_PRED` where `M[b] = 1` if there's an onset at bin
offset `b` from the cursor, else 0. The activation map is built
with Gaussian σ=2-frame smoothing around GT onset positions; the
denoiser is a 1D Conv stack with FiLM modulation and a per-bin
audio-feature input channel; the decoder thresholds the final
predicted map and emits **all** positive bins at once. STOP is
implicit (empty positive set ⇒ STOP_HOP=20 bins).

The reframing changes three things at once:

1. **AR step semantics**. Each cursor advance now emits 0..N
   events (vs exactly 1 in #014/#015). With ~3 events/s typical
   density and a 500-bin (2.5s) window, expect 5-10 events per AR
   step. Total AR steps per chart drops ~5-10× compared to
   #014/#015.
2. **STOP class gone**. The activation map encodes "no event
   here" as low activation values; there is no separate STOP
   class to bias.
3. **Denoiser sees future audio**. The Conv1D denoiser consumes
   the full future-half audio token sequence (125 tokens, post
   stride-4 conv stem) upsampled linearly to 500 per-bin channels,
   in addition to the cursor token and time embed. The #014/#015
   MLP denoiser only saw the cursor token.

Past framewise approaches in taiko1 (pre-event-embedding) failed
because single-shot framewise outputs are blurry — there's no
opportunity to commit to a sharp peak. The diffusion-refinement
adds T inference steps of explicit mass-redistribution, which is
the same mechanism that makes diffusion successful at structured
output generation in the literature (Analog Bits, DiffusionDet).
This is the first taiko2 attempt to combine framewise output with
iterative refinement.

## Citations

- Direct baselines:
  - [#015 — diffusion-patched](../015-diffusion-patched/). Best gt
    `matched_rate` 0.6468 [015-diffusion-patched/ablations/ddim_4_e0_n4_off1/gt_cond/comparisons_summary.json:fields.matched_rate.median],
    `error_median_ms` 11.0 [same file:fields.error_median_ms.median],
    `stop_f1` 0.7663 [exp_015_diffusion_patched, step 186,066,
    val/single/onset/stop_f1] (current taiko2 ATH). Diffusion-stack
    machinery (`GaussianContinuousProcess`, `DDIMSampler`,
    self-conditioning, Min-SNR γ=5) reused unchanged.
  - [#007 — TimeStretch](../007-time-stretch/). Best gt
    `matched_rate` 0.7028 [exp_007_time_stretch, step 413,480,
    val/single/corpus/gt_cond_cmp/matched_rate_mean]. The
    softmax-CE next-bin baseline this design ultimately compares
    against.
  - [#012 — onset-feature channels](../012-onset-channels/). Best
    gt `matched_rate` 0.7080 [exp_012_onset_channels, step
    308,550, infer_corpus/eval_308550/gt_cond/comparisons_summary.json:fields.matched_rate.median]
    — current taiko2 all-time-best.
- Earlier framewise precedents in taiko2:
  - [#011 — onset feature survey](../011-onset-feature-survey/).
    Classical mel-domain ODFs (spectral flux, HFC, SuperFlux,
    sub-band SF) at frame-wise F1 = 0.679 against chart-author GT
    at ±10 frames [011-onset-feature-survey/results/summary.json:by_algo.spectral_flux.by_tolerance.10].
    Single-shot framewise upper bound at the classical-algorithm
    level; #016 tests whether iterative diffusion lifts it.
  - [#011b — onset disagreement](../011b-onset-disagreement/).
    Cross-group ODF pairs reach recall 0.905 (`hfc_mel +
    spectral_flux`) [011b-onset-disagreement/results/summary.json:pairwise.pairs.hfc_mel+spectral_flux.recall_union].
    Suggests there is enough signal in the audio + context for a
    well-tuned framewise output to recover most events.
- Reframing literature:
  - [Analog Bits: Generating Discrete Data Using Diffusion Models with Self-Conditioning (Chen, Zhang, Hinton, ICLR 2023)](https://arxiv.org/abs/2208.04202)
    — closest direct analog. Continuous Gaussian diffusion on a
    [0, 1] continuous representation of discrete targets. Source
    for self-conditioning + asymmetric time intervals (both kept
    from #015's stack).
  - [Min-SNR Weighting Strategy (Hang et al., ICCV 2023)](https://arxiv.org/abs/2303.09556).
    γ=5 weighting kept from #015.
  - [DiffusionDet (Chen et al., ICCV 2023)](https://arxiv.org/abs/2211.09788)
    — uses diffusion over per-position object queries; the closest
    structural analog to "diffuse a per-position activation map."
  - [BeatThis! (Foscarin et al., ISMIR 2024)](https://arxiv.org/abs/2407.21658)
    — transformer-based framewise beat tracking. Single-shot
    framewise baseline for comparison.
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

If we replace #014/#015's next-bin softmax-style diffusion with a
**framewise activation-map diffusion** — same trunk, x_0-Gaussian
DDPM with cosine T=64 schedule, x_0-parameterized 1D Conv denoiser
with FiLM + audio-feature input channel, 16-step DDIM at inference,
threshold-based decoder that emits multiple events per AR step —
then **AR `matched_rate` (gt_cond, median) will reach ≥ 0.720** at
the best ablation variant (clears #007's 0.7028 baseline by +1.7 pp
and ties / beats #012's 0.7080 ATH), with `error_median_ms ≤ 12 ms`
(within #007's 10.2 ms band) and frame-F1 at the canonical operating
point (τ=0.5, ±2 frames) ≥ 0.70 on the sampled-t leaky metric.

### Mechanism

Three independent effects on top of #015's design:

1. **Removes the AR error-compounding axis.** #014/#015 emit one
   event per AR step over ~120 steps per chart. Each step is
   conditioned on the output of all previous steps (via the
   updated event-offset context). Errors compound. #016 emits
   5-10 events per step over ~10-30 steps per chart — the AR
   feedback path is ~10× shorter. If error compounding is part of
   the #014/#015 ceiling, this should lift matched_rate.
2. **Removes the STOP-class bias.** #014/#015 had a single
   "no event here" outcome contesting probability mass with all
   500 bin-offset options. The STOP class biased the model toward
   "predict STOP" when uncertain — exactly the missing-heavy
   failure mode (#015 missed/hallucinated ratio = 2.7×). #016's
   activation map encodes "no event here" as low activation;
   there is no class to bias toward.
3. **Audio-feature input channel for the denoiser.** #014/#015's
   MLP denoiser conditioned only on the cursor token — a global
   summary. The Conv1D denoiser receives per-bin audio context
   (upsampled future audio tokens). Each bin's denoising decision
   can use audio features at the corresponding time position, not
   just a global representation.

The diffusion iterative refinement still serves its main purpose:
turn a noisy initial activation map into sharp peaks at GT onset
positions. The mass-redistribution that #014/#015 used to commit
to ONE bin is now used to commit to a SUBSET of bins. The same
patches that mattered in #015 (Min-SNR γ=5, self-conditioning) are
all retained.

### Predicted numbers

Reference: best variants of [#007](../007-time-stretch/),
[#012](../012-onset-channels/), [#015](../015-diffusion-patched/).
gt_cond medians.

| Metric | #007 best | #012 best | #015 best | Predicted (#016, best variant) | Notes |
|---|---:|---:|---:|---:|---|
| AR `matched_rate` (gt_cond, median) | 0.7028 | 0.7080 | 0.6468 | **≥ 0.720** must / **0.74-0.76** nice | must-have: clears #007 baseline |
| AR `error_median_ms` | 10.2 | 8 | 11.0 | **≤ 12** | timing should hold; sharp diffusion peaks at the GT positions |
| AR `density_ratio` | 0.865 | n/a | 0.802 | **0.88-0.95** | no STOP bias; activation map directly encodes density |
| AR `hallucination_rate` | 0.146 | 0.135 | 0.145 | **0.13-0.16** | NMS + threshold controls |
| AR `hi_pspace` | 0.909 | 0.894 | 1.000 | ≥ 0.95 | should hold |
| AR `dc_human` | 0.928 | 0.918 | 0.918 | 0.91-0.93 | might recover #007 ground |
| frame F1 at (τ=0.5, ±2 frames) (sampled-t, leaky) | n/a — different metric | n/a | n/a | **≥ 0.70** | new metric; diffusion baseline target |
| frame AUC-PR (sampled-t, leaky) | n/a | n/a | n/a | ≥ 0.80 | threshold-free integrated quality |
| `rollout/best_k_step` at converged eval | n/a | n/a | n/a | ≥ 12 / 16 | model uses most of its inference budget |
| `rollout/final_vs_best_delta` | n/a | n/a | n/a | ≈ 0 (±0.02) | model should not over-denoise past final step |
| `rollout/monotone_fraction` | n/a | n/a | n/a | ≥ 0.85 | rollout should monotonically refine on average |
| Total params | 16.35 M (#007) | n/a | 24.47 M | **22.27 M** | Conv1D denoiser (5.91 M) < MLP denoiser (8.11 M); 5.92 M trunk-side carried |
| Wall time / eval | 2.18 h (#007) | n/a | 2.30 h (#015) | 2.5-3.0 h | EVAL_K rollout pass adds ~5% per eval; AR is ~10× fewer cursor advances but each does the same 16-step diffusion |

Observational (not gated):

- **Per-step rollout convergence shape.** Does each diffusion
  step monotonically increase F1? Does F1 plateau before T=16?
  Tracked in `convergence_curves.png` + the EVAL_K NPZ.
- **Per-density bucket F1.** Sparse charts are the hardest for
  classical ODFs ([#011b](../011b-onset-disagreement/)); does
  diffusion fix that or reproduce it? Tracked in
  `convergence_by_density.png`.
- **Per-onset-kind F1.** Does DON vs KA differ? Diffusion should
  not specialize on either (no kind-conditioning in the loss).
- **Multi-tolerance matched_rate curve.** At ±5/10/20/40/100 ms,
  how does matched_rate scale? Tells us where on the precision/
  recall tradeoff each variant sits.

## Success criteria

- **Must have:** AR `matched_rate` ≥ 0.720 at the best AR-corpus
  ablation variant (clears #007's 0.7028 baseline by ≥ 1.7 pp;
  ties or beats #012's 0.7080 ATH).
- **Must have:** training stable, no NaN, runs to E15+ without
  divergence; positive-class rate prediction (`frame/pos_rate_pred_50`)
  > 0.005 and < 0.1 (no class collapse to all-zeros or all-ones).
- **Must have:** `frame/separation` ≥ 0.4 at any post-warmup eval —
  the model is producing distinguishably-different activations at
  GT vs non-GT positions.
- **Must have:** `rollout/best_k_step` ≥ 4 — the model is using
  the diffusion steps for something (vs collapsing to single-shot
  behavior with the rest as no-ops).
- **Must have:** AR `density_ratio` ≥ 0.80 at the best variant —
  the absence of STOP class is delivering on its predicted effect.
- **Nice-to-have:** AR `matched_rate` ≥ 0.740 — clear architectural
  win over #007; new taiko2 SOTA by a meaningful margin.
- **Nice-to-have:** AR `error_median_ms` ≤ 10 — matches or beats
  #007 on timing precision.
- **Nice-to-have:** frame AUC-PR ≥ 0.85 at a late eval —
  threshold-free quality measure beats the classical ODF baseline
  from #011 (single-channel SF AUC-PR isn't directly reported but
  is bounded by the F1 = 0.679 at ±10 frames).
- **Fails if:** AR `matched_rate` < 0.65 at every eval (worse than
  #015) — the framewise reframing didn't help and the design
  is structurally inferior to next-bin diffusion.
- **Fails if:** `frame/pos_rate_pred_50` < 0.005 at every eval —
  model collapsed to predicting all-zeros (the classic framewise
  failure mode).
- **Fails if:** `rollout/best_k_step` = 0 at every eval — the
  initial random x_T is no worse than any denoised state, meaning
  the denoiser isn't denoising.
- **Fails if:** training diverges or loss curve plateaus before E5.

## Changes from baseline

Baseline: [#015 — diffusion-patched](../015-diffusion-patched/).

Code changes (committed alongside this experiment, ~2200 LOC + 122
new tests):

- **`osu/taiko2/domain/framewise.py`** (NEW) — `FramewiseTarget`
  frozen dataclass holding `(target_map_binary, target_map_smoothed,
  gt_bins_padded, n_gt)`. Factory `make_framewise_target` builds
  the binary + Gaussian-σ-smoothed maps from per-sample GT bin
  offsets.
- **`osu/taiko2/domain/diffusion.py`** — `DenoiserHead.forward`
  gained an optional `audio_features: torch.Tensor | None = None`
  kwarg after `prev_x0_hat`. Backward-compatible — #015's MLP
  denoiser still works.
- **`osu/taiko2/diffusion/processes.py`** — new
  `FramewiseActivationProcess` + `FramewiseActivationProcessConfig`.
  Parallels `GaussianContinuousProcess` but `encode_x0` accepts
  the activation map directly (rank-2, not a (B,) int) and
  `decode_to_logits` clips to [0, 1] instead of dividing by
  `x0_scale`. Only x_0-parameterization supported (config rejects
  noise / v).
- **`osu/taiko2/diffusion/denoisers.py`** — new `Conv1DDenoiser` +
  `Conv1DDenoiserConfig`. 1D Conv stack with FiLM (zero-init);
  input concat of x_t, prev_x0_hat (if self_cond), sinusoidal
  positional embedding, audio features (linearly upsampled from
  125 to 500 along the time axis), broadcast cursor projection,
  broadcast time embed projection. Default conv stack: 3 blocks
  of kernels {31, 15, 15}, 256 channels, GroupNorm(8).
- **`osu/taiko2/diffusion/samplers.py`** — `DDPMSampler.sample` /
  `DDIMSampler.sample` gained an optional `audio_features` kwarg.
  `DDIMSampler.sample_with_intermediates` (new) returns the full
  per-step `M_k` tensor for the rollout hook.
- **`osu/taiko2/models/framewise_diffusion_detector.py`** (NEW) —
  `FramewiseDiffusionDetector(EventEmbeddingDetector)` + output
  dataclass `FramewiseModelOutput` (cursor_token + audio_features
  + the training-mode fields). `_trunk_forward` extracts both the
  cursor token and the full audio token sequence in one pass;
  `get_audio_features` returns the future-half tokens
  [125:250]. `forward_diffusion` is identical to #015's
  diffusion forward except takes `target_map` (B, n_bins) and
  passes `audio_features` to the denoiser.
- **`osu/taiko2/training/framewise_adapter.py`** (NEW) —
  `FramewiseSampleAdapter` converts a `DataSample` into
  `(EventEmbeddingInput, FramewiseTarget)`. Filters future events
  to those in `[0, b_pred)`, calls `make_framewise_target` with
  σ=2-frame smoothing.
- **`osu/taiko2/training/framewise_diffusion_loss.py`** (NEW) —
  `FramewiseDiffusionLoss`. Weighted MSE with per-sample
  positive-class upweighting (clamped [10, 200]) + Min-SNR γ=5.
  Reports 19 scalar metrics including frame F1 at the canonical
  operating point, AUC-PR, AUC-ROC, per-t-quartile losses,
  separation, pos/neg rates.
- **`osu/taiko2/training/framewise_curve_metrics.py`** (NEW) —
  pure-function curve computations: 101-threshold precision /
  recall / f1 sweep, 5-tolerance × 101-threshold grid (~5/10/20/
  40/100 ms tolerances), threshold-free AUC-PR and AUC-ROC.
- **`osu/taiko2/training/framewise_rollout_hook.py`** (NEW) —
  the big one. Runs at every Nth eval. EVAL_K + NOAUG_K full
  T_inf-step rollouts on ~32 charts × ~5 windows. Saves
  per-sample per-step M_k tensors to `rollout_maps.npz`. Renders
  `convergence_curves.png` with mean + p10/p25/p75/p90 bands.
  Renders per-bucket convergence (density / star / kind).
  Renders 5 representative GIFs (best / p75 / p50 / p25 / worst
  by final F1) + one population-summary GIF. Aggregate metrics:
  `rollout/best_k_step`, `final_vs_best_delta`,
  `convergence_step_90`, `monotone_fraction`. Mini-chart metrics
  at step K threshold-decoded across 101 thresholds × 5
  tolerances.
- **`osu/taiko2/training/framewise_artifacts.py`** (NEW) —
  framewise versions of the per-eval artifacts:
  `FramewiseHeatmapArtifact` (predicted M_0_hat vs target),
  `FramewiseDistributionArtifact` (predicted-value histograms at
  GT vs non-GT positions), `FramewiseTrainingHeatmapArtifact`
  (train_noaug counterpart).
- **`osu/taiko2/inference/autoregressive/framewise_diffusion_decoder.py`**
  (NEW) — `FramewiseDiffusionDecoder`. Runs the sampler against
  cursor_token + audio_features, applies optional 1D max-pool NMS
  (`nms_kernel ≥ 3` keeps only local maxima), thresholds at
  `decode_threshold`, applies `min_emit_gap_bins`, returns
  `ARDecision` with multi-bin `bin_offsets`. The AR loop's
  existing semantics (iterate `bin_offsets`, advance cursor to
  last) handle the multi-bin case correctly without modification.
- **`osu/taiko2/inference/multi_tolerance_compare.py`** (NEW) —
  multi-tolerance chart comparison wrapper. `compare_at_tolerances`
  runs `Chart.compare` at 5 tolerances and returns per-tolerance
  results. `aggregate_multi_tolerance_summaries` produces the
  expected `comparisons_summary_tol.json` shape.
- **`osu/taiko2/inference/corpus.py`** — `InferCorpusConfig` gained
  `tolerances_ms` field. `_run_one_mode` now emits per-tolerance
  scalars (`*_at_tol_1/2/4/8/20`) alongside the canonical
  comparison.
- **`osu/taiko2/cli/train.py`** — framewise mode detection;
  dispatches `FramewiseSampleAdapter`, `FramewiseDiffusionLoss`,
  `FramewiseHeatmapArtifact`, `FramewiseDistributionArtifact`,
  `FramewiseRolloutHook`. Disables weighted sampling (no STOP
  class so no class-balance need). Skips `OnsetMetric` (defined
  for single-bin output). New CLI flags for the rollout hook:
  `--rollout-eval-n-charts`, `--rollout-t-inf-steps`, etc.
- **Tests** (NEW): 122 new tests across
  `test_framewise_diffusion.py` (43, Chunk A),
  `test_framewise_pipeline.py` (36, Chunk B),
  `test_framewise_decoder.py` (20, Chunk C),
  `test_framewise_rollout.py` (15, Chunk C),
  `test_multi_tolerance_compare.py` (8, Chunk C).
  Full suite at 596/596 passing.

Config snapshots ([`config/`](./config/)):

- `config/model.json` — `FramewiseDiffusionDetectorConfig` with the
  Conv1D denoiser (256 channels, kernels {31, 15, 15}, self_cond
  on, audio_feature_dim=384, audio_token_count=125).
- `config/loss.json` — `FramewiseDiffusionLossConfig`. MSE +
  `snr_weighting=true` (γ=5) + positive-class weight clamp
  [10, 200] + canonical operating point (τ=0.5, ±2 frames).
- `config/data.json` — `d_events: 1 → 100` (the only change from
  #015's data config; framewise adapter needs to see all future
  events in window).
- `config/adapter.json` — `FramewiseSampleAdapterConfig` with
  b_pred=500, sigma_frames=2.0, max_events_per_window=100.
- `config/trainer.json` — byte-identical to #015.
- `config/infer.json` — `FramewiseDiffusionDecoder` + DDIM 16/0/0/0
  + threshold=0.5 + nms_kernel=1 default.
- `config/ablation_matrix.json` — 10 variants: 3 threshold sweeps
  + 2 NMS variants + 3 step-count variants + 1 DDPM-64 + 1
  combined-Pareto.

## Dropped metrics from #014/#015

The framewise framing makes these metrics undefined or
meaningless:

- `onset/hit`, `onset/good`, `onset/miss`, `onset/exact` — these
  measured "single predicted bin vs single target bin" semantics.
  No longer apply: the output is now a vector of independent
  per-bin activations.
- `onset/rhit`, `onset/rgood`, `onset/fhit`, `onset/fgood`,
  `onset/ihit`, `onset/igood`, etc. — same.
- `onset/stop_f1`, `onset/stop_precision`, `onset/stop_recall`,
  `pred_stop_rate`, `pred_stop_fp_rate` — no STOP class.
- `onset/frame_err_mean`, `onset/frame_err_median`,
  `onset/frame_err_p90` — single-prediction-vs-target metric.
  Replaced by `error_median_ms` per onset in AR-corpus
  comparisons, kept there.
- `ratio_hit`, `metronome_*`, `ratio_error.*` graphs — defined on
  predicted-vs-prev-event ratios from single-bin output.

The headline AR-corpus comparison metrics
(`matched_rate`, `error_median_ms`, `density_ratio`,
`hallucination_rate`, `hi_pspace`, `dc_human`, `oc_human`,
`over_pspace_self`, etc.) are unchanged in definition — they
operate on the emitted onset list, which the framewise decoder
produces just like the next-bin decoder did. They're now extended
to 5 tolerances (`*_at_tol_1/2/4/8/20`) for the headline reports.

## New metrics added for #016

Per training eval, 5 passes run:

1. **EVAL_1 / NOAUG_1** (sampled-t, leaky on full val 5% / train_noaug 5%).
2. **EVAL_K / NOAUG_K** (full T_inf-step rollout on 32 charts × 5 windows).
3. **AR-corpus** (full AR loop on val 10%).

Scalar metrics emitted to `metrics.jsonl`:

- Loss-level: `loss`, `loss/snr_weighted`, `loss/pos_only`,
  `loss/neg_only`, `loss/pos_neg_ratio`, `loss/per_t_q0..3`.
- Framewise (sampled-t, leaky): `frame/f1_τ_50_tol_2`
  (canonical op point), `frame/precision_τ_50_tol_2`,
  `frame/recall_τ_50_tol_2`, `frame/auc_pr`, `frame/auc_roc`,
  `frame/mean_act_pos`, `frame/mean_act_neg`, `frame/separation`,
  `frame/pos_rate_pred_50`, `frame/pos_rate_target`.
- Rollout (EVAL_K aggregates): `rollout/best_k_step`,
  `rollout/best_k_f1`, `rollout/final_vs_best_delta`,
  `rollout/convergence_step_90`, `rollout/monotone_fraction`.
- AR-corpus multi-tolerance: `corpus/gt_cond_cmp_tol/*_at_tol_1/2/4/8/20`
  for matched_rate, hallucination_rate, error_median_ms,
  close_rate, far_rate.

Per-eval artifacts in `runs/exp_016_framewise_diffusion/eval_{step}/`:

- `curves.npz` — full 101-threshold × 5-tolerance grids for every
  framewise / mini-chart metric. Stored at float16 where
  appropriate.
- `rollout_maps.npz` — per-sample per-step `M_k` tensors (shape
  `(n_samples, K+1, B_PRED)` float16) + metadata (chart_id,
  window_bin, n_gt, density_bucket, star_bucket, final_f1).
- `noaug_rollout_maps.npz` — train_noaug equivalent.
- `convergence_curves.png` — mean + p10/p25/p75/p90 bands over k,
  three panels (F1 / MSE / mass_at_target_fraction).
- `convergence_by_density.png`, `convergence_by_star.png`,
  `convergence_by_kind.png` — per-bucket variants.
- `rollout_gifs/` — 5 representative GIFs (best / p75 / p50 /
  p25 / worst by final F1) + `summary_histogram.gif`.
- `train_noaug/` — mirror of all the above for train_noaug.

## Run config

- Run name: `exp_016_framewise_diffusion`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1` (80-row mel; same as #007/#013/#014/#015).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
      --run-name exp_016_framewise_diffusion \
      --config-dir osu/taiko2/experiments/016-framewise-diffusion/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --infer-corpus-spec osu/taiko2/experiments/016-framewise-diffusion/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```
  Note: no `--benchmarks` flag — benchmarks are run separately
  post-run via a forthcoming CLI script (per the experiment plan;
  the 5-pass eval cadence is already heavy enough without bench).
- Post-run sampler ablation:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.diffusion_sampler_ablation \
      --base-spec osu/taiko2/experiments/016-framewise-diffusion/config/infer.json \
      --matrix osu/taiko2/experiments/016-framewise-diffusion/config/ablation_matrix.json \
      --dataset taiko2_v1 \
      --out-dir osu/taiko2/experiments/016-framewise-diffusion/ablations
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

_(table populated post-run)_

### Per-eval progression

_(generated post-run from `runs/exp_016_framewise_diffusion/metrics.jsonl`)_

Machine-readable copies: [`metrics.json`](./metrics.json).

## Visualizations

_(post-run; auto-generated PNGs from `runs/exp_016_framewise_diffusion/curves/`,
`eval_{best}/`, and the rollout-hook outputs)_

## Custom analyses

- [Sampler / decoder ablation matrix](ablations/) — output of
  `cli.diffusion_sampler_ablation`. 10 variants covering
  threshold ∈ {0.3, 0.5, 0.7}, NMS kernel ∈ {1, 3, 5},
  step counts ∈ {4, 8, 16, 32, 64}, asymmetric time
  offset ∈ {0, 1}, DDPM-64 reference.
- Per-eval `rollout_maps.npz` — full per-sample per-step
  activation maps. Queryable via numpy.
- Per-eval `rollout_gifs/` — visual progressions of the
  denoising trajectory at best / median / worst quality
  samples.

## Vs prediction

_(post-run)_

## Takeaways

_(post-run)_

## Followup questions

_(post-run; expected candidates: #016b — transformer denoiser if
Conv1D plateaus; #016c — wider prediction window 500→1000 bins;
#016d — combined head ensembling framewise + #015's next-bin
diffusion.)_
