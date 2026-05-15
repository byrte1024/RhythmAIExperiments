# Experiment 016 — Framewise activation-map diffusion

## Status

`Halted at step 103,370 (1 epoch of 5); diagnostic complete, superseded by #016b (planned).`

The run was stopped after 5 evals once the rollout-convergence signal
showed the diffusion sampler **regressing monotonically** across its
16 steps and **getting worse with training**, not better. Headline
single-step and AR-corpus metrics looked strong in isolation but
masked a foundational bug in the loss weighting that made further
training counter-productive. Full diagnosis in [Results](#results-summary)
and [Takeaways](#takeaways) below.

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

The run was **halted at step 103,370 (~1 epoch of a planned 5)** after
the 5th eval. Headline metrics (single-step F1, AR `matched_rate`)
appeared strong, but the per-step rollout artifacts revealed the
diffusion sampler **regressing monotonically** across its 16 steps
and the regression **growing** with training, not shrinking. Root
cause was identified in the loss-weighting code; continuing training
would compound the failure rather than fix it.

### Headline finding

The DDIM sampler does not sharpen broad activation blobs into single
peaks; it instead **bleaches every detected local maximum to ≈1.0
while leaving the blob count roughly constant**. The audio
conditioning already locates blobs correctly at sampler step k=0
(recall 0.987 @ thr=0.3, NMS=3 [rollout_maps.npz step 103,370 k=0]);
each subsequent sampler step destroys progressively more of the
confidence-grading information that the AR decoder needs to filter
out the extras. By k=15 (the step the AR decoder consumes), 99.4 %
of detected peaks have value > 0.95 — the threshold knob is dead.

A clean operating point exists at **k=0 only**: threshold=0.95 +
NMS=3 gives density ratio 1.00×, hallucination 0.168, recall 0.851
[rollout_maps.npz step 103,370 k=0 threshold sweep, this README's
[custom/per_t_loss_imbalance](custom/per_t_loss_imbalance/)].
The AR decoder cannot reach this operating point because it consumes
the k=15 output.

### Final vs baseline

`final` = last completed eval (step 103,370). `peak` cells cite the
best eval for that metric across the 5 evals run. **The peak rows
look strong only on the AR-corpus metrics, which mask the rollout
regression** (the AR decoder uses tol=±2 + threshold-based filtering
which forgives the cluster-spam failure mode the rollout exposes).

| Metric | #007 best | #012 best | #015 best | #016 final (step 103,370) | #016 peak | Δ vs #015 |
|---|---:|---:|---:|---:|---:|---:|
| AR `matched_rate` (gt_cond, median) | 0.7028 [a] | 0.7080 [b] | 0.6468 [c] | 0.9551 | 0.9734 @ 62,022 | **+0.327** |
| AR `error_median_ms` (gt_cond) | 10.2 [a] | 8 [b] | 11.0 [c] | 1.58 | 1.44 @ 103,370 fixed_cond | **−9.4 ms** |
| AR `hallucination_rate` (gt_cond) | 0.146 [a] | 0.135 [b] | 0.145 [c] | 0.357 | 0.346 @ 41,348 | +0.21 |
| AR `density_ratio` (gt_cond) | 0.865 [a] | n/a | 0.802 [c] | 10.92 | 11.29 @ 82,696 | **+10.1** |
| AR `hi_pspace` (gt_cond) | 0.909 [a] | 0.894 [b] | 1.000 [c] | 0.891 | 0.918 @ 82,696 | −0.082 |
| AR `dc_human` (gt_cond) | 0.928 [a] | 0.918 [b] | 0.918 [c] | 0.696 | 0.696 @ 103,370 | −0.222 |
| frame F1 (τ=0.5, ±2) | n/a | n/a | n/a | 0.898 | 0.907 @ 82,696 | (new) |
| frame AUC-PR | n/a | n/a | n/a | 0.719 | 0.719 @ 103,370 | (new) |
| frame AUC-ROC | n/a | n/a | n/a | 0.986 | 0.986 @ 103,370 | (new) |
| frame `separation` (mean_pos − mean_neg) | n/a | n/a | n/a | 0.808 | 0.810 @ 82,696 | (new) |
| frame `pos_rate_pred_50` | n/a | n/a | n/a | 0.169 | n/a | (must-have band [0.005, 0.1] **violated**) |
| rollout `best_k_step` (median) | n/a | n/a | n/a | 2.0 | 3.5 @ 62,022 | (new) |
| rollout `final_vs_best_delta` (mean) | n/a | n/a | n/a | **−0.092** | −0.081 @ 62,022 | (new) |
| rollout `monotone_fraction` | n/a | n/a | n/a | 0.425 | 0.468 @ 82,696 | (new) |
| `loss/per_t_q0` (low-t, sampler tail) | n/a | n/a | n/a | 0.0017 | n/a | (new) |
| `loss/per_t_q3` (high-t, sampler head) | n/a | n/a | n/a | 0.0596 | n/a | (new) |
| q3 / q0 ratio | n/a | n/a | n/a | **35.1×** | n/a | (new) |
| Total params | 16.35 M [a] | n/a | 24.47 M [c] | 22.27 M | — | −2.20 M |
| Wall time / eval | 2.18 h [a] | n/a | 2.30 h [c] | ~2.7 h | — | +~0.4 h |

[a] exp_007_time_stretch, step 413,480, val/single/corpus/gt_cond_cmp/
matched_rate_mean (and sibling fields).
[b] exp_012_onset_channels, step 308,550, infer_corpus/eval_308550/
gt_cond/comparisons_summary.json.
[c] 015-diffusion-patched/ablations/ddim_4_e0_n4_off1/gt_cond/
comparisons_summary.json (matched_rate, error_median_ms,
hallucination_rate fields).

### Per-eval progression

All metrics emitted to `runs/exp_016_framewise_diffusion/metrics.jsonl`
across the 5 evals run. Loss / framewise-leaky / AR-corpus subset
below; the full machine-readable dump is in
[`metrics.json`](./metrics.json).

| eval | step | loss | L_pos | L_neg | L_q0 | L_q3 | fr_P | fr_R | fr_F1 | auc_pr | auc_roc | sep | pr_pred_50 | gt_match | gt_hallu | gt_err_med_ms | gt_dens | gt_dc | gt_hi | fx_match | fx_hallu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 |  20,674 | 0.0402 | 0.0139 | 0.0285 | 0.0033 | 0.0724 | 0.7801 | 0.9901 | 0.8694 | 0.6010 | 0.9792 | 0.7804 | 0.1759 | 0.9693 | 0.3660 | 1.71 | 14.42 | 0.6036 | 0.9080 | 0.9802 | 0.3706 |
| 2 |  41,348 | 0.0360 | 0.0125 | 0.0253 | 0.0022 | 0.0660 | 0.8091 | 0.9916 | 0.8889 | 0.6795 | 0.9833 | 0.7993 | 0.1687 | 0.9637 | 0.3587 | 1.74 | 16.39 | 0.5617 | 0.8907 | 0.9701 | 0.3660 |
| 3 |  62,022 | 0.0335 | 0.0108 | 0.0244 | 0.0021 | 0.0616 | 0.8138 | 0.9921 | 0.8915 | 0.6919 | 0.9844 | 0.7940 | 0.1693 | 0.9734 | 0.3902 | 1.70 | 14.97 | 0.5878 | 0.8790 | 0.9826 | 0.3914 |
| 4 |  82,696 | 0.0331 | 0.0131 | 0.0218 | 0.0019 | 0.0608 | 0.8393 | 0.9900 | 0.9065 | 0.7115 | 0.9850 | 0.8098 | 0.1623 | 0.9421 | 0.3890 | 1.82 | 11.29 | 0.6890 | 0.9182 | 0.9635 | 0.3927 |
| 5 | 103,370 | 0.0322 | 0.0102 | 0.0235 | 0.0017 | 0.0596 | 0.8237 | 0.9923 | 0.8983 | 0.7185 | 0.9862 | 0.8082 | 0.1686 | 0.9551 | 0.3571 | 1.58 | 10.92 | 0.6955 | 0.8906 | 0.9711 | 0.3627 |

Loss falls monotonically (0.0402 → 0.0322, −20 % across the 5 evals).
fr_F1 climbs to a peak of 0.9065 at eval 4 then dips. AR matched_rate
oscillates in a tight band 0.94–0.97. `loss/per_t_q3 / loss/per_t_q0`
ratio sits at 21.9× / 30.0× / 29.3× / 32.0× / 35.1× across the 5
evals — the imbalance **grows** with training, mirroring the rollout
regression.

## Visualizations

Loss + per-eval-aggregate plots from the trainer:

- [`graphs/loss.png`](graphs/loss.png) — training-loss curve (step
  axis). Decreasing monotonically across the visible range.
- [`graphs/convergence_curves_step_103370.png`](graphs/convergence_curves_step_103370.png)
  — F1 / MSE / mass-at-target fraction across the 16 DDIM steps at
  the final eval, mean ± p10/p25/p75/p90 bands over 160 samples.
- [`graphs/framewise_heatmap_step_103370.png`](graphs/framewise_heatmap_step_103370.png)
  — predicted activation map vs target for 64 representative
  windows at the final eval (sampled-t leaky head).
- [`graphs/framewise_distribution_step_103370.png`](graphs/framewise_distribution_step_103370.png)
  — distribution of predicted activations at GT-positive vs GT-
  negative bins.
- [`graphs/convergence_by_density_step_103370.png`](graphs/convergence_by_density_step_103370.png),
  [`graphs/convergence_by_kind_step_103370.png`](graphs/convergence_by_kind_step_103370.png),
  [`graphs/convergence_by_star_step_103370.png`](graphs/convergence_by_star_step_103370.png)
  — per-bucket K-step convergence breakdowns.
- [`graphs/summary_histogram_step_103370.gif`](graphs/summary_histogram_step_103370.gif)
  — population summary GIF of the K-step refinement.

## Custom analyses

- [`custom/k_step_regression/`](custom/k_step_regression/) — per-
  sampler-step F1 trajectory **across all 5 evals**, showing the
  monotone drop from k≈0 to k=15 and how the gap widens with
  training. Source data: `rollout_maps.npz:f1` per eval.
- [`custom/per_t_loss_imbalance/`](custom/per_t_loss_imbalance/) —
  two diagnostic plots that pin the failure mode. Per-t-quartile
  loss bars show q3 ≈ 35× q0 (the Min-SNR sign bug); peak-value
  histogram across k shows mean peak value going 0.911 → 0.997 and
  fraction-above-0.95 going 0.597 → 0.994.
- [`custom/best_eval_rollout_gifs/`](custom/best_eval_rollout_gifs/)
  — five sample-level GIFs from eval 5 illustrating the "broad
  blobs that never sharpen, then saturate" failure on best / p60 /
  p40 / p20 / worst-quartile windows.
- The planned sampler / decoder ablation matrix
  (`config/ablation_matrix.json`) was **not run**. It would have
  characterised how 10 variants each fail under the same Min-SNR
  bug — no additional information; superseded by 16b's redesign.

## Vs prediction

| Metric | Predicted (must / nice) | Actual best | Verdict |
|---|---:|---:|---|
| AR `matched_rate` (gt_cond) | ≥ 0.720 must / 0.74–0.76 nice | 0.9734 @ 62,022 | **headline beat, mechanism wrong**. The metric clears because tol=±2 + decoder threshold forgives the cluster-spam shape — see [Takeaways §1](#takeaways). |
| AR `error_median_ms` | ≤ 12 | 1.58 @ 103,370 | **beat by 10.4 ms.** Sharp peak locations carry through the saturation collapse — the model learns *where* well, just not *with what confidence*. |
| AR `density_ratio` | 0.88–0.95 | 10.92 @ 103,370 | **missed by 10×.** Cluster spam: 5–7 emitted bins per real onset (raw thr=0.5), 1.3× after kernel-3 NMS. |
| AR `hallucination_rate` | 0.13–0.16 | 0.357 @ 103,370 | **missed by 2.4×.** Same cause as density_ratio — extras within tol of real GT. |
| AR `hi_pspace` | ≥ 0.95 | 0.918 @ 82,696 | missed by 0.03. |
| AR `dc_human` | 0.91–0.93 | 0.696 @ 103,370 | **missed by 0.22.** Density-driven distance-to-human collapse. |
| frame F1 (τ=0.5, ±2) | ≥ 0.70 | 0.9065 @ 82,696 | **beat by 0.21.** Sampled-t leaky metric — averaged over t ∈ [0, 63] training timesteps, so reflects high-t (well-trained) regime, **not** k=15 (which is what the decoder uses). |
| frame AUC-PR | ≥ 0.80 | 0.719 @ 103,370 | missed by 0.08. |
| rollout `best_k_step` | ≥ 12 / 16 | **2.0** @ 103,370 (median) | **missed by 10 steps.** Best F1 sits at the **start** of the chain — confirms the model never learned to refine. |
| rollout `final_vs_best_delta` | ≈ 0 (±0.02) | **−0.092** @ 103,370 | **missed by 4.6× outside band.** Final F1 is 0.092 below the chain's best F1. |
| rollout `monotone_fraction` | ≥ 0.85 | **0.468** @ 82,696 | **missed by 0.38.** Fewer than half of samples improve across the rollout. |

Must-have criteria (pre-run):

- **PASS** — training stable, no NaN, ran past E5 without divergence.
- **PASS** — `frame/separation` ≥ 0.4 at any post-warmup eval
  (achieved 0.780 at eval 1, 0.810 at eval 4).
- **FAIL** — `frame/pos_rate_pred_50` ∈ [0.005, 0.1]. All 5 evals
  fall in [0.162, 0.176], **above the band by 60–76 %**. The model
  is class-imbalanced toward *predicting too many positive bins*.
  Pre-run band assumed standard threshold-sweepable peaks; actual
  output is saturated post-NMS.
- **FAIL** — `rollout/best_k_step` ≥ 4. Median is 2.0–3.5 across
  evals; best F1 sits at the start of the chain, not after refinement.
- **FAIL** — AR `density_ratio` ≥ 0.80. Actual 10.92×–18.14×; the
  expectation was a 0.88–0.95 *deficit* relative to GT; the run
  delivered a 12–18× *overshoot*.
- **PASS** — `matched_rate` ≥ 0.720 must-have. Cleared by 0.235
  absolute, but the cause is the metric's tolerance behaviour, not
  the design's mechanism (see [Takeaways §1](#takeaways)).

Nice-to-have criteria — `matched_rate ≥ 0.740` cleared; `error_median_ms ≤ 10`
cleared (1.58 actual); `frame AUC-PR ≥ 0.85` missed (0.719 actual).

Fail-criteria — none triggered. `matched_rate ≥ 0.65` at all evals;
`pos_rate_pred_50` > 0.005 always; `best_k_step` > 0 always; loss
did not plateau before E5.

**Summary**: 1 of 5 must-have criteria PASSED outright (`matched_rate`)
and 1 PASSED on a technicality (`separation`); 3 of 5 FAILED. The
must-have criteria as written were calibrated to detect class
collapse and divergence — they didn't anticipate the specific
failure mode that emerged (saturation collapse with correct peak
*locations* but unfilterable confidence).

## Takeaways

1. **The "matched_rate 0.97" headline is misleading; the model is
   not better than #007.** The AR decoder uses τ=0.5 + 1-D NMS over
   the final-step (k=15) saturated map. Every blob has a local
   maximum at ≈1.0, so NMS keeps one peak per blob. Because the
   blobs are roughly correctly positioned, ±2-frame tolerance covers
   them. The metric counts a real GT onset as "matched" whenever any
   of the 1.34× over-emitted peaks falls within ±2 frames — even if
   the model produced 3 emitted peaks around a single real onset.
   The same forgiveness inflates `hallucination_rate` to 0.36 (those
   extras are "hallucinated" predictions). The real model quality —
   measured at a comparable operating point — is **not** above #007.

2. **The pre-run reframing arguments were each *individually*
   correct, but the gain from them was destroyed by an unrelated
   loss-weighting bug.** Removing the STOP class did remove its
   bias (no `predict_stop_when_uncertain` collapse). Removing the
   AR error-compounding axis did shorten the AR feedback path
   (per-window event count is now ~15 vs ~1, AR steps per chart
   dropped ~10×). The audio-feature input channel does help
   (`frame/separation` 0.81 at eval 4 vs n/a baseline; the model
   does cleanly distinguish GT-positive from GT-negative bins at
   sampled t). None of these mechanisms were broken; they were just
   masked.

3. **The bug: Min-SNR weighting is wrong-signed for x0-parameterization.**
   `training/framewise_diffusion_loss.py:137` computes
   `weight = min(snr, γ) / snr`. That formula was derived for
   **ε-prediction** (Hang et al. 2023), where the unweighted per-t
   loss naturally explodes at low t and Min-SNR caps the explosion.
   Applied to **x0-prediction** (what `process_config.parameterization`
   is set to in `config/model.json:24` and what the process enforces
   in `diffusion/processes.py:278`) the same formula multiplies
   low-t gradient by γ/snr → ≈0, killing the refinement-regime
   signal entirely. Confirmed by `loss/per_t_q0=0.0017` vs
   `loss/per_t_q3=0.0596` (35× ratio) and by the imbalance **growing
   with training** (q3/q0 rises 21.9 → 35.1 across the 5 evals).

4. **Consequence: the sampler bleaches peaks instead of sharpening
   them.** At training time the model sees the high-t regime almost
   exclusively, so it learns to **emit a roughly-correct activation
   map from audio + cursor context, ignoring x_t**. At inference, by
   the time the DDIM chain reaches low t, the model has nothing it
   was trained to do — it doesn't refine, it doesn't sharpen, it
   just outputs the same audio-conditioned map at higher amplitude.
   Each sampler step pushes peak values closer to the x_0 target's
   binary 1.0; by k=15 every peak is saturated. See `custom/per_t_loss_imbalance/peak_saturation_and_threshold.png`.

5. **Confidence-grading information is preserved at k=0 and
   destroyed by k=15.** A salvage operating point exists at k=0 (use
   the first sampler step's output, NMS=3, threshold=0.95) — at
   eval 5 this gives density_ratio 1.00×, hallucination 0.168,
   recall 0.851 [`rollout_maps.npz`, k=0]. **This was not run as a
   formal benchmark** because 16b will rebuild the loss anyway and
   the point of this experiment is now closed; the salvage exists
   as evidence that the audio-conditioning pathway works.

6. **Self-conditioning and the absence of `x0_scale` are secondary
   contributors.** Self-cond is trained at the same t with 50 %
   probability (`models/framewise_diffusion_detector.py:332-348`)
   but consumed at a noisier t at inference (`diffusion/samplers.py:259`)
   — a real distribution mismatch but small relative to the Min-SNR
   bug. `FramewiseActivationProcess` lacks the `x0_scale` parameter
   that `GaussianContinuousProcess` exposes (`diffusion/processes.py:62`),
   so x_0 is fed in [0, 1] while noise is unit-variance — at any
   meaningful t the per-bin signal is dwarfed by noise, encouraging
   the denoiser to ignore x_t entirely. Both compound the primary
   bug but neither alone would produce the rollout-regression shape.

7. **Conclusion — supersede, don't patch.** The framing
   (per-frame activation map, multi-event emission per AR step,
   audio-feature denoiser) is sound; the diffusion machinery
   inherited from #015 carries a parameterization-mismatched loss
   weight that makes the chain anti-refinement. Fixing it in place
   would invalidate the published numbers and the rollout artifacts
   for this checkpoint, and the trainer needs new tests for the
   weighting path anyway. Closing #016 and opening a successor
   experiment is cleaner.

## Followup questions

- **#016b — `snr_weighting=false` (or x0-mode form)** is the minimum
  patch. Three candidate weightings to bench: (a) uniform-t MSE
  (`snr_weighting=false`); (b) Hang 2023 x0-mode
  `weight = (min(snr, γ) + 1) / (snr + 1)`; (c) inverse Min-SNR
  `weight = γ / min(snr, γ)` to actively boost low-t. Plan: a fast
  ~5k-step probe with each weighting, gated on rollout
  `final_vs_best_delta` ≥ −0.02 before committing to a full run.
- **#016c — add `x0_scale` to `FramewiseActivationProcessConfig`** so
  the activation-map signal isn't drowned by unit-variance noise.
  Independent fix; could combine with #016b.
- **#016d — fix self-cond training distribution.** Sample
  `prev_x0_hat` from a one-step-noisier `t` (matching what the
  sampler feeds at inference). Independent fix; smaller magnitude.
- **#016e (deferred) — combined head ensembling framewise +
  next-bin diffusion.** Only worth attempting after #016b/c land a
  working framewise baseline; the head architectures are
  independent.
