# Experiment 017 — Framewise BCE (non-diffusion control)

## Status

`Planned`

## Context

[#014](../014-diffusion/), [#015](../015-diffusion-patched/),
[#016](../016-framewise-diffusion/), and
[#016b-mini](../016b-mini/) explored diffusion as a replacement output
head for the [#007](../007-time-stretch/) softmax-CE next-bin
classifier. All four plateaued below #007's gt `matched_rate` 0.7028
[exp_007_time_stretch, step 413,480,
val/single/corpus/gt_cond_cmp/matched_rate_mean] (or, in #016's case,
produced a 0.97 headline that masked cluster-spam with density_ratio
10.9x [exp_016_framewise_diffusion, step 103,370,
val/single/corpus/gt_cond_cmp/density_ratio_mean]).

#016's post-mortem identified that the audio-conditioned Conv1D
pathway itself works — `frame/separation` 0.810
[exp_016_framewise_diffusion, step 82,696,
val/single/frame/separation], k=0 recall 0.987 at threshold 0.3 —
but the Gaussian diffusion chain destroys confidence grading and the
DDIM sampler monotonically regresses. #016b confirmed that Min-SNR
sign was not load-bearing; the saturation traces to the sigma=2
Gaussian-smoothed target shape.

This experiment strips the diffusion machinery entirely and trains the
same framewise activation-map pathway as a single-shot BCE detector.
The goal is to establish whether the **framewise framing on its own**
beats [#007](../007-time-stretch/) — independent of the diffusion
question. If it does, diffusion was never the bottleneck and future
work pivots to "what does iterative refinement actually buy us, if
anything?" (D3PM absorbing-state, #018 candidate). If it does not,
the framewise framing itself is the ceiling.

**Platform note:** This run executes on CachyOS Linux (Arch-based,
kernel 7.0.6-1-cachyos) rather than the Windows environment used for
#001-#016b. Same GPU (RTX 5070), same PyTorch nightly
(2.12.0.dev20260307+cu128), same CUDA 12.8. Python micro-version
bumped 3.13.12 to 3.13.13. No effect on results is expected — all
numerically-relevant components (torch, CUDA, numpy) are identical
builds.

## Citations

- Direct baseline:
  - [#007 -- TimeStretch](../007-time-stretch/). Best gt
    `matched_rate` 0.7028 [exp_007_time_stretch, step 413,480,
    val/single/corpus/gt_cond_cmp/matched_rate_mean]. Best
    `onset/miss` 0.2406 [exp_007_time_stretch, step 372,132,
    val/single/onset/miss]. The softmax-CE next-bin baseline this
    experiment compares against.
  - [#012 -- onset-feature channels](../012-onset-channels/). Best gt
    `matched_rate` 0.7080 [exp_012_onset_channels, step 308,550,
    infer_corpus/eval_308550/gt_cond/comparisons_summary.json:fields.matched_rate.median]
    -- current taiko2 all-time-best on AR `matched_rate`.
- Parents of the framewise design:
  - [#016 -- framewise activation-map diffusion](../016-framewise-diffusion/).
    Source of the Conv1D+audio-features pathway. Best `frame/separation`
    0.810 [exp_016_framewise_diffusion, step 82,696,
    val/single/frame/separation]. k=0 single-shot F1 0.761 with 1/5
    epoch and wrong-signed loss.
  - [#016b-mini -- Min-SNR sign probe](../016b-mini/). Confirmed
    saturation is target-shape-dependent, not loss-weighting-dependent.
- Framewise precedents:
  - [#011 -- onset feature survey](../011-onset-feature-survey/).
    Classical mel-domain ODFs at F1 = 0.679 against chart-author GT
    at +/-10 frames [011-onset-feature-survey/results/summary.json:by_algo.spectral_flux.by_tolerance.10].
  - [#011b -- onset disagreement](../011b-onset-disagreement/).
    Cross-group ODF pairs reach recall 0.905 [011b-onset-disagreement/results/summary.json:pairwise.pairs.hfc_mel+spectral_flux.recall_union].
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

If the diffusion machinery is removed from #016's framewise design and
the model is trained end-to-end with binary-target BCE + positive-class
upweighting, then **AR `matched_rate` (gt_cond, median) will reach
>= 0.720** (clears #007's 0.7028 baseline), with `error_median_ms
<= 12 ms` and `density_ratio` in [0.85, 1.10], because the single-
shot audio-conditioned pathway already locates onsets correctly (k=0
recall 0.987 in #016) and removing the diffusion chain eliminates the
saturation/regression pathology without losing any useful refinement.

### Mechanism

Three effects relative to #016:

1. **No saturation.** The model outputs raw logits; `sigmoid` produces
   graded confidences by construction. The sigma=2 Gaussian target
   that caused k=15 saturation in #016 is replaced by binary {0, 1}
   targets — the model learns to be confident at GT bins and
   uncommitted elsewhere, without any chain to bleach the gradation.
2. **No regression.** There is no multi-step sampler chain to degrade
   the prediction. The single forward pass is the final output.
3. **BCE is numerically stable and well-understood for sparse binary
   classification.** Positive-class upweighting (clamp [10, 200])
   handles the ~1 % positive-bin rate. No Min-SNR, no per-t quartile
   imbalance, no loss-weight asymmetry.

The Conv1D head reuses #016's architecture minus the diffusion-specific
channels (x_t, time embed, self-cond, FiLM-from-time), so the head
is ~5 M params smaller than #016's denoiser.

### Predicted numbers

Reference: best variants of [#007](../007-time-stretch/),
[#012](../012-onset-channels/). gt_cond medians where applicable.

| Metric | #007 best | #012 best | Predicted (#017, best eval) | Notes |
|---|---:|---:|---:|---|
| AR `matched_rate` (gt_cond, median) | 0.7028 [a] | 0.7080 [b] | **>= 0.720** must / **0.74-0.78** nice | must-have: clears #007 baseline |
| AR `error_median_ms` | 10.2 [a] | 8 [b] | **<= 12** | timing should hold — single-shot peak locations are accurate per #016 k=0 |
| AR `density_ratio` | 0.865 [a] | n/a | **0.85-1.10** | no STOP bias, no cluster spam — BCE threshold controls density |
| AR `hallucination_rate` | 0.146 [a] | 0.135 [b] | **0.10-0.18** | NMS + threshold + min-gap controls |
| AR `hi_pspace` | 0.909 [a] | 0.894 [b] | >= 0.90 | should hold |
| AR `dc_human` | 0.928 [a] | 0.918 [b] | 0.91-0.93 | might recover #007 ground |
| frame F1 (tau=0.5, +/-2 frames) | n/a | n/a | **>= 0.85** at converged eval | #016 sampled-t leaky 0.91 / k=0 rollout 0.76 at 1/5 epoch |
| frame AUC-PR | n/a | n/a | **>= 0.85** | #016 = 0.72 (wrong-signed loss, 1 epoch) |
| frame `pred_hedge_frac` | n/a | n/a | **<= 0.15** | model should commit — bimodal prediction distribution |
| frame `brier` | n/a | n/a | **<= 0.05** | well-calibrated confident predictions |
| mini/tau50/matched_rate (tol=25ms) | n/a | n/a | **>= 0.80** | per-window onset matching at canonical threshold |
| Total params | 16.35 M [a] | n/a | **~21 M** | 16.35 M trunk + ~5 M Conv1D head |
| Wall time / eval | 2.18 h [a] | n/a | **1.5-2.0 h** | no sampler loop — single forward pass |

[a] exp_007_time_stretch, step 413,480, val/single/corpus/gt_cond_cmp/
(and sibling fields).
[b] exp_012_onset_channels, step 308,550, infer_corpus/eval_308550/
gt_cond/comparisons_summary.json.

## Success criteria

- **Must have:** AR `matched_rate` >= 0.720 at the best eval
  (clears #007's 0.7028 baseline by >= 1.7 pp; ties or beats #012's
  0.7080 ATH).
- **Must have:** AR `density_ratio` in [0.70, 1.50] at the best eval
  -- no cluster-spam regime (rules out a #016-style tolerance illusion).
- **Must have:** training stable, no NaN, runs to E10+ without
  divergence; `frame/separation >= 0.4` at any post-warmup eval.
- **Must have:** `frame/pred_hedge_frac <= 0.25` at any post-warmup
  eval -- the model is producing committed (not hedged) predictions.
- **Nice-to-have:** AR `matched_rate` >= 0.740 -- clear architectural
  win over #007; new taiko2 SOTA.
- **Nice-to-have:** frame F1 (tau=0.5, +/-2) >= 0.90 at a late eval.
- **Nice-to-have:** AR `error_median_ms` <= 10 -- matches or beats
  #007 on timing precision.
- **Fails if:** AR `matched_rate` < 0.65 at every eval -- framewise
  framing is structurally inferior to next-bin softmax even without
  diffusion overhead.
- **Fails if:** `frame/pred_hedge_frac > 0.50` at every eval -- model
  hedges instead of committing (BCE not sharp enough for this task).
- **Fails if:** AR `density_ratio > 3.0` at every eval -- cluster
  spam returns, meaning the decoder (not the diffusion chain) was the
  problem in #016.

## Changes from baseline

Baseline: [#007 -- TimeStretch](../007-time-stretch/).

Code changes (7 new files, 4 edited files, 24 new tests; 621/621
passing):

- **`osu/taiko2/models/framewise_detector.py`** (NEW) --
  `FramewiseDetector(EventEmbeddingDetector)` +
  `FramewiseDetectorConfig` + `FramewiseDetectorOutput`. Conv1D-on-
  bin-axis head with per-bin audio features (linearly upsampled from
  125 future audio tokens to 500 bins), sinusoidal positional embedding
  (32-dim), broadcast cursor projection (32-dim). 3 blocks of Conv1d
  (kernels {31, 15, 15}, 256 channels), GroupNorm(8), SiLU, Dropout.
  Output `(B, n_bins)` logits + `sigmoid(logits).detach()` as
  `confidence_map`.
- **`osu/taiko2/training/framewise_bce_loss.py`** (NEW) --
  `FramewiseBCELoss`. `BCEWithLogitsLoss` with per-sample `pos_weight =
  clamp(n_neg/max(n_gt,1), [10, 200])`. Target: `target_map_binary`
  (strict {0,1}, no sigma smoothing). Reports 18 scalar metrics
  including frame F1, AUC-PR, AUC-ROC, separation, hedging fraction,
  Brier score, confidence-by-outcome medians.
- **`osu/taiko2/training/framewise_metric.py`** (NEW) --
  `FramewiseMetric(Metric)`. Accumulates across eval batches. Mini-
  chart comparison via `gt_match_metrics` at 5 thresholds
  {0.3, 0.4, 0.5, 0.6, 0.7} x 5 tolerances {5, 10, 25, 50, 100 ms}.
  Per-threshold NMS (kernel=3) + greedy match using the same matching
  function the AR corpus uses.
- **`osu/taiko2/training/framewise_diagnostics_artifact.py`** (NEW) --
  `FramewiseDiagnosticsArtifact`. 5 per-eval outputs: per-bin rate
  plot, target value histogram, prediction value histogram (linear +
  log), confidence-by-outcome overlaid histograms, reliability /
  calibration plot with ECE + Brier. Designed for subclassing by a
  future `DiffusionFramewiseDiagnosticsArtifact`.
- **`osu/taiko2/inference/autoregressive/framewise_decoder.py`** (NEW)
  -- `FramewiseDecoder` + shared `framewise_decision_from_map`. Same
  threshold + NMS + min-gap logic as `FramewiseDiffusionDecoder` but
  without a sampler -- operates on `output.confidence_map` directly.
  `FramewiseDiffusionDecoder._decision_from_map` refactored to use the
  shared function.
- **`osu/taiko2/domain/chart.py`** -- new public `gt_match_metrics()`
  with configurable `tolerances_ms` parameter (default 5/10/25/50/100
  ms). `_gt_match_metrics` delegates to it. Returns per-tolerance
  `matched_rate_at_tol_X` and `halluc_rate_at_tol_X` keys.
- **`osu/taiko2/domain/framewise.py`** -- `make_framewise_target`
  accepts `sigma=None` to skip Gaussian smoothing (smoothed = binary).
- **`osu/taiko2/training/framewise_adapter.py`** -- `binary_only: bool`
  on `FramewiseSampleAdapterConfig`. Routes `sigma=None` when set.
- **`osu/taiko2/training/framewise_artifacts.py`** --
  `_extract_pred_target` prefers `output.confidence_map` over raw logits.
- **`osu/taiko2/training/loop.py`** -- `_framewise_batch_stats` prefers
  `output.confidence_map`.
- **`osu/taiko2/cli/train.py`** -- `is_framewise_bce` dispatch branch.
  Wires model, loss, adapter, `FramewiseMetric`, diagnostics artifact,
  benchmarks (`--benchmarks all`), `InferCorpusHook`.
- **Tests** (NEW): 24 new tests in `test_framewise_bce.py`. Full suite
  at 621/621 passing.

Config snapshots ([`config/`](./config/)):

- `config/model.json` -- `FramewiseDetectorConfig` with Conv1D head
  (256 channels, kernels {31, 15, 15}, pos_embed 32, cursor_proj 32).
- `config/loss.json` -- `FramewiseBCELossConfig`.
- `config/adapter.json` -- `FramewiseSampleAdapterConfig` with
  `binary_only=true`.
- `config/data.json` -- identical to #016 (`d_events=100`).
- `config/trainer.json` -- 15 epochs, batch 64, lr 3e-4, watched
  metric `loss` (lower is better).
- `config/infer.json` -- `FramewiseDecoder` + AR predictor, threshold
  0.5, NMS kernel 3.

## Run config

- Run name: `exp_017_framewise_bce`.
- Config snapshots: [`config/`](./config/).
- Dataset: `taiko2_v1` (80-row mel; same as #007).
- Command:
  ```bash
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_017_framewise_bce \
      --config-dir osu/taiko2/experiments/017-framewise-bce/config \
      --dataset taiko2_v1 --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --infer-corpus-spec osu/taiko2/experiments/017-framewise-bce/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

─────────────────────────────────────────────────────────────────────
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
─────────────────────────────────────────────────────────────────────

## Results summary

### Final vs baseline

| Metric | Baseline (exp N) | This run (final) | Delta | Direction |
|---|---:|---:|---:|:---:|
| — | — | — | — | — |

### Per-eval progression

{Generated from `runs/exp_017_framewise_bce/metrics.jsonl`.}

Machine-readable copies (both tables): [`metrics.json`](./metrics.json).

## Visualizations

![Training loss](graphs/01_train_loss.png)
*Training loss over steps (log-y).*

![Validation progression](graphs/02_val_progression.png)
*Watched metric across evals.*

## Vs prediction

## Takeaways

## Followup questions
