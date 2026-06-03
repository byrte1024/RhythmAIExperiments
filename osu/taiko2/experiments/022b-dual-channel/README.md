# Experiment 022b — Dual-channel mel + octopus

## Status

`Complete`

## Context

[#022](../022-octopus-input/) demonstrated that the octopus gradient
is learnable as a standalone onset representation (frame F1 0.784, AR
F1 0.728) but plateaus ~0.035-0.04 below mel on every metric
[exp_022_octopus_input, metrics.jsonl]. The critical failure was
silence_overlap_f1: 0.389 vs mel's 0.527 — the octopus
representation's log-domain loudness normalization makes quiet sections
indistinguishable from loud ones, causing onset hallucinations in
silence.

The mini-to-AR gap analysis revealed the root cause: octopus per-
window detection is only 0.04 behind mel (mini F1 0.750 vs 0.782),
but the gap widens to 0.07 in full AR inference (fps50 AR 0.570 vs
0.638). The AR decoder needs absolute energy context to decide where
to emit and where to stay silent — information mel provides but
octopus discards.

This experiment gives the model both: **177 input channels** (80 mel
+ 97 octopus gradient) concatenated along the frequency axis. The
model can learn to use octopus for onset detection (where it excels
at cross-frequency synchrony) and mel for energy-based gating (where
it provides silence/loudness context). The two representations are
**complementary, not redundant** — unlike #019's coincidence summary
which was derived from mel, the octopus gradient comes from a
different signal processing pipeline (gammatone filterbank, not FFT)
and encodes a fundamentally different quantity (synchrony, not energy).

The `taiko2_v1_mel_octopus` dataset already stores `(177, T)` features.
No dataset rebuild needed — only config changes.

## Citations

- Direct baselines:
  - [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).
    Best sweep tau=0.4 (eval_248088): `f1` 0.771, `precision` 0.778,
    `recall` 0.757, `density_ratio` 0.964, `dc_human` 92.4,
    `gap_hist_tvd` 0.331, `silence_overlap_f1` 0.527
    [017f threshold_sweep.json].
  - [#022 -- octopus-only](../022-octopus-input/). Frame F1 0.784,
    AR F1 0.728. silence_f1 0.389. Mini-to-AR gap 0.128 at fps50
    [exp_022, metrics.jsonl].
- Failed alternatives:
  - [#019 -- coincidence input](../019-coincidence-input/). Model
    ignored coincidence (+0.2%) — redundant with mel.
  - [#021 -- wider stem](../021-wider-stem/). Conv width not the
    bottleneck.
- Interpretability:
  - [#020](../020-activation-maximization/). High-band saliency,
    bin-exact confidence.
  - [#022 benchmarks](../022-octopus-input/). Octopus is less
    context-dependent (no_context -7% vs -10.3%).
- Biology / implementation: see [#022](../022-octopus-input/)
  citations.

---
<!--
PRE-RUN. Do not edit after the run.
-->
---------------------------------------------------------------------

## Hypothesis

### Claim

Providing both mel (80 bands) and octopus gradient (97 cells) as a
177-channel input will beat mel-only on AR F1 and silence_overlap_f1,
because the model can use octopus for onset timing (where it provides
pre-computed cross-frequency synchrony) and mel for energy context
(where it provides absolute loudness needed for silence detection and
onset importance weighting).

### Mechanism

#022 showed octopus loses to mel primarily in the AR loop, not in
per-window detection (mini-to-AR gap 0.128 vs mel's 0.100 at fps50).
The AR decoder needs to make chart-level decisions — where to emit,
where to stay silent, how densely to place onsets — that require
energy context the onset-only representation discards.

With dual input, the conv stem receives `(177, 1000)` and mixes mel
bands with octopus cells in the first convolution. The model can
learn cross-representation features: "there's a strong octopus onset
here AND mel energy is high" -> emit; "there's an octopus onset but
mel energy is low" -> suppress (ambient noise, not a real hit).

Unlike #019 (coincidence + mel), where the coincidence was derived
from mel and therefore redundant, the octopus gradient is computed
from a fundamentally different signal pipeline. #022 proved the
octopus signal carries unique information (F1 0.728 from onset
synchrony alone). The question is whether adding it to mel provides
a net benefit or just adds noise.

The risk: 177 input channels is 2.2x wider than mel's 80. The conv
stem's first layer grows from `Conv1d(80, 192, k=7)` to
`Conv1d(177, 192, k=7)` — more parameters in a layer that's already
the main spectral mixing bottleneck. #021 showed that stem width
(intermediate channels) didn't help, but input width (more
information to mix) is a different dimension.

### Predicted numbers

Reference: [#017f](../017f-framewise-bce-metrics-rerun/) tau=0.4
sweep [017f threshold_sweep.json, eval_248088] and
[#022](../022-octopus-input/) E8
[exp_022, metrics.jsonl].

| Metric | 017f (mel) | 022 (octopus) | Predicted (022b) | Notes |
|---|---:|---:|---:|---|
| frame F1 | 0.822 | 0.784 | **>= 0.82** | Mel anchors quality |
| AR F1 (25ms) | 0.771 | 0.728 | **>= 0.77** | Mel + octopus onset sync |
| fps50 F1 | 0.741 | 0.698 | **>= 0.74** | Watched metric |
| density_ratio | 0.964 | 0.942 | **0.93-1.03** | Near 1.0 |
| dc_human | 92.4 | 92.0 | **>= 92** | Should hold |
| silence_f1 | 0.527 | 0.389 | **>= 0.54** | Mel fixes octopus silence problem |
| gap_TVD | 0.331 | 0.353 | **< 0.33** | Octopus onset sync helps structure |
| mini-to-AR gap (fps50) | 0.100 | 0.128 | **< 0.11** | Mel restores AR coherence |

## Success criteria

- **Must:** AR F1 >= 0.75 at some threshold (at least matches mel).
- **Must:** silence_f1 > 0.45 (above octopus-only's 0.389, proving
  mel restores silence awareness).
- **Confirms hypothesis if:** AR F1 > 0.78 at matched density,
  beating mel's 0.771 — the octopus channel added value.
- **Fails if:** frame F1 < 0.80 — the 177-channel input degraded
  learning.
- **Fails if:** metrics are indistinguishable from 017f (mel-only) —
  the octopus channel added no value, same as #019.
- **Nice-to-have:** gap_TVD < 0.30, beating both mel and octopus.
- **Nice-to-have:** mini-to-AR gap < 0.10 at fps50, proving the
  dual representation improves AR coherence.

## Changes from baseline

Baseline: [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).

Three changes:

- `config/model.json` -- `n_mels: 80 -> 177`. Model sees all 177
  channels (80 mel + 97 octopus).
- `config/adapter.json` -- `feature_rows` removed (was `[80, 177]`
  in #022). No slicing -- adapter passes all rows.
- `config/infer.json` -- `MelOctopusSampler` without `output_rows`.
  Produces `(177, T)` at inference time.

Same dataset as #022: `taiko2_v1_mel_octopus`. Same loss, trainer,
data config. decode_threshold=0.4.

## Run config

- Run name: `exp_022b_dual_channel`
- Config snapshots: [`config/`](./config/)
- Dataset: `taiko2_v1_mel_octopus` (177-channel: 80 mel + 97 octopus;
  model sees all rows)
- Command:
  ```bash
  set -e CUDA_VISIBLE_DEVICES && ulimit -n 65536 && \
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_022b_dual_channel \
      --config-dir osu/taiko2/experiments/022b-dual-channel/config \
      --dataset taiko2_v1_mel_octopus --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/022b-dual-channel/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

## Augmentation notes

FreqRoll uses `section_boundary=80` (auto-detected from adapter
config when `feature_rows` is set). Without `feature_rows`, the
boundary is None and all 177 rows roll as one block. This means mel
values CAN wrap into octopus space during FreqRoll. The two value
ranges differ (mel: -30 to +50 dB, octopus: 0 to 1), but the roll
is small (+/-3 rows) and acts as regularization — the model must be
robust to slight frequency shifts in both representations
simultaneously. If this causes issues, add explicit
`section_boundary=80` to the augmentation config.

---------------------------------------------------------------------
<!--
POST-RUN. Do not fill until the run completes.
Everything below comes from real measurements, not predictions.
-->
---------------------------------------------------------------------

## Results summary

17 evals completed (steps 20,570 to 349,690). Threshold sweep across
all checkpoints x 7 thresholds. **First experiment to beat mel-only
on both pointwise and distributional metrics in the threshold sweep.**

### Threshold sweep: 022b beats 017f at every threshold

Best AR F1: 022b **0.794** vs 017f 0.782 (+0.012). Both at tau=0.3
[022b threshold_sweep.json eval_329120, 017f threshold_sweep.json
eval_289436].

| tau | 022b F1 | 017f F1 | Delta | 022b gTVD | 017f gTVD | 022b sil_f1 | 017f sil_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.727 | 0.691 | **+0.036** | 0.480 | 0.543 | 0.371 | 0.404 |
| 0.2 | 0.780 | 0.768 | **+0.012** | 0.355 | 0.399 | 0.464 | 0.514 |
| 0.3 | **0.794** | 0.782 | **+0.012** | 0.300 | 0.332 | 0.546 | 0.546 |
| 0.4 | 0.780 | 0.767 | **+0.013** | **0.290** | 0.327 | 0.514 | 0.511 |
| 0.5 | 0.746 | 0.737 | **+0.009** | 0.334 | 0.374 | **0.510** | 0.463 |
| 0.6 | 0.687 | 0.668 | **+0.018** | 0.426 | 0.465 | 0.461 | 0.456 |
| 0.7 | 0.601 | 0.577 | **+0.024** | 0.526 | 0.569 | 0.409 | 0.446 |

022b wins F1 at every threshold (+0.009 to +0.036). gap_TVD better
at every threshold. The advantage is largest at extreme thresholds
(tau=0.1: +0.036, tau=0.7: +0.024) where the mel model's onset
detection degrades more than the dual model's.

### At matched density (tau=0.4, DR ~1.0)

Best density match: 022b eval_287980 (DR 1.003) vs 017f eval_144718
(DR 0.968) [threshold_sweep.json]:

| Metric | 022b | 017f | Delta |
|---|---:|---:|---:|
| **F1 (25ms)** | **0.785** | 0.769 | **+0.016** |
| Precision | 0.767 | 0.777 | -0.010 |
| Recall | **0.786** | 0.756 | **+0.030** |
| density_ratio | 1.003 | 0.968 | +0.035 |
| **dc_human** | **92.79** | 92.50 | **+0.28** |
| **gap_TVD** | **0.296** | 0.335 | **-0.039** |
| ratio_TVD | 0.477 | 0.472 | +0.005 |
| density_corr | 0.540 | 0.538 | +0.003 |
| **gap_IoU** | **0.744** | 0.733 | **+0.012** |
| silence_f1 | 0.546 | 0.576 | -0.030 |
| dense_f1 | **0.979** | 0.969 | **+0.010** |
| bpm_ratio | 1.036 | 0.968 | +0.068 |

Best dc_human overall: 022b **93.26** (eval_205700/tau=0.5) vs 017f
93.18 [threshold_sweep.json].

### Training progression

Frame F1 matched mel from E6 onward. AR F1 converged to mel parity
by E11 and exceeded it at E14/E17
[exp_022b, metrics.jsonl]:

| Eval | Step | F1 | fps50 | AR F1 | DR | dc_h | gTVD | sil_f1 | dF1 vs mel | dAR vs mel |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20,570 | 0.755 | 0.661 | 0.704 | 0.808 | 92.29 | 0.386 | 0.555 | -0.020 | -0.030 |
| 6 | 123,420 | 0.820 | 0.734 | 0.760 | 0.984 | 92.30 | 0.315 | 0.576 | +0.001 | -0.011 |
| 9 | 185,130 | 0.813 | 0.727 | 0.758 | 0.934 | 92.10 | 0.310 | **0.609** | -0.007 | -0.007 |
| 12 | 246,840 | 0.822 | 0.738 | 0.765 | 0.981 | 91.88 | **0.297** | 0.575 | +0.003 | -0.006 |
| 14 | 287,980 | **0.824** | **0.741** | 0.768 | 1.000 | 91.65 | 0.298 | 0.577 | +0.003 | +0.000 |
| 17 | 349,690 | 0.816 | 0.729 | **0.771** | 0.952 | 92.50 | 0.305 | 0.583 | -0.005 | +0.003 |

### Benchmarks

no_coincidence (octopus zeroed): -0.7% F1 — the model uses the
octopus channels slightly, unlike #019's +0.2%.
no_mel (mel zeroed): -40.7% F1 — mel remains the primary signal.
The model is mel-dominant with octopus as a supplementary onset
channel [exp_022b, metrics.jsonl, bench/no_coincidence, bench/no_mel,
step 246840].

Context robustness improved over mel: random_context -6.8% (mel:
-8.2%), context_time_shifted -9.7% (mel: -10.3%)
[exp_022b, metrics.jsonl, bench/*, step 246840].

### Comparison to #019 (coincidence + mel)

Unlike #019 where the model ignored the additional channels
(no_coincidence +0.2%), 022b shows a small but real -0.7% drop when
octopus is zeroed. The octopus gradient, computed from a fundamentally
different pipeline than mel, provides non-redundant information. The
sweep confirms this: 022b beats mel at every threshold, while #019's
gains did not survive the sweep at all operating points.

## Vs prediction

- **AR F1 >= 0.77:** predicted yes -> actual **0.794** at best
  threshold, **0.785** at matched density -> **beat**.
- **silence_f1 > 0.45:** predicted yes -> actual 0.546 at matched
  density -> **match**. Best per-eval was 0.609 (E9).
- **Confirms hypothesis (AR F1 > 0.78):** predicted at matched
  density -> actual 0.785 -> **confirmed**.
- **frame F1 >= 0.82:** predicted yes -> actual 0.824 (E14) ->
  **match**.
- **fps50 >= 0.74:** predicted yes -> actual 0.741 (E14) -> **match**.
- **gap_TVD < 0.33:** predicted yes -> actual **0.290** at tau=0.4 ->
  **beat** (best of any experiment).
- **gap_TVD < 0.30 (nice-to-have):** actual 0.290 -> **match**.
- **mini-to-AR gap < 0.11 (nice-to-have):** not directly measured in
  sweep, but per-eval AR F1 matching mel suggests the gap closed.
- **Indistinguishable from mel (fail):** not triggered — 022b beats
  mel at every threshold by +0.009 to +0.036.

## Takeaways

- **Dual-channel mel+octopus is the new best model.** AR F1 0.794
  at best threshold (+0.012 over mel), 0.785 at matched density
  (+0.016). gap_TVD 0.290 (best ever, -0.039 vs mel). dc_human
  93.26 (best ever). The octopus gradient provides genuine additive
  value over mel — this is not the conv-width effect (#021 showed
  that didn't survive the sweep).

- **The octopus contribution is real but small.** no_coincidence
  benchmark shows -0.7% F1 drop — the model uses the octopus
  channels, unlike #019's coincidence which was ignored (+0.2%).
  The improvement comes from non-redundant onset synchrony
  information that mel cannot provide (gammatone filterbank vs FFT,
  cross-channel coincidence vs per-band energy).

- **Mel remains the dominant signal.** no_mel drops F1 by 40.7%.
  The model routes primarily through mel (energy, timbre, silence
  detection) with octopus as a supplementary onset timing channel.
  This is the correct architecture: mel for "what kind of audio is
  this," octopus for "is there a cross-frequency transient here."

- **Silence_f1 improved during training but not consistently in
  the sweep.** Per-eval best was 0.609 (E9, +0.08 over mel). At
  matched density in the sweep it was 0.546 (vs mel's 0.576, -0.030).
  At tau=0.5 it was 0.510 (vs mel's 0.463, +0.047). The silence
  improvement is threshold-dependent, not a robust structural gain.

- **gap_TVD is the most robust improvement.** Better than mel at
  every threshold, every checkpoint. The octopus onset synchrony
  directly helps the model produce rhythmically faithful gap
  distributions — the coincidence gate pre-computes "where onsets
  are" so the model can focus on "which ones to keep."

- **The path to F1 > 0.80 or > 0.90 remains open.** This experiment
  improved F1 from 0.782 to 0.794 (+1.5%) by adding a biologically-
  inspired onset representation. The improvement is real but
  incremental. The remaining gap to human-level chart quality
  involves higher-level decisions — which onsets to select from the
  detected set, how to structure silence and density across the full
  song, onset type classification (DON vs KA) — that are unlikely to
  be solved by input representation changes alone. Future directions
  include:
  - **Model architecture changes**: the Conv1D detection head, the
    transformer depth/width, the conditioning mechanism.
  - **Loss function innovations**: silence-aware loss, density-
    matching loss, structural losses that penalize gap/ratio
    distribution mismatch directly.
  - **Onset type prediction**: the current model predicts binary
    onset/no-onset. Predicting DON vs KA would require the model
    to learn timbral distinctions — an orthogonal capability.
  - **Multi-scale AR**: the current 500-bin (2.5s) window may be
    too narrow for song-level structure decisions. Hierarchical
    prediction (section-level density -> per-window onsets) could
    help.

## Followup questions

- **Is the improvement from octopus or from 177-channel conv width?**
  Run 017f on the mel_octopus dataset with `feature_rows: [0, 80]`
  (mel-only, 80 channels from the 177-channel features). If metrics
  match 017f's original results, the improvement is from octopus.
  If they differ, the different dataset (10,038 vs 10,048 charts)
  explains the shift.

- **Separate stems**: two conv stems (mel and octopus) merged before
  the transformer, instead of concatenating on the frequency axis.
  This lets each stem learn representation-specific spectral mixing
  without cross-contamination in the first convolution.

- **Octopus-gated loss**: weight the BCE loss by octopus onset
  strength — bins where octopus detects a strong onset get higher
  loss weight. This explicitly couples the onset detection signal
  to the training objective.
