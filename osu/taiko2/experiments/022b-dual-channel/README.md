# Experiment 022b — Dual-channel mel + octopus

## Status

`Planned`

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

<!-- TODO: fill after run -->

## Visualizations

<!-- TODO: fill after run -->

## Vs prediction

<!-- TODO: fill after run -->

## Takeaways

<!-- TODO: fill after run -->
