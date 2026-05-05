# Experiment 012 — Onset-feature channels appended to mel input

## Status

`Planned`

## Context

[#007](../007-time-stretch/) is the standing direct-bin baseline at
val miss 0.2406 (best, E18, step 372,132). Every experiment since
([#008](../008-log-emd/), [#009](../009-mdn/),
[#010](../010-ratio-decomposition/) family) has tried to break that
plateau by changing the loss, the head architecture, or the gradient
routing. **None of them changed the input the model sees.** Audio
has been a fixed 80-band log-mel at 5 ms / frame since #001.

[#011](../011-onset-feature-survey/) and
[#011b](../011b-onset-disagreement/) measured what classical
mel-domain onset detection algorithms can recover from the same
log-mel features the model already gets. Single-channel
`spectral_flux` reaches frame-wise F1 = 0.679 against GT at
±10 frames; sub-band variants are tied; cross-group pairs hit
recall 0.905 at K = 2. The signal exists in the cached features —
the question is whether handing it to the model as a pre-computed
input plane (rather than asking the conv stem to derive it from
scratch alongside everything else) reduces miss / lifts hit.

This experiment is the smallest viable test of that. **Four extra
input rows: sub-band spectral flux split into 4 mel-band groups,
range-matched to the log-mel dB scale.** The model receives an
84-row input instead of 80; everything else is identical to #007.

## Citations

- **Direct baseline**: [#007 — TimeStretch](../007-time-stretch/).
  Best val miss 0.2406, hit 0.7333, exact 0.5568, frame_err_p90 30,
  stop_f1 0.5831 (E18, step 372,132). Identical model, loss,
  adapter, augmentation pipeline, dataset build options. Only the
  **audio feature pipeline** and the **conv stem input row count**
  change.
- **Channel set decision**: [#011](../011-onset-feature-survey/).
  `spectral_flux` chosen as the ODF; sub-band split chosen for
  per-row representation weight (single broadband row ≈ 1 % of an
  84-row input, sub-band split gives 4 rows ≈ 5 %).
- **Channel multiplicity decision**:
  [#011b](../011b-onset-disagreement/). Per-band recall does NOT
  specialize on DON vs KA (refuting the pre-run prediction), but
  the sub-band rows still carry per-frequency-window flux that
  encodes *where* in the spectrum the activation came from — useful
  inductive bias for the conv stem.
- **Encoding choice — "as extra mel bands" vs "as separate
  channels"**: framing chosen during #012 pre-run discussion. ODF
  rows are range-matched to log-mel dB scale (-80 .. 0 dB) so the
  conv stem treats them like additional log-mel bands rather than a
  signal in a different distribution.
- **MIR onset detection literature**:
  [Onset Detection Revisited (Dixon, 2006)](https://www.dafx.de/paper-archive/2006/papers/p_133.pdf),
  [librosa.onset.onset_strength_multi](https://librosa.org/doc/main/generated/librosa.onset.onset_strength_multi.html),
  general consensus that spectral-flux-family ODFs hit F1 ≥ 0.85 on
  solo-drums at ±50 ms tolerance.

---

## Hypothesis

### Claim

If we append four sub-band-spectral-flux rows to the existing 80
mel bands at the same time grid, range-matched to log-mel dB scale,
**val/single/onset/miss will improve by 1.0-2.0 pp at the matched
step / matched eval count vs #007** (target: miss ≤ 0.225 by E18).
The improvement comes from the conv stem getting a pre-computed
"audio onset here, in this frequency region" signal that today's
80-band log-mel + 8-layer transformer has to derive from scratch.

train_noaug — val gap is predicted to be roughly the same as
#007's (≤ 2.5 pp), because the channels don't add new training
data — only re-package what the model already sees.

### Mechanism

1. **The conv stem already learns spectral flux as a feature.**
   ``Conv1d(80 → 192, k=7, s=2)`` over log-mel must be learning
   per-band time differences of magnitude as a low-level feature
   on its way to onset detection. By feeding pre-computed
   sub-band flux as additional input rows, we move that
   computation out of the conv stem's weight budget. The conv
   stem gets ~5 % more input bandwidth and four rows of high-
   quality "spectral change" signal it doesn't have to learn.

2. **Range-matching keeps the dynamics balanced.** Per-chart
   99th-percentile normalization of the sub-band flux maps to
   ``[0, 1]``; linear stretch to ``[-80, 0]`` matches log-mel's
   dB range. Conv-stem first-layer weights init'd via Kaiming
   normal see input variance comparable across rows from step 0,
   so the onset rows aren't down-weighted just because they're
   different scale.

3. **Frame alignment.** [#011](../011-onset-feature-survey/)
   showed every classical ODF collapses below ±2 frames because
   their activation peaks 2-3 frames after the attack. We
   considered bucket-pooling to ±5 or ±10 frame windows to
   collapse this lag, but for the first pass we serve the
   activation at the raw 5 ms grid — the conv stem already
   handles ±2 frame alignment naturally via its kernel = 7
   and stride = 2.

4. **Sub-band rather than broadband.** A single broadband
   spectral_flux row is ~1.2 % of an 81-row input — risk that
   the conv stem learns to ignore it. Four sub-band rows
   (~5 % of input) put the signal at meaningful weight while
   giving the conv stem per-frequency-region structure: each
   row encodes "spectral change at frequency band X" rather
   than "spectral change anywhere." Per [#011b](../011b-onset-disagreement/),
   the per-band recall does not specialize on DON vs KA, but
   the bands still encode different frequency localizations
   that the conv stem can use.

### Predicted numbers

Reference: [#007](../007-time-stretch/) E18, step 372,132 (best
val miss).

| Metric | #007 @ E18 | Predicted (#012, mature eval) | Notes |
|---|---:|---:|---|
| val/single/onset/miss | 0.2406 | 0.220 - 0.232 | 1-2 pp improvement |
| val/single/onset/hit | 0.7333 | 0.745 - 0.755 | 1-2 pp |
| val/single/onset/exact | 0.5568 | 0.560 - 0.580 | matched or +1-2 pp |
| val/single/onset/fhit (±2 fr) | ~0.50 | ~0.50 | unchanged — channels collapse < ±2 fr |
| val/single/onset/fgood (±7 fr) | ~0.66 | 0.66 - 0.69 | small lift in mid-tolerance |
| val/single/onset/stop_f1 | 0.5831 | 0.58 - 0.62 | small lift; channels indicate "audio onset present" |
| val/single/onset/frame_err_p90 | 30 | 28 - 30 | unchanged or marginal |
| ratio-banding ridges (`±log 2/3`) | present | present | channels don't encode tempo |
| train_noaug gap | -2.55 pp | -2.0 to -3.0 pp | similar overfit dynamics |

#### Predicted ranking surprises

- **Strict-frame metrics (fhit, exact) are unlikely to move much.**
  Per #011, classical ODFs collapse at strict tolerances. The
  channels carry "near this frame" signal, not "at this frame"
  signal. So we expect miss / hit to lift but exact / fhit to be
  near-flat.
- **STOP F1 is the second-most-likely-to-move metric** because
  channels firing densely vs sparsely encodes "many vs few audio
  onsets ahead" — directly informative for the STOP decision.
- **Ratio-banding ridges will persist.** Channels encode "onset
  presence", not "tempo octave." The octave-confusion failure
  mode that motivated #010+ is structurally orthogonal to channel
  inputs.
- **Slight risk of over-reliance.** If the channels are
  consistently strong on training data, the conv stem may
  over-rely on them and fail to learn robust direct-mel features.
  Watch for: large train_noaug→val gap (channels' precision is
  per-chart-normalized, so val distribution should match), or
  metrics regressing on charts with quiet music.

## Success criteria

- **Must have:** val/single/onset/miss ≤ 0.235 by eval 18 (i.e. at
  least 0.6 pp better than #007's 0.2406). Below that we can't
  distinguish the channel effect from training-noise.
- **Must have:** training stable, no NaN, no divergence.
- **Must have:** train_noaug gap not materially worse than #007's
  (-2.55 pp at E10) — channels shouldn't *increase* the
  generalization gap.
- **Nice-to-have:** val miss ≤ 0.225 at any eval (1.6 pp
  improvement).
- **Nice-to-have:** stop_f1 ≥ 0.60 (+1.7 pp over #007).
- **Fails if:** val miss > 0.245 at every eval — channels are
  net-noise rather than net-signal.
- **Fails if:** train_noaug → val gap > 4 pp at any post-warmup
  eval — would indicate channel over-reliance / memorization.

## Changes from baseline

Baseline: [#007](../007-time-stretch/).

- `config/model.json` — replaced ``EventEmbeddingConfig`` with
  ``OnsetAugmentedConfig`` (drop-in subclass). ``n_mels`` raised
  from 80 to 84; ``n_onset_channels`` set to 4 for explicit
  semantics. Conv stem first layer auto-expands from
  ``Conv1d(80, 192)`` to ``Conv1d(84, 192)``; everything else
  identical.

- `config/data.json` — unchanged (the data sampler is unaware of
  the feature row count; it just mmaps whatever the manifest
  records).

- `config/loss.json`, `config/adapter.json`, `config/trainer.json`,
  `config/infer.json` — all unchanged from #007.

- **Dataset rebuild required.** Existing ``taiko2_v1`` has 80-row
  log-mel features cached. We need 84-row features for #012. The
  build is via the canonical ``prepare_dataset.py`` pipeline using
  the new ``mel_onset`` audio-sampler alias, producing a sister
  dataset ``taiko2_v1_onset`` with augmented features. No source
  data changes; only the audio feature pipeline.

- New audio sampler:
  - ``samplers/mel_onset.py:MelOnsetSampler`` (subclass of
    ``MelSampler``). Identical mel pipeline + appended sub-band
    spectral flux rows.
  - Config alias ``mel_onset`` registered in
    ``cli/prepare_dataset.py:AUDIO_SAMPLERS``.
  - Default config in ``configs/mel_onset_default.json``.

- New model class:
  - ``models/onset_augmented.py:OnsetAugmentedDetector`` (subclass
    of ``EventEmbeddingDetector``, marker-only). Validates the
    config and inherits all behavior.
  - ``OnsetAugmentedConfig`` adds ``n_onset_channels`` field; the
    parent's ``n_mels`` is the **total** input row count.

- All 14 training augmentations and CursorShift settings identical
  to #007. TimeStretch (p=0.3, max_scale=1.4); EventJitter,
  EventDropout, EventInsertion, etc. as exp 45 set.

## Run config

- Dataset: `taiko2_v1_onset`. Built fresh from the same .osz packs
  used for `taiko2_v1`, via ``prepare_dataset.py
  --audio-sampler mel_onset --audio-config
  configs/mel_onset_default.json``.
- Run name: `exp_012_onset_channels`.
- Config snapshots: [`config/`](./config/).

### Build the augmented dataset

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.prepare_dataset \
    --name taiko2_v1_onset \
    --charts-dir <path-to-osz-pack-root> \
    --audio-sampler mel_onset \
    --audio-config osu/taiko2/configs/mel_onset_default.json
```

Re-decodes every audio file and computes 84-row features. Slow
(hours on full set) but the canonical pipeline.

### Train

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.train \
    --run-name exp_012_onset_channels \
    --config-dir osu/taiko2/experiments/012-onset-channels/config \
    --dataset taiko2_v1_onset --device cuda \
    --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
    --benchmarks all --benchmark-fraction 0.05 \
    --train-noaug-fraction 0.05 \
    --infer-corpus-spec osu/taiko2/experiments/012-onset-channels/config/infer.json \
    --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
```

Note: no `--cursor-shift-prob` (that flag was added in #010 for the
ratio decomposition; #007 didn't have it and #012 is direct-bin
again).

---
<!-- Post-run below -->
---

## Results summary

_(To fill post-run.)_

## Visualizations

_(Post-run.)_

## Vs prediction

_(Post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
