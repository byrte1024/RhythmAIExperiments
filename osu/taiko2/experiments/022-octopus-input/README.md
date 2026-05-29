# Experiment 022 — Octopus gradient onset representation (drop-in)

## Status

`Planned`

## Context

Every experiment in the 017 series uses a log-mel spectrogram as the
model's audio input. The mel spectrogram encodes *energy* across
frequency bands and time — it answers "how loud is each frequency
right now?" The model must then learn to extract onset information
from this energy representation: detecting energy transients, learning
cross-frequency coincidence patterns, and distinguishing onset-like
energy changes from sustained energy, vibrato, and tremolo.

[#020](../020-activation-maximization/) showed the model succeeds at
this — saliency at bands 70-79 is 2-4x bands 0-9
[020-activation-maximization/custom/saliency], meaning the model
learned to attend to high-frequency transient attack. But this onset
detection is learned from scratch using model capacity that could
instead be spent on higher-level chart decisions (which onsets to
select, how to space them, silence structure).

[#019](../019-coincidence-input/) attempted to provide pre-computed
onset features alongside mel via a 13-row coincidence summary. The
model ignored them entirely (no_coincidence benchmark +0.2%) because
the coincidence features were *derived from mel* and therefore
informationally redundant
[019-coincidence-input, metrics.jsonl, bench/no_coincidence].

This experiment takes a fundamentally different approach: **replace
mel entirely** with a biologically-inspired onset representation that
is computed from a different signal processing pipeline (gammatone
filterbank, not FFT) and encodes a different quantity (cross-frequency
synchrony, not energy). The representation cannot be derived from mel
because it depends on per-channel phase timing information that the
mel power spectrum discards.

### The octopus cell representation

The representation is modeled after **octopus cells** in the mammalian
cochlear nucleus (Golding, Robertson & Bhatt, 1995). These are
dedicated onset detection neurons with three properties:

1. **Ultra-low input resistance (~7 MOhm)** and fast membrane time
   constant (~200 us). Only strong, synchronous input produces a
   response. Weak or asynchronous input doesn't sum (Golding et al.
   1995).

2. **Broad dendritic span** across ~1/3 of the tonotopic (frequency)
   array, receiving 60-200+ auditory nerve fiber inputs. The cell
   "listens" across a wide frequency range simultaneously (McGinley &
   Bhatt, 2024).

3. **One spike per onset** with submillisecond temporal precision
   (jitter 20-40 us). During sustained sounds, auditory nerve fibers
   fire asynchronously across frequency, so the octopus cell stays
   silent. At a transient onset, fibers fire together within ~1 ms —
   the octopus cell detects this coincidence and fires once (Golding
   et al. 2022).

The key difference from mel: the octopus cell doesn't ask "is there
energy?" It asks **"did many frequency channels activate
simultaneously?"** — a fundamentally different and more onset-specific
question. Sustained sounds produce no response. Vibrato and tremolo
produce no response (frequency sweeps activate channels sequentially,
not simultaneously). Only transient onsets with broadband energy
produce a spike.

### The gradient implementation

The algorithm (implemented in `domain/octopus.py`, ported from
`twoof/octopus_repr/`):

```
Raw audio (22kHz)
  -> Gammatone filterbank (128 ERB-spaced channels, 50-8000 Hz)
     Hybrid IIR (>220 Hz) / FIR (<220 Hz), 8-thread parallel.
  -> Per-channel envelope (rectify + max-pool to 1ms frames)
  -> Log-domain onset function: max(0, log(E[t]) - log(E[t-3]))
     Log domain models the ear's compressive nonlinearity.
  -> Group delay compensation (align broadband transients)
     Shifts each channel forward by its analytical group delay
     (up to 19ms at 50 Hz). Without this, low-freq channels lag
     high-freq by up to 18ms and the coincidence window fails.
  -> Cross-channel synchrony detection
     Per cell: find onset peaks -> count channels with peaks in
     +/-1.5ms window -> nonlinear response (count/total)^1.5
  -> Gradient: slide cell window by 1 channel -> (97, T) at 1ms
  -> Max-pool to 5ms frames -> (97, T) at 200 fps
```

The gradient produces 97 cells from 128 filters with
`gradient_cell_width_frac=0.25` and `gradient_step=1`. Each cell
spans 32 channels (~1/4 of the tonotopic array) and slides by 1
channel, producing a dense frequency-localized onset map normalized
to [0, 1] per cell.

### Why this could work where coincidence failed

The coincidence map (#019) was derived from mel via spectral flux and
was therefore informationally redundant — the model could extract the
same information directly from mel. The octopus gradient is computed
from a **different signal processing pipeline**:

- **Gammatone filterbank vs FFT**: the gammatone preserves per-channel
  temporal envelope information that the FFT power spectrum discards.
  The onset function operates on individual channel envelopes, not on
  the power spectrum.
- **Cross-channel synchrony vs spectral flux**: the octopus
  representation explicitly computes whether multiple frequency
  channels activated *simultaneously* within a 1.5 ms window. Mel
  has no mechanism for this — it measures per-band energy
  independently.
- **Group delay compensation**: the gammatone filterbank's
  frequency-dependent group delay (50 Hz lags 8 kHz by 18 ms) is
  compensated analytically before coincidence detection. This is
  critical for low-frequency onset detection — without it, bass
  onsets cannot trigger the coincidence gate because their energy
  arrives too late relative to treble (Golding et al. 1995 measured
  ~1 ms effective integration window).
- **Vibrato/tremolo immunity**: vibrato sweeps frequency channels
  sequentially (not simultaneously), so the coincidence gate
  naturally suppresses it. Mel spectrograms show vibrato as
  energy modulation that models must learn to ignore.

### Dataset strategy

The dataset `taiko2_v1_mel_octopus` stores both mel and octopus as a
single `(177, T)` feature array per chart (80 mel + 97 octopus).
The `feature_rows` config on the adapter and `output_rows` on the
inference sampler select which rows the model sees:

| Experiment | feature_rows | n_mels | Input |
|---|---|---:|---|
| **022 (this)** | [80, 177] | 97 | Octopus only |
| Future: mel only | [0, 80] | 80 | Mel only (control) |
| Future: dual channel | null | 177 | Mel + octopus |

One dataset build supports all three experiments.

## Citations

- Direct baseline:
  - [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).
    Best sweep tau=0.4 (eval_248088): `f1` 0.771, `precision` 0.778,
    `recall` 0.757, `density_ratio` 0.964, `dc_human` 92.4,
    `gap_hist_tvd` 0.331, `silence_overlap_f1` 0.527
    [017f threshold_sweep.json].
- Failed alternative input:
  - [#019 -- coincidence input](../019-coincidence-input/).
    Model ignored coincidence (no_coincidence +0.2%). Redundant with
    mel [019 metrics.jsonl, bench/no_coincidence].
- Conv stem capacity:
  - [#021 -- wider stem](../021-wider-stem/). stem_width 192->256
    did not reproduce 019's gains. Conv width is not the bottleneck
    [021 threshold_sweep.json].
- Interpretability:
  - [#020 -- activation maximization](../020-activation-maximization/).
    Model uses high-frequency transient attack (saliency 2-4x at
    bands 70-79 vs 0-9) [020 custom/saliency].
  - [#020b -- legible dreams](../020b-legible-dreams/). Model dreams
    low-frequency-dominant onsets (ratio 1.3-1.5x) [020b custom].
- Biology:
  - Golding NL, Robertson D, Bhatt DK. "Detection of synchrony in
    the activity of auditory nerve fibers by octopus cells of the
    mammalian cochlear nucleus." PNAS, 1995. Ultra-low input
    resistance, ~1 ms integration, one spike per onset.
  - McGinley MJ, Bhatt DK. "An anatomical and physiological basis
    for flexible coincidence detection in the auditory system."
    bioRxiv/eLife, 2024. Glycinergic inhibition extends coincidence
    window, E/I ratio 5:2.
  - Golding NL et al. "Mammalian octopus cells are direction
    selective to frequency sweeps by excitatory synaptic sequence
    detection." PNAS, 2022. Kv1 channel shunting, submillisecond
    jitter.
- Signal processing:
  - Glasberg BR, Moore BCJ. "Derivation of auditory filter shapes
    from notched-noise data." Hearing Research, 1990. ERB formula,
    gammatone parameterization.
  - Bello JP et al. "A Tutorial on Onset Detection in Music
    Signals." IEEE TSAP, 2005. Onset detection survey.
  - Boeck S, Widmer G. "Maximum Filter Vibrato Suppression for
    Onset Detection." DAFx, 2013. SuperFlux benchmark.
- Implementation: `domain/octopus.py`, `samplers/mel_octopus.py`.

---
<!--
PRE-RUN. Do not edit after the run.
-->
---------------------------------------------------------------------

## Hypothesis

### Claim

Replacing the 80-band log-mel spectrogram with the 97-cell octopus
gradient as the model's sole audio input will produce onset detection
quality comparable to mel (F1 within 0.05 of 017f at matched density)
while improving distributional metrics — specifically
`silence_overlap_f1` and `gap_hist_tvd` — because the octopus
representation pre-computes cross-frequency onset synchrony that the
mel model must learn from scratch, and it naturally suppresses
sustained energy and vibrato that cause mel-based false positives.

### Mechanism

The octopus gradient encodes a fundamentally different quantity than
mel. Where mel says "there is 30 dB of energy in the 1-2 kHz band at
time t," the octopus gradient says "at time t, 75% of frequency
channels in the 1-2 kHz region experienced a simultaneous onset
within 1.5 ms." This is a pre-computed onset feature, not a raw
energy measurement.

The mel model's hallucination problem (silence_overlap_f1 0.527 at
tau=0.4 [017f threshold_sweep.json]) stems from responding to energy
transients that are acoustically real but not chart-worthy — the
model detects every audio onset and must learn which ones to suppress.
The octopus representation has two properties that should help:

1. **Sustained energy suppression**: the log-domain onset function
   `max(0, log(E[t]) - log(E[t-k]))` produces zero output during
   sustained sounds. Only energy *increases* register. This
   eliminates a class of false positives that mel produces.

2. **Vibrato/tremolo immunity**: vibrato sweeps frequency channels
   sequentially (each channel peaks at a different time), so the
   coincidence gate (which requires simultaneous activation within
   1.5 ms) naturally suppresses it. Mel shows vibrato as energy
   modulation that the model must learn to ignore.

However, the octopus representation discards harmonic structure,
timbral detail, and absolute energy levels. The model loses the
ability to distinguish different instrument timbres and may struggle
with soft onsets that don't trigger enough channels to pass the
coincidence gate. This is the core risk.

### Predicted numbers

Reference: [#017f](../017f-framewise-bce-metrics-rerun/) tau=0.4
sweep [017f threshold_sweep.json, eval_248088].

| Metric | #017f (tau=0.4) | Predicted (#022) | Notes |
|---|---:|---:|---|
| AR `f1` (25ms) | 0.771 | **0.72-0.77** | May be slightly lower; onset timing is pre-computed but onset *selection* is harder without timbral context |
| AR `precision` | 0.778 | **>= 0.75** | Octopus suppresses non-onset energy -> fewer FPs |
| AR `recall` | 0.757 | **0.65-0.75** | May miss soft onsets that don't trigger coincidence gate |
| `density_ratio` | 0.964 | **0.85-1.05** | Unknown operating point |
| `dc_human` | 92.4 | **>= 91** | Pattern quality should hold |
| `gap_hist_tvd` | 0.331 | **< 0.33** | Pre-computed onset structure should help rhythmic fidelity |
| `silence_overlap_f1` | 0.527 | **>= 0.55** | Sustained energy suppression reduces silence FPs |
| `density_corr` | 0.546 | **>= 0.54** | Should hold |
| frame F1 | 0.822 | **>= 0.78** | May be lower; 97 octopus cells carry less total information than 80 mel bands |
| fps50 F1 | 0.741 | **>= 0.70** | Watched metric |

## Success criteria

- **Must:** frame F1 >= 0.75. The octopus input must produce a
  trainable model that achieves reasonable onset detection.
- **Must:** AR F1 >= 0.65 at some threshold. The model must
  produce usable charts from octopus input alone.
- **Confirms hypothesis if:** `silence_overlap_f1` > 0.55 at
  matched density (above 017f's 0.527), demonstrating that the
  onset-focused representation reduces silence false positives.
- **Fails if:** frame F1 < 0.50 — the representation doesn't carry
  enough information for onset detection.
- **Fails if:** AR F1 < 0.50 at every threshold — the model cannot
  produce coherent charts.
- **Nice-to-have:** AR F1 >= 0.75 at some threshold, matching mel.
- **Nice-to-have:** `gap_hist_tvd` < 0.30, beating both 017f (0.331)
  and 019 (0.294).

## Changes from baseline

Baseline: [#017f -- framewise BCE metrics rerun](../017f-framewise-bce-metrics-rerun/).

Four changes:

- **New dataset `taiko2_v1_mel_octopus`** — built with
  `MelOctopusSampler` (`--audio-sampler mel_octopus`). Features on
  disk are `(177, T)` float16: 80 mel bands + 97 octopus gradient
  cells.
- `config/model.json` — `n_mels: 80 -> 97`. Matches the 97 octopus
  cells selected by `feature_rows`.
- `config/adapter.json` — `feature_rows: [80, 177]`. Selects octopus
  rows from the `(177, T)` features at training time.
- `config/infer.json` — `audio_sampler` changed to
  `MelOctopusSampler` with `output_rows: [80, 177]`. Produces
  `(97, T)` octopus-only features at inference time.

All other configs (loss.json, trainer.json, data.json) identical to
#017f. `decode_threshold=0.4` (017f sweep optimal).

New code:
- `domain/octopus.py` — pure compute functions: gammatone filterbank
  (hybrid IIR/FIR, 8-thread parallel), group delay compensation,
  octopus gradient with cumulative-sum O(1) cell computation. Ported
  from `twoof/octopus_repr/`.
- `samplers/mel_octopus.py` — `MelOctopusSampler(AudioSampler)`.
  Produces `(177, T)` by concatenating mel + octopus gradient.
  `output_rows` config selects which rows to output (for inference).
- `training/framewise_adapter.py` — added `feature_rows` config
  field for row selection at training time.
- `training/augmentations.py` — FreqRoll no longer hardcodes 80 mel
  bands; rolls all feature rows.

Dataset preparation:

```bash
osu/taiko2/.venv/bin/python -m osu.taiko2.cli.prepare_dataset \
    --name taiko2_v1_mel_octopus \
    --charts-dir /home/drore/charts/repos/BeatDetector/osu/taiko/charts/ \
    --audio-sampler mel_octopus

osu/taiko2/.venv/bin/python -m osu.taiko2.cli.fetch_stars --dataset taiko2_v1_mel_octopus
```

## Run config

- Run name: `exp_022_octopus_input`
- Config snapshots: [`config/`](./config/)
- Dataset: `taiko2_v1_mel_octopus` (177-channel: 80 mel + 97 octopus;
  model sees rows 80-176 only = 97 octopus cells)
- Command:
  ```bash
  set -e CUDA_VISIBLE_DEVICES && ulimit -n 65536 && \
  osu/taiko2/.venv/bin/python -m osu.taiko2.cli.train \
      --run-name exp_022_octopus_input \
      --config-dir osu/taiko2/experiments/022-octopus-input/config \
      --dataset taiko2_v1_mel_octopus --device cuda \
      --time-stretch-prob 0.3 --time-stretch-max-scale 1.4 \
      --train-noaug-fraction 0.05 \
      --benchmarks all \
      --compile \
      --infer-corpus-spec osu/taiko2/experiments/022-octopus-input/config/infer.json \
      --infer-corpus-config osu/taiko2/configs/infer_corpus_per_eval.json
  ```

## Augmentation notes

Augmentations apply to the full `(177, T)` feature array before the
adapter slices to `[80, 177]`. The augmentations that affect the
octopus rows (80-176):

- **Time stretch** (30%): resamples along time axis. Valid for onset
  maps — stretching time stretches onset spacing.
- **FreqRoll** (15%): rolls all rows including octopus cells. Shifts
  which frequency region each cell represents. Biologically
  reasonable — equivalent to the model handling onset detection at
  slightly different frequency positions.
- **SpecAug freq mask** (20%): zeros some octopus cells. Simulates
  missing onset information in a frequency region.
- **SpecAug time mask** (20%): zeros a time window. Same as for mel.
- **MelGainJitter** (30%): adds dB offset to all rows including
  octopus. On [0, 1] octopus values this shifts the range, which
  may need tuning. For the first run, keep default parameters and
  monitor if octopus values clip.
- **MelGaussianNoise** (15%): adds Gaussian noise. On [0, 1] octopus
  values, the default std 0.1-0.3 is relatively large. Monitor and
  tune if needed.

## Future directions

The `taiko2_v1_mel_octopus` dataset supports three follow-up
experiments without rebuilding:

- **Mel-only control**: `feature_rows: [0, 80]`, `n_mels: 80`.
  Verify mel performance on this dataset matches 017f (control for
  dataset differences).
- **Dual channel**: `feature_rows: null`, `n_mels: 177`. Model sees
  both mel and octopus simultaneously. The model can learn to use
  whichever representation is better for each onset type.
- **Separate stems**: two conv stems (one for mel rows 0-79, one for
  octopus rows 80-176) merged before the transformer. Most flexible
  but requires model code changes.

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
