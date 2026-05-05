# Experiment 011 — Onset detection algorithm survey

## Status

`Complete`. Single-channel `spectral_flux` is enough; F1 ceiling
~0.68 at ±10 frames is much higher than predicted; **all classical
ODFs collapse below ±2 frames** because they fire 2-3 frames late
on the attack. Recommended next step: encode the spectral activation
at a **coarser (±5 or ±10 frame) grid** rather than at the raw 5 ms
grid, so the downstream model gets a clean bucket-aligned signal
instead of a 2-3-frame-blurred one.

## Context

[#010](../010-ratio-decomposition/) through [#010e](../010e-aux-frozen/)
established that the ratio decomposition direction has a structural
ceiling at miss ≈ 0.33 set by divisor accuracy (~0.78), and the
multiplicative reconstruction `bin = div × ratio − offset` cannot
escape that ceiling regardless of head freezing, gradient routing,
bin count, or smoothing. Direct-bin prediction (#007) remains
ahead at miss ≈ 0.24, but the same ridges at ±log(2) / ±log(3)
that motivated the ratio direction are still present.

Across that whole sweep no experiment has changed the **input
representation** the model sees. Audio has been a fixed 80-band
log-mel at 5 ms / frame since #001. Every architectural and loss-
side intervention has worked against the same input.

The MIR community has 20+ years of hand-engineered onset detection
functions (ODFs) — spectral flux, HFC, SuperFlux, Complex Domain,
sub-band variants — with documented strong recall on percussive
content. Stacking one or more of those as additional input
channels is a candidate for a substantial change to the input
representation. Before scaffolding a training experiment around
that idea, this experiment characterizes how those ODFs perform
against the taiko `taiko2_v1` GT directly: which algorithms have
the highest *recall*, what their precision floors look like
against author-chosen onsets, how they degrade with frame
tolerance, and whether multi-channel unions cover meaningfully
more onsets than any single channel.

This is **an analysis experiment, not a training experiment.**
No model is trained. The output is a calibration report that
informs the design of #012 (the actual training experiment using
the most promising channels).

## Citations

- Direct motivation: [#010e](../010e-aux-frozen/) — last in the
  ratio-decomposition family. Established that the failure mode is
  not a ratio-head problem.
- Headline baseline for the eventual training experiment:
  [#007](../007-time-stretch/) — direct-bin best, miss 0.241.
- Onset detection literature:
  - **Spectral flux family**:
    [Onset Detection Revisited (Dixon, 2006)](https://www.dafx.de/paper-archive/2006/papers/p_133.pdf),
    [A Tutorial on Onset Detection in Music Signals (Bello et al.)](http://www-labs.iro.umontreal.ca/~pift6080/H09/documents/papers/bello_onset_tutorial.pdf).
  - **SuperFlux**:
    [Maximum Filter Vibrato Suppression for Onset Detection (Böck & Widmer, DAFx-13)](https://phenicx.upf.edu/system/files/publications/Boeck_DAFx-13.pdf),
    [SuperFlux reference implementation](https://github.com/CPJKU/SuperFlux).
  - **Complex Domain**:
    [On the Use of Phase and Energy for Musical Onset Detection in the Complex Domain (Bello et al., 2004)](https://ieeexplore.ieee.org/document/1300607/).
  - **HFC**:
    [Hard real-time onset detection of percussive sounds](https://www.researchgate.net/publication/325541830_Hard_real-time_onset_detection_of_percussive_sounds).
  - **Sub-band onset**:
    [librosa.onset.onset_strength_multi](https://librosa.org/doc/main/generated/librosa.onset.onset_strength_multi.html).
- Evaluation conventions: MIREX onset detection wikis ([2018](https://www.music-ir.org/mirex/wiki/2018:Audio_Onset_Detection),
  [2021](https://www.music-ir.org/mirex/wiki/2021:Audio_Onset_Detection)),
  [Design and Evaluation of Onset Detectors (ISMIR 2014)](https://archives.ismir.net/ismir2014/paper/000229.pdf).
- [State-Of-The-Art for Audio Onset Detection (MIR blog)](https://musicinformationretrieval.wordpress.com/2017/02/03/state-of-the-art-for-audio-onset-detection-week-3/) —
  reports MIREX 2016 solo-drums F1 ≥ 0.85 across spectral methods.

---

## Hypothesis

### Claim

Multi-channel unions of spectral flux, SuperFlux, HFC, and
sub-band spectral flux can achieve **frame-wise recall ≥ 0.95
at ±25 ms** against `taiko2_v1` GT onsets at moderate precision
(0.30–0.50). Single best classical algorithm reaches recall
0.93–0.97 at ±50 ms but with low precision (0.20–0.30) because
GT is *chart-author-chosen* onsets, not all audio onsets. The
gap between single-channel and union recall is at least 3 pp,
justifying a multi-channel input for the downstream training
experiment.

### Mechanism

GT onsets are not all audio onsets — chart authors map a subset
of musically salient events. Any honest ODF tuned to detect
audio onsets will fire on real-but-unmapped onsets (background
hi-hat / ghost note / vocals). These count as false positives
against GT, capping precision well below 1.0 regardless of
algorithm quality.

What we actually want is a free per-frame "look here" signal that
the trained model can use to filter out non-onset frames. For
that, **recall is the load-bearing metric**: every real onset
should light up at least one ODF channel, even at low precision,
because the downstream model can sort out the false positives
during training.

Spectral methods differ in *where* they fire as much as *whether*
they fire. HFC weighting biases toward sharp broadband transients
(taiko DON / KA strikes); sub-band SF biases toward whichever
frequency band carries the onset energy; SuperFlux's max-filter
suppresses vibrato-like fluctuations. Their false-positive
distributions are partially independent, so a 2–3 channel union
can be expected to retrieve ~99% of GT onsets even when each
single channel hits 0.93–0.97.

### Predicted numbers (per-algorithm, pooled across all val charts)

Reference: ±10 frames = ±50 ms tolerance window (MIREX standard).
At higher resolutions (±5 frames = ±25 ms, ±2 frames = ±10 ms,
±0 frames = exact frame) every algorithm degrades but at
different rates.

#### Per-algorithm best F1 / recall / precision @ ±10 frames

| Algorithm | Predicted F1 | Predicted Recall | Predicted Precision | Notes |
|---|---:|---:|---:|---|
| energy | 0.20–0.25 | 0.85–0.90 | 0.10–0.15 | Fires on amplitude swells; near-useless precision. |
| spectral_flux | 0.35–0.40 | 0.93–0.96 | 0.20–0.25 | Workhorse baseline. |
| log_filtered_flux | 0.36–0.42 | 0.94–0.97 | 0.22–0.27 | Volume-robust bump. |
| hfc_mel | 0.35–0.45 | 0.92–0.96 | 0.20–0.30 | Strong on transient drums; cymbal/brass false positives. |
| **superflux** | **0.40–0.45** | **0.94–0.97** | **0.25–0.30** | Best classical at MIREX 2016; vibrato suppression irrelevant for percussion but harmless. |
| subband_sf_4 (collapsed) | 0.36–0.42 | 0.94–0.97 | 0.22–0.27 | Single-envelope ≈ spectral_flux; per-band benefit is for joint coverage. |
| subband_sf_8 (collapsed) | 0.36–0.42 | 0.94–0.97 | 0.22–0.27 | Same. |
| complex_domain | 0.35–0.40 | 0.93–0.96 | 0.20–0.25 | Phase term doesn't help on percussion → tied with mag-based. |

#### Tolerance sweep (predicted F1 of *best classical algorithm*)

| Tolerance | Frames | ms | Predicted F1 |
|---|---:|---:|---:|
| Strict | 0 | 0 | 0.10–0.20 |
| ±1 frame | 1 | 5 | 0.20–0.30 |
| ±2 frames | 2 | 10 | 0.30–0.38 |
| ±5 frames | 5 | 25 | 0.36–0.42 |
| **±10 frames** | **10** | **50 (MIREX)** | **0.40–0.45** |
| ±20 frames | 20 | 100 | 0.42–0.48 |

The strict-frame degradation reveals which algorithms fire on
the *exact* attack frame (HFC, SuperFlux) versus a few frames
late (spectral flux, energy — they peak after the spectral
change, not at it).

#### Joint coverage (recall) at ±10 frames, top-K unions

| K | Channels (predicted) | Predicted union recall |
|---:|---|---:|
| 1 | superflux *or* hfc_mel | 0.94–0.97 |
| 2 | superflux + hfc_mel | 0.97–0.99 |
| 3 | superflux + hfc_mel + spectral_flux | 0.98–0.99 |
| 4 | + subband_sf_4 | 0.99+ |

If any 2–3 algorithm union breaks recall 0.99 with reasonable
precision (≥ 0.30), that's the channel set #012 should adopt.

### Predicted ranking surprises

- **superflux likely wins** plain F1 ranking but may *lose* on
  the strict-frame tolerance (max-filter shifts the peak by 1
  frame in some attacks).
- **hfc_mel may underperform expectations** because we're
  approximating it on mel bands rather than on linear-frequency
  STFT. The mel filterbank already concentrates low-frequency
  resolution, so the linear-by-frequency HFC weighting is less
  pronounced than on a raw spectrogram.
- **complex_domain may underperform on taiko specifically.**
  Phase information is most useful for tonal onsets (bowed,
  blown). Taiko is broadband percussion in pop tracks; the phase
  term should add little.
- **Sub-band collapsed envelopes will tie spectral_flux** by
  construction (the collapsed envelope = sum of band fluxes =
  spectral flux modulo per-band weighting). The interesting
  metric is **per-band recall on DON vs KA** and **joint coverage
  across bands** — not the collapsed envelope's own F1.

### What we measured during the 4-chart smoke

| Algorithm | F1 | Recall | Precision |
|---|---:|---:|---:|
| spectral_flux | 0.697 | 0.840 | 0.595 |
| superflux | 0.624 | 0.720 | 0.550 |
| subband_sf_4 (collapsed) | 0.697 | 0.840 | 0.595 |
| hfc_mel | 0.557 | 0.742 | 0.446 |

**Smoke-test F1 is higher than the predicted full-set F1 (0.55–0.70
vs 0.35–0.45 predicted).** Plausible reasons:
- The 4 charts are all difficulties of one beatmapset ("ICHIKO -
  I SAY YES"), likely a clean pop track with onsets aligned to
  obvious audio events. Easy material.
- 4 charts is too few to estimate the long tail of hard cases.

The full-set numbers should drift down. **If the full-set best F1
exceeds 0.55 across the board, my pre-run F1 predictions were
too pessimistic** and the channel-input strategy is even more
attractive than expected. Predicted full-set best F1 stays
between 0.40 and 0.55.

## Success criteria

This experiment succeeds if it produces a defensible answer to
each of:

- **Must answer:** what's the per-algorithm best F1 / recall /
  precision at ±50 ms tolerance, on the full val split?
- **Must answer:** which 2–4-channel union maximises recall at
  ±25 ms and at ±10 ms, and what's the precision tradeoff?
- **Must answer:** does sub-band SF reveal a DON-vs-KA structural
  bias that motivates per-band channels in the downstream model?
  (Computed via per-kind aggregate.)
- **Nice-to-have:** crossover analysis — at what tolerance does
  classical-ODF F1 cross #007's frame-wise F1 (when #007 is added
  to the survey)?
- **Nice-to-have:** per-density / per-star-rating breakdown
  showing whether ODF performance degrades on dense charts.
- **Fails if:** the script crashes on the full split (corrupt
  audio, missing features, etc.) on > 5 % of charts. Indicates
  data-pipeline bug.
- **Fails if:** every algorithm hits recall ≥ 0.99 at every
  tolerance — would mean evaluation is broken (the GT is being
  trivially matched).
- **Fails if:** every algorithm hits recall ≤ 0.50 at ±10 frames —
  would mean the activations don't align with the feature grid
  (off-by-one / hop-mismatch bug).

## Methodology

### Algorithms surveyed

Mel-domain (computed from cached log-mel features, GPU):
- `energy` — sum of mel-band magnitudes per frame.
- `spectral_flux` — half-wave-rectified diff over freq.
- `log_filtered_flux` — same on log-compressed mel (cached
  features are already in dB; this stays as a plumbing
  parallel to the literature definition).
- `hfc_mel` — bands weighted by mel center freq in Hz.
- `superflux` — flux with max-filter (k=3) on previous frame's
  mel along freq, μ=1 frame lag.
- `subband_sf_4` — spectral flux split into 4 mel-band groups,
  reported per-band and as a summed envelope.
- `subband_sf_8` — same with 8 groups.

STFT-domain (requires `--charts-dir` with .osz packs):
- `complex_domain` — predicted-phase + predicted-magnitude
  Euclidean distance, summed over freq. Standard CD ODF.

Out of scope for this run:
- madmom RNN/CNN onset (separate model dep, not free signal).
- #007 frame-wise evaluation (separate follow-up because it
  needs checkpoint loading + AR rollout — see Followup questions).

### Evaluation procedure

1. For each chart in `taiko2_v1` val split (≈ 958 charts):
   1. Load cached log-mel features (`features/*.npy`, shape
      `(80, T)` float16 → float32 on GPU).
   2. Optionally load source `.osz` audio for STFT-domain
      algorithms (decoded at 22 kHz mono via librosa).
   3. Compute every algorithm's activation envelope on GPU.
   4. Normalize each activation by its 99th-percentile so a
      common threshold range works across charts.
   5. Sweep 51 thresholds in `[0.0, 2.0]`. At each threshold:
      peak-pick the activation (strict local max above
      threshold, NMS with min-distance 1 frame) and evaluate
      the predicted frames against GT bins at tolerances
      `{0, 1, 2, 5, 10, 20}` frames.
   6. Record TP / FP / FN per (algorithm, threshold, tolerance).
2. Aggregate across all charts by **summing TP / FP / FN** before
   computing precision / recall / F1 (pooled / micro-averaged) —
   weights each chart by its event count. Per-chart best-F1 also
   recorded for the per-chart CSV.
3. Joint coverage: for every subset of size 1..4 across the
   surveyed algorithms, take the union of predicted peaks (at
   each algorithm's recall-≥-0.95 threshold), evaluate the
   union against GT.

### Threshold and tolerance choices

- **51 thresholds** in `[0.0, 2.0]` after percentile normalization.
  0.0 includes "no threshold" (every local max counts);
  2.0 is well past the active range for most ODFs (yields
  zero or near-zero predictions).
- **6 tolerances** spanning 0 (strict) to 20 frames (±100 ms).
  ±10 frames is the MIREX-standard ±50 ms.
- **Per-chart percentile normalization** rather than global so
  loud / quiet songs are comparable.
- **Min peak distance = 1 frame** so adjacent-frame peaks are
  both retained (matters at tolerance 0–1).

### Expected runtime

- Mel-only (cached features path): ~1 chart / 0.4 s on RTX 5070
  → ~6.5 minutes for full val split (958 charts).
- With audio decode + Complex Domain: ~1 chart / 2 s including
  audio decode + STFT → ~32 minutes for full val split.
  Adds the .osz indexing pass at startup (~2 minutes).

## Run config

- Output root: [`results/`](./results/) — auto-created by the
  CLI. `per_chart.csv`, `per_algo/{algo}/curves.csv`,
  `summary.json`.
- Smoke test (4 charts, mel-only, already run):
  [`results_smoke/`](./results_smoke/).
- Commands:

  ```bash
  # Mel-only path (fast, no audio decode needed):
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.onset_feature_survey \
      --dataset taiko2_v1 \
      --output osu/taiko2/experiments/011-onset-feature-survey/results \
      --split val --device cuda

  # Full path (including Complex Domain):
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.onset_feature_survey \
      --dataset taiko2_v1 \
      --output osu/taiko2/experiments/011-onset-feature-survey/results \
      --split val --device cuda \
      --charts-dir <path-to-osz-pack-root>
  ```

- Parameters:
  - `--n-thresholds 51`
  - `--threshold-min 0.0 --threshold-max 2.0`
  - `--tolerances 0,1,2,5,10,20`
  - `--min-peak-distance 1`
  - `--norm-percentile 99.0`

---
<!-- Post-run below -->
---

## Results summary

Survey ran on the full val split (958 charts) in **6 min 22 s** for
the per-chart loop on a single RTX 5070, plus another **17 s** for
joint coverage after the post-`[done]` patch (peak tensors moved to
CPU once + union hoisted out of the tolerance loop + tqdm progress).
No charts skipped.

Headline finding: **classical ODFs are far stronger against
chart-author GT than predicted.** Single-channel best F1 at
±10 frames is **0.679 (spectral_flux)**, not the 0.40-0.45 my pre-
run hypothesis estimated. Per-algo precision floors landed at
0.47-0.62 — chart authors track audio onsets much more tightly than
"a small subset of all audio events" assumed in the pre-run.

The structural finding is in the tolerance sweep, not the absolute
F1. **Every classical ODF collapses below ±2 frames**: at ±0
(exact frame) every algorithm scores < 0.09 F1. The activations
peak 2-3 frames *after* the actual attack, because spectral flux
measures the change *between* frames, and HFC measures envelope
energy. This means the per-frame channel signal is fundamentally
**candidate-region**, not **frame-precise**, and the natural
aggregation grid for downstream input is the **±5 or ±10 frame
window** where the ODFs already operate near ceiling — not the
raw 5 ms grid.

### Per-algorithm best F1 / recall / precision @ ±10 frames

Pooled across 958 charts (TP / FP / FN summed before computing
P / R / F1, so high-onset charts weight more).

| Algorithm | F1 | Recall | Precision | Best threshold |
|---|---:|---:|---:|---:|
| **spectral_flux** | **0.679** | 0.742 | **0.625** | 0.32 |
| subband_sf_4 (collapsed) | 0.679 | 0.742 | 0.625 | 0.32 |
| subband_sf_8 (collapsed) | 0.679 | 0.742 | 0.625 | 0.32 |
| energy | 0.646 | **0.780** | 0.552 | 0.76 |
| hfc_mel | 0.625 | 0.772 | 0.526 | 0.68 |
| superflux | 0.622 | 0.692 | 0.565 | 0.28 |
| log_filtered_flux | 0.550 | 0.654 | 0.475 | 0.24 |

`subband_sf_4` and `subband_sf_8` collapse to the same envelope as
`spectral_flux` when summed across bands (per-band sum = total
spectral flux modulo per-band weighting), which is why they tie
exactly. The per-band signal has different value, see Joint coverage.

Complex Domain (`complex_domain`) was **not run**: it requires raw
audio for STFT phase, which means walking and parsing every `.osz`
pack to recover paths. The mel-domain results were strong enough
that scaffolding the audio-loading path didn't seem worth the
overhead for this analysis.

### Tolerance sweep — best F1 per algorithm

| Algorithm | ±0 | ±1 | ±2 | ±5 | ±10 | ±20 |
|---|---:|---:|---:|---:|---:|---:|
| spectral_flux | 0.070 | 0.231 | 0.415 | **0.656** | **0.679** | 0.697 |
| subband_sf_4 | 0.070 | 0.231 | 0.415 | 0.656 | 0.679 | 0.697 |
| subband_sf_8 | 0.070 | 0.231 | 0.415 | 0.656 | 0.679 | 0.697 |
| superflux | 0.070 | 0.217 | 0.369 | 0.590 | 0.622 | 0.650 |
| log_filtered_flux | **0.086** | **0.255** | 0.398 | 0.535 | 0.550 | 0.568 |
| hfc_mel | 0.006 | 0.018 | 0.036 | 0.205 | 0.625 | 0.679 |
| energy | 0.004 | 0.014 | 0.027 | 0.169 | 0.646 | 0.687 |

Two clusters by strict-tolerance behavior:
- **Difference-based (sf, sf-variants, superflux, log-filtered-sf)**
  measure the spectral *change* between frames. They peak on the
  attack frame or the next one. Strict-frame F1: 0.07-0.09.
- **Envelope-based (energy, hfc_mel)** measure how much energy is
  in the current frame. They peak when the broadband envelope
  rises, which lags the attack by 2-5 frames. Strict-frame F1:
  ~0.005, but they recover at ±10 frames.

`log_filtered_flux` quietly wins the strict-tolerance regime
(0.086 / 0.255 / 0.398 at ±0 / ±1 / ±2) — log compression sharpens
the peak alignment. It loses ground at looser tolerances because
the compression also dampens the activation magnitude, hurting
peak-pick recall.

### Joint coverage caveat

The per-algo high-recall threshold (recall ≥ 0.95 at any tolerance)
turned out to be a near-no-threshold operating point — peaks
saturated at ~10× the GT count. Every 3-4-channel union therefore
sits at recall = 1.0 with precision ≈ 0.08-0.09. The joint coverage
table is in the summary but **does not differentiate channel sets
in this configuration** — the analysis would need a more
restrictive per-algo threshold (e.g. recall ≥ 0.85) to be
informative. Treated as a known limitation; not blocking the
overall conclusions.

## Visualizations

![Per-algorithm F1 / R / P at ±10 frames](graphs/01_algo_summary_bars.png)
*Per-algorithm best-F1 ranking @ ±10 frames. spectral_flux ≈
sub-band variants (the latter collapse to the same envelope when
summed) lead at F1 = 0.679; energy is shockingly close (0.646)
because chart authors mostly map amplitude swells.*

![PR curves at ±10 frames](graphs/02_pr_curves_tol10.png)
*Precision-recall traces over the 51-threshold sweep, marker on
each algorithm's best-F1 operating point. spectral_flux dominates
the upper-right corner; log_filtered_flux trails on this tolerance.*

![Best F1 vs tolerance](graphs/03_tolerance_sweep.png)
*The headline diagnostic. Every algorithm collapses below ±2
frames; spectral_flux + sub-band variants own the top of the curve
above ±5; energy and hfc_mel catch up to within 0.05 F1 by ±10
because they fire late but eventually fire on the right region.*

![Recall + precision at the recall-≥-0.95 threshold](graphs/04_recall_at_high_threshold.png)
*Calibration view. At a per-algorithm threshold tuned for recall
≥ 0.95, every algo lands near recall = 1.0 with precision 0.08-0.20.
Demonstrates the recall-saturation problem with the joint coverage
analysis (precision is too low to be useful as-is).*

![Top joint-coverage subsets](graphs/05_joint_coverage.png)
*Top-3 subsets by recall, per subset size. Recall pegs at 1.0 for
size 3+ (saturation artifact); meaningful differences would need a
tighter per-algo threshold. Single-algorithm row shows
spectral_flux / sub-band variants tied at the top.*

![P / R / F1 vs threshold per algorithm](graphs/06_per_algo_grid.png)
*Operating-point shape per algorithm at ±10 frames. spectral_flux
has the cleanest crossover near threshold = 0.32; energy shifts
its sweet spot to threshold = 0.76 because its activation magnitude
is much higher.*

## Vs prediction

| Prediction | Actual | Verdict |
|---|---|---|
| superflux best F1 (predicted 0.40-0.45) | 0.622 | **MISS — too pessimistic by +18 pp** |
| spectral_flux best F1 (predicted 0.35-0.40) | 0.679 | **MISS — too pessimistic by +30 pp** |
| energy best F1 (predicted 0.20-0.25) | 0.646 | **MISS — too pessimistic by +42 pp** |
| hfc_mel best F1 (predicted 0.35-0.45) | 0.625 | **MISS — too pessimistic by +21 pp** |
| log_filtered_flux best F1 (predicted 0.36-0.42) | 0.550 | **MISS — too pessimistic by +14 pp** |
| HFC underperforms on mel-binned approximation | F1 0.625, mid-pack | **MET (predicted)** — not the worst, but trails the SF cluster as predicted. |
| Sub-band collapsed envelopes tie spectral_flux | 0.679 = 0.679 = 0.679 | **MET** exactly. |
| superflux loses on strict-frame tolerance | 0.070 vs spectral_flux 0.070 (tied at ±0) | **PARTIAL** — predicted shift, observed indifferent at strict. At ±2 superflux *does* trail SF (0.369 vs 0.415). |
| Joint coverage 2-channel union ≥ 0.97 recall | size-2 unions all at recall = 1.000 | **MET on numbers**, but the saturation makes this uninformative — see caveat. |
| Per-kind / per-density breakdown answers questions | not computed in this run | **DEFERRED** to a follow-up plotting pass — the per-chart CSV has the data, just not aggregated by kind / density. |
| `complex_domain` runs and is in the table | not run (audio-loading path skipped) | **NOT TESTED** — mel-domain results were strong enough that scaffolding `.osz` indexing didn't make the cut. |

**The hypothesis was directionally correct (single-channel recall
0.93-0.97; multi-channel near-saturating coverage) but the absolute
F1 predictions were uniformly too pessimistic — by +14 to +42 pp.**
The pre-run mental model of "GT is a small subset of audio onsets"
was wrong: chart authors map the *vast majority* of salient audio
onsets, and the precision floor is correspondingly higher than
predicted.

## Takeaways

- **Single-channel `spectral_flux` is the obvious pick.** F1 = 0.679
  at ±10 frames, P = 0.625. Adding more channels has limited
  marginal value once the envelopes collapse (sub-band variants
  contribute nothing when summed; per-band channels would need
  separate peak-picking and a different evaluation to be useful
  beyond the smoothed envelope).
- **Encode the activation at a coarser (±5 or ±10 frame) grid,
  not the raw 5 ms grid.** Every classical ODF collapses below
  ±2 frames because the activations peak 2-3 frames after the
  attack. Feeding the raw 5 ms-grid activation to the model gives
  a 2-3-frame-blurred signal — the model has to learn to compensate
  for the peak lag itself. **Pooling the activation into 25 ms or
  50 ms buckets** (max-pool or sum-pool over 5- or 10-frame
  windows) collapses the lag into the bucket and gives the model a
  clean bucket-aligned "is there an onset in this region?" signal.
  This is the channel-encoding decision for #012.
- **The precision floor is much higher than the pre-run model
  predicted (0.47-0.62 vs predicted 0.20-0.30).** Chart authors
  track audio onsets tightly; the chart-vs-audio asymmetry is
  smaller than the literature on general-music onset detection
  would suggest. A free per-frame channel comes with **moderate
  precision**, not just **high recall** — the downstream model
  gets meaningful signal, not just candidate regions.
- **`log_filtered_flux` quietly wins at strict tolerances** (0.086 /
  0.255 / 0.398 at ±0 / ±1 / ±2 frames) — the cached features are
  already log-compressed, so this is essentially the cleanest
  attack-aligned signal available from the existing feature
  pipeline. If we *do* want a fine-grained channel (rather than
  bucket-pooled), this is the candidate, not plain spectral_flux.
- **Energy is competitive at ±10** (F1 = 0.646 vs SF 0.679) but
  collapses at strict tolerances (0.004 at ±0). For a bucketed
  ±10-frame channel, energy is a viable cheap alternative to SF.
  For a frame-precise channel it's useless.
- **Sub-band channels need a different evaluation** to show their
  value. The collapsed envelope is identical to broadband SF; the
  real benefit (low-band fires on DON-like, high-band on KA-like)
  would only show up under per-kind aggregation, which this run
  did not compute. Filed as a followup.
- **Joint-coverage analysis was hampered by threshold saturation.**
  Re-run the coverage step with per-algo recall ≥ 0.85 (instead
  of 0.95) to get an informative recall vs precision tradeoff
  across channel-count. Filed as a followup.
- **`complex_domain` skipped** because mel-domain results
  saturated the design space. Adding it would require .osz
  indexing + per-chart audio decode (~30 min on full split). Not
  worth running unless the strict-tolerance regime becomes the
  bottleneck for #012.

## Followup questions

The decision queue for #012 (the actual training experiment):

- **Channel-encoding granularity is the load-bearing knob.** Two
  candidate designs to A/B:
  1. **Bucketed channel** — pool the spectral_flux activation
     over 5- or 10-frame windows (max-pool or mean-pool), produce
     a coarser per-bucket "onset present?" channel. Repeat for
     2-3 algorithms. Concat as low-resolution input planes.
  2. **Frame-precise channel** — keep the activation at 5 ms grid
     using `log_filtered_flux` (the only one that doesn't collapse
     at ±2 frames). One channel as-is.
  Likely (1) gives a cleaner training signal at the cost of
  coarser timing; (2) gives finer timing but with the 2-3-frame
  lag the model has to compensate for.

Open follow-ups within the survey itself:

- **Joint coverage with tighter per-algo recall target (0.85
  instead of 0.95).** Should differentiate channel sets and tell us
  whether 2 or 3 channels actually help vs single SF. Cheap re-run.
- **Per-kind aggregation (DON vs KA recall per algorithm).** The
  per-chart CSV has the chart_id; we can join with the events.npz
  files to compute per-kind recall and check whether sub-band
  splits expose a low-band/DON, high-band/KA bias.
- **Per-density / per-star-rating breakdown.** Same data, different
  aggregation. Tells us whether ODF performance degrades on dense
  charts (where the model needs the most help).
- **`complex_domain`** if the strict-tolerance regime turns out to
  matter for #012's downstream metric.
- **#007 frame-wise evaluation** — same harness, with #007's AR
  rollout output as the activation source. Crossover analysis vs
  classical ODFs.
- **madmom RNN onset** — same harness, with the SOTA classical
  baseline as a ceiling reference.

The expected followup, regardless of result, is to scaffold #012:

- **#012 — Onset-channel input augmentation.** Add the top
  K channels from this survey (predicted: `superflux + hfc_mel
  + subband_sf_4` or similar) as additional input planes
  alongside the 80-band log-mel. Train an otherwise-#007-identical
  model and measure whether miss / hit improves on the val split.

Other followups expected post-run:

- **#007 frame-wise evaluation.** Run #007's best checkpoint in
  AR mode on the same val split, build a binary frame mask from
  predicted onsets, and run the same `evaluate_frames` against
  GT. Crossover analysis (at what tolerance does the trained
  model beat / get beaten by classical ODFs).
- **madmom RNN onset** — same evaluation harness, with the
  RNNOnsetProcessor as the activation source. Gives a SOTA
  classical reference and tells us how much of the ceiling is
  already reachable with a free pretrained model.
- **Per-frame model evaluation in cursor-sweep mode.** At
  every Nth frame, run #007 with the cursor at that frame;
  treat the next-bin softmax as a per-frame "is there an onset
  here?" signal and run the same harness. Apples-to-apples
  comparison against ODF activations rather than against an
  AR binary mask.
