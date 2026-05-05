# Experiment 011b — Onset detection disagreement + sub-analysis

## Status

`Planned`

## Context

[#011](../011-onset-feature-survey/) measured single-algorithm
F1 / recall / precision against GT and ran a joint-coverage pass.
The single-algorithm numbers were strong — best F1 ≈ 0.679 at
±10 frames for `spectral_flux` — but the joint-coverage table
saturated at recall = 1.0 because the per-algo high-recall
threshold (recall ≥ 0.95) over-fired peaks by ~10×, making every
3-4-channel union pegged at 100 % recall with ~9 % precision.

Two questions #011 didn't answer:

1. **Are the single-algorithm misses correlated or complementary?**
   If `spectral_flux` and `superflux` both score recall = 0.74,
   that does *not* mean their union is also 0.74. If their misses
   are largely disjoint — e.g. SF misses softer high-frequency
   KA-style attacks that SuperFlux catches — the union might be
   0.85+ for almost no extra cost. The right metric is
   **per-onset complementarity** (Venn-diagram-style), not pooled
   recall.
2. **Is the F1 ceiling uniform across content?** The pooled F1 of
   0.679 hides per-kind, per-density, and per-difficulty
   variation. If some classes (e.g. KA, dense charts, hard
   difficulties) have systematically lower recall, that's where
   #012's downstream model gets the most headroom from the
   channel input.

This experiment answers both, and adds activation-strength
distributions (how confident is each algorithm at the frames it
catches vs misses?) so #012 can pick a sensible threshold per
channel and know what the input-feature distribution looks like.

## Citations

- Direct parent: [#011](../011-onset-feature-survey/) — same
  algorithms, same data, same evaluation harness for the per-
  algorithm part. New analyses layered on top.
- Onset detection literature: same set as #011's references.
- Disagreement / fusion conceptually related to:
  [Design and Evaluation of Onset Detectors Using Different Fusion Policies (ISMIR 2014)](https://archives.ismir.net/ismir2014/paper/000229.pdf).

---

## Hypothesis

### Claim

The mel-domain ODFs are **largely redundant** because they all
operate on the same log-mel signal and share the same dominant
failure modes: peak-pick NMS collapses adjacent onsets into one,
soft attacks with slow rise-time produce no strong activation in
any algorithm, and chart-author choices that don't track audio
events look identical to every algorithm. Pairwise marginal
recall gains will be **small (1-4 pp typical, up to ~6 pp for the
most differentiated pair)**, and the 2-channel best union will
sit only **3-5 pp above single-channel recall**, not the 10+ pp
the naive "different algorithms → different misses" intuition
suggests.

The K-vs-recall curve is therefore expected to flatten quickly:
single-channel ≈ 0.74 → 2-channel ≈ 0.78 → 3-channel ≈ 0.81 →
4-channel ≈ 0.83. Channel-stacking helps, but with sharply
diminishing returns.

Per-kind, per-density, and per-difficulty breakdowns are still
expected to show divergence (DON vs KA recall gaps, density
degradation), but those reflect *different content being harder*
rather than *different algorithms catching different things on
the same content*. The per-kind cut may be the only place
sub-band channels show real specialization.

### Mechanism — why redundancy dominates

- **All algorithms share the input.** Every ODF here reads the
  same 80-band log-mel — there's no independent information
  source. Differences are functions of the same signal. When the
  log-mel doesn't have a clear local maximum at an onset frame
  (soft attack, noisy passage, vocal masking), no transformation
  of that mel can manufacture one.
- **Peak-picker NMS is the same for every channel.** A
  ``min_distance=1`` local-max with absolute threshold suppresses
  the same clusters of adjacent activations regardless of what
  produced the activation. Two flux-shaped envelopes peak at the
  same frame on a sharp attack, and both are NMS'd to one peak —
  catching exactly the same GT.
- **Threshold-bound misses dominate.** At each algorithm's
  best-F1 threshold (~0.30-0.80 of normalized activation), most
  GT-aligned activations either clear the threshold (TP) or sit
  near zero (FN). The "near-miss" band — local max just below
  threshold — is narrow. So the FN sets are determined more by
  which onsets *have any activation at all* than by *which
  algorithm's transformation amplifies them most*.
- **Chart-author miss is unfixable.** A meaningful share of GT
  onsets don't correspond to clear audio events at all (chart
  authors mapping vocals or off-beat patterns). Those are
  uniformly missed by every algorithm; they show up as
  ``n_neither`` in the pairwise breakdown, capping union recall.
- **Where complementarity *might* be real:** the few onset
  characteristics that algorithms genuinely treat differently —
  soft high-frequency attacks (HFC weights high freq more),
  slow-rise envelopes (energy peaks late but eventually peaks),
  log-amplified soft onsets (log-filtered SF) — represent a
  small fraction of total onsets. So the marginal gains exist
  but are bounded.

### Predicted numbers

#### Pairwise complementarity (best-F1 threshold per algo, ±10 frames)

Heavy emphasis on redundancy. Marginal gain expressed in
**percentage points of GT recall**, not relative percentages.

| Pair | Predicted union recall | Predicted marginal gain (B|A) | Predicted Jaccard | Notes |
|---|---:|---:|---:|---|
| spectral_flux + subband_sf_4 | 0.742 | 0.0 pp | ~1.00 | Identical envelopes when collapsed. |
| spectral_flux + subband_sf_8 | 0.742 | 0.0 pp | ~1.00 | Identical envelopes when collapsed. |
| subband_sf_4 + subband_sf_8 | 0.742 | 0.0 pp | ~1.00 | Same. |
| spectral_flux + superflux | 0.76-0.78 | 2-4 pp | 0.88-0.93 | Max-filter shifts a few peaks; mostly redundant. |
| spectral_flux + hfc_mel | 0.76-0.78 | 2-4 pp | 0.85-0.92 | HFC catches some high-freq misses SF blurred over. |
| spectral_flux + energy | 0.75-0.77 | 1-3 pp | 0.88-0.95 | Energy mostly fires where SF fires. |
| spectral_flux + log_filtered_flux | 0.77-0.80 | 3-6 pp | 0.82-0.90 | The most differentiated pair; log compression catches some softer onsets. |
| superflux + log_filtered_flux | 0.74-0.78 | 2-5 pp | 0.85-0.92 | Different compressions but both diff-based. |
| hfc_mel + log_filtered_flux | 0.78-0.82 | 4-7 pp | 0.78-0.86 | Possibly the strongest pair — different temporal shape AND different freq weighting. |
| hfc_mel + energy | 0.79-0.82 | 1-3 pp | 0.92-0.96 | Both envelope-based, lag-aligned; redundant. |

#### Best 2 / 3 / 4-channel union (predicted F1 / recall)

| K | Predicted best set | Recall | Precision (lower bound) | F1 |
|---:|---|---:|---:|---:|
| 1 | spectral_flux | 0.742 | 0.625 | **0.679** |
| 2 | spectral_flux + log_filtered_flux *or* hfc_mel + log_filtered_flux | 0.78-0.80 | 0.50-0.55 | 0.62-0.65 |
| 3 | + hfc_mel (or + energy) | 0.81-0.83 | 0.42-0.48 | 0.56-0.60 |
| 4 | + superflux | 0.83-0.85 | 0.36-0.42 | 0.51-0.56 |

**Predicted shape:** F1 *peaks at K=1*, drops monotonically as K
grows. Recall climbs but precision (even by the lower-bound
estimate) drops faster. **For #012 the right takeaway is likely
"single channel" or "two channels" — not "stack everything."**

If the claim above is wrong and 2-channel union actually reaches
recall 0.85+ with marg ≥ 8 pp, that's strong evidence of
genuine complementarity and #012 should plan for 3-4 channels.

#### Per-kind recall (best-F1 threshold, ±10 frames)

Predicted **gaps between algorithms within each kind to be
small (≤ 5 pp)** — same-input redundancy applies here too. The
predicted *kind-by-kind* differences (BIG variants higher than
plain because louder; DRUMROLL/SPINNER different because extended)
are real and present across all algorithms similarly.

| Algorithm | DON | KA | BIG_DON | BIG_KA |
|---|---:|---:|---:|---:|
| spectral_flux | 0.74 | 0.74 | 0.85 | 0.83 |
| superflux | 0.70 | 0.69 | 0.83 | 0.80 |
| hfc_mel | 0.74 | 0.76 | 0.85 | 0.84 |
| energy | 0.78 | 0.74 | 0.88 | 0.83 |
| log_filtered_flux | 0.66 | 0.66 | 0.78 | 0.76 |

Possible weak signal: HFC very slightly better on KA (high-freq-
weighted), energy slightly better on DON / BIG_DON (loud broadband),
log-filtered-flux uniformly worse. Magnitude of any specialization
predicted ≤ 4 pp.

Sub-band 8 per-band recall (the clearest test of frequency
specialization):
- low bands (0-1) on DON: 0.55-0.70
- low bands (0-1) on KA: 0.30-0.50
- high bands (6-7) on DON: 0.30-0.50
- high bands (6-7) on KA: 0.55-0.70

If this DON↔KA crossover is observed at sub-band granularity
(predicted gap 15-25 pp between low-band-DON-recall and
low-band-KA-recall), that's the best-evidence case for **per-band
channels feeding the model separately** rather than collapsed.

#### Per-density-bucket F1 (spectral_flux only, ±10 frames)

Density bucket = chart's `density_mean` events / s. Quartile cuts
on val split.

| Bucket | Range (events/s) | Predicted F1 |
|---|---|---:|
| Sparse | < 1.5 | 0.75 |
| Medium | 1.5-3.0 | 0.70 |
| Dense | 3.0-5.0 | 0.62 |
| Very-dense | ≥ 5.0 | 0.50 |

Dense charts predicted to be ≥ 10 pp lower than sparse; that's
where the channel input most helps the downstream model.

#### Per-star-rating F1 (spectral_flux only, ±10 frames)

| Bucket | Stars | Predicted F1 |
|---|---|---:|
| Easy | < 3 | 0.72 |
| Medium | 3 - 4 | 0.69 |
| Hard | 4 - 5 | 0.65 |
| Insane+ | ≥ 5 | 0.58 |

#### Activation-strength distributions

Predicted shape: at TP frames, activation values cluster near the
99th-percentile-of-chart (since these are the picked peaks).
At FP frames, values cluster around the threshold (just-barely-
above). At FN frames, values are distributed more uniformly across
the full range — many FN onsets exist where the activation simply
didn't peak (no local max above threshold) versus near-misses
where the local max was just below threshold.

The shape of the FN distribution determines whether a lower
threshold would help: if FN values cluster near the threshold,
yes; if they cluster low, the algo just doesn't have signal there
and a different algorithm or channel is the only fix.

## Success criteria

- **Must answer:** at the best-F1 threshold, what are the pairwise
  union-recall and marginal-gain numbers for every (A, B) pair?
  Specifically: which pair is most differentiated, and what's the
  marginal gain?
- **Must answer:** does any 2 / 3-channel union materially beat
  single-channel recall at ±10 frames? Predicted answer: 2-channel
  beats by 3-6 pp recall, 3-channel by 6-9 pp; F1 declines
  monotonically.
- **Must answer:** are per-kind recalls divergent across
  algorithms? Predicted answer: small (≤ 5 pp) divergences
  except for sub-band per-band cuts.
- **Must answer:** does F1 degrade systematically on dense charts
  / hard difficulties? Predicted answer: yes, ~10-20 pp gap
  between sparse and very-dense.
- **Nice-to-have:** activation distributions per (algo, frame
  category) — useful for picking thresholds in #012.
- **Nice-to-have:** per-band sub-band aggregation showing
  DON-vs-KA specialization for the sub-band split channels
  individually (rather than the collapsed envelope). Predicted
  to be the *only* place sub-band channels show real value.
- **Fails (refutes redundancy claim) if:** any pair's marginal
  gain exceeds **8 pp** — would mean the algorithms have
  meaningfully complementary miss sets and channel-stacking
  is worth more than predicted.
- **Fails (low-information run) if:** every pair's marginal gain
  is < 1 pp — would mean the algorithms are *perfectly*
  redundant and there's literally nothing to gain from any
  combination beyond single-channel.
- **Fails (data-pipeline bug) if:** sum of caught flags across
  all algos is constant per chart (every algo catches *exactly*
  the same set) — would indicate threshold normalization is
  collapsing the signal.

## Methodology

### Algorithms surveyed

Same set as [#011](../011-onset-feature-survey/), mel-domain only:

- `energy`
- `spectral_flux`
- `log_filtered_flux`
- `hfc_mel`
- `superflux`
- `subband_sf_4` (collapsed; per-band breakdown for sub-analyses)
- `subband_sf_8` (collapsed; per-band breakdown for sub-analyses)

### Operating point

Each algorithm's threshold is fixed at its **best-F1 operating
point @ ±10 frames** as measured in #011 (see
`#011/results/per_algo/{algo}/curves.csv`). The thresholds are:

| Algorithm | Threshold |
|---|---:|
| spectral_flux | 0.32 |
| log_filtered_flux | 0.24 |
| hfc_mel | 0.68 |
| superflux | 0.28 |
| energy | 0.76 |
| subband_sf_4 (collapsed) | 0.32 |
| subband_sf_8 (collapsed) | 0.32 |

This is the apples-to-apples operating point for every analysis
in this experiment — pairwise complementarity, per-kind recall,
per-density bucket, etc.

### Per-onset detection record

Per chart, per algorithm, per GT onset, record `caught_in_window`:
1 if the algorithm's predicted peaks include any frame within
±10 of this GT bin (matching the canonical tolerance), 0 else.

That gives a `(n_onsets_total × n_algos)` boolean matrix from
which every aggregate analysis derives:
- pairwise complementarity by indexing two columns
- per-kind by joining with `events.npz['kind_ids']`
- per-density / per-star by joining with the chart's manifest
  fields

### Pairwise complementarity matrix

For each pair `(A, B)` of algorithms, count over all GT onsets:
- `n_AB` — caught by both A and B
- `n_A_only` — caught by A but not B
- `n_B_only` — caught by B but not A
- `n_neither` — caught by neither

Compute:
- Recall(A) = (n_AB + n_A_only) / total
- Recall(B) = (n_AB + n_B_only) / total
- Recall(A ∪ B) = (n_AB + n_A_only + n_B_only) / total
- Marginal gain of B given A = n_B_only / total
- Jaccard similarity of caught sets
  = n_AB / (n_AB + n_A_only + n_B_only)

The per-onset matrix lets us also evaluate any 3- or 4-channel
union recall and precision exactly the same way.

### Per-kind / per-density / per-star aggregation

- `per_kind`: aggregate the detection matrix by `kind_ids` → recall
  per (algo, kind). Also computed for each sub-band separately so
  we can see if low-band fires on DON specifically.
- `per_density_bucket`: bucket charts by `density_mean` quartile
  on the val split → F1 per (algo, bucket).
- `per_star_bucket`: bucket charts by `star_rating` (cuts at
  3, 4, 5) → F1 per (algo, bucket).

For the bucket-level F1 we need precision too, so we also count
predicted peaks per chart and per (algo, bucket).

### Activation distributions

For each algorithm, sample activation values at:
- TP frames (the picked peaks that matched a GT)
- FP frames (the picked peaks that didn't match a GT)
- FN frames (GT bins where no peak fired within ±10 frames)
- "near-miss" frames (GT bins where the activation had a local
  max within ±10 frames but below the threshold)

Histogram each, plot overlaid. Tells us how separable the
distributions are and where threshold-tuning could help.

### Compute

Mel-domain only, GPU-accelerated as in #011. Same per-chart
loop, with the per-onset matrix accumulated alongside the
TP/FP/FN counters. Estimated runtime: ~7-8 min on the full val
split.

## Run config

- Output root: [`results/`](./results/) — auto-created by the CLI.
- Smoke test: 4 charts pass before launching the full run.
- Command:

  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.onset_feature_survey_b \
      --dataset taiko2_v1 \
      --output osu/taiko2/experiments/011b-onset-disagreement/results \
      --split val --device cuda
  ```

Plot script:

```bash
osu/taiko2/.venv/Scripts/python.exe \
    osu/taiko2/experiments/011b-onset-disagreement/plot_results.py
```

---
<!-- Post-run below -->
---

## Results summary

_(To fill post-run.)_

## Visualizations

_(Post-run.)_

Planned graphs:

- `01_complementarity_matrix.png` — pairwise marginal recall gain
  heatmap.
- `02_complementarity_jaccard.png` — pairwise Jaccard similarity
  heatmap (1 = perfectly redundant).
- `03_union_vs_size.png` — best union recall / precision / F1
  by subset size (1 / 2 / 3 / 4) at the best-F1 operating point.
- `04_per_kind.png` — recall per (algo, onset kind).
- `05_per_density.png` — F1 per (algo, density bucket).
- `06_per_star.png` — F1 per (algo, star bucket).
- `07_activation_distributions.png` — overlaid histograms of
  activation values at TP / FP / FN / near-miss frames per
  algorithm.
- `08_subband_per_kind.png` — sub-band 8 per-band recall, split
  by onset kind. Tests the low-band-DON / high-band-KA prediction.

## Vs prediction

_(Post-run.)_

## Takeaways

_(Post-run.)_

## Followup questions

_(Post-run.)_
