# Experiment 011b — Onset detection disagreement + sub-analysis

## Status

`Complete`. **Pairwise complementarity is much larger than the
"redundancy dominates" hypothesis predicted** — best 2-channel
union hits **recall 0.905** at `hfc_mel + spectral_flux`,
**+16.3 pp over the best single channel** (energy, R = 0.780)
and well above the 8-pp "fails-if-this-high" threshold the pre-
run set. The structural finding is that **difference-based**
ODFs (spectral flux family, SuperFlux, log-filtered) and
**envelope-based** ODFs (energy, HFC) catch genuinely different
onset subsets — within-group Jaccard 0.81-0.88, **cross-group
Jaccard 0.57-0.69**. Sub-band frequency specialization on
DON vs KA was **not observed** (≤ 5 pp gap in any band, no
crossover) — abandons the sub-band-as-separate-channel design
idea.

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

Survey ran on the full val split (958 charts, **654,333 GT
onsets**) in **27.5 s** including all aggregations. No charts
skipped.

**Headline:** complementarity exists and is much higher than the
pre-run "redundancy dominates" hypothesis predicted. The best
2-channel union (`hfc_mel + spectral_flux`) reaches recall 0.905
— +16.3 pp over the best single channel (`energy` at 0.780),
and +33.2 pp over `log_filtered_flux` at 0.654. The maximum
pairwise marginal gain is **13.3 pp**, well above the predicted
4-7 pp ceiling. The structural reason is what we missed in the
pre-run mechanism analysis: difference-based and envelope-based
ODFs do detect different onsets even though they read the same
log-mel.

The F1 picture, however, **gets worse with more channels** because
precision drops faster than recall rises. Single-channel best F1
is 0.679 (`spectral_flux`); 2-channel best F1 is 0.495 (`hfc_mel +
spectral_flux`); 4-channel drops to 0.29. For #012's downstream
model design, the recall vs precision tradeoff is the design
question — and **recall is what matters**, since the model is
expected to filter the channel-injected false positives during
training.

The per-density and per-star results are inverted from the pre-run
prediction. F1 *increases* with density and difficulty:
sparse charts have F1 = 0.35; very-dense F1 = 0.73. The driver is
precision, not recall — sparse charts contain many audio onsets
that the chart author chose *not* to map (background hi-hats,
ghost notes), so ODF activations get counted as false positives.
Dense / insane charts map nearly every salient audio onset, so
ODF activations match GT.

The sub-band-8 DON-vs-KA per-band recall, predicted to show a
strong low-band-on-DON / high-band-on-KA crossover (15-25 pp gap),
**did not appear**: every band has KA recall slightly above DON
recall by 0-7 pp; no crossover. This is an unambiguous design
data point — sub-band channels don't carry kind-specific
information beyond what broadband flux already encodes.

### Per-algorithm single-channel @ ±10 frames (best-F1 threshold)

| Algorithm | Recall | Precision (lower) | F1 | n_pred |
|---|---:|---:|---:|---:|
| **spectral_flux** | 0.7425 | **0.6254** | **0.6789** | 776,820 |
| subband_sf_4 (collapsed) | 0.7425 | 0.6254 | 0.6789 | 776,820 |
| subband_sf_8 (collapsed) | 0.7425 | 0.6254 | 0.6789 | 776,820 |
| **energy** | **0.7802** | 0.5517 | 0.6463 | 925,426 |
| hfc_mel | 0.7718 | 0.5257 | 0.6254 | 960,740 |
| superflux | 0.6923 | 0.5653 | 0.6224 | 801,240 |
| log_filtered_flux | 0.6536 | 0.4751 | 0.5502 | 900,221 |

`energy` has the highest single-channel **recall** (0.780),
confirming the #011 surprise; `spectral_flux` has the highest
**precision** (0.625) and best single-channel F1.

### Pairwise marginal gain (B | A) — sorted descending

A small selection (full table in `pairwise.csv`). `B | A` is the
fraction of GT onsets that B catches that A missed:

| Pair | R(A) | R(B) | R(union) | marg(B \| A) | Jaccard |
|---|---:|---:|---:|---:|---:|
| **hfc_mel + spectral_flux** | 0.772 | 0.742 | **0.905** | **0.133** | 0.673 |
| hfc_mel + log_filtered_flux | 0.772 | 0.654 | 0.905 | 0.133 | 0.575 |
| hfc_mel + superflux | 0.772 | 0.692 | 0.898 | 0.127 | 0.630 |
| energy + log_filtered_flux | 0.780 | 0.654 | 0.904 | 0.124 | 0.586 |
| energy + spectral_flux | 0.780 | 0.742 | 0.900 | 0.120 | 0.691 |
| energy + superflux | 0.780 | 0.692 | 0.895 | 0.115 | 0.645 |
| log_filtered_flux + spectral_flux | 0.654 | 0.742 | 0.761 | 0.107 | 0.835 |
| log_filtered_flux + superflux | 0.654 | 0.692 | 0.742 | 0.088 | 0.814 |
| energy + hfc_mel | 0.780 | 0.772 | 0.835 | 0.055 | 0.859 |
| spectral_flux + superflux | 0.742 | 0.692 | 0.765 | 0.022 | 0.876 |
| spectral_flux + subband_sf_4 | 0.742 | 0.742 | 0.742 | 0.000 | 1.000 |

**Pattern: cross-group is the win.** Pairs of one envelope-based
algorithm (energy / HFC) and one difference-based algorithm (SF /
log-SF / SuperFlux) consistently push union recall to ~0.90+
with marginal gains 11-13 pp. Within-group pairs only gain 2-9
pp. The sub-band collapsed envelopes are mathematically identical
to broadband SF — Jaccard 1.000, marginal gain 0.000.

### Best K-channel union by recall and by F1

| K | Best by recall | Recall | Precision (lower) | F1 |
|---:|---|---:|---:|---:|
| 1 | spectral_flux | 0.742 | 0.625 | **0.679** |
| 2 | hfc_mel + spectral_flux | **0.905** | 0.341 | 0.495 |
| 3 | energy + hfc_mel + log_filtered_flux | **0.932** | 0.219 | 0.354 |
| 4 | energy + hfc_mel + log_filtered_flux + superflux | 0.939 | 0.171 | 0.290 |

**F1 peaks at K=1; recall climbs at K=2 and again at K=3.** The
recall-precision tradeoff is sharp: every additional channel
adds ~700K predicted peaks but only catches a few percentage
points more GT onsets, dropping precision proportionally. Pre-
run prediction was directionally right (F1 peaks at K=1) but the
recall climb is steeper than predicted.

### Per-kind recall (best-F1 threshold per algo)

GT distribution: DON 303,821 (46.4 %); KA 309,597 (47.3 %);
BIG_DON 19,844 (3.0 %); BIG_KA 19,450 (3.0 %); SPINNER 1,164;
DRUMROLL 457.

| Algorithm | DON | KA | BIG_DON | BIG_KA | DRUMROLL | SPINNER |
|---|---:|---:|---:|---:|---:|---:|
| energy | 0.760 | 0.785 | **0.914** | 0.905 | 0.731 | 0.535 |
| hfc_mel | 0.757 | 0.775 | 0.874 | 0.851 | 0.759 | 0.547 |
| log_filtered_flux | 0.617 | 0.670 | 0.807 | 0.823 | 0.711 | 0.539 |
| spectral_flux | 0.703 | 0.757 | 0.936 | **0.942** | 0.718 | 0.502 |
| superflux | 0.648 | 0.711 | 0.884 | 0.896 | 0.628 | 0.488 |

**Every algorithm scores higher on KA than on DON by 1-5 pp**,
and BIG variants by 15-20 pp. Inter-algorithm differences within
a kind are 5-10 pp — `log_filtered_flux` is uniformly worst,
`spectral_flux` wins on BIG variants, `energy` wins on plain DON
and ties KA.

The predicted asymmetry (HFC stronger on KA, energy stronger on
DON) **is not present** — all algorithms have the same
DON < KA < BIG_DON ≈ BIG_KA ordering. The 5-pp range across
algorithms within a kind is the redundancy I expected to see at
the *pair* level — it's there at the *kind-stratified-recall*
level instead.

### Per-density-bucket F1 (spectral_flux only)

Bucket cuts: sparse < 1.5 events/s; medium 1.5-3.0; dense 3.0-5.0;
very_dense ≥ 5.0.

| Bucket | F1 | Recall | Precision |
|---|---:|---:|---:|
| sparse | 0.350 | **0.913** | 0.217 |
| medium | 0.505 | 0.902 | 0.351 |
| dense | 0.684 | 0.829 | 0.582 |
| very_dense | **0.732** | 0.689 | **0.780** |

**F1 increases with density. Reverse of the pre-run prediction.**
Recall actually *decreases* with density (peak picker NMS more
likely to suppress real onsets in dense passages), but precision
climbs much faster (sparse charts have many unmapped audio onsets;
dense charts map almost everything). Net effect: F1 climbs.

### Per-star-rating-bucket F1 (spectral_flux only)

| Bucket | Stars | F1 | Recall | Precision |
|---|---|---:|---:|---:|
| easy | < 3 | 0.551 | 0.881 | 0.401 |
| medium | 3 - 4 | 0.718 | 0.801 | 0.650 |
| hard | 4 - 5 | 0.730 | 0.719 | 0.741 |
| insane_plus | ≥ 5 | 0.728 | 0.663 | 0.807 |

Same shape as density. Easy charts are sparse → many FPs → low
F1. Insane+ charts are dense → few FPs → high F1.

### Sub-band 8 per-band recall — DON vs KA

Tests the predicted low-band-fires-on-DON / high-band-fires-on-KA
crossover (predicted 15-25 pp gap):

| Band | DON recall | KA recall | KA − DON |
|---:|---:|---:|---:|
| 0 (low) | 0.687 | 0.690 | +0.003 |
| 1 | 0.673 | 0.725 | +0.052 |
| 2 | 0.674 | 0.727 | +0.053 |
| 3 | 0.650 | 0.707 | +0.057 |
| 4 | 0.607 | 0.668 | +0.061 |
| 5 | 0.573 | 0.634 | +0.061 |
| 6 | 0.533 | 0.600 | +0.067 |
| 7 (high) | 0.505 | 0.565 | +0.060 |

**No crossover.** Every band catches KA slightly *better* than DON
by ~5-7 pp; no band catches DON better than KA. The predicted
specialization (low-band-DON / high-band-KA) does not appear at
this granularity. Two read-outs:
1. Sub-band channels do not carry kind-specific information
   the way the pre-run hypothesis assumed.
2. Recall *decreases* with band index across both kinds — high-
   freq bands have less mass to fire on; predictions get sparser.
   So the high-band channel mostly misses both kinds rather than
   specializing on one.

This abandons the sub-band-as-channel design idea. **The
broadband collapsed envelope (which is exactly `spectral_flux`)
captures everything sub-band channels collectively offer at this
granularity.**

## Visualizations

_(Plots not yet generated — script runs in seconds when needed.
The above tables capture the headline numbers.)_

Planned graphs:

- `01_complementarity_matrix.png` — pairwise marginal recall gain
  heatmap.
- `02_complementarity_jaccard.png` — pairwise Jaccard similarity
  heatmap (1 = perfectly redundant).
- `03_union_vs_size.png` — best union recall / precision / F1
  by subset size (1 / 2 / 3 / 4).
- `04_per_kind.png` — recall per (algo, onset kind).
- `05_per_density.png` — F1 per (algo, density bucket).
- `06_per_star.png` — F1 per (algo, star bucket).
- `07_activation_distributions.png` — overlaid histograms of
  activation values at TP / FP / FN / near-miss frames per
  algorithm.
- `08_subband_per_kind.png` — sub-band 8 per-band recall by kind.

## Vs prediction

| Prediction | Actual | Verdict |
|---|---|---|
| Pairwise marginal gains 1-4 pp typical, max ~6 pp | 11/21 pairs ≥ 8 pp; max **13.3 pp** | **MISS — refuted.** Cross-group complementarity dominates. |
| 2-channel best union at recall 0.78-0.80 | recall **0.905** | **MISS by 10+ pp.** |
| 3-channel union at recall 0.81-0.83 | recall **0.932** | **MISS by 10+ pp.** |
| F1 peaks at K=1; declines with K | F1 = 0.679 → 0.495 → 0.354 → 0.290 | **MET** directionally. |
| Sub-band per-band DON vs KA gap 15-25 pp with crossover | gap 0-7 pp, no crossover | **MISS — refuted.** No frequency specialization. |
| Per-kind algorithm gaps ≤ 5 pp | observed 5-10 pp range | **PARTIAL** — slightly larger than predicted. |
| `log_filtered_flux` uniformly worst on per-kind | yes, by 5-10 pp | **MET**. |
| Density: F1 drops on dense charts (~0.50) | F1 *rises* on dense (0.73) | **MISS — wrong direction.** |
| Star rating: F1 drops on insane+ (~0.58) | F1 *rises* on insane+ (0.73) | **MISS — wrong direction.** |
| `spectral_flux + subband_sf_4` near-zero gain | gain 0.000 exactly (Jaccard 1.000) | **MET**. |
| Activation distribution analysis | not yet computed; data in summary.json | **DEFERRED**. |

**Hypothesis "redundancy dominates" is rejected.** The pre-run
mental model — same input, same NMS, same threshold-bound
failures → near-redundant outputs — was wrong because difference-
based ODFs and envelope-based ODFs respond to fundamentally
different audio events. A flux-based algorithm fires on the
attack frame; an envelope-based algorithm fires on the energy
peak (which can be 2-5 frames later, on a different *type* of
onset profile). Their miss sets concentrate on different
onset shapes.

## Takeaways

- **Cross-group channel pairs are the strongest signal.** A
  2-channel input of `spectral_flux + hfc_mel` captures
  recall 0.905 — almost 16 pp above any single channel. That's
  the candidate channel set for #012's first attempt.
- **Within-group additional channels add little.** SF + SuperFlux
  Jaccard 0.876 → marginal gain 2.2 pp. SF + log-SF marginal gain
  10.7 pp (looks high in isolation but only because log-SF has
  poor recall on its own — 0.654 — and most of its gain is from
  catching onsets the others fire on too with different precision).
  For channel selection, prefer one diff-based + one envelope-
  based over two of the same family.
- **F1 is not the right metric for channel selection.** Recall
  is. F1 collapses at K ≥ 2 because the channels each contribute
  ~800K-960K predicted peaks, dropping precision to 0.34. But
  the channels are *inputs* to the model, not predictions —
  the model can filter them. The real metric for channel design
  is **maximum recall at K channels** with reasonable
  per-channel precision (~0.50+).
- **Sparse / easy-difficulty charts are where the channel
  signal is weakest in absolute precision** (sparse: P = 0.22)
  but the chart's recall is highest (R = 0.91 on sparse). The
  channel's job is "look here" — sparse charts are exactly
  where the model needs that hint most, and the channel still
  provides it (recall 0.91), just at noisy precision.
- **Sub-band channels are not worth the architectural cost.**
  No DON / KA frequency specialization. Broadband flux already
  encodes everything sub-band channels collectively encode. Drop
  the sub-band-as-multi-channel idea from #012.
- **`energy` is a real candidate channel.** Highest recall of
  any single channel (0.780); high complementarity with SF / SF
  family (Jaccard 0.69, 12 pp marginal gain). Cheap to compute.
  Worth including in #012 as a second channel alongside SF.
- **`log_filtered_flux` is consistently the weakest signal**
  (lowest recall, lowest F1, lowest per-kind recall on every kind).
  The log compression apparently hurts more than it helps for
  taiko percussion. Drop from the channel candidate list for #012.
- **Per-density inversion is a content insight, not a bug.** Easy /
  sparse charts have many *unmapped* audio onsets relative to
  mapped ones. The ODF predictions are full of "musically valid"
  but chart-irrelevant peaks. **Implication for #012**: the
  channel signal carries different information by chart density,
  so the model may need to learn density-dependent gating.

## Followup questions

For #012 (the channel-input training experiment):

- **Recommended channel set: `spectral_flux + energy + hfc_mel`**
  — 3-channel union recall 0.927 (or 0.932 if `log_filtered_flux`
  is added; F1 marginally worse). 2-channel `spectral_flux +
  hfc_mel` at recall 0.905 is the lower-cost variant.
- **Frame encoding (from #011):** bucket-pool the activations
  into ±5 or ±10 frame windows, not raw 5 ms grid. Combined with
  the multi-channel set above.
- **Open: should each channel be max-pool or sum-pool when
  bucketing?** Max-pool preserves peak strength; sum-pool
  preserves total activation density. Worth a small ablation in
  #012's pre-run.

Open follow-ups within the survey itself:

- **Generate the planned plots** from `summary.json` (matrix
  heatmaps, K-vs-recall curve, etc.) — script can be one-shot
  off this experiment's data.
- **Activation distributions** — the data is sampled and stored
  in `summary.json["activation_distributions"]`; just need a
  histogram plot to see TP/FP/FN/near-miss separability.
- **Threshold sensitivity** of the complementarity result — does
  the cross-group complementarity hold if we use each algo's
  recall-≥-0.85 threshold instead of best-F1? Cheap re-run, would
  tell us whether the 13 pp marginal gains generalize across
  operating points.
- **Three-way and four-way Jaccard / complementarity decomposition**
  — for the top 3-channel set, what fraction of GT onsets are
  caught by exactly 1 / 2 / 3 of the algorithms? Tells us whether
  the 0.93 union recall is "any one of three fires" (best-case
  union) or "two of three" (degenerate redundancy).
