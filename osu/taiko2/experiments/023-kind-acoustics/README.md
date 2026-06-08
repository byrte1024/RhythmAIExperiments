# Experiment 023 — Kind acoustics and pattern analysis

## Status

`Planned`

## Context

Every taiko2 experiment to date predicts **when** onsets occur but not
**what kind** they are (DON, KA, BIG_DON, BIG_KA, DRUMROLL, SPINNER).
The AR inference loop hardcodes `default_kind=DON` for every predicted
onset. Before building a typing model, a fundamental question must be
answered: **is D/K assignment driven by audio timbre or by rhythmic
pattern structure?**

Corpus statistics from [#003](../003-gap-ratio-corpus/) and the chart
metrics summary show that `don_ratio` has median 0.494 with IQR
0.47-0.51 -- essentially 50/50 across ~6.5M normal hits. BIG variants
mirror exactly (50.5/49.5). This near-perfect symmetry, combined with
the observation that identical rhythmic phrases often appear as DKDD
in one chart and KDKK in another, suggests D/K is interchangeable at
the audio level and determined by pattern context.

This experiment runs a corpus-wide analysis to test that hypothesis
before committing to a typing architecture.

## Citations

- Corpus shape reference: [#003](../003-gap-ratio-corpus/) --
  `don_ratio` median 0.494
  [taiko2_v1/chart_metrics_summary.json:don_ratio.median].
- Kind distribution: corpus-wide estimates from chart_metrics_summary --
  DON 46.8%, KA 47.6%, BIG_DON 2.8%, BIG_KA 2.7%, DRUMROLL 0.1%,
  SPINNER 0.2%.
- Kind storage format: `persistence/events.py` -- `kind_ids` stored as
  uint8 in every `.npz`, loaded by `TaikoDetectionSampler`.

---

## Hypothesis

### Claim

If DON and KA are acoustically indistinguishable at the onset frame
(Fisher LDA separability < 0.01 on the mel spectrum), then the typing
model must operate on **pattern context** (surrounding D/K sequence +
IOI structure), not on per-onset audio classification. Conversely, if
BIG variants show Fisher LDA > 0.05 vs their NORMAL counterparts,
BIG/NORMAL can be classified from audio alone.

### Mechanism

In osu!taiko, DON and KA map to center vs rim drum hits. Chart authors
("mappers") assign D/K to create rhythmic patterns -- alternation
(DKDK), runs (DDDD), and mixed patterns. The musical audio itself
does not distinguish between a pattern that should be D vs K; the
distinction is aesthetic and structural. The 50/50 corpus split and
the DKDD / KDKK symmetry both predict that mel spectra at D onsets
and K onsets will be drawn from the same distribution.

BIG variants (finish sounds in osu! terminology) are typically placed
on strong beats or accents -- musically prominent positions. These
should correlate with higher spectral energy or stronger transients,
producing measurable acoustic separation.

### Predicted numbers

| Metric | Predicted | Notes |
|---|---:|---|
| D vs K Fisher LDA (mel at onset) | < 0.01 | near-zero separability |
| D vs K mean \|mel diff\| per band | < 0.1 dB | indistinguishable |
| BIG vs NORMAL Fisher LDA | > 0.05 | moderate separability |
| BIG vs NORMAL energy diff | > 5% | louder onsets |
| P(K\|D) transition probability | 0.45-0.55 | near-symmetric |
| DKDK 4-gram share of all 4-grams | > 10% | alternation is common |
| Pattern repeat: same D/K rate | < 60% | frequent flipping |
| Pattern repeat: flipped D/K rate | > 20% | confirms symmetry |
| DRUMROLL/SPINNER vs hits Fisher | > 0.1 | clearly different |

## Success criteria

- **Must have:** `summary.json` with all five question blocks (Q1-Q5)
  populated -- mel separability, transition matrices, n-grams, IOI by
  kind, pattern repeat analysis.
- **Must have:** at least 5 graphs rendering without error.
- **Must have:** D vs K Fisher LDA score computed on > 3M onsets each.
- **Nice-to-have:** D vs K Fisher LDA < 0.01 (confirms the symmetry
  hypothesis cleanly).
- **Nice-to-have:** pattern repeat analysis covers > 200 charts.
- **Fails if:** D vs K Fisher LDA > 0.10 -- would mean D/K IS
  acoustically distinguishable and the symmetry hypothesis is wrong.

## Changes from baseline

No baseline -- first corpus analysis of onset kind acoustics.

- New CLI: `cli/analyze_kinds.py`. Reads `taiko2_v1` manifest +
  events + features. Per-onset mel extraction, per-kind aggregation,
  transition analysis, n-gram counting, pattern repeat detection.
- Output under `experiments/023-kind-acoustics/results/`.

## Run config

- Dataset: `taiko2_v1`, split `all` (all 10,048 charts).
- Command:
  ```bash
  uv run --directory osu/taiko2 \
      python -m osu.taiko2.cli.analyze_kinds \
      --dataset taiko2_v1 --split all \
      --output osu/taiko2/experiments/023-kind-acoustics/results
  ```

---
<!-- Everything below written after the run. Do not pre-populate. -->
---

## Results summary

### Final vs baseline

N/A (corpus analysis, not a training run).

### Per-eval progression

N/A (corpus analysis, not a training run).

Machine-readable copy: [`results/summary.json`](./results/summary.json).

## Visualizations

{Post-run}

## Vs prediction

{Post-run}

## Takeaways

{Post-run}

## Followup questions

{Post-run}
