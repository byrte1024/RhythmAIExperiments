# Experiment 003 — corpus gap / ratio shape reference

## Status

`Planned`

## Context

Earlier in the session we added four pairs of shape metrics to
[`ChartMetrics`](../../domain/chart.py): a `gap_*` set over the IOI
histogram, and a parallel `ratio_*` set over the
`gap[i] / gap[i-1]` log2-bucketed distribution. Each set has
**peak_count**, **peak_falloff**, **random_distance** (TVD from
uniform) and **metronome_distance** (TVD from a delta distribution —
anchored at the observed mode for gaps, and anchored at the 1.0× bucket
for ratios).

The metrics were designed with a specific geometry in mind — 200
buckets at 10 ms (gaps) or 0.03 log2 (ratios), ±20 ms / ±0.15 log2
smoothing, 30 ms / 0.30 log2 min separation — but those parameters
were chosen on intuition, not data. Before we start using these
numbers as an evaluation axis for a trained model's output, we need a
**reference distribution**: what does the corpus of ~10 k real human-
made taiko charts look like on each axis? Is the smoothing window
too wide, so we only see 1 peak everywhere? Too narrow, so we see 20
noise peaks on every chart? Are the random-distance and metronome-
distance fields actually spread across their [0, 1] range, or do they
saturate at the extremes?

This is a **data-gathering experiment** — no model is trained. We run
`cli.analyze_charts` over `taiko2_v1`, collect the four + four + two
(peak-list) metrics per chart, and emit reference histograms for each.

## Citations

- Metric design and implementation:
  [`domain/chart.py#ChartMetrics`](../../domain/chart.py) +
  [`cli/analyze_charts.py`](../../cli/analyze_charts.py).
- Baseline per-chart metrics (the fields that existed before these
  shape additions): this experiment inherits the same analysis
  scaffolding as the dataset-wide runs under
  [`analysis/taiko2_v1/`](../../analysis/taiko2_v1/).

---

## Hypothesis

### Claim

If we compute the new gap + ratio shape metrics over every chart in
`taiko2_v1`, the **ratio distribution** will show `1.0×`
overwhelmingly dominating — most charts are metronomic most of the
time, so consecutive-gap ratios cluster hard at 1.0. Specific gap
peak centers (in ms) won't be of much use as a universal reference
because different songs use different tempos, but the **shape** of
the gap histogram — how many peaks each chart has, how steeply they
fall off — will be a useful signal. I expect **1–4 gap peaks** and
**1–3 ratio peaks** per chart on average.

### Mechanism

Human-made taiko charts are written around a BPM: the dominant IOI
is the 16th/8th/quarter note interval, and most of the chart sits at
that interval or at simple halving/doubling variants. A steady 16th-
note passage produces only a 1.0× ratio; a transition from 16ths to
8ths produces a single 2.0× ratio at the seam. Polyrhythmic sections
(drumrolls, triplets) produce 3:2 and 4:3 ratios but are uncommon
and usually short. So:

- `ratio_peak_count` concentrated around 1–3.
- `ratio_metronome_distance` (anchored at 1.0×) **low** for most
  charts — maybe median ≤ 0.2 — because 1.0× holds the lion's share
  of every chart's ratio mass.
- `gap_peak_count` concentrated around 1–4, with higher-star charts
  leaning to the upper end (more rhythm variety).
- `gap_random_distance` very high — always close to 0.995 — because
  no taiko chart has a flat uniform IOI histogram.
- Specific gap-peak-center locations (ms) spread across many bucket
  positions due to per-song BPM variety, so the corpus-wide gap-peak
  histogram won't have sharp modes.

### Predicted numbers

Not much to predict precisely — this is the pass that *establishes*
the reference. The one-liners below are my priors; they're there to
be falsified.

| Metric | Current | Predicted (median across corpus) | Notes |
|---|---:|---:|---|
| `gap_peak_count`   | — | 1–4, median ≈ 2–3 | higher on dense / varied charts |
| `ratio_peak_count` | — | 1–3, median ≈ 1–2 | heavily skewed toward 1 |
| `gap_peak_falloff` | — | 0.3–0.7 when ≥2 peaks | sub-rhythms usually carry 1/2–1/4 of the dominant peak's mass |
| `ratio_peak_falloff` | — | ≤ 0.3 when ≥2 peaks | 1.0× dwarfs everything else |
| `gap_random_distance` | — | > 0.9 for nearly all charts | no chart is uniform |
| `ratio_random_distance` | — | > 0.9 for nearly all charts | ratios cluster at 1.0 |
| `gap_metronome_distance` | — | 0.3–0.7 median | varied but not metronomic |
| `ratio_metronome_distance` | — | 0.1–0.4 median | 1.0× dominance → low distance |

## Success criteria

- **Must have:** `cli.analyze_charts` runs to completion over the
  full `taiko2_v1` corpus without errors.
- **Must have:** all 38 graphs under
  `analysis/taiko2_v1/chart_metrics_graphs/` are produced, with at
  least 200 data points each (loose sanity check that we actually
  processed the corpus).
- **Must have:** `chart_metrics.csv` contains finite, non-NaN
  numbers in all eight new scalar columns for every chart that had
  ≥ 2 gaps.
- **Nice-to-have:** a clearly-skewed `top_ratio_peak` histogram
  (mode at 1.0×) — confirms the 1.0×-dominance intuition.
- **Nice-to-have:** some of the correlation plots (25–38) show `|r|
  < 0.3` — evidence the new metrics carry signal independent of
  existing ones.
- **Fails if:** any of the eight scalar columns comes out constant
  across the corpus — would imply the metric saturates and offers no
  signal.
- **Fails if:** `gap_peak_count` or `ratio_peak_count` routinely
  comes out > 10 — would suggest the smoothing / merge parameters are
  too lax and we're detecting noise peaks.

## Changes from baseline

First observational pass over these metrics; nothing forks from
anything.

## Run config

- Run name: the script writes under
  `analysis/taiko2_v1/chart_metrics/` (per-chart JSON) and
  `analysis/taiko2_v1/chart_metrics_graphs/` (PNGs); both are
  re-generated on each run.
- Dataset: `taiko2_v1`, full (`--split all`, no subsample).
- Command:
  ```bash
  osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.analyze_charts \
      --dataset taiko2_v1
  ```

─────────────────────────────────────────────────────────────────────
<!-- Everything below written after the run. Do not pre-populate. -->
─────────────────────────────────────────────────────────────────────

## Results summary

### Per-metric summary table

| Metric | min | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| gap_peak_count   | — | — | — | — | — | — |
| gap_peak_falloff | — | — | — | — | — | — |
| gap_random_distance | — | — | — | — | — | — |
| gap_metronome_distance | — | — | — | — | — | — |
| ratio_peak_count   | — | — | — | — | — | — |
| ratio_peak_falloff | — | — | — | — | — | — |
| ratio_random_distance | — | — | — | — | — | — |
| ratio_metronome_distance | — | — | — | — | — | — |

Machine-readable copy: [`metrics.json`](./metrics.json).

### Correlation highlights

Top correlations between new and existing metrics (Pearson `|r|`
ranked):

| New metric | Existing metric | r | Interpretation |
|---|---|---:|---|
| — | — | — | — |

## Visualizations

{Copy the most informative PNGs from
`analysis/taiko2_v1/chart_metrics_graphs/` into `./graphs/` after the
run, one-liner captions on each.}

![](graphs/01_ratio_peak_count.png)
*Ratio peak count distribution across the corpus.*

![](graphs/02_top_ratio_peak.png)
*Where the #1 ratio peak lands (log2 axis).*

![](graphs/03_gap_peak_count.png)
*Gap peak count distribution.*

![](graphs/04_gap_peak_histogram.png)
*Corpus-wide gap-peak bucket occupancy.*

## Vs prediction

- `gap_peak_count` predicted 1–4: actual range `—`, median `—` → **—**
- `ratio_peak_count` predicted 1–3: actual range `—`, median `—` → **—**
- `ratio_metronome_distance` predicted median ≤ 0.2: actual `—` → **—**
- `gap_random_distance` predicted > 0.9 for most: actual fraction `—` → **—**

## Takeaways

- {Concrete sentence after the run.}

## Followup questions

- If a metric saturates (most charts at 0 or 1), do we need to
  re-tune its bucket geometry? — investigate in a follow-up analysis
  pass.
- Which of the eight metrics correlate strongest with star rating?
  Those might substitute for `over_pspace_self` in difficulty
  modeling — exp TBD.
- Do the new ratio metrics correlate with the new ratio-hit /
  metronome artifacts we plot per eval during training? Cross-
  reference with [#002](../002-exp45-full/)'s per-eval numbers.
