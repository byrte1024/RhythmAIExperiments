# Experiment 003 — Corpus gap / ratio shape reference · Architecture

Self-contained spec for the corpus analysis pass. No model training,
no inference. Everything below is reproducible from this file alone
plus the taiko2 codebase.

## Goal

Run `Chart.calculate_metrics()` over every chart in the `taiko2_v1`
dataset and aggregate the new gap-distribution and ratio-distribution
shape metrics into corpus-level graphs + summary statistics.

## Inputs

- Dataset: `taiko2_v1` under `osu/taiko2/datasets/taiko2_v1/`.
- Built from 10,048 osu!taiko charts; manifest + event / feature
  files already cached on disk. No re-parsing needed.

## Pipeline

1. `TaikoDetectionSampler` is constructed with
   `split="all", min_cursor_bin=0, allowed_overlap_forward=0,
   allowed_overlap_back=0`. The overlap filter being off is
   irrelevant here — we're iterating charts, not samples.
2. For each chart `i` in range `count_charts()`:
   a. `chart = sampler.get_chart(i)` reconstructs the `Chart` from
      cached events + track metadata. Audio bytes are not needed.
   b. `metrics = chart.calculate_metrics()` computes the full
      `ChartMetrics` dataclass.
   c. `metrics` is serialised to `analysis/taiko2_v1/chart_metrics/
      {safe_chart_id}.json`.
   d. A scalar subset is appended to an in-memory CSV row.
   e. The chart's `gap_peaks` / `ratio_peaks` tuples fold into four
      corpus aggregators (raw + mass, both distributions) keyed by
      200-bucket support.
   f. The chart's `ioi_histogram_10ms` folds into a separate
      corpus-wide IOI aggregator.
3. After all charts:
   - Write `chart_metrics.csv` (flat scalar summary).
   - Write `chart_metrics_summary.json` (per-field min / p25 /
     median / p75 / p95 / max).
   - Write 40+ graphs to `chart_metrics_graphs/`.

## ChartMetrics shape metrics (exact definitions)

### Gap histogram

Dense 200-bucket histogram over `[0, 2000) ms` at 10 ms resolution.
Built from positive inter-onset gaps; gaps ≥ 2000 ms are dropped.

Peak detection: bucket `i` is a peak if:
- `h[i] >= h[j]` for all `j ∈ [i-2, i+2]` (smoothing radius 2).
- `h[i] >= max(0.05 * h.max(), 3)` (height threshold).
- In a plateau run of equal-count buckets, only the leftmost is kept.
- After that, a greedy merge pass walks kept candidates in
  descending count and drops any within 3 buckets (30 ms) of an
  already-kept stronger peak.

Metrics derived from the kept peak set:
- `gap_peak_count` — number of kept peaks.
- `gap_peak_falloff` — mean `c_{i+1} / c_i` of sorted-desc peak
  counts; `0.0` when fewer than 2 peaks.
- `gap_random_distance` — TVD between the normalised histogram and a
  uniform distribution over all 200 buckets.
- `gap_metronome_distance` — TVD from a delta-at-mode reference
  distribution = `1 - mode_share`.
- `gap_peak_mass_total` — sum of counts across all kept peaks.
- `gap_peaks` — per-peak tuple `(bucket_center_ms, count)` sorted by
  count descending.

### Ratio histogram

For consecutive positive gaps `g[i-1], g[i]`, form
`ratio = g[i] / g[i-1]`. Take `log2(ratio)` and bucket across the
fixed support `[-3, +3] log2 = [0.125×, 8×]` into 200 buckets of
width 0.03 log2 each.

Peak detection rules: same shape as the gap histogram except
**wider** smoothing / merge radii:
- smoothing radius 5 buckets (±0.15 log2 ≈ ±10.5 % ratio)
- min separation 10 buckets (±0.30 log2 ≈ ±23 % ratio)

This ensures ratios 1.01 and 0.95 collapse to one peak at 1.0× while
distinct rhythmic categories (0.67×, 1.0×, 1.33×) stay separate.

Metrics derived:
- `ratio_peak_count`, `ratio_peak_falloff`, `ratio_random_distance`
  — same formulas as the gap versions.
- `ratio_metronome_distance` = `1 - h[bucket_at_1.0x] / total`.
  Anchored at the 1.0× bucket, **not** at the observed mode: a
  chart whose ratios are all 2.0× has
  `ratio_metronome_distance = 1.0`. This encodes the specific
  question "how far is this chart from pure metronome continuation?".
- `ratio_peak_mass_total` — sum of counts across all kept peaks.
- `ratio_peaks` — per-peak tuple `(bucket_center_ratio, count)` in
  linear ratio units, sorted by count descending.

## Corpus aggregators

Four parallel arrays accumulate over every chart's kept peak list.
All are 200-element int64 arrays keyed by the same bucket geometry
as the per-chart histograms.

| Aggregator | Contribution per peak |
|---|---|
| `gap_peak_raw_hist[b]`   | +1 per peak occurrence at bucket `b` |
| `gap_peak_mass_hist[b]`  | +`count` per peak occurrence at bucket `b` |
| `ratio_peak_raw_hist[b]` | +1 per peak occurrence at bucket `b` |
| `ratio_peak_mass_hist[b]`| +`count` per peak occurrence at bucket `b` |

The "raw" vs "mass" distinction matters because one chart with a
1000-gap dominant peak and zero others looks identical under "raw"
to a chart with a 3-gap peak at the same bucket; under "mass" the
first contributes 1000, the second contributes 3.

## Graphs produced

40+ PNGs. The ones this experiment specifically reads:

| # | Graph | Reads which prediction |
|---|---|---|
| 13–20 | Per-metric distribution histograms (8 total)            | P6, P7 |
| 21 | `top_gap_peak_ms` — dominant IOI per chart                 | — |
| 22 | `top_ratio_peak` — dominant ratio per chart, log2 axis     | P1 |
| 23 | `gap_peak_raw`, corpus-wide                                | — |
| 24 | `gap_peak_mass`, corpus-wide (log y)                       | — |
| 25 | `ratio_peak_raw`, corpus-wide                              | — |
| 26 | `ratio_peak_mass`, corpus-wide (log y)                     | P2, P3, P4, P5 |
| 27–34 | Correlations vs star rating                            | — |
| 35–36 | Correlations vs streak-event fraction                  | P8, P9 |
| 37 | pspace vs ratio_peak_count                                 | — |
| 38 | density_mean vs gap_peak_falloff                           | — |
| 39 | gap_peak_count vs ratio_peak_count                         | — |
| 40 | gap_metronome_distance vs ratio_metronome_distance         | — |
| 41 | gap: raw vs mass                                           | — |
| 42 | ratio: raw vs mass                                         | — |

P2, P3, P4, P5 are read off graph 26 (`ratio_peak_mass`) by looking
at the bar heights at specific log2 positions:
- 1.0x → bucket at log2 = 0 → index 100
- 0.5x / 2.0x → log2 = ±1 → indices 66 / 133
- 0.25x / 4.0x → log2 = ±2 → indices 33 / 166
- 1/3 / 3x → log2 = ±log2(3) ≈ ±1.585 → indices ≈ 47 / 152
- 1/5 / 5x → log2 = ±log2(5) ≈ ±2.322 → indices ≈ 22 / 177

## Environment

Same as the training runs — see the shared
`osu/taiko2/pyproject.toml` pin. Relevant for this experiment:

| Component | Version |
|---|---|
| Python    | 3.13.13 |
| numpy     | 2.4.2 |
| matplotlib| 3.10.8 |

No GPU required. Runtime is dominated by I/O over the cached chart
manifest; the full corpus analysis should complete in under 10
minutes.

## Output layout

```
osu/taiko2/analysis/taiko2_v1/
    chart_metrics/                    # per-chart full JSON (10,048 files)
    chart_metrics.csv                 # flat scalar summary
    chart_metrics_summary.json        # per-field min/p25/median/p75/p95/max
    chart_metrics_graphs/             # 40+ PNGs
```

The experiment folder (`experiments/003-gap-ratio-corpus/`) copies
the specific graphs referenced in the results table into its own
`graphs/` folder post-run.

## Addenda

_(None before the run.)_
