# Experiment 003 — corpus gap / ratio shape reference · Architecture

This is a data-gathering pass. No model trains. "Architecture" here
means the exact script, inputs, metric definitions, and outputs
needed to reproduce the run from scratch.

## Pipeline

```
dataset on disk  →  TaikoDetectionSampler.get_chart(i)
                 →  Chart.calculate_metrics()
                 →  per-chart JSON  +  flat CSV row  +  aggregates
                 →  summary JSON + 38 PNG graphs
```

All data flow lives in `osu/taiko2/cli/analyze_charts.py`.

## Dataset

| Field | Value |
|---|---|
| Name | `taiko2_v1` |
| Root | `osu/taiko2/datasets/taiko2_v1/` |
| Charts | 10 048 (same as #001/#002 train + val set combined) |
| Split | `all` (every chart analyzed, train + val together) |
| `allowed_overlap_forward` / `allowed_overlap_back` | 0 (taiko1 parity; analysis isn't affected by overlap filtering) |
| `min_cursor_bin` | 0 |

No audio is decoded during analysis; only the `Track.onsets` sequence
per chart is used.

## Per-chart metrics computed

`Chart.calculate_metrics()` returns a `ChartMetrics` dataclass with
every field documented at
`osu/taiko2/domain/chart.py:ChartMetrics`. The pre-existing fields are
out of scope for this experiment — they were already established by
prior analysis passes under `analysis/taiko2_v1/`. The new-in-this-
experiment fields are the eight below.

### Gap-distribution shape (IOI histogram)

Build a dense histogram `h` over the IOI-per-sample sequence:

- Support: `[0, 2000)` ms, open at the high end.
- Bucket width: 10 ms.
- Number of buckets: 200.
- Gaps ≥ 2000 ms are dropped (they already live in
  `long_gap_count`).
- Gaps are collected from `np.diff(onset.time_ms)` restricted to
  positive values (drop duplicates / out-of-order pairs).

Peak detection on `h`:

- **Smoothing radius** `R = 2` buckets (±20 ms). Bucket `i` is a
  local max iff `h[i] >= h[j]` for `j ∈ [max(0, i−R), min(n, i+R+1))`.
- **Height threshold** `T = max(h.max() × 0.05, 3)`. A candidate
  must clear this (5 % of global max, or 3 events absolute,
  whichever is larger).
- **Plateau resolution**: in a run of equal-count buckets at the
  local-max value, only the leftmost index is kept.
- **Merge pass**: candidates are iterated in descending `h[i]`
  order. A candidate is discarded if it sits within `D = 3` buckets
  (30 ms) of an already-kept peak.

Metrics derived from `h` and the kept peak indices `P`:

| Field | Definition |
|---|---|
| `gap_peak_count` | `len(P)` |
| `gap_peak_falloff` | If `\|P\| >= 2`: mean of `c_{i+1} / c_i` over the peak counts sorted descending (`c_i = h[P_i]`). Otherwise 0.0. |
| `gap_random_distance` | `0.5 × Σ \|h[i]/total − 1/200\|` — total variation distance from the uniform distribution over the 200-bucket support. |
| `gap_metronome_distance` | `1 − max(h) / total` — TVD from a delta distribution anchored at the observed mode. |
| `gap_peaks` | Tuple of `(bucket_center_ms, count)` over every kept peak, sorted by count descending. `bucket_center_ms = i × 10 + 5`. |

### Ratio-distribution shape

Consecutive-gap ratios `r[i] = gaps[i] / gaps[i-1]` are bucketed in
log2 space:

- Support: `log2(r) ∈ [−3, +3)` (linear ratios `[0.125×, 8×)`).
- Bucket width: `0.03` in log2 (~2.1 % ratio per bucket).
- Number of buckets: 200.
- Ratios outside the support are dropped.
- At least 2 in-range ratios required — otherwise all ratio metrics
  return 0.

Peak detection uses the same rule structure as the gap histogram but
wider windows — musically "1.01× and 0.95× are the same thing" while
"0.67× and 1.33× are distinct rhythmic categories":

| Parameter | Value | Interpretation |
|---|---|---|
| Smoothing radius `R` | 5 buckets | ±0.15 log2 ≈ ±10.5 % ratio |
| Height threshold `T` | `max(0.05 × h.max(), 3)` | same as gaps |
| Min separation `D` | 10 buckets | ±0.30 log2 ≈ ±23 % ratio |

Metrics:

| Field | Definition |
|---|---|
| `ratio_peak_count` | `len(P)` |
| `ratio_peak_falloff` | same formula as gaps |
| `ratio_random_distance` | TVD from uniform over the 200-bucket support |
| `ratio_metronome_distance` | `1 − h[one_bucket] / total`, where `one_bucket = int((0 − (−3)) / 0.03) = 100` is the 1.0× bucket. **Anchored at 1.0× regardless of observed mode** — "how far is this chart's ratio distribution from one where every consecutive pair of gaps has identical length." |
| `ratio_peaks` | Tuple of `(bucket_center_ratio, count)` over every kept peak (ratios are linear, already exponentiated from log2), sorted by count descending. |

## Corpus aggregates (computed inline by `analyze_charts`)

- `aggregated_ioi`: per-bucket count summed over every chart's
  `ioi_histogram_10ms`.
- `gap_peak_hist`: integer array of shape `(200,)`; each kept gap
  peak from every chart increments the bucket it landed in.
- `ratio_peak_hist`: same for ratio peaks over the log2 support.
- `top_gap_peak_ms`: list with one entry per chart = center_ms of
  that chart's #1 peak.
- `top_ratio_peak`: list with one entry per chart = linear ratio of
  that chart's #1 peak.

## Outputs

Written atomically under `osu/taiko2/analysis/taiko2_v1/`:

```
chart_metrics/
    <chart_id>.json          # one per chart, full ChartMetrics
                             # including gap_peaks and ratio_peaks
                             # tuples
chart_metrics.csv            # flat scalar summary; the eight new
                             # shape scalars are columns alongside
                             # the pre-existing set.
chart_metrics_summary.json   # min / p25 / median / p75 / p95 / max
                             # per numeric column
chart_metrics_graphs/
    01..12 — existing metric graphs (BPM, streaks, density, etc.)
    13_gap_peak_count.png
    14_gap_peak_falloff.png
    15_gap_random_distance.png
    16_gap_metronome_distance.png
    17_ratio_peak_count.png
    18_ratio_peak_falloff.png
    19_ratio_random_distance.png
    20_ratio_metronome_distance.png
    21_top_gap_peak_ms.png       # #1 gap-peak location per chart
    22_top_ratio_peak.png        # #1 ratio-peak location (log2)
    23_gap_peak_histogram.png    # every kept gap peak, corpus-wide
    24_ratio_peak_histogram.png  # same for ratios
    25..32 — each of the 8 new metrics vs star rating
    33, 34 — streak_event_fraction vs gap/ratio metronome_distance
    35     — over_pspace_self vs ratio_peak_count
    36     — density_mean vs gap_peak_falloff
    37     — gap_peak_count vs ratio_peak_count
    38     — gap_metronome_distance vs ratio_metronome_distance
```

Each scatter title carries Pearson `r` so redundancy vs novelty is
visible at a glance.

## Environment

| Component | Version |
|---|---|
| Python | 3.13.13 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.3 |

## Addenda

_(None before the run.)_
