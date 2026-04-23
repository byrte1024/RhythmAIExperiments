# Experiment 004 — Engagement × chart-metric corpus · Architecture

Self-contained spec for the engagement fetch + correlation analysis
pass. No model training, no inference. Reproducible from this file
plus the taiko2 codebase.

## Goal

1. Fetch popularity / engagement metadata for every chart in the
   `taiko2_v1` dataset from osu! API v2.
2. Cross-correlate those engagement scalars against the intrinsic
   chart metrics already produced by `cli/analyze_charts` (from
   experiment #003).
3. Rank pairs by `|r|`, render the strongest scatters, and dump the
   full correlation matrix to JSON + CSV.

## Inputs

| Artifact | Path | Produced by |
|---|---|---|
| Dataset manifest | `datasets/taiko2_v1/manifest.json` | `cli/prepare_dataset` |
| Chart metrics CSV | `analysis/taiko2_v1/chart_metrics.csv` | `cli/analyze_charts` (exp #003) |
| OAuth credentials | env `OSU_CLIENT_ID` / `OSU_CLIENT_SECRET`, or `osu/taiko2/.env`, or `osu/taiko2/secrets.json` | user-supplied |

## Pipeline

### Stage 1 — `cli/fetch_engagement`

1. Load `manifest.json`.
2. Extract `beatmap_id` for every chart.
3. Dedupe and sort — `/api/v2/beatmaps?ids[]=...` returns one
   payload per id, so repeats waste quota.
4. Batch 50 ids at a time (the API's hard limit) via
   `fetch.client.OsuV2Client.get_beatmaps`. Client-credentials
   OAuth token is cached in-process; one token per run.
5. Parse each beatmap response into an `EngagementRow` dataclass:

   Per-difficulty:
     - `playcount`, `passcount` → derived `pass_rate = passcount / playcount`
     - `status`, `last_updated`

   Per-beatmapset (copied onto every diff's row):
     - `play_count_set`, `favourite_count`
     - `ratings` (11-element 0..10 bucket list) → derived
       `rating_mean = Σ(i * ratings[i]) / Σ(ratings[i])`,
       `rating_count = Σ(ratings[i])`
     - `bpm`, `genre.name`, `language.name`, `user_id`,
       `nominations_summary.current`, `nsfw`

6. Write two sidecars next to the manifest (does NOT modify the
   manifest itself):
   - `manifest_engagement.json` — structured, keeps the raw 11-
     element `rating_buckets` list.
   - `manifest_engagement.csv` — flat scalars only (drops
     `rating_buckets`) for easy pandas / spreadsheet ingest.

### Stage 2 — `cli/analyze_engagement`

1. Load `chart_metrics.csv` (n_charts rows) and
   `manifest_engagement.csv` (m rows).
2. Inner-join by `beatmap_id`. Report how many chart rows had no
   engagement match (usually: graveyarded / deleted beatmaps).
3. Compute summary stats per engagement scalar: `n`, `min`, `p25`,
   `median`, `p75`, `p95`, `max`, `mean`. Write to
   `engagement_summary.json`.
4. Also write top-20 value-counts for categorical fields (`status`,
   `genre`, `language`) to the same summary.
5. Compute the full correlation matrix:
   - For each of 9 engagement scalar fields × 32 chart scalar fields:
     - Drop pairs where either side is blank, NaN, or non-numeric.
     - Skip if joint sample count < 50 (too small to trust).
     - Skip if either side has zero std (degenerate column).
     - Compute Pearson `r = corrcoef(x, y)[0, 1]`.
   - Collect all pairs as `{engagement_field, chart_field, n, r,
     abs_r}` records.
   - Sort by `abs_r` descending.
   - Write to `correlations.json` (pretty) and
     `correlations_ranked.csv` (flat).
6. Render the top-K (default K=24) scatter plots to
   `engagement/graphs/{NN}_{e_field}__vs__{c_field}.png`. Plots use
   log-scale x when the engagement side is a play / pass / favourite
   / rating count (all three orders of magnitude on typical
   distributions).
7. Render one overview bar chart per engagement metric showing that
   metric's 10 highest-|r| pairs with chart metrics. File:
   `_overview_{e_field}.png`. Blue = positive r, red = negative.

## Fields catalog

### Engagement scalars (9 correlated, 3 categorical)

| Field | Type | Source | Notes |
|---|---|---|---|
| `playcount`            | int   | beatmap            | per-difficulty plays |
| `passcount`            | int   | beatmap            | per-difficulty completions |
| `pass_rate`            | float | derived            | passcount / playcount |
| `play_count_set`       | int   | beatmapset         | set-aggregate plays |
| `favourite_count`      | int   | beatmapset         | user favourites on the set |
| `rating_mean`          | float | derived (ratings) | Σ(i * r[i]) / Σ(r[i]) |
| `rating_count`         | int   | derived (ratings) | Σ(r[i]) |
| `bpm_set`              | float | beatmapset         | mapper-declared BPM |
| `nominations_current`  | int   | beatmapset         | mod nomination progress |
| `status`               | str   | beatmap            | ranked / loved / graveyard / …  (categorical) |
| `genre`                | str   | beatmapset.genre   | "Anime" / "Pop" / … (categorical) |
| `language`             | str   | beatmapset.language| "Japanese" / … (categorical) |

### Chart scalars correlated against (32 total)

Sourced from `chart_metrics.csv` — the flat scalar subset of
`ChartMetrics`. Covers:

- Basic: `total_events`, `duration_s`, `events_per_sec`
- Type breakdown: `don_ratio`
- IOI: `ioi_{mean,median,std,p95,p99}_ms`, `short_ioi_pct`, `long_gap_count`
- Density: `density_{mean,peak,std,cv}`
- Streaks: `longest_streak`, `mean_streak_len`, `streak_event_fraction`
- BPM: `estimated_bpm`, `dominant_ioi_ms`
- Pattern-space: `over_pspace_self`
- Gap shape (from #003): `gap_peak_count`, `gap_peak_mass_total`,
  `gap_peak_falloff`, `gap_random_distance`, `gap_metronome_distance`
- Ratio shape (from #003): `ratio_peak_count`,
  `ratio_peak_mass_total`, `ratio_peak_falloff`,
  `ratio_random_distance`, `ratio_metronome_distance`
- External: `star_rating`

## Output layout

```
osu/taiko2/analysis/taiko2_v1/engagement/
    engagement_summary.json      # scalar percentiles + categorical top-20
    correlations.json            # full matrix, sortable JSON
    correlations_ranked.csv      # flat, sorted by |r| desc
    graphs/
        01_{e}__vs__{c}.png      # top-K scatters (K=24)
        02_...
        …
        _overview_playcount.png  # per-engagement bar chart
        _overview_favourite_count.png
        _overview_rating_mean.png
        _overview_pass_rate.png
        …
```

The experiment folder copies the specific graphs referenced in its
Results section into its own `graphs/` folder post-run.

## Credentials

- Read via `osu.taiko2.credentials.require(name)` — it tries env
  vars, then `osu/taiko2/.env`, then `osu/taiko2/secrets.json`.
- Never passed as CLI arguments.
- Only `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` are needed (OAuth
  client-credentials flow). No user-scoped token.

## Rate limiting

- API v2 limits unauthenticated traffic to 60 req/min; OAuth client-
  credentials gets higher limits but not unbounded.
- Batching 50 ids/call reduces the full 10,048-chart corpus to
  ~201 API calls. At ~1 call/sec this is ~3.5 minutes.
- On HTTP 429 the shared client sleeps 60 s and retries once
  (`fetch/client.py`). Failures past that are logged and the batch
  is skipped — ids are just missing from the result.

## Environment

| Component | Version |
|---|---|
| Python    | 3.13.13 |
| numpy     | 2.4.2 |
| matplotlib| 3.10.8 |
| requests  | 2.32.5 |

No GPU. Full corpus pass takes ~5 minutes (API-bound).

## Addenda

_(None before the run.)_
