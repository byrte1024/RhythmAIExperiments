# Experiment 63b — TaikoNation Apples-to-Apples Comparison

## Purpose

Exp 63 ran TaikoNation on our val songs (10.2% close, 399ms error). But they used different songs for their published results (DCHuman 75.0%). This isn't a fair comparison — their model may work better on their songs, ours on ours.

Emily Halina (TaikoNation author) shared their exact 10 evaluation beatmaps. Now we can run our models on the SAME charts and compare directly.

## TaikoNation Evaluation Charts

| BeatmapSet | Beatmap ID | Notes |
|---|---|---|
| 1116903 | 2333205 | |
| 202040 | 491412 | |
| 304553 | 682296 | |
| 535179 | 1133589 | |
| 562299 | 1574370 | |
| 565959 | 1200360 | |
| 724269 | 1529288 | |
| 748803 | 1580130 | |
| 821700 | 1722197 | |
| 821745 | 1722264 | |

## Method

1. Download all 10 charts (audio + GT onsets)
2. Run our best models (exp58, S3) on these exact songs
3. Compute TaikoNation metrics (Over. P-Space, HI P-Space, DCHuman, OCHuman, DCRand)
4. Compute GT matching metrics (close%, hallucination%, etc.)
5. Compare directly to TaikoNation's published numbers

## TaikoNation Published Results (on these charts)

| Model | Over. P-Space | HI P-Space | DCHuman | DCRand |
|---|---|---|---|---|
| TaikoNation | 21.3% | 94.1% | 75.0% | 50.4% |
| DDC | 15.9% | 83.2% | 77.9% | 49.9% |
| Human Taiko | 14.5% | — | — | 50.2% |

## Launch

```bash
cd osu/taiko
python experiments/experiment_63b/download_taikonation_charts.py
python experiments/experiment_63b/run_comparison.py
```

## Status: RUNNING

Charts downloaded, inference running on exp14, exp44, exp45, exp58.

## Expectations

We expect:
1. **Our Human GT Over. P-Space should be ~14.5%** — matching TaikoNation's published 14.453%. This validates we're computing the metric correctly.
2. **Our DCRand should be ~50%** — both theirs and ours should be near 50% (random baseline).
3. **Our DCHuman should be significantly higher than TaikoNation's 75%** — exp58 gets 90.8% on our val songs. Even on their songs, we expect 85%+.
4. **Our Over. P-Space will be lower than TaikoNation's 21.3%** — our models tend toward 10-12%, closer to human GT (14.5%). TaikoNation's high P-Space may indicate over-diverse patterns.
5. **Our close rate / timing will be much better** — TaikoNation got 10.2% close on our songs (exp63). Our models should be 60%+ on any songs.
6. **exp58 should be the best of our models** — it's our ATH per-sample model.

The key question: does TaikoNation perform better on its own evaluation songs than it did on ours (10.2% close)? If so, by how much?

## Result

*Pending*

## Lesson

*Pending*
