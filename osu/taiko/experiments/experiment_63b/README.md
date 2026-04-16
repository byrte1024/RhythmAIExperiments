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

### Our Models on TaikoNation's Charts

| Model | Close% | Far% | Hall% | d_ratio | Err Med | Over.PS | HI-PS | DCHuman |
|---|---|---|---|---|---|---|---|---|
| **exp58** | **79.6%** | **7.4%** | **8.5%** | **0.92** | **11ms** | 15.5% | 87.8% | 84.2% |
| exp45 | 75.7% | 7.7% | 9.4% | 0.88 | 12ms | 15.4% | 90.2% | 83.7% |
| exp44 | 74.2% | 10.3% | 7.6% | 0.83 | 11ms | 13.8% | 82.0% | 85.5% |
| exp14 | 71.9% | 13.2% | 9.4% | 0.83 | 10ms | 17.1% | 88.1% | 85.8% |

### TaikoNation on Its Own Charts (our run)

| Metric | Our run | Published | Gap |
|---|---|---|---|
| Over. P-Space | 41.3% | 21.3% | +20pp |
| HI P-Space | 89.7% | 94.1% | -4.4pp |
| DCHuman | 81.9% | 75.0% | +6.9pp |
| DCRand | 49.6% | 50.4% | -0.8pp |
| Close (<50ms) | 9.8% | — | — |
| Hallucination | 46.5% | — | — |
| Error median | 413ms | — | — |
| Density ratio | 0.25 | — | — |

### Human GT Self-Metrics (validation)

| | Our measurement | Their published |
|---|---|---|
| Over. P-Space | 13.8% | 14.5% |
| DCRand | 50.1% | 50.2% |

Within 0.7pp — confirms our metric computation is correct.

### Caveat: TaikoNation Reproduction Likely Incorrect

The dramatic discrepancy between our TaikoNation run and their published results (P-Space 41.3% vs 21.3%, density ratio 0.25) strongly suggests we are running their model incorrectly. Possible causes:

1. **Wrong checkpoint** — their repository may contain multiple checkpoints; we may have the wrong one
2. **Preprocessing mismatch** — they use essentia for feature extraction, we use librosa. Normalization, hop size, or frequency range differences could produce incompatible features
3. **Architecture reconstruction error** — we rebuilt their TFLearn model from paper/code; a small difference in layer connections would produce degraded output
4. **Inference logic mismatch** — how we feed context back (note queue) or decode the 4-step output may not match their implementation

The 0.25 density ratio (287 predicted events vs 1082 GT) is the smoking gun — the model is dramatically under-predicting, suggesting it's not receiving features in the format it expects.

**Our model results on these charts are valid** — they use our own pipeline end-to-end. The TaikoNation comparison should be treated as approximate until reproduction issues are resolved.

## Lesson

1. **Our models dominate on TaikoNation's own charts.** exp58 achieves 79.6% close rate, 11ms timing, 8.5% hallucination on these songs. This is consistent with our val set results — our models generalize.

2. **TaikoNation's model doesn't work in our reproduction.** 9.8% close, 413ms error, same as on our songs (exp63). This is likely a reproduction issue, not necessarily TaikoNation's actual performance.

3. **Our metric computation is validated.** Human GT P-Space matches their published value within 0.7pp. The metrics are correct; the model reproduction is the problem.

4. **exp58 beats TaikoNation's published DCHuman.** Even comparing our 84.2% to their published 75.0% (giving them the benefit of correct reproduction), we win by +9.2pp on their metric, on their songs.

5. **Pattern diversity is comparable.** Our models' P-Space (13.8-17.1%) is in the same range as Human GT (13.8%) and TaikoNation's published (21.3%). We're closer to human patterns; their high diversity may indicate noise.
