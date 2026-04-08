# Experiment 63 - TaikoNation Direct Comparison

## Purpose

Run TaikoNation (Halina & Guzdial, FDG 2021) with its original pretrained weights on the same 30 val songs used in our experiments (59-H, 60, 61). Direct apples-to-apples comparison instead of comparing against published numbers on different songs.

## Setup

TaikoNation requires Python 3.7 + TensorFlow 1.15 + TFLearn. A separate venv is set up in `taikonation_env/`:

```
taikonation_env/
  python37/          # Python 3.7.9 local install
  venv37/            # venv with TF 1.15, tflearn, librosa
```

Pretrained weights extracted from the TaikoNation repo to `external/TaikoNationV1/extracted_model/`.

### Launch

```bash
cd osu/taiko
experiments/experiment_63/taikonation_env/venv37/Scripts/python.exe experiments/experiment_63/run_taikonation.py
```

## Method

1. Load TaikoNation's exact architecture and pretrained weights
2. For each of 30 val songs:
   - Extract mel features (80 bands, 23ms windows, normalized per band)
   - Run sliding window inference (16-frame context, 4-step output)
   - Average overlapping predictions, sample from distribution
   - Post-process (remove double positives < 23ms apart)
3. Compare predicted onsets to ground truth using our standard metrics
4. Compare to our models' results from exp 59-HB

### Key differences from our approach

| Aspect | TaikoNation | BeatDetector (exp58) |
|---|---|---|
| Resolution | 23ms | 5ms |
| Audio window | 368ms (16 × 23ms) | 5000ms (1000 × 5ms) |
| Context | Previous 4 predictions | 128 past events with gap/ratio embeddings |
| Output | 4 × 7-class (note types) | 251-class (bin offset) or 4 × 251 (exp62) |
| Note types | Predicted (don/kat/big/roll/denden) | Not predicted (onset timing only) |
| Density control | None | FiLM conditioning |
| Dataset | ~100 curated charts | 10,048 charts |

For this comparison we only measure onset timing accuracy (ignoring note types).

## Result

30 val songs, same set as exp 59-H/60/61.

### TaikoNation actual vs our models vs DDC:

| Model | Close% | Far% | Hall% | d_ratio | err_med | P-Space | HI-PS | DCHuman |
|---|---|---|---|---|---|---|---|---|
| **exp58 (ours)** | **75.9%** | **16.6%** | **15.6%** | **0.92** | **8ms** | 10.1% | 81.1% | **90.8%** |
| DDC Oracle | 77.1% | 14.8% | 19.9% | 1.00 | 27ms | — | — | — |
| TaikoNation | 10.2% | 81.5% | 50.9% | 0.39 | 399ms | **34.8%** | **92.9%** | 86.4% |

### TaikoNation actual vs paper:

| Metric | Actual (our songs) | Paper (their songs) |
|---|---|---|
| Over. P-Space | 34.8% | 21.3% |
| HI P-Space | 92.9% | 94.1% |
| DCHuman | 86.4% | 75.0% |

### Key findings:

1. **TaikoNation fails on onset timing for our songs.** 10.2% close rate, 399ms median error, 81.5% far rate. Only 39% of expected density. Half its predictions are hallucinations.

2. **Our models are ~7.5x better on close rate** (75.9% vs 10.2%) and **~50x better on timing precision** (8ms vs 399ms).

3. **DCHuman is misleadingly high for TaikoNation** (86.4%) because most 23ms timesteps are "no note" — and both model and GT agree on "no note" for those. The metric doesn't penalize missing actual onsets enough.

4. **P-Space is high (34.8%) because of near-random placement** on unfamiliar songs. High pattern diversity from a model that can't place notes correctly is not meaningful.

5. **The paper's results (21.3% P-Space, 75.0% DCHuman) were on their curated dataset.** On arbitrary val songs, the model doesn't generalize.

### Important caveats:

- **Dataset mismatch is the primary factor.** TaikoNation was trained on ~100 curated high-difficulty charts. Our val songs span all difficulties and genres. The model was never designed to generalize to arbitrary songs.
- **TaikoNation may produce higher quality charts on songs similar to its training data.** The 10.2% close rate could partly reflect different charting styles (high difficulty vs mixed difficulty) rather than pure failure.
- **TaikoNation predicts note types (don/kat/big/etc.)** which we ignore in this comparison. Its patterning strength is in note type sequences, not onset timing.
- **A fair comparison would require running TaikoNation on its own training distribution** or running our model on their curated dataset — pending Emily Halina's response about their evaluation songs.

## Lesson

1. **TaikoNation doesn't generalize to arbitrary songs.** A small curated dataset + lightweight LSTM produces charts that are well-patterned within its training distribution but fail on unseen songs. Our 10K-chart dataset with a larger transformer generalizes much better.

2. **DCHuman is a weak metric for sparse events.** When 95%+ of timesteps are "no note", a model that predicts mostly "no note" gets high DCHuman regardless of onset accuracy. Our close/far/hallucination metrics are more informative.

3. **Pattern diversity without placement accuracy is meaningless.** TaikoNation's 34.8% P-Space is the highest we've seen, but with 81.5% of notes in the wrong place, it's diversity from confusion, not creativity.

4. **Future AR evaluation should include TaikoNation, DDC, and human GT** as baselines alongside our best model. Human judges would reveal whether TaikoNation's note-type patterning compensates for poor timing, or if timing accuracy dominates perception.
