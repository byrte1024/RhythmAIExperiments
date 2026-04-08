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

*Pending*

## Lesson

*Pending*
