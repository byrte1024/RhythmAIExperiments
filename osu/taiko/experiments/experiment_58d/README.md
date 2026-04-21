# Experiment 58d — Difficulty-Filtered Training (Stars 4-6)

## Hypothesis

Our models train on ALL difficulty levels (Kantan through Inner Oni). Kantan/Futsuu charts follow simple "hit every strong beat" conventions — they teach the model metronomic behavior. Oni+ charts use complex patterns, syncopation, streams — the actual charting conventions we want to learn.

By training ONLY on stars 4-6 (Oni to Inner Oni), we remove the metronomic training signal. The model should learn more complex pattern structures instead of defaulting to "repeat the beat."

## Data

| Full dataset | Stars 4-6 subset |
|---|---|
| 10,048 charts | **2,548 charts** (25.4%) |
| ~5.25M samples | **~2.5M samples** |
| Density 0.5-14.8 | Density 3.0-10.5 |
| Stars 0-13.8 | Stars 4.0-6.0 |

Stars 4-6 covers Oni and lower Inner Oni — the "sweet spot" of complex but well-charted content. Excludes:
- Kantan/Futsuu (1-3 stars): simple downbeat following
- Muzukashii (3-4 stars): intermediate, still metronomic
- Extreme (7+ stars): niche/unusual patterns, small sample

## Architecture

Identical to exp58: ProposeSelectDetector. Only the training data changes.

| Param | Value |
|---|---|
| Model | ProposeSelectDetector (same as exp58) |
| A_BINS / B_BINS | 500 / 500 |
| B_PRED | 250, N_CLASSES=251 |
| Proposer | 4 layers |
| Selector | 8 layers (enc 4 + fusion 4) |
| d_model | 384, 8 heads |
| Gap ratios | ON |
| Density jitter | ±10% at 30% |
| Proposer freeze | **4 evals** (2x data = 2x longer warmup) |
| **Star filter** | **4.0 ≤ stars < 6.0** |
| ramp_alpha | 2.5 (from exp44e) |

## Launch

```bash
cd osu/taiko
python detection_train.py taiko_v2 --run-name detect_experiment_58d \
    --model-type event_embed_propose \
    --a-bins 500 --b-pred 250 --gap-ratios \
    --density-jitter-rate 0.30 --density-jitter-pct 0.10 \
    --proposer-freeze-evals 4 \
    --min-stars 4 --max-stars 6 \
    --ramp-alpha 2.5 \
    --epochs 50 --batch-size 48 --evals-per-epoch 4 --workers 3
```

## Expected Behavior

- HIT% may be lower initially (less data, harder charts)
- Anti-metronome accuracy should improve (no metronomic training signal)
- Pattern diversity (P-Space) should increase in AR
- The model should make more "interesting" mistakes (wrong pattern vs metronomic repeat)

## Comparison

Train two models side by side:
1. **exp58d**: stars 4-6 only (this experiment)
2. **exp58**: all stars (existing, 74.6% HIT ATH)

Compare on the SAME val set (which includes all difficulties) to see if difficulty filtering helps generalization.

Running on CachyOS (RTX 4060, 8GB).

## Result

*Pending*

## Lesson

*Pending*
