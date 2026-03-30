# Experiment 45 - Reliable Density Conditioning

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp 44 benchmarks showed random density costs 14pp accuracy and zero density costs 17.9pp. The model uses density conditioning significantly — but the current ±10% jitter at 30% rate during training may be teaching the model to distrust density, weakening a signal that could help more.

At inference time, density conditioning is set by the user and should be reliable — there's no reason to train the model to be robust against bad density values. If we make density more trustworthy during training, the model should learn to rely on it more strongly, which could:
- Improve per-sample accuracy (density is free information)
- Help with metronome breaking (density tells the model "this section should be sparse/dense", overriding pattern continuation)
- Make the random_density/zero_density benchmarks DROP more (model trusts density more = bigger penalty when it's wrong, but better performance when it's right)

### Changes from exp 44

**Density conditioning:**
- Jitter: ±10% @ 30% → **±2% @ 10%** (barely perceptible noise, almost always correct)

**Event embeddings — gap ratio features (new):**

Previous embeddings per event: presence + gap_before + gap_after (3 × d_model → MLP → d_model)

Now: presence + gap_before + gap_after + **gap_ratio_before** + **gap_ratio_after** (5 × d_model → MLP → d_model)

- `gap_ratio_before[i]` = gap_before[i-1] / gap_before[i] — is the rhythm accelerating or decelerating coming into this event?
- `gap_ratio_after[i]` = gap_after[i+1] / gap_after[i] — is the rhythm accelerating or decelerating leaving this event?

Both are sinusoidal-encoded after scaling (ratio × 50, clamped to [0.1, 10.0]). Ratio=1.0 (value 50) means constant rhythm (metronome). <1 means speeding up, >1 means slowing down.

This gives the model explicit rhythm-change information that it previously had to infer from raw gap sequences. It directly encodes whether the pattern is metronomic (all ratios ≈ 1.0) or varied.

### Risks

- **Gap ratios may worsen metronome behavior.** If the model's metronome tendency is a loss-minimizing strategy (44-B showed 47% of targets ask for continuation), giving it explicit ratio features makes it even easier to detect "ratio ≈ 1.0 = continue the pattern." The model could learn to lean on the ratio signal to confidently repeat, rather than using it to detect when to break. This would show up as lower metronome benchmark scores and higher pred_continues_target_breaks in the metronome stats.
- **More reliable density could have the same effect.** If the density signal is more trustworthy, and the density matches a metronomic section, the model has even more reason to continue.

### Architecture
EventEmbeddingDetector with expanded event projection (5 × d_model input instead of 3 × d_model). Slightly more parameters in the event_proj MLP.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_45 --model-type event_embed --gap-ratios --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Stopped at eval 8. No meaningful improvement over exp 44.**

| Metric | Exp 45 (eval 8) | Exp 44 (eval 8) | Exp 44 ATH (eval 19) |
|---|---|---|---|
| HIT | 71.9% | 72.5% | 73.6% |
| MISS | 27.5% | 27.0% | 25.9% |
| Ctx delta | 7.6pp | 7.5pp | 5.6pp |
| AR step0 | 73.5% | 75.4% | 76.7% |
| Metronome | 44.8% | 40.0% | 42.9% |
| Adv metronome | 49.7% | 50.2% | 50.1% |
| Random density | 32.3% | ~37% | 40.7% |
| Zero density | 29.1% | ~24% | 36.8% |

Consistently ~0.6pp behind exp 44 on HIT. Small differences in metronome benchmarks are noise. Random density dropped (32.3% vs exp 44's ~37%) confirming the model relied more on density, but this didn't translate to better per-sample accuracy.

### AR density adherence

| Exp | Eval | Conditioned | Actual | Ratio |
|---|---|---|---|---|
| 44 | 20 | 5.69 | 6.45 | 1.132 (13% over) |
| 45 | 7 | 5.69 | 6.39 | 1.122 (12% over) |

Both overshoot by 12-29% in AR benchmarks. In manual full-song AR testing, exp 45 showed notably better density adaptation — responding proportionally to different density inputs with a consistent ~10-20% overshoot. The tighter density jitter made the model more responsive to density conditioning in real AR generation, even though per-sample metrics didn't improve.

## Lesson

- **Gap ratio features don't help per-sample metrics.** The model can already infer rhythm acceleration/deceleration from raw gap sequences via attention. Making it explicit adds parameters but no signal.
- **Tighter density jitter IS valuable for AR generation.** Per-sample metrics didn't improve, but real AR testing showed much better density adherence. The model follows requested density more faithfully with less jitter during training. This is adopted going forward.
- **Per-sample metrics continue to understate AR quality differences.** Same story as exp 42-AR — what matters in generation isn't fully captured by validation accuracy.
