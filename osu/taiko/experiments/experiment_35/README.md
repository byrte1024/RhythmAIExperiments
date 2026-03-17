# Experiment 35 - Mel-Embedded Event Ramps

## Hypothesis

20 experiments (14-34) tried every fusion approach — concat self-attention, cross-attention, FiLM conditioning — and all fail to make the model use context. The core problem: context features enter through a separate pathway that the model can route around.

**Embed events directly into the mel spectrogram.** Encode gradient ramps between consecutive event positions in reserved mel bands. The audio encoder sees both audio features AND event timing through the exact same conv + self-attention pathway. The model literally cannot process audio without also processing the event ramps — they're in the same tensor.

### How it works

Between each pair of consecutive events (and from the last event to the cursor), a linear ramp is encoded in reserved mel bands:
- Ramp goes from 1.0 at the event position to 0.0 at the next event
- This creates a sawtooth pattern whose frequency = the rhythm pattern
- The slope of each ramp encodes the gap length (steep = short gap, shallow = long gap)
- Events older than the mel window (>2.5s ago) still show their gradient entering from the left edge — the slope carries gap information even without seeing the origin

Reserved bands with fading intensity:
```
Band 0  (bottom): 100% ramp intensity
Band 1:            50%
Band 2:            25%
...
Band 77:           25%
Band 78:            50%
Band 79 (top):    100% ramp intensity
```

### Why this is different from all prior approaches

1. **Context travels through the audio pathway** — same conv stem, same self-attention, same cursor extraction. No separate encoder to atrophy.
2. **The model can't ignore it** — the ramps are in the mel tensor. The conv layers process them alongside audio. There's no routing decision.
3. **Temporal precision preserved** — events are at their actual frame positions. Self-attention at position 125 naturally sees nearby event ramps.
4. **Survives benchmarks correctly** — embedded inside `forward()`, so `no_audio` (zeroed mel) kills audio but the ramps are added AFTER zeroing. `no_events` (masked events) produces no ramps. Both ablations work correctly.
5. **Zero architectural changes** — same OnsetDetector as exp 27. This is a data representation change, not an architecture change.

### Architecture

Same as exp 25-27 (unified fusion, ~19M params). The only change is `_embed_events_in_mel()` called at the start of `forward()`.

### Expected outcomes

1. **Fast convergence** — identical architecture to exp 27, audio bootstraps normally.
2. **Context delta should be high** — the model physically processes event ramps through its audio features. no_events removes the ramps, which should noticeably hurt.
3. **No banding** — audio self-attention is unchanged, no coarse features injected into attention.
4. **Pattern disambiguation** — the sawtooth rhythm pattern is visible in the mel. For `150 150 75 75`, the ramp pattern has two slow slopes then two fast slopes, directly encoding the rhythm.

### Risk

- The conv stem may learn to filter out the ramps (they're in specific bands, easily separable). If the model learns "ignore bands 0-2 and 77-79," we're back to audio-only.
- 10 dB ramp intensity may be too weak or too strong relative to actual mel values.
- The GapEncoder still runs alongside — the model may use GapEncoder features and ignore the ramps, or use ramps and ignore GapEncoder. May want to try without GapEncoder in a follow-up.
- The per-sample loop in `_embed_events_in_mel` may be slow. Could optimize with vectorized ops if it becomes a bottleneck.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_35 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Context delta 5.0% — better than FiLM but still low. Ramps in edge bands only too subtle.** Killed after eval 1.

| eval | epoch | HIT | Miss | Score | Acc | Unique | Val loss | no_events | Ctx Δ | no_audio |
|------|-------|-----|------|-------|-----|--------|----------|-----------|-------|----------|
| 1 | 1.25 | 65.7% | 33.7% | 0.298 | 47.7% | 449 | 2.674 | 42.8% | 5.0% | 0.4% |

Ramps in reserved edge bands (0-2, 77-79) are too easy for the conv to filter out. Next: apply ramps to ALL bands with halved audio.

## Lesson

- **Edge-band ramps provide some signal** (5.0% delta, better than FiLM's 4.2%) but the conv learns to ignore specific bands.
- **Ramps need to be everywhere** — the model must process them alongside audio features, not in separable frequency bands.
