# Experiment 54 - B_AUDIO/B_PRED Split + STOP Query Token

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 53-B](../experiment_53b/README.md) established the best window configuration (A=500, B_AUDIO=500, B_PRED=250) reaching 73.4% HIT with the best audio-only accuracy ever (49.7%). However, its STOP behavior was unstable — no_audio_stop bounced between 12-95% across evals, and stop F1 plateaued at 0.562.

[Exp 47-E](../experiment_47e/README.md) proved the STOP query token architecture works: a learned token participating in all transformer layers, with separate stop head MLP. It achieved 81.5% no_audio_stop and 0.469 stop F1 at eval 4 (early stopped). Key enablers: 20x STOP sampling boost + balanced BCE.

**This experiment combines them:** 53-B's window configuration + 47-E's STOP query token. With B_PRED=250, STOP prevalence is ~0.8% (events beyond 250 bins), higher than the 0.3% in 47-E's 500-bin setup. The higher STOP rate plus the 20x sampling boost should give the STOP token even more gradient signal.

Key changes from 53-B:
- STOP removed from onset head's classes (250 instead of 251)
- Learned STOP query token appended to sequence (250 audio + 1 STOP = 251 tokens)
- Separate stop head MLP reads from STOP token after transformer
- Stop loss: balanced BCE averaged separately for STOP/onset samples
- 20x STOP sampling boost in balanced sampler

### Architecture

```
Mel window: 1000 frames (500 past + 500 future)
Conv stem: 1000 → 250 audio tokens + 1 STOP query token = 251 tokens
Cursor: token 125 (500 // 4)
Onset head: 250 classes (bins 0-249, no STOP)
STOP head: binary logit from STOP query token
```

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_54 --model-type event_embed --a-bins 500 --b-bins 500 --b-pred 250 --stop-token --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

Stopped at eval 1 (epoch 1.2). STOP head architecture not worth the cost.

### Eval 1 metrics:

| Metric | Exp 54 | 53-B (eval 1) | 47-E (eval 1) |
|--------|--------|---------------|---------------|
| HIT% | 65.9% | 68.4% | 66.8% |
| Stop F1 | 0.392 | 0.479 | 0.377 |
| Stop precision | 0.261 | — | 0.240 |
| Stop recall | 0.794 | — | 0.880 |
| no_audio_stop | 97.2% | 95.4% | 99.9% |
| no_events_stop | 21.0% | — | — |
| Val loss | 2.806 | 2.637 | 2.732 |

### Why stopped

1. **Stop F1 (0.392) is worse than 53-B's softmax STOP (0.479)** at the same eval. The separate binary head underperforms the integrated STOP class.
2. **Stop precision is terrible (26.1%)** — 74% of STOP predictions are false positives. The head is trigger-happy.
3. **no_events_stop only 21%** — when context is stripped, the model barely uses STOP. It learned "STOP when audio is quiet" but not "STOP when uncertain."
4. **HIT 2.5pp behind 53-B** — the 20x STOP sampling boost steals onset training samples (~13% of batches are STOP), slowing onset learning with no payoff.
5. **STOP is still not the bottleneck.** The softmax STOP class in 53-B already works adequately (F1=0.562 at peak). A separate head adds complexity without benefit.

## Lesson

The STOP query token architecture from [exp 47-E](../experiment_47e/README.md) does not improve when combined with the B_AUDIO/B_PRED split. Despite STOP being more prevalent (0.8% vs 0.3%) and 60% of STOPs being "informed" (next onset visible in audio beyond B_PRED), the separate binary head still underperforms the simple softmax STOP class.

The fundamental issue: the STOP head learns audio-based shortcuts ("silence = stop") rather than uncertainty-based reasoning ("I can't confidently predict = stop"). The softmax STOP class naturally competes with onset predictions, giving it implicit uncertainty awareness that a separate binary head lacks.

**STOP head experiments are exhausted.** 47-47D (binary gate) failed. 47-E (query token) worked but wasn't needed. Exp 54 (query token + B_PRED split) underperformed the baseline. The softmax STOP class is sufficient — inference hop spacing handles the rest.
