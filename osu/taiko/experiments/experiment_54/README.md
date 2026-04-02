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

*Pending*

## Lesson

*Pending*
