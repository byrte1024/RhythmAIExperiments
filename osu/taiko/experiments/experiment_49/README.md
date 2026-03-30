# Experiment 49 - Virtual Tokens for Out-of-Window Context

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

Exp 48 showed 14.2% of failures are universal across all architectures, with 2x/0.5x metric confusion as the dominant error. The model can't distinguish beat from sub-beat because it only sees 2.5s of audio + 2.5s of in-window events. A full musical phrase at 120 BPM is 8 seconds — the model can't see one.

The model HAS 128 context events spanning ~25 seconds of history, but events outside the 2.5s audio window are discarded because there's no audio token to place them on.

**Virtual tokens** solve this by adding V extra tokens before the audio sequence. Out-of-window events get placed on these tokens via the same scatter-add mechanism. The transformer attends to virtual tokens alongside audio tokens — one unified sequence, no cross-attention.

### Architecture

```
[V virtual tokens] + [250 audio tokens] = V+250 total tokens

Virtual tokens:  learned watermark embedding + position embedding + event embeddings (scatter-add)
Audio tokens:    conv stem output + position embedding + event embeddings (scatter-add, unchanged)
Cursor:          token V+125 (was 125)
```

Virtual tokens are NOT silence — they carry a learned watermark that signals "this is context, not audio." The transformer can attend to them for rhythm history without confusing them with quiet audio sections.

### Event mapping

- In-window events (offset >= -500): mapped to audio tokens 0-124 (unchanged)
- Out-of-window events (offset < -500): mapped to virtual tokens 0 to V-1
  - Linear mapping: earlier events → lower token indices
  - More recent out-of-window events → closer to the audio boundary (token V-1)

### Key parameters

- `--n-virtual-tokens V` (default 32): number of virtual tokens
- Default V=32 covers ~25s of history at ~0.8s per token resolution

### Risks

- Virtual tokens add V*d_model parameters and increase attention cost by (V/250)^2
- The watermark must be distinctive enough that the model doesn't confuse virtual tokens with silence
- Out-of-window events have no audio association — the model must learn to use rhythm-only information

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_49 --model-type event_embed --n-virtual-tokens 32 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Stopped at eval 9. Virtual tokens work — anti-metronome behavior achieved, but over-prediction is the new problem.**

### Per-sample metrics

| Metric | Eval 3 | Eval 9 | Exp 44 eval 9 |
|---|---|---|---|
| HIT | 71.2% | 72.6% | 72.7% |
| MISS | 28.4% | 27.0% | 26.8% |
| Ctx delta | 5.1pp | 3.4pp | 6.7pp |
| Metronome | 40.8% | 46.6% | 43.9% |
| Time shifted | 39.7% | 45.0% | 44.6% |
| stop_f1 | 0.533 | 0.530 | 0.523 |

Per-sample HIT matches exp 44 at the same eval point. Metronome and time-shifted resilience are better. Context delta is notably lower (3.4pp vs 6.7pp).

### AR generation — the key finding

```
Events: HIT=36.7% GOOD=48.5% MISS=4.1%  (5795/15774 found)
Preds:  HIT=18.1% GOOD=24.0% HALL=52.7%  (5795/31968 valid)
Surv@10: 100.0%  @30: 100.0%  Density: 5.7->6.9 (1.21x)
```

- **100% survival at step 30** — the model NEVER metronomes. Unprecedented. Every previous model collapsed.
- **Event finding: 48.5% GOOD, only 4.1% MISS** — it knows where the real events are
- **52.7% hallucination** — over half of predictions don't match real events. The model places ~2x as many notes as needed
- **Density 1.21x** — reasonably close to target despite the hallucinations

In manual AR testing, the model almost never falls into metronome patterns. But it misses and hallucinates notes frequently — placing notes at sub-beat positions that don't have mapped onsets.

### The tradeoff

| | Exp 44 | Exp 49 |
|---|---|---|
| Metronome | Frequent collapse | Almost never |
| Precision | High (few hallucinations) | Low (52% hallucinated) |
| AR survival | Degrades after step 10 | 100% at step 30 |
| Error type | Locks into patterns | Over-predicts sub-beats |

Exp 44's errors compound (metronome cascade). Exp 49's errors are varied and recoverable but numerous.

## Lesson

- **Virtual tokens solve metronome collapse.** With 25s of rhythm history visible, the model maintains variety instead of locking into patterns. This was the #1 quality problem identified in exp 42-AR.
- **But they cause over-prediction.** The model finds the right rhythmic grid but at double resolution — inserting sub-beat notes that don't exist. The 2x/0.5x confusion from exp 48 manifests in the opposite direction.
- **32 virtual tokens may be too coarse.** Each token covers ~0.8s = ~4 events. Multiple events blend into one token, losing individual gap information. The model gets meter-level structure but not precise gap sequences. This may explain the low context delta (3.4pp) and hallucinations.
- **Context delta dropped but that's not necessarily bad.** The model may be using virtual tokens for rhythm structure while relying on audio for timing — a healthier split than over-relying on recent context (which causes metronome lock-in).
- **Next directions:**
  - More virtual tokens (64 or 128) for finer resolution
  - Logarithmic mapping — more tokens for recent history, fewer for distant
  - Combine with density tightening to constrain over-prediction
  - Human evaluation: despite worse per-sample metrics, AR charts may sound better than exp 44 due to no metronome
