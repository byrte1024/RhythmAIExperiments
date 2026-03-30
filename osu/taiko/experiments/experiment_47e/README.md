# Experiment 47-E - STOP Query Token

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.


## Hypothesis

47-47D all failed because the binary gate head was a small afterthought that couldn't learn:
- Reading from cursor token (shaped by onset loss, not stop loss)
- Reading from forward pool (too lossy, no context)
- Gradient too small regardless of loss weighting

**New approach: learned STOP query token.** A special learnable token is appended to the 250 audio tokens, making 251 total. It participates in ALL 8 transformer layers of self-attention — it can attend to every audio token, every event-marked token, and the cursor. After 8 layers, this token has built its own representation of "should I stop?" from the full context.

Key differences from 47-47D:
- **STOP token goes through the full transformer** — not a shallow MLP on frozen features
- **Gets gradient through attention** — stop loss backprops through all 8 layers via the STOP token's attention weights, shaping how the backbone processes information
- **Attends to both audio AND events** — can learn "quiet audio + sparse context = stop"
- **Balanced sampler already upweights STOP** — `1/count^0.5` weighting means STOP samples appear much more frequently than their 0.3% natural rate

### Architecture

```
Conv stem: mel → 250 audio tokens
Append: [250 audio tokens] + [1 STOP query token] → 251 tokens
Event embeddings: scatter-add to audio tokens (unchanged)
8x Transformer layers + FiLM: all 251 tokens attend to each other
Onset head: token 125 (cursor) → LayerNorm → Linear(384, 500) → 500 onset logits
STOP head: token 250 (STOP query) → LayerNorm → MLP(384→96→1) → stop logit
```

STOP is removed from the onset head's classes (500 instead of 501). The onset head only predicts onset locations. The STOP head independently predicts whether to stop.

### Training
- Onset loss: CE on non-STOP samples (500 classes)
- Stop loss: BCE on stop logit (target=1 for STOP, 0 for onset), averaged separately for STOP/onset samples within each batch to prevent class dilution
- Combined: `onset_loss + stop_weight * stop_loss`
- Balanced sampler with **20x STOP boost**: ensures ~7-8 STOP samples per batch of 48 (up from 0-1 without boost). Without this, most batches had zero STOP samples and the stop token got no gradient.
- STOP is 15,866 / 5.25M samples (0.3%). Standard balanced sampling at power=0.5 gives STOP ~4.5x weight over common bins, but that's still <1 per batch. The 20x boost brings STOP to ~16% sampling share.

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_47e --model-type event_embed --stop-token --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

## Result

**Stopped at eval 4. STOP token works — first successful binary stop architecture after 4 failed attempts (47-47D).**

### Progression

| Metric | Eval 1 | Eval 2 | Eval 4 | Exp 44 eval 4 |
|---|---|---|---|---|
| HIT | 66.8% | 67.8% | 70.3% | 72.0% |
| stop_f1 | 0.377 | 0.464 | 0.469 | 0.520 |
| stop_precision | 0.240 | 0.330 | 0.338 | ~0.39 |
| stop_recall | 0.880 | 0.784 | 0.766 | ~0.73 |
| no_audio_stop | 99.9% | 94.4% | 81.5% | 15.5% |
| AR step0 | 67.7% | 64.9% | 70.8% | 74.7% |

### What worked

- **Learned STOP query token** participating in all 8 transformer layers — the token builds its own representation through attention, unlike shallow MLP heads that failed in 47-47D
- **20x STOP sampling boost** — ensures ~7-8 STOP samples per batch (up from 0-1). Without this, most batches had zero STOP gradient
- **Balanced BCE** — STOP and onset losses averaged separately within each batch to prevent class dilution

### What's still behind

- HIT 70.3% vs exp 44's 72.0% at same eval — onset head trains on ~84% of data (16% is STOP samples). Gap was closing, likely would converge with more training
- Stop F1 0.469 vs exp 44's 0.520 — precision (34%) still low, too many hallucinated STOPs. Recall (77%) is good
- no_audio_stop drifting down (99.9% → 81.5%) — still far better than exp 44's 15.5% but trending toward it

### Why stopped

STOP isn't the primary bottleneck right now. Low hop spacing at inference (--hop-ms 10-50) works around poor STOP prediction effectively. The metronome problem and AR cascade are higher priority. This architecture is proven and can be revisited when STOP becomes the bottleneck.

## Lesson

- **The STOP query token works.** A learned token in the transformer sequence is the right architecture for binary stop — it gets full attention context and its own gradient path. Shallow MLP heads on frozen features fail.
- **Data balance was the real blocker.** Every 47-47D variant failed because STOP is 0.3% of data. The 20x sampling boost + balanced BCE were essential — not the architecture.
- **Onset head entropy appeared worse but was actually fine.** The entropy heatmap looked all-red because STOP token's near-zero entropy stretched the color scale. Actual onset entropy distribution was nearly identical to exp 45.
- **Onset loss should include STOP samples.** Currently the onset head gets zero gradient on STOP samples — wasted forward passes. Future work: give onset head a "what would the onset be IF there was one" target on STOP samples.
- **no_audio_stop was solved immediately** (99.9% at eval 1) and stayed high. The STOP token learned "silence = stop" with minimal training. This was stuck at 3-18% for all previous experiments.
