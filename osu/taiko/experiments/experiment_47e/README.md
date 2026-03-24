# Experiment 47-E - STOP Query Token

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

*Pending*

## Lesson

*Pending*
