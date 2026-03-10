# Experiment 17 - Top-K Reranking Architecture

## Hypothesis

Experiments 15-16 proved that loss-function approaches cannot activate the context path. The root cause is architectural: with additive logits (`audio + context → 501 classes`), context's optimal strategy is always to be a no-op (output zeros/uniform). Standard CE (exp 15) had zero effect. Rank-weighted CE (exp 16) forced wrong opinions that actively degraded audio's correct rankings (-5pp top-K).

**The fix — architectural constraint:** Replace the 501-way context path with a top-K reranking selector. Audio proposes K=20 candidates, context must pick one. Rubber-stamping and no-op are architecturally impossible: context must output a K-way distribution and every choice affects the prediction.

### Architecture changes

**Old ContextPath (exp 14-16):**
- Event self-attn + audio cross-attn → query token → 501-way logits
- Combined via `audio_logits + context_logits` (additive in logit space)
- Context could output zeros = no effect

**New ContextPath (top-K reranker):**
1. Audio produces 501 logits → take top-K=20 candidates (STOP always included)
2. For each candidate, build rich features:
   - Sinusoidal bin position embedding (where in time)
   - Audio score + rank (how confident audio is, what rank)
   - Audio feature at that temporal position (what audio "sees" there)
3. Combine into d_model-dim candidate embeddings via MLP
4. Process event history through transformer decoder (event self-attn + audio cross-attn → query vector)
5. Score each candidate via scaled dot product: `q_proj(query) · k_proj(candidate) / √d_score`
6. K-way softmax → selection probability

**Final prediction:** scatter selection logits onto 501-way tensor at candidate positions. `argmax` gives the predicted bin.

**Selection loss:** Soft K-way CE with trapezoid targets (same distance logic as OnsetLoss). Candidates near the true target get proportional credit. Multiple good candidates can share weight.

**Audio loss:** Full OnsetLoss on audio_logits (weight 1.0, not 0.2 — audio is now the sole proposer and needs strong gradient).

### Why this works

- **No-op impossible**: context must select one of K candidates. Even uniform selection shifts prediction.
- **Rubber-stamping impossible**: picking audio's #1 every time will be wrong ~33% of the time (miss rate). The selection loss pushes context to identify when #2/#3 is correct.
- **Information-rich candidates**: context sees audio's score, rank, AND the audio feature at each candidate position — directly enabling "I see audio ranked bin 48 and bin 96, and from event spacing I know it should be ~48, so pick #1."
- **Audio path unchanged**: identical to exp 14. Audio aux at full weight keeps proposal quality stable.

### New metrics & charts

1. **Audio proposal quality**: audio-only top-K HIT rates (separate from final), to track proposer independent of selector
2. **Candidate selection histogram**: which of the K candidates does context pick? (should NOT be flat at #0)
3. **Accuracy by selected rank**: when context picks candidate #0/#1/#2/etc, what's the HIT rate?
4. **Override rate**: how often does context pick something other than audio's #1?
5. **Override accuracy**: when it overrides, how often is the override correct?
6. **Target availability**: % of samples where correct answer is in top-K (expected ~97%)

### Expected outcomes

- **Context MUST engage**: no_events accuracy should drop below full accuracy since context can't be a no-op
- **Audio proposals stable**: audio top-K HIT rates should match exp 14 (~95% at top-10)
- **Accuracy ≥ exp 14**: context selecting well from audio's candidates should beat audio alone (~50%)
- **Override rate > 0**: context should learn to override audio's #1 on 10-30% of samples
- **Override accuracy > random**: when overriding, context should be right more often than chance among top-K

### Risk

- Selection loss might be too easy (always picking #0 gives ~67% HIT) — if context doesn't learn to override, we get exp 14 performance with extra overhead
- K=20 might miss some targets (~3% not in top-20 based on exp 14 data) — those samples get no useful context gradient
- The candidate feature MLP adds parameters and computation — watch for training speed regression

## Result

*Pending.*

## Lesson

*Pending.*
