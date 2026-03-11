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

**Partial success — context activated but not helpful.** Stopped at E7.

### Trajectory

| Epoch | train_loss | val_loss | accuracy | hit_rate | miss_rate | override_rate | override_acc | target_in_topk |
|-------|-----------|----------|----------|----------|-----------|---------------|-------------|----------------|
| 1 | 6.048 | —* | 45.1% | 63.7% | 35.3% | 25.8% | 38.5% | 97.6% |
| 2 | 5.713 | —* | 44.5% | 64.1% | 34.7% | 35.2% | 44.0% | 97.4% |
| 3 | 5.589 | —* | 40.8% | 63.1% | 34.6% | 46.2% | 47.1% | 97.1% |
| 4 | 5.502 | —* | 42.6% | 63.9% | 34.5% | 46.6% | 48.3% | 96.9% |
| 5 | 5.442 | —* | 42.9% | 64.4% | 33.8% | 45.5% | 50.4% | 96.8% |
| 6 | 5.394 | —* | 42.1% | 65.3% | 32.5% | 50.0% | 52.5% | 96.4% |
| 7 | 5.360 | 5.178 | 43.0% | 65.3% | 32.3% | 47.1% | 51.5% | 96.3% |

*E1-E6 val_loss was miscalculated due to a bug (OnsetLoss computed on scattered 501-way logits with -100 at non-candidate positions). Fixed for E7+.

### What worked

- **Context path activated for the first time ever.** Override rate rose from 26% (E1) to 50% (E6), proving the architectural constraint forced engagement. Exp 11-16 never achieved this.
- **no_events benchmark diverged from full accuracy** — E1: 47.7% vs full 45.1%, E7: 42.0% vs full 43.0%. Context IS contributing signal (unlike exp 14-16 where no_events ≈ full).
- **Architecture > loss tricks confirmed.** Exp 15 aux CE = zero engagement. Exp 16 rank-weighted CE = wrong opinions. Exp 17 top-K constraint = immediate engagement by E1.

### What didn't work

- **Override accuracy plateaued at ~51-52%**, barely above coin flip for overrides. Context learned to override but not WHEN to override correctly.
- **Accuracy dropped from E1 (45.1%) as context became more active.** The model was better at E1 with 26% override than at E7 with 47% override. Context overrides are net-negative.
- **Hit rate stuck at 65.3%**, still 3-4pp below exp 14's audio-only 69%. The reranking overhead costs more than the selection gains.
- **Rank distribution heavily skewed**: ~53% pick #0, ~24% pick #1, ~12% pick #2. Context mostly agrees with audio and when it disagrees, it's wrong half the time.
- **Selection stats plateau by E5-E6**: override_accuracy, override_rate, and target_in_topk all flat. No sign of continued learning.

### Benchmarks (E7)

| Benchmark | Accuracy | Notes |
|-----------|----------|-------|
| full (normal) | 43.0% | Below exp 14's 50.5% |
| no_events | 42.0% | Context engaged but not helpful |
| no_audio | 0.3% | Audio is essential (expected) |
| random_events | 43.3% | ≈ full — context isn't using events well |
| static_audio | 1.7% | Audio temporal structure needed |
| metronome | 42.8% | ≈ full |
| time_shifted | 42.6% | ≈ full |
| advanced_metronome | 41.7% | ≈ full |
| zero_density | 10.3% | Density still load-bearing |
| random_density | 34.4% | ~9pp drop from full |

### Key numbers vs exp 14

| Metric | Exp 14 (E8) | Exp 17 (E7) | Delta |
|--------|-------------|-------------|-------|
| accuracy | 50.5% | 43.0% | -7.5pp |
| hit_rate | 69.0% | 65.3% | -3.7pp |
| miss_rate | 30.0% | 32.3% | +2.3pp |
| p99 frame error | ~150 | 162 | worse |

## Lesson

- **Architectural constraint activates context, but activation ≠ value.** The top-K reranking successfully broke rubber-stamping — a first across 6 experiments. But the context path learned to override without learning WHEN to override. 51% override accuracy means context is essentially flipping coins on its disagreements.
- **The reranking bottleneck hurts audio.** Restricting final output to K=20 candidates with context selection overhead costs 7.5pp accuracy vs audio-only (exp 14). The selection loss competes with the audio loss for shared encoder capacity.
- **Shared encoders create gradient interference.** Audio and context share the AudioEncoder and EventEncoder. Context's selection loss gradient flows back through these shared paths, potentially degrading audio's proposal quality. This explains why accuracy is BELOW exp 14 despite identical audio architecture.
- **Next direction: full path separation.** Give audio and context completely separate losses with stop-gradient between paths. Audio optimizes top-1 proposal quality only. Context optimizes selection quality only. No gradient leakage between the two objectives.
