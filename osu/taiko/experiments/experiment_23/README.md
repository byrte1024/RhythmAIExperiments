# Experiment 23 - Shuffled Candidates with Confidence Scores

## Hypothesis

Exp 22 proved that shuffling eliminates positional bias and that context makes reasonable picks (low inaccurate_topK). But without audio confidence, context overrides 50%+ of predictions — it can't tell when audio is confident (and likely correct) vs uncertain (and worth overriding). Delta bottomed out at -4.6pp.

Exp 21 showed that giving context full audio info (scores + rank ordering) leads to conservatism bias — context learns "k=0 is usually right" rather than independently judging quality. Delta -0.95pp with only 28-37% override rate.

**The middle ground: shuffled candidates with confidence scores.**

Each candidate gets its softmax probability from audio as a scalar feature. But candidates are shuffled randomly — context can't learn positional shortcuts. It sees:

> "Here are 20 candidate positions in random order. Each has a gap embedding, a mel snippet, and a confidence value. Pick the best one."

This gives context the "when to override" signal (low confidence = worth investigating) without the "which one to pick by default" bias (no rank ordering).

### Changes from exp 22

**1. Restore score feature as softmax probability**

Each candidate gets `score_proj(softmax_prob)` — a single scalar (the audio model's probability for that bin) projected to d_ctx. This replaces exp 21's `score_proj(score, rank)` which gave both raw logit AND rank position.

Key difference from exp 21: only confidence magnitude, no rank ordering. Context knows "this candidate has 40% probability" but not "this was audio's #1 pick."

**2. Keep shuffle (from exp 22)**

Candidates randomly permuted per sample during training. Context cannot exploit position.

**3. Keep simplified SelectionLoss (from exp 22)**

Soft CE on quality-weighted candidates, skip when no HIT. No baseline comparison.

**4. Fix decision_categories chart bug**

`false_topK` was defined as "overrode & top1 was correct" without checking if final pick was wrong — causing overlap with `true_topK` and sum > 1.0. Fixed to "overrode & final wrong & top1 was correct." Five categories now mutually exclusive, sum to 1.0.

### Architecture

Identical to exp 22 but with `score_proj` restored:

| Component | Params | Training |
|-----------|--------|----------|
| AudioEncoder | 8.0M | **Frozen** (from exp 14) |
| EventEncoder | 0.5M | **Frozen** (from exp 14) |
| AudioPath | 5.0M | **Frozen** (from exp 14) |
| cond_mlp | ~8K | **Frozen** (from exp 14) |
| Context gap encoder | 0.9M | Training |
| Context snippet encoder | 0.2M | Training |
| Context selection head | 1.2M | Training |
| Context scoring + score_proj | ~0.05M | Training |
| **Total trainable** | **~2.5M** | |

### What context sees per candidate

| Signal | Exp 21 | Exp 22 | Exp 23 |
|--------|--------|--------|--------|
| Gap embedding | Yes | Yes | Yes |
| Mel snippet | Yes | Yes | Yes |
| Audio softmax prob | Yes (+ rank) | **No** | **Yes** |
| Rank position | Yes | No | **No** |
| Candidate order | Fixed (#1 first) | Shuffled | **Shuffled** |

### Expected outcomes

1. **Audio HIT = 69.5%** — frozen.
2. **Override rate 20-40%** — between exp 21's conservative 28-37% and exp 22's wild 50-56%. Confidence signal should let context be selective.
3. **Override accuracy > 55%** — with both rhythm features AND confidence, context should make better override decisions than either alone.
4. **Delta closer to 0 than either exp 21 or 22** — the sweet spot between "too conservative" and "too aggressive."
5. **Decision categories chart fixed** — five mutually exclusive categories summing to 1.0.

### Risk

- Softmax probability alone may not be enough — the absolute probability matters less than relative (is this the top pick or #15?). But relative info is exactly what we're avoiding to prevent positional bias.
- Context might learn to threshold on confidence ("override below 0.3, keep above 0.3") which is a simple heuristic, not deep understanding. This could plateau quickly.
- The score_proj(1 → d_ctx) projection may overweight the scalar confidence vs the richer gap/snippet features.

## Result

*Pending*
