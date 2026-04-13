# Experiment 66-1b — Corruption Evaluator as AR Quality Metric

## Hypothesis

The corruption detector from 66-1 phase 1 (no human rating fine-tuning) may already be a useful chart quality metric for evaluating our generators. Generated charts have corruption-like artifacts — metronomic patterns, timing jitter, density errors — that the model learned to detect. If the evaluator ranks our generators in the same order as GT matching metrics, it's a valid automatic quality proxy.

**Three questions:**
1. Does the model score GT charts higher than generated charts?
2. Does the model rank generators in the correct order (exp 62 > 58 > 45 > 14)?
3. Does gen_score correlate with established metrics (close rate, hallucination, pattern variety)?

**Secondary:** Does the phase 2 (rating-finetuned) model do better or worse than phase 1 (corruption-only)?

## Method

### AR inference

Run AR inference on 30 val songs (standard `select_30_val_songs`, seed 42) for four detector models spanning the project's history:

| Model | Exp | Key Feature | Per-sample HIT% |
|---|---|---|---|
| exp 14 | Audio-only baseline | No context, first correct data | 69.0% |
| exp 45 | Event embeddings + gap ratios | Context + density jitter | 73.6% |
| exp 58 | Two-stage propose-select | S1 proposals + S2 context | 74.6% |
| exp 62 | Multi-onset (4 simultaneous) | Pattern diversity win | 74.9% |

### Quality evaluation

Score each generated chart AND its ground truth with both evaluator checkpoints:
- **P1:** `eval_experiment_66_1/checkpoints/best.pt` — corruption-only (phase 1)
- **P2:** `eval_experiment_66_1_p2/checkpoints/best.pt` — corruption + rating (phase 2)

8 windows per chart, uniformly spaced.

### Metric correlation

For each generated chart, compute all `analyze_ar` metrics:
- **GT matching:** matched_rate, close_rate, far_rate, hallucination_rate, error_mean, density_ratio
- **TaikoNation:** Over. P-Space, HI P-Space, DCHuman, OCHuman, DCRand
- **Pattern variety:** gap_std, gap_cv, gap_entropy, dominant_gap_pct, max_metro_streak

Correlate gen_score (from the evaluator) with each metric via Spearman. A strong evaluator should correlate positively with close_rate, DCHuman, gap_cv, and negatively with hallucination_rate, dominant_gap_pct, max_metro_streak.

## Launch

```bash
bash run_eval_66_1b.sh
```

Runs all three steps: AR inference → quality scoring → cross-model summary.

## Expected results

- GT scores higher than generated in most songs, but not by a huge margin (our best models are good)
- Evaluator ranks generators: exp 62 > exp 58 > exp 45 > exp 14 (matching GT metrics)
- gen_score correlates with close_rate and hallucination_rate (obviously — less corrupted charts match GT better)
- Interesting: whether gen_score also correlates with pattern variety (gap_cv, metro_streak) — this would mean the corruption training implicitly learned "metronomic = bad"
- P1 (corruption-only) likely performs as well as or better than P2 for this task, since generated chart artifacts are closer to corruption than to rating differences

## Result

### GT vs Generated: model reliably prefers real charts

| Exp | P1 GT win% | P1 Diff | P2 GT win% | P2 Diff |
|---|---|---|---|---|
| 14 | 90.0% | +8.51 | 93.3% | +7.07 |
| 45 | 93.3% | +9.47 | 90.0% | +8.41 |
| 58 | 93.3% | +8.43 | 93.3% | +7.83 |
| 62 | 96.7% | +14.46 | 93.3% | +13.13 |

GT scores higher in 90-97% of songs. The model is a reliable real-vs-generated discriminator. P1 and P2 perform similarly.

### Generator ranking: WRONG

| Exp | Per-sample HIT% | P1 gen_score | Expected rank |
|---|---|---|---|
| 14 | 69.0% | +6.83 | 4th (worst) |
| 58 | 74.6% | +6.90 | 2nd |
| 45 | 73.6% | +5.87 | 3rd |
| 62 | 74.9% | **+0.88** | **1st (best)** |

The evaluator ranks exp 62 (our best generator) as producing the **worst** charts. Exp 14 (audio-only baseline) scores higher than exp 62. The ranking is backwards.

### Why: the model thinks metronomic = good

Correlation of P1 gen_score with AR metrics (pooled across all 4 experiments, n=120):

| Metric | Spearman | Significance | Problem |
|---|---|---|---|
| **metro_streak** | **+0.312** | p=5e-4 *** | Higher score = MORE metronomic. **Backwards.** |
| **DCHuman** | **-0.262** | p=4e-3 ** | Higher score = worse human match. **Backwards.** |
| **gap_std** | **-0.251** | p=6e-3 ** | Higher score = less variety. **Backwards.** |
| **gap_entropy** | **-0.194** | p=0.03 * | Higher score = less entropy. **Backwards.** |
| close_rate | +0.188 | p=0.04 * | Correct direction, weak |
| density | +0.193 | p=0.03 * | Higher = denser |

The corruption training taught: GARBAGE (random) is bad, CLEAN (structured) is good. The model generalized this to: **regularity = quality**. But real chart quality is an inverted U — too regular (metronomic) is just as bad as too random (noise).

Exp 62 scores lowest because it has the highest pattern diversity (metro_streak=12.9, dominant_gap_pct=47%, P-Space=12.4). The evaluator penalizes exactly what makes it good.

### GT matching detail

| Exp | close% | hall% | d_ratio | gap_cv | metro_streak | dom_gap% |
|---|---|---|---|---|---|---|
| 14 | 70.6% | 16.6% | 0.87 | 0.705 | 25.7 | 51.1% |
| 45 | 68.0% | 15.6% | 0.81 | 0.699 | 18.2 | 49.1% |
| 58 | 75.9% | 15.6% | 0.92 | 0.690 | 15.5 | 48.6% |
| 62 | 75.0% | 15.9% | 0.97 | 0.712 | 12.9 | 47.0% |

Exp 62 has the best density_ratio (0.97) and lowest metro_streak (12.9) — it's the most human-like generator. But the evaluator scores it lowest.

### P1 vs P2: no meaningful difference

Rating fine-tuning (P2) barely changed behavior. Same orderings, similar win rates. The noisy rating signal didn't overcome the corruption bias.

## Lesson

**Unidirectional corruption creates a regularity bias.** Training only on CLEAN → RANDOM teaches "structured = good" which is the opposite of what makes charts creative. The evaluator needs bidirectional corruption — both "too random" AND "too metronomic" should score lower than real charts. This is the motivation for exp 66-2.
