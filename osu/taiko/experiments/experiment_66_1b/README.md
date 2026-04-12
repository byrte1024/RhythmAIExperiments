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

*(awaiting results)*
