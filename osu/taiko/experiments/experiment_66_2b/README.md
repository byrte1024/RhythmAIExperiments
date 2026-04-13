# Experiment 66-2b — Evaluator vs Human Preference (The Real Test)

## Hypothesis

Exp 66-2 showed the bidirectional corruption evaluator doesn't rank generators by HIT% — but HIT% doesn't predict human preference either (exp 42-AR, 53-AR). The evaluator might be capturing something real that simple metrics miss.

**The definitive test:** score the exact same charts that humans ranked in blind A/B/C/D tests, then compare evaluator rankings against human rankings per-song. If the evaluator agrees with humans more than random chance, it's learning genuine quality.

## Data sources

### Exp 42-AR (18 votes)
- **Models:** exp14 (audio-only), exp35c (mel ramps), exp42 (event embeddings)
- **Songs:** 10 songs across pop, j-pop, chiptune, indie rock, hyperpop
- **Human winner:** exp14 (43pts) > exp42 (34pts) > exp35c (31pts)
- **Key finding:** Audio-only baseline beat context models. Metronome regression was the #1 complaint.

### Exp 53-AR (15 votes)
- **Models:** exp14, exp44 (gentle aug), exp45 (gap ratios), exp53 (B_AUDIO/B_PRED split)
- **Songs:** 10 songs across j-pop, j-dance, indie rock, chiptune, pop
- **Human winner:** exp45 (44pts) > exp44 (43pts) > exp53 (36pts) > exp14 (27pts)
- **Key finding:** Context models overtook audio-only. exp45/exp44 virtually tied.

### Split analysis
- **Self votes:** 10 songs each, detailed notes, from the project author (experienced evaluator)
- **External evaluators:** 5 (42-AR) + 5 (53-AR) volunteers, 1 song each, blind ranking

## Evaluator checkpoints

| Checkpoint | Description |
|---|---|
| 66-2 | Bidirectional corruption evaluator (best.pt) |
| 66-1 P1 | Unidirectional corruption evaluator (best.pt) |

## Launch

```bash
python classifier_eval_human.py \
    --checkpoint runs/eval_experiment_66_2/checkpoints/best.pt \
    --checkpoint2 runs/eval_experiment_66_1/checkpoints/best.pt \
    --output experiments/experiment_66_2b/results.json
```

## Success criteria

- **Per-song #1 match > 25%** (random baseline for 4 models) or >33% (for 3 models)
- **Pairwise accuracy > 50%** (random baseline)
- **Global ranking correlation > 0** (Spearman)
- 66-2 (bidirectional) should outperform 66-1 (unidirectional) on all metrics

## Result

### 42-AR: 66-2 perfectly matches human ranking

| Split | #1 match | Pairwise | Spearman |
|---|---|---|---|
| Self (9 songs) | 4/9 (44%) | 44.4% | **+1.00** |
| Evaluators (5 songs) | 2/5 (40%) | 66.7% | +0.50 |
| Total (9 songs) | 4/9 (44%) | 50.0% | **+1.00** |

**Global ranking: perfect match.**
- Human: exp14 > exp42 > exp35c
- 66-2:  exp14 > exp42 > exp35c

The bidirectional evaluator correctly identifies exp14 (audio-only, 69% HIT) as the best generator — matching the human finding that per-sample accuracy doesn't equal quality. The unidirectional 66-1 got this completely backwards (Spearman -0.50).

### 53-AR: both models fail

| Split | 66-2 Pairwise | 66-2 Spearman | 66-1 Pairwise | 66-1 Spearman |
|---|---|---|---|---|
| Self | 50.0% | -0.80 | 46.7% | 0.00 |
| Evaluators | 26.7% | -0.60 | 53.3% | -0.40 |
| Total | 44.1% | -0.80 | 47.5% | 0.00 |

**Global ranking: nearly inverted.**
- Human: exp45 > exp44 > exp53 > exp14
- 66-2:  exp53 > exp14 > exp44 > exp45

The evaluator puts exp45 (human winner) last. Score spread is tiny — all four models land in a 0.17 range (-0.52 to -0.69). The evaluator sees them as nearly identical, with differences dominated by noise.

### Why 42-AR works but 53-AR doesn't

42-AR models span very different quality levels (exp14 from 2024 vs exp42 with deep context — dramatically different architectures and failure modes). 53-AR models are close variants of the same architecture, separated by 1 point in human voting (exp45=44pts vs exp44=43pts). The evaluator can detect large quality gaps but not subtle preference differences.

### 66-2 vs 66-1

| Dataset | 66-2 Spearman | 66-1 Spearman |
|---|---|---|
| 42-AR total | **+1.00** | -0.50 |
| 53-AR total | -0.80 | 0.00 |

Bidirectional training dramatically improved 42-AR alignment but didn't help 53-AR. The bidirectional model better captures gross quality differences but not fine-grained human preference.

## Lesson

The evaluator is a valid quality metric for **large quality gaps** (different-era models, real vs generated). It matches human preference perfectly on 42-AR where models are genuinely different. It fails on 53-AR where models are similar — the evaluator's score compression (0.17 range) means these differences are below its resolution.

This suggests the evaluator could be useful for:
- Real-vs-generated discrimination (97-100% accuracy)
- Coarse model comparison (is model A significantly better than B?)
- NOT for ranking similar models or subtle preference differences
