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

*(awaiting results)*
