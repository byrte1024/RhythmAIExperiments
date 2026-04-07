# Experiment 59-HB - Ground Truth Comparison for 59-H Models

## Hypothesis

[Exp 59-H](../experiment_59h/README.md) showed exp51 dominating synthetic evaluation despite 67.5% HIT. Before investing in human evaluation, check the actual AR quality vs ground truth — matched/close/far rates, hallucination, density adherence — using the CSVs already generated in 59-H.

If exp51 has extremely high hallucination or very low match rates, its high gap variety is likely noise. If it maintains reasonable match rates while having varied gaps, it's genuinely interesting.

### Launch

```bash
cd osu/taiko
python experiments/experiment_59hb/gt_comparison.py
```

## Result

*Pending*

## Lesson

*Pending*
