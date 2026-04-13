# Experiment 65-S2v2b — Both-Miss Failure Analysis

## Purpose

16.4% of onset bins are missed by both S1 (audio) and S2v2 (context). Are these genuinely hard cases spread across the dataset, or do they cluster in specific problematic charts/songs?

If clustered: likely a data quality issue (bad annotations, unusual audio, non-standard charts). Fixable by data cleaning.

If uniform: structurally hard — no audio transient AND no rhythmic pattern predicts them. These are the fundamental ceiling.

## Method

Run S1 and S2v2 on the full val set. For each sample, identify bins where both models miss. Track:

1. **Per-chart both-miss rate**: what % of each chart's onsets are missed by both?
2. **Distribution**: is the both-miss rate uniform or heavy-tailed?
3. **Outlier charts**: which charts have the highest both-miss rate?
4. **Chart properties**: do outlier charts share characteristics (density, duration, star rating, genre)?
5. **Per-sample concentration**: are both-miss bins spread evenly within a chart, or concentrated in specific sections?

## Result

*Pending*

## Lesson

*Pending*
