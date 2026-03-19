# Experiment 39-E - Audio Analysis of Failure Cases

## Hypothesis

Exp 39-D showed that when the model is wrong, the correct answer is at rank 1-2 with only 43% the confidence of the wrong pick. The model genuinely believes the wrong answer is correct despite heavy training loss penalizing these errors.

**Theory: the audio at the target and predicted positions will be nearly identical.** Both positions are real onsets (exp 39 showed 83% of overpredictions match future onsets), so both have real transients. The model isn't picking a louder transient over a quieter one — it's choosing between two equally valid audio events and can't determine which is nearer.

If the audio energy/flux/onset strength is similar at both positions, the failure isn't an audio saliency problem — it's a proximity/ordering problem. The model detects onsets correctly but lacks the mechanism to rank them by distance from cursor.

### Method

For each failure case where the correct answer is in top-K:
1. Compare mel energy (mean across bands, ±5 frame window) at target vs predicted position
2. Compare spectral flux (frame-to-frame change) at both positions
3. Compare onset strength (max flux in ±3 window) at both positions
4. Export 10 visual mel windows with target and predicted positions marked

### Expected: audio features will be nearly identical at target and predicted positions

## Result

*Pending*

## Lesson

*Pending*
