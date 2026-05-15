# custom: per-t loss imbalance + peak saturation

Two diagnostic plots that together pin the failure mode.

## `per_t_quartile_loss.png`

Per-t-quartile **unweighted** MSE at each eval. Source:
`runs/exp_016_framewise_diffusion/eval_{step}/eval.json:metrics.loss/per_t_q{0..3}`.

q0 covers t ∈ [0, T/4) (near-clean, end of sampler chain). q3 covers
t ∈ [3T/4, T) (near-noise, start of chain). At step 103,370 the
quartiles are q0=0.0017, q1=0.0205, q2=0.0464, q3=0.0596 — q3 is
**~35× q0**.

That disparity is **not** the natural shape of an MSE-on-x0 loss.
The Min-SNR weighting in `training/framewise_diffusion_loss.py:137`
applies `weight = min(snr, γ) / snr`, which is the formula derived
for **ε-parameterization** (where natural per-t loss explodes at low
t and Min-SNR caps the explosion). Applied to **x0-parameterization**
(what #016 uses — see `models/framewise_diffusion_detector.py:97`)
this multiplies the low-t gradient by γ/snr → ~0. The model trains
almost exclusively on the high-t "predict from pure noise + audio
conditioning" regime; the low-t "refine an almost-clean map" regime
gets vanishing gradient.

## `peak_saturation_and_threshold.png`

Left panel: across the 16 DDIM sampler steps, mean value of detected
local-maximum peaks (NMS kernel=3, raw threshold>0.5) at eval step
103,370. Mean peak value climbs from 0.911 at k=0 to 0.997 at k=15;
the fraction of peaks with value > 0.95 climbs from 59.7% at k=0 to
**99.4%** at k=15.

Right panel: kept-bin count vs decode threshold for k=0 and k=15.
At k=0 the threshold knob works (kept count drops 28.67 → 15.57 as
threshold rises 0.30 → 0.95). At k=15 the knob is destroyed
(20.97 → 20.80 over the same threshold range). Every detected peak
has saturated to ~1.0, so a confidence-based filter cannot
distinguish true peaks from extra ones.

## Combined finding

The sampler doesn't sharpen broad blobs into singular peaks — it
**bleaches every detected peak to ≈1.0 while leaving the blob count
roughly constant**. The audio conditioning already locates blobs
correctly at k=0 (recall 0.987 at thr=0.3); the diffusion chain then
destroys the confidence signal that would let a downstream decoder
filter out the extras.

The clean threshold sweep at k=0 with NMS=3 reaches density ratio
1.00× and hallucination 0.168 at threshold 0.95 — a usable operating
point exists **only at the first sampler step**, which the AR
decoder doesn't use because the DDIM chain runs to k=15.
