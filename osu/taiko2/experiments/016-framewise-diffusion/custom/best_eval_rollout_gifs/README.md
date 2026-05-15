# custom: best-eval rollout GIFs

Verbatim copies of `runs/exp_016_framewise_diffusion/eval_103370/rollout_gifs/*.gif`
— five representative DDIM rollouts at the final eval (step 103,370).
File names encode quintile and final F1, e.g.
`sample_best_idx008_f10.930.gif` is the highest-F1 sample at idx 8.

Each GIF shows the 16-step progression of the predicted activation
map `M_k` (overlaid on the GT binary map and target Gaussian-σ=2
smoothing). The visual story matches the numbers in
[`../per_t_loss_imbalance/`](../per_t_loss_imbalance/):

- Early frames (k=0–3) — broad, graded peaks near GT locations.
- Middle frames (k=4–10) — peaks rising toward 1.0; blob widths
  staying roughly constant.
- Late frames (k=11–15) — every peak saturated to 1.0; no
  sharpening; the AR decoder's threshold knob has nothing to bite.

The worst-quartile sample (`sample_worst_idx155_f10.109.gif`) shows
the same failure shape on a dense chart where the broad peaks collide
into one blob plateau across many bins, making singular peaks
impossible to recover by thresholding.
