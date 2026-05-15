# custom: K-step regression

`k_step_f1_per_eval.png` — mean F1 (τ=0.5, ±2 frames) at each of the
16 DDIM sampler steps, plotted across all 5 evals. Built from
`runs/exp_016_framewise_diffusion/eval_{step}/rollout_maps.npz:f1`
(shape `(160 samples, 16 steps)`, mean over the sample axis).

The line for each eval peaks at k≈0–3 and decreases monotonically to
k=15. The gap between best and final widens across evals — at step
20,674 final F1 sits 0.083 below best; at step 103,370 it sits 0.092
below best. More training makes the K-step trajectory regress more,
not less, because the high-t end of the schedule (where k=0 lives)
gets the bulk of the gradient signal while the low-t end (where k=15
lives) stays untrained — see [`../per_t_loss_imbalance/`](../per_t_loss_imbalance/).
