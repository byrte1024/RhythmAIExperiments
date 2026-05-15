# custom: Min-SNR sign delta vs #016

Three plots that compare eval 1 of this probe against eval 1 of
#016, isolating the effect of the formula switch.

- `k_step_f1_eval1_vs_016.png` — mean F1 across the 16 DDIM sampler
  steps. Both runs regress monotonically; the gap from k=0 to k=15
  shrinks from −0.066 (#016 eval 1) to −0.050 (#016b-mini eval 1),
  i.e. the sign fix produces a flatter regression curve but not a
  rising one.
- `per_t_quartile_eval1_vs_016.png` — `loss/per_t_q0..3` at eval 1
  for both runs. The bars show the formula switch **flips which
  half is starved**: under the ε-form, q0 is 0.0033 (low-t under-
  trained but task is easy → small loss anyway) and q3 is 0.0724
  (high-t trained); under the x0-form, q0 is 0.0044 (low-t now
  trained) and q3 is 0.1591 (high-t starved → loss climbed 2.2×).
  Both forms produce an asymmetric per-t profile, just in opposite
  directions. q3/q0 ratio went 21.9× → 35.9×.
- `peak_value_dist_k0_k15.png` — histograms of local-max-bin values
  (NMS=3, raw > 0.5) at k=0 and k=15 for both runs. At k=0 both
  runs have graded distributions. At k=15 both runs collapse to
  ≈1.0 — the saturation pathology is **unchanged** by the formula
  switch. Mean value at k=15: #016 0.997 vs #016b-mini 0.997.
  Frac > 0.95 at k=15: #016 99.4 % vs #016b-mini 98.7 %.

Combined finding: the Min-SNR sign was a real bug (the F1 chain is
~25 % less negative on `final_vs_best_delta` and the q3/q0 ratio
swung past balance to the opposite extreme) but it was **not the
dominant cause** of #016's failure. The saturation behaviour
survives the fix.
