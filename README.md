# RhythmAIExperiments

Experiments on machine learning applied to music and rhythm.

## Projects

- **[osu/taiko2](osu/taiko2/)** — Active. Onset detection for
  osu!taiko chart generation. Layered codebase with domain ABCs +
  concrete implementations, stable on-disk formats, and per-
  experiment self-contained specs. Best per-step val miss 0.2406
  and best AR `matched_rate` 0.7061, both at
  [#007](osu/taiko2/experiments/007-time-stretch/)
  [exp_007_time_stretch, step 372,132, val/single/onset/miss
  for the val-miss number; step 413,480,
  infer_corpus/eval_413480/gt_cond/comparisons_summary.json:fields.matched_rate.median
  for the AR number]. See
  [`osu/taiko2/README.md`](osu/taiko2/README.md).

- **[osu/taiko](osu/taiko/)** — Legacy, no longer under active
  development. Migrated to taiko2. The 124 experiment directories
  here [`ls osu/taiko/experiments/ | wc -l`] remain reproducible
  (74.6 % HIT, 8 ms median AR error per
  [`osu/taiko/PERFORMANCE.md`](osu/taiko/PERFORMANCE.md)) and
  the trained model still works for inference / training against
  the legacy dataset pipeline. Older work — open
  [`osu/taiko/README.md`](osu/taiko/README.md) for the design and
  results from that codebase.

## License

Source code is licensed under the [PolyForm Noncommercial
License 1.0.0](LICENSE). See [`LICENSE.md`](LICENSE.md) for a
project-specific notice about scope and training-data
responsibility. You are responsible for the legal use of
training data and generated content in your jurisdiction.
