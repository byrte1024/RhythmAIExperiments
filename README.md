# RhythmAIExperiments

Experiments on machine learning applied to music and rhythm.

## Projects

- **[osu/taiko2](osu/taiko2/)** — Active. Onset detection for
  osu!taiko chart generation. Layered codebase with domain ABCs +
  concrete implementations, stable on-disk formats, and per-
  experiment self-contained specs. Best per-step val miss 0.241
  ([#007](osu/taiko2/experiments/007-time-stretch/)); best AR
  `matched_rate` 0.612
  ([#010e](osu/taiko2/experiments/010e-aux-frozen/)). See
  [`osu/taiko2/README.md`](osu/taiko2/README.md).

- **[osu/taiko](osu/taiko/)** — Legacy, no longer under active
  development. Migrated to taiko2. The 121 experiments here remain
  reproducible (74.6 % HIT, 8 ms median AR error) and the trained
  model still works for inference / training against the legacy
  dataset pipeline. Older work — open
  [`osu/taiko/README.md`](osu/taiko/README.md) for the design and
  results from that codebase.

## License

See [`LICENSE.md`](LICENSE.md). Research and educational use only.
You are responsible for the legal use of training data and
generated content in your jurisdiction.
