# Experiments

Each folder contains a README with hypothesis, results, and key graphs.

| # | Name | Status | Key Result |
|---|------|--------|------------|
| 05 | [Gaussian Soft Targets](experiment_05/) | Baseline | First soft target approach, metronome behavior |
| 06 | [Trapezoid Soft Targets + Ablation Benchmarks](experiment_06/) | Best of early arch | 55.8% HIT E1, introduced 8 ablation benchmarks |
| 07 | [Heavy Context Augmentation](experiment_07/) | Failed | 25% dropout + time-warp killed event path entirely |
| 09 | [Reverted to Light Augmentation](experiment_09/) | Mixed | Revealed model over-relies on events over audio |
| 10 | [Two-Path Architecture](experiment_10/) | Bugged | NaN from all-masked attention; benchmarks broken |
| 11 | [Two-Path, NaN Fixed](experiment_11/) | Best so far | 47.1% acc, 64.8% HIT, top-3 86%, audio > events |
| 12 | [Stronger Context Path + AR Aug](experiment_12/) | Failed | Starved audio proposer → mode collapse |
| 13 | [AR Augmentation Only](experiment_13/) | Stopped early | AR aug works (+8-10% corruption resilience), but found BIN_MS data alignment bug |
| 14 | [Corrected Data Alignment](experiment_14/) | Pending | Regenerated dataset with exact BIN_MS; expect miss rate 10-30% |

## Key Lessons

- **BIN_MS=5.0 was wrong** — actual mel frame is 4.9887ms, causing 408ms drift at 3min. Likely the ~46% accuracy ceiling across all experiments (exp 13 discovery)
- **Audio aux loss (0.2) is load-bearing** — reducing it collapses the proposer (exp 12)
- **AR augmentations improve robustness** — recency-scaled jitter + insertions/deletions give +8-10% on corruption benchmarks (exp 13)
- **Heavy event augmentation kills context** — model ignores events entirely (exp 07)
- **NaN from all-masked attention** is silent and devastating (exp 10 → 11)
- **Audio must dominate events** — no_events > no_audio is the healthy regime (exp 11+)
