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
| 14 | [Corrected Data Alignment](experiment_14/) | **New best** | 50.5% acc, 69% HIT, 30% miss. E1 beat all prior exps. Context path dormant — 50% is audio-only ceiling |
| 15 | [Context Aux Loss + Density Benchmarks](experiment_15/) | Failed | 0.1 context aux didn't break rubber-stamping. Density benchmarks revealed FiLM is load-bearing (~25pp) |
| 16 | [Rank-Weighted Context Loss](experiment_16/) | Failed | Forced opinions degraded combined output. Val loss increasing, top-K dropped 3-5pp. Wrong opinions worse than no opinions |
| 17 | [Top-K Reranking Architecture](experiment_17/) | Partial | First context activation ever (50% override), but override accuracy ~51% (coin flip). 43% acc, 65.3% HIT — below exp 14's audio-only 69% |
| 18 | [Gradient-Isolated Context + Two-Stage Event Focus](experiment_18/) | Failed | Stop-gradient works (audio protected), but context overrides net-harmful (-0.94pp). 35% override accuracy, worse than coin flip. Reranking paradigm may be fundamentally flawed |
| 19 | [Gap-Based Context with Own Encoders](experiment_19/) | Hopeful | First context to beat audio during training. Gap repr + snippets + own encoders. Delta -0.18pp at E2 (best ever), but plateaued. Unstable proposer poisons selector |
| 20 | [Warm-Start + Frozen Audio, Context-Only](experiment_20/) | Infra win | Warm-start + freeze work (69.5% HIT, 2x speed). Context bolder (11% override, F1=22%) but delta -1.18pp. Loss function is now the bottleneck |
| 21 | [Relative Quality Selection Loss](experiment_21/) | Running | Soft targets weighted by closeness improvement over audio's #1. Asymmetric miss_penalty=2x for conservatism. Same warm-start + freeze infra |

## Key Lessons

- **Data quality > model complexity** — fixing BIN_MS (5.0→4.9887) had more impact than every architecture/loss/augmentation change combined (exp 14)
- **BIN_MS=5.0 was wrong** — actual mel frame is 4.9887ms, causing 408ms drift at 3min. Was the ~46% accuracy ceiling across exp 05-13
- **The model was rational about bad data** — it relied on events over audio because audio was genuinely misaligned. Heavy augmentation (exp 07) was catastrophic because it corrupted the only reliable signal
- **Audio aux loss (0.2) is load-bearing** — reducing it collapses the proposer (exp 12)
- **AR augmentations improve robustness** — recency-scaled jitter + insertions/deletions give +8-10% on corruption benchmarks (exp 13)
- **NaN from all-masked attention** is silent and devastating (exp 10 → 11)
- **Context path is currently dormant** — no_events ≈ full accuracy at exp 14 E8. The ~50% ceiling is audio-only; breaking it requires activating context + density
- **Density conditioning is load-bearing** — zero_density drops accuracy by ~25pp, random_density by ~8pp. FiLM conditioning is the second most important signal after audio (exp 15)
- **Can't aux-loss out of rubber-stamping** — 0.1 context aux CE had zero effect on context engagement over 4 epochs (exp 15). The local minimum of "agree with audio" is stable under standard CE
- **Wrong opinions worse than no opinions** — rank-weighted context loss forced context to have strong opinions, which corrupted audio's correct rankings and dropped top-K by 3-5pp (exp 16). Loss-function approaches can't solve a structural problem
- **Activation ≠ value** — top-K reranking forced context to engage (50% override rate, first ever), but override accuracy plateaued at 51% (coin flip). Shared encoder gradients degraded audio quality, netting -7.5pp accuracy vs audio-only exp 14 (exp 17). Next: full path separation with stop-gradient
- **Stop-gradient works, two-stage doesn't help** — gradient isolation protects audio HIT (67.5% at E1), proven reliable (exp 18). But two-stage architecture with shared encoder features still produced harmful overrides (35% accuracy, -0.94pp). The problem isn't architecture — it's that shared encoder features aren't shaped for context's task
- **Gap representation is the right framing** — inter-onset intervals + local audio snippets + own encoders produced the first context that beat audio during training (exp 19). Delta reached -0.18pp at E2 (best ever). But an unstable proposer (audio still learning) poisons the selector — context wastes capacity on noisy early proposals
- **Warm-start + freeze are proven infrastructure** — audio at 69.5% from step 1, 2x training speed, no degradation risk (exp 20). Context overrides more (11%) with best F1 ever (22%), but delta -1.18pp. Architecture and training dynamics are solved — the loss function is the bottleneck. Hard CE only rewards exact matches, not "improvement over audio"
