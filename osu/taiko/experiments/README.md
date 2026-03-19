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
| 14 | [Corrected Data Alignment](experiment_14/) | **New best** | 50.5% acc, 69% HIT, 30% miss. E1 beat all prior exps. Context path dormant - 50% is audio-only ceiling |
| 15 | [Context Aux Loss + Density Benchmarks](experiment_15/) | Failed | 0.1 context aux didn't break rubber-stamping. Density benchmarks revealed FiLM is load-bearing (~25pp) |
| 16 | [Rank-Weighted Context Loss](experiment_16/) | Failed | Forced opinions degraded combined output. Val loss increasing, top-K dropped 3-5pp. Wrong opinions worse than no opinions |
| 17 | [Top-K Reranking Architecture](experiment_17/) | Partial | First context activation ever (50% override), but override accuracy ~51% (coin flip). 43% acc, 65.3% HIT - below exp 14's audio-only 69% |
| 18 | [Gradient-Isolated Context + Two-Stage Event Focus](experiment_18/) | Failed | Stop-gradient works (audio protected), but context overrides net-harmful (-0.94pp). 35% override accuracy, worse than coin flip. Reranking paradigm may be fundamentally flawed |
| 19 | [Gap-Based Context with Own Encoders](experiment_19/) | Hopeful | First context to beat audio during training. Gap repr + snippets + own encoders. Delta -0.18pp at E2 (best ever), but plateaued. Unstable proposer poisons selector |
| 20 | [Warm-Start + Frozen Audio, Context-Only](experiment_20/) | Infra win | Warm-start + freeze work (69.5% HIT, 2x speed). Context bolder (11% override, F1=22%) but delta -1.18pp. Loss function is now the bottleneck |
| 21 | [Relative Quality Selection Loss](experiment_21/) | Promising | Best override quality ever (F1=46%, acc=61%), but delta still -0.95pp. Conservatism bias in loss design. Context shouldn't know audio's preferences |
| 22 | [Blind Selection (Shuffled, No Scores)](experiment_22/) | Too aggressive | Shuffling works, but no confidence signal → 50%+ override rate, delta -4.6pp. Context picks reasonably but overrides too often. Need scores back as scalar |
| 23 | [Shuffled Candidates + Confidence](experiment_23/) | Best reranking | Best override quality ever (F1=67%, acc=64%, 4:1 ratio). But delta still -3.2pp. Proves reranking paradigm fundamentally limited after 9 experiments |
| 24 | [Additive Context Logits](experiment_24/) | Best delta | Additive logits safer than reranking (best delta -0.64pp), but context without audio access caps at ~53% HIT. Proves separate paths cannot break even - unification needed |
| 25 | [Unified Audio + Gap Fusion](experiment_25/) | Matched exp 14 | Matched ~68.6% HIT but overfitting from E2, context delta collapsed 6.8%→2.3%. Unified fusion alone doesn't force context usage |
| 26 | [Heavy Audio Augmentation](experiment_26/) | Same ceiling | Overfitting delayed ~3 epochs but same ~68.8% HIT ceiling. Context delta still collapsed (1.7%). Audio augmentation orthogonal to context usage |
| 27 | [Full Dataset (No Subsample)](experiment_27/) | **New best** | **69.8% HIT** — broke ~69% ceiling. Top-10 acc 96%, but context delta still 1.5%. Overfitting delayed further. Data diversity helps but doesn't fix context |
| 27-B | [Context Pattern Analysis](experiment_27b/) | Diagnostic | 95% of misses have target in context. Strict pattern matching catches 22.5%, but manual inspection shows far more are solvable. Context has the answer — model doesn't use it |
| 28 | [Focal Loss](experiment_28/) | Better calibration | Best entropy separation ever, Stop F1 0.552, but HIT ceiling 68.6% (~1pp below exp 27). Context delta zero. Loss reweighting doesn't force context usage |
| 29 | [Auxiliary Context Loss](experiment_29/) | Weight too low | ctx_loss_weight=0.2 too weak — aux head barely learned (loss 4.2→4.1), context delta collapsed as usual. Gradient dominated by fusion |
| 29-B | [Aux Context Loss (Weight 1.0)](experiment_29b/) | Same failure | Weight 1.0 still can't teach aux head — 501-class gap-only prediction is unsolvable. Ctx loss barely dropped |
| 30 | [Cursor-Region Audio Masking](experiment_30/) | Killed early | Masking works as regularization but context delta still collapsing (6.8%→3.3%). Model detects zeroed mel as a mode switch. 16 experiments confirm: training tricks can't overcome architectural audio dominance |
| 31 | [Dual-Stream Cross-Attention](experiment_31/) | Context works, bottleneck | 18.8% context delta (highest ever!) but only ~80 unique predictions — 2 cross-attention layers too narrow for fine-grained info flow |
| 31-B | [Dual-Stream 4x Cross-Attention](experiment_31b/) | Worse + NaN | 4 layers too deep — 18.6% HIT, 37 unique preds, NaN instability. Gap activations (±20) overwhelm audio (±7) through residual path |
| 32 | [Dual-Stream + Audio Skip Connection](experiment_32/) | Banding fixed, ctx killed | 363 unique preds (banding gone) but -1.8% context delta — skip connection becomes audio shortcut bypassing cross-attention |
| 33 | [Interleaved Self+Cross Attention](experiment_33/) | Cold start failure | 19% HIT after 5 evals — model can't bootstrap. Cross-attn between random streams at every layer prevents learning. Cross-attention is the wrong fusion mechanism |
| 34 | [Context as FiLM Conditioning](experiment_34/) | Too weak | Clean architecture (467 unique, 66.5% HIT) but 4.2% context delta — FiLM bottleneck (64-dim) can't encode sequential patterns. Context needs temporal embedding |
| 35 | [Mel-Embedded Event Ramps](experiment_35/) | Subtle | 5.0% context delta — edge-band ramps too easy for conv to filter. Ramps need to be everywhere |
| 35-B | [Full-Band Mel Ramps (Nuclear)](experiment_35b/) | Best sustained Δ | 3.5-5% context delta (best non-cross-attn). Linear ramps too gradual, high entropy from fixed 0.5x audio |
| 35-C | [Exponential Ramps + Amplitude Jitter](experiment_35c/) | **BREAKTHROUGH** | **71.6% HIT** (new ATH), sustained 4.5-5.7% context delta. First to break 70% AND keep context. Entropy/2.0x errors remain |
| 35-D | [Exponential Ramps + Focal γ=3](experiment_35d/) | Too aggressive | 64.0% HIT (-2.2pp vs 35-C). Focal gamma=3 suppressed easy-sample gradients too early, decreased entropy globally (not just hard cases). 2.0x band is structural, not a loss problem |
| 36 | [Multi-Target + Threshold Inference](experiment_36/) | Threshold bottleneck | Nearest HIT=66.2% (=35-C) but event recall 8.2%. Normalized soft targets dilute per-onset gradient |
| 36-B | [Multi-Target + Recall Loss](experiment_36b/) | Softmax bottleneck | Precision +10.8pp but recall still 9.5%. Softmax competition prevents multi-target — needs per-bin sigmoid |
| 37 | [Per-Bin Sigmoid Multi-Target](experiment_37/) | Over/underprediction | Focal γ=2 → nothing (3% HIT). No focal + pos_weight=5 → everything (468 preds/win). Soft targets + pos_weight too aggressive |
| 37-B | [Sigmoid (pos_weight=1.0)](experiment_37b/) | Same overprediction | 460 preds/win even at pos_weight=1.0. Soft targets + sigmoid BCE fundamentally broken — doesn't incentivize sparsity |
| 37-C | [Focal Dice Multi-Target](experiment_37c/) | Slow, architecture wrong | Dice avoids extremes but 5.4% HIT. Single cursor → 501 logits is wrong for multi-target. Need architectural redesign |
| 38 | [Framewise Onset Detection](experiment_38/) | Dice degenerate | Dice smooth term → all-zero predictions (0% recall). Dice wrong for sparse binary detection |
| 38-B | [Framewise + Weighted BCE](experiment_38b/) | Overpredicts | 24% recall (model learns!) but 46.7 preds/win (3x real). Fixed causal mask bug. pos_weight=7 too aggressive |
| 38-C | [Framewise + Unweighted BCE](experiment_38c/) | F1 too low | Better precision (11.7% vs 8.4%) but F1=0.156 still impractical. Framewise approach exhausted — returning to single-target (35-C) |
| 39 | [Overprediction Analysis](experiment_39/) | **Key insight** | **83.2% of overpredictions match real future onsets.** Model sees onset landscape but picks wrong one. Theoretical ceiling: 86.5% (+14.9pp) |
| 39-B | [Top-K Reranking Sweep](experiment_39b/) | Pending | Rerank top-K by confidence × proximity. Find weight combo that maximizes HIT without breaking existing hits |

## Key Lessons

- **Data quality > model complexity** - fixing BIN_MS (5.0→4.9887) had more impact than every architecture/loss/augmentation change combined (exp 14)
- **BIN_MS=5.0 was wrong** - actual mel frame is 4.9887ms, causing 408ms drift at 3min. Was the ~46% accuracy ceiling across exp 05-13
- **The model was rational about bad data** - it relied on events over audio because audio was genuinely misaligned. Heavy augmentation (exp 07) was catastrophic because it corrupted the only reliable signal
- **Audio aux loss (0.2) is load-bearing** - reducing it collapses the proposer (exp 12)
- **AR augmentations improve robustness** - recency-scaled jitter + insertions/deletions give +8-10% on corruption benchmarks (exp 13)
- **NaN from all-masked attention** is silent and devastating (exp 10 → 11)
- **Context path is currently dormant** - no_events ≈ full accuracy at exp 14 E8. The ~50% ceiling is audio-only; breaking it requires activating context + density
- **Density conditioning is load-bearing** - zero_density drops accuracy by ~25pp, random_density by ~8pp. FiLM conditioning is the second most important signal after audio (exp 15)
- **Can't aux-loss out of rubber-stamping** - 0.1 context aux CE had zero effect on context engagement over 4 epochs (exp 15). The local minimum of "agree with audio" is stable under standard CE
- **Wrong opinions worse than no opinions** - rank-weighted context loss forced context to have strong opinions, which corrupted audio's correct rankings and dropped top-K by 3-5pp (exp 16). Loss-function approaches can't solve a structural problem
- **Activation ≠ value** - top-K reranking forced context to engage (50% override rate, first ever), but override accuracy plateaued at 51% (coin flip). Shared encoder gradients degraded audio quality, netting -7.5pp accuracy vs audio-only exp 14 (exp 17). Next: full path separation with stop-gradient
- **Stop-gradient works, two-stage doesn't help** - gradient isolation protects audio HIT (67.5% at E1), proven reliable (exp 18). But two-stage architecture with shared encoder features still produced harmful overrides (35% accuracy, -0.94pp). The problem isn't architecture - it's that shared encoder features aren't shaped for context's task
- **Gap representation is the right framing** - inter-onset intervals + local audio snippets + own encoders produced the first context that beat audio during training (exp 19). Delta reached -0.18pp at E2 (best ever). But an unstable proposer (audio still learning) poisons the selector - context wastes capacity on noisy early proposals
- **Warm-start + freeze are proven infrastructure** - audio at 69.5% from step 1, 2x training speed, no degradation risk (exp 20). Context overrides more (11%) with best F1 ever (22%), but delta -1.18pp. Architecture and training dynamics are solved - the loss function is the bottleneck. Hard CE only rewards exact matches, not "improvement over audio"
- **Relative quality loss works but framing is wrong** - soft targets weighted by closeness doubled override F1 (22%→46%) and pushed accuracy above coin flip (61%) for the first time (exp 21). But any loss referencing audio's baseline creates conservatism. Context should be blind to audio's preferences - shuffle candidates, strip scores, make it "pick the best position" not "should I override?"
- **Audio confidence is informative, not just bias** - fully blind context (no scores, shuffled) overrides 50%+ uniformly and can't distinguish high-confidence from low-confidence audio picks (exp 22). Delta -4.6pp despite reasonable per-override quality. Confidence as a per-candidate scalar (shuffled) gives context the "when to override" signal without rank-ordering bias
- **Reranking is fundamentally limited** - 9 experiments (15-23) across 4 loss functions, 3 info levels, 4 architectures. Best result: 64% override accuracy, 4:1 good:bad ratio, still -3.2pp delta (exp 23). The discrete override interface can't break even against a 70%-correct base model. Context features (gaps, snippets, own encoders) are proven useful - the output interface is the bottleneck
- **Separate paths cannot break even** - 10 experiments (15-24) across reranking AND additive logits. Additive was safer (best delta -0.64pp vs reranking's -0.77pp) but still negative. Context without audio access caps at ~53% HIT - it can learn rhythm patterns but can't know when audio is already correct (70%). Context's influence is random w.r.t. audio correctness, guaranteeing helped ≈ hurt. The architecture must be unified: audio and context features jointly attended, not separate paths combined post-hoc
- **Unification doesn't guarantee fusion** - putting audio and gap tokens in the same self-attention doesn't mean the model learns to use both. Audio is a shortcut: more immediately informative, stronger gradients. Context delta collapses from ~7% to ~1.5% in every unified experiment (exp 25-27)
- **Audio augmentation is orthogonal to context usage** - heavier augmentation delays overfitting (~3 extra epochs) but doesn't increase context contribution. The model adapts to noise within the audio pathway (exp 26)
- **Data diversity raises the ceiling modestly** - full dataset (4x) pushed 68.9%→69.8% HIT and delayed overfitting, but context delta still collapsed. Worth keeping for cleaner signal (exp 27)
- **Context has the answer** - 95% of misses with sufficient history have the target gap value in context. Top-10 accuracy is 96%. The model narrows to the right candidates but can't pick between them. This is the pattern disambiguation problem (exp 27-B). See [THE_CONTEXT_ISSUE.md](../THE_CONTEXT_ISSUE.md)
