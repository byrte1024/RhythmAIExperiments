# Experiments

Each folder contains a README with hypothesis, results, and key graphs.

*Experiments 01-04 predate experiment tracking. No data remains.*

| # | Name | Status | Key Result |
|---|------|--------|------------|
| 05 | [Gaussian Soft Targets](experiment_05/) | Baseline | First soft target approach, metronome behavior |
| 06 | [Trapezoid Soft Targets + Ablation Benchmarks](experiment_06/) | Best of early arch | 55.8% HIT E1, introduced 8 ablation benchmarks |
| 07 | [Heavy Context Augmentation](experiment_07/) | Failed | 25% dropout + time-warp killed event path entirely |
| 08 | *(crashed mid-run, identical to 09 — typo renamed it to 09, already committed)* | Crashed | — |
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
| 39-B | [Top-K Reranking Sweep](experiment_39b/) | +0.9pp only | Global proximity bias too blunt — 1,299 regressions per 1,994 improvements. Needs confidence-aware reranking |
| 39-C | [Entropy-Weighted Reranking](experiment_39c/) | +1.0pp ceiling | Entropy weight adds negligible gain. Post-hoc reranking of model's own top-K limited to ~+1pp |
| 39-D | [Top-K Depth Analysis](experiment_39d/) | Key data | 3 HITs per sample (74.5%), correct at rank 1-2 (73.8%) but 2.3x less confident. Confidence gap too large for reranking |
| 39-E | [Audio Analysis of Failures](experiment_39e/) | Saliency bias | Mel energy identical (50/50) but 78-81% of overpredictions have sharper transients at the wrong position. Model picks sharpest, not nearest |
| 40 | [Stronger Balanced Sampling](experiment_40/) | Worse | power=0.7 hurt common predictions (-2.8pp HIT) without helping rare. Distant predictions are inherently ambiguous, not undertrained |
| 41 | [Deep Entropy Analysis](experiment_41/) | **Key insight** | Skip 0=93.7% HIT, skip 1+=0% (11.2% of samples). Underpred=46.5% HIT (19.3%). Context helps (r=-0.21). Fixing skip-1 → ~83% HIT |
| 41-B | [Entropy Progression](experiment_41b/) | Skip rate structural | Entropy/confidence improve with training but overpred rate stuck at ~28%, skip-1 at ~11%. Can't train away — need context or inference fix |
| 42 | [Event Embedding Detector](experiment_42/) | **New ATH 73.2%** | Deepest context (metronome 25.4%). HIT +1.6pp vs [35-C](experiment_35c/). But entropy unchanged — improvements on easy cases only. Skip-1 still 11% |
| 42-AR | [Human Evaluation](experiment_42ar/) | Epic Fail | Blind A/B/C test: 10 people rank AR-generated charts from exp [14](experiment_14/), [35-C](experiment_35c/), [42](experiment_42/). Which model sounds best in practice?, T'was [14](experiment_14/) |
| 42-B | [Pure Hard CE](experiment_42b/) | **Entropy -45%** | hard_alpha=1.0 slashes entropy (2.39→1.32) and flattens distance correlation (+0.58→+0.36). But HIT -4.5pp and skip rate +2.1pp. Confidence ≠ accuracy |
| 43 | [AR-Resilient Training](experiment_43/) | Backfired | ~43% context corruption rate → model distrusts context (-0.4% delta), metronomes from step 0 (11 unique preds). Worse on both per-sample (-3.7pp vs [42](experiment_42/)) and AR. Augmentation must be much gentler |
| 43-B | [AR Resilience Comparison](experiment_43b/) | **Surprise** | Exp [42](experiment_42/) (deepest context) is MOST AR-resilient, not least. Beats exp [14](experiment_14/) at every step. Metronome collapse universal across all models |
| 44 | [Gentle AR Augmentation](experiment_44/) | **New ATH 73.6%** | ~14% context corruption. Distort don't destroy. 2x metronome resilience vs exp [42](experiment_42/). Best HIT/MISS ever. Longer training broke through plateau |
| 44-B | [Metronome Data Analysis](experiment_44b/) | **Key insight** | 43.9% of training targets are "continue the pattern." At streak 8+, 83% correct to continue. Metronome is the statistically optimal prediction |
| 44-C | [Top-K vs Top-U Oracle](experiment_44c/) | Analysis | Model has 2-4 real options (median 4 at 5% conf). Top-U 3 oracle=91.8% HIT. The model knows the answer, can't pick it |
| 44-D | [Temperature Sampling](experiment_44d/) | Analysis | Temperature strictly worse on per-sample metrics. No sweet spot. Value is in AR diversity, not accuracy |
| 45 | [Reliable Density + Gap Ratios](experiment_45/) | Mixed | Gap ratios don't help per-sample. Tighter density jitter improves AR density adherence. Both adopted going forward |
| 46 | [Hard/Soft Loss Ratio Sweep](experiment_46/) | Analysis | hard_alpha is a precision knob, not behavior knob. Same error structure at all settings. 0.50 has worst metronome delta. Over-prediction bias at high alpha |
| 47 | [Binary STOP Head](experiment_47/) | Failed | pos_weight backwards. STOP rate 0% |
| 47-B | [Binary STOP + Focal](experiment_47b/) | Failed | Focal loss mean() drowns STOP. F1=0.066, recall=3% |
| 47-C | [Binary STOP + Balanced Focal](experiment_47c/) | Failed | Balanced averaging fixed per-class, but gate_weight too low + cursor token wrong for STOP |
| 47-D | [Binary STOP + Forward-Pool](experiment_47d/) | Failed | Forward-pool gate, still no learning. Binary head approach fundamentally flawed — STOP wins by elimination in softmax, not active prediction |
| 47-E | [STOP Query Token](experiment_47e/) | Success | Learned STOP token in transformer (251 tokens). 20x STOP sampling boost. F1=0.469, no_audio_stop=81.5%. First working STOP architecture. Shelved — STOP not the bottleneck |
| 48 | [Cross-Model Failure Analysis](experiment_48/) | **Key insight** | 14.2% of failures universal across ALL models. 2x/0.5x metric confusion on ordinary audio. Models predict same wrong bin 80%. Meter awareness needed |
| 49 | [Virtual Tokens](experiment_49/) | Promising | 32 virtual tokens for out-of-window context. 100% AR survival (no metronome ever), but 52% hallucination rate. Solves metronome, creates over-prediction. 72.6% HIT matches exp [44](experiment_44/) |
| 50 | [Anti-Entropy Loss (w=0.1)](experiment_50/) | Sidegrade | Better corruption resilience (+2.6pp metronome, +1.9pp adv met) but HIT capped at 73.2% vs exp [44](experiment_44/)'s [73.6%](experiment_44/README.md). Anti-entropy is a robustness tool, not accuracy tool |
| 50-B | [Anti-Entropy Loss (w=0.5)](experiment_50b/) | Similar | Bimodal entropy (eliminated disambiguation zone). Same HIT ceiling as w=0.1. Stronger pressure doesn't help further |
| 51 | [Streak-Ratio Loss Weighting](experiment_51/) | Failed | Per-sample loss weight based on streak-ratio cell frequency. HIT dropped 6pp (67.5% vs 73.6%). Too much capacity diverted to rare cells. Anti-entropy (exp 50) achieves similar metronome resilience without HIT cost |
| 52 | [Audio Window Size Sweep](experiment_52/) | **Key finding** | 6 configs tested. 250 past sufficient (saves 50% compute). 500 future optimal. 1000 future breaks STOP. 33 future spams. B=250 gives healthiest density dependence. Best: 250/500 matches exp 45 at 0.56x cost |
| 53 | [B_AUDIO/B_PRED Split](experiment_53/) | Plateau | A=250, B_AUDIO=500, B_PRED=250. 72.1% HIT (=exp 45). Better benchmarks but A_BINS=250 may be bottleneck |
| 53-AR | [Human Evaluation Round 2](experiment_53ar/) | Complete | 15 votes: exp45 wins (44pts), exp44 close 2nd (43pts). Context models overtake audio-only. Expert vs volunteer preference diverges |
| 53-B | [B_AUDIO/B_PRED + A_BINS=500](experiment_53b/) | Confirmed | A=500, B_AUDIO=500, B_PRED=250. A_BINS=250 was the bottleneck — 73.4% HIT breaks 72.1% ceiling. Best audio-only acc ever (49.7%) |
| 54 | [B_AUDIO/B_PRED + STOP Query Token](experiment_54/) | Failed | Separate STOP head (F1=0.39) underperforms softmax STOP (F1=0.48). 20x boost steals onset samples for no gain. STOP head experiments exhausted |
| 55 | [Auxiliary Ratio Head](experiment_55/) | Modest gain | Training-only log10-ratio head (201 bins). 73.6% HIT (ties ATH), best val loss ever (2.461), faster convergence. Doesn't break ceiling |
| 56 | [Density Conditioning AR Analysis](experiment_56/) | Complete | AR on 48 val songs. Model under-predicts density (0.83x avg). Low density = high hallucination, high density = conservative |
| 56-B | [Density Sensitivity Sweep](experiment_56b/) | Complete | Model IS density-sensitive (1.53x ratio). Needs ~1.2x density to match reality. Under-prediction is calibration, not deafness |
| 57 | [1:1 Virtual Context Tokens](experiment_57/) | Failed | 128 vtokens contribute 0%. Future audio is everything (NA_B=1.3%). Gap encoding is foundation of event embeddings. 8 new benchmarks added |
| 58 | [Two-Stage Propose-Select](experiment_58/) | **New ATH 74.6%** | Stage 1 proposes, Stage 2 selects. Breaks 73.7% ceiling. 7 consecutive improvements, zero oscillations. S1 proposals load-bearing (50pp delta) |
| 58-B | [Propose-Select, Precision S1](experiment_58b/) | 1pp below | S1 pos_weight 2.0. Fewer proposals (19 vs 67), S2 more independent (override 53% vs 49%), but peak 73.6% vs 58's 74.6% |
| 59 | [AR Quality Metric Discovery](experiment_59/) | No signal | Raw chart metrics don't correlate with human preference. Per-song confound dominates |
| 59-B | [Within-Song Normalized Metrics](experiment_59b/) | **Key finding** | gap_std (+0.30), gap_cv (+0.29), dominant_gap_pct (-0.27), max_metro_streak (-0.27) all significant. Pattern variety predicts human preference |
| 59-C | [Synthetic Human Evaluator](experiment_59c/) | Works | gap_std + gap_cv predicts #1 at 52% (2x random). Volunteers 60%. Spearman r=0.35, p=0.001 |
| 59-D | [Self vs Volunteer Metric Split](experiment_59d/) | Divergent | Expert=gap_cv (+0.38), Volunteers=dominant_gap_pct (-0.42). Expert values proportional variety, volunteers punish boredom |
| 59-E | [Split Synthetic Evaluators](experiment_59e/) | Validated | Tuned evaluators outperform on target: volunteer=60% #1 (r=0.47), expert=47% (tau=0.42). Cross-eval drops to random |
| 59-F | [Evaluator Weight Sweep](experiment_59f/) | Optimized | Expert=top-2 (47%, any temp). Volunteer=top-7 temp=0.5 (70% #1 match). Temperature barely matters |
| 59-G | [Fresh Data Validation](experiment_59g/) | Partial | Expert+song_density nails top-2. Others swap exp44/exp45 (1pt human margin). Coarse ranking generalizes, tight races need humans |
| 59-H | [Extended Model Comparison](experiment_59h/) | Surprising | exp51 (67.5% HIT, worst per-sample) dominates synthetic eval by 100+ pts. Per-sample accuracy inversely correlated with pattern variety. Needs human validation |
| 59-HB | [GT Comparison for 59-H](experiment_59hb/) | Resolved | exp51's variety is from under-prediction (0.64 density ratio), not creativity. exp58 is actual best (75.9% close, 0.92 d_ratio). Synthetic evaluator needs density filter |
| 60 | [DDC Onset Comparison](experiment_60/) | Complete | DDC Oracle (density-matched): 77.1% close vs exp58 75.9%. But exp58 has 3.4x better timing (8ms vs 27ms) and 4.3pp less hallucination. All difficulties tested |
| 61 | [TaikoNation Eval Metrics](experiment_61/) | Complete | DCHuman 90.8% vs TaikoNation 75.0% (we win placement). Over.P-Space 10.1% vs 21.3% (they win diversity, but human is 11.7% — we're closer to human) |
| 44-RE | [Reproducibility Verification](experiment_44re/) | **Pending** | Exact exp 44 reproduction on CachyOS/RTX 4060. Verifies cross-machine reproducibility |
| 62 | [Multi-Onset Prediction](experiment_62/) | **Running** | Predict 4 onsets simultaneously (inspired by TaikoNation). o1=74.0% at eval 4, o4=40.1%. strict_increasing=67.4% |
| 63 | [TaikoNation Direct Comparison](experiment_63/) | **Pending** | Run TaikoNation with original weights on our 30 val songs. Same-song comparison instead of cross-paper |

## Key Lessons

- **Data quality > model complexity** — fixing BIN_MS (5.0→4.9887) had more impact than every architecture/loss/augmentation change combined. Source: [exp 14](experiment_14/README.md)
- **BIN_MS=5.0 was wrong** — actual mel frame is 4.9887ms, causing 408ms drift at 3min. Was the ~46% accuracy ceiling. Source: [exp 13](experiment_13/README.md) (discovered), [exp 14](experiment_14/README.md) (fixed)
- **The model was rational about bad data** — it relied on events over audio because audio was genuinely misaligned. Heavy augmentation was catastrophic because it corrupted the only reliable signal. Source: [exp 07](experiment_07/README.md), [exp 09](experiment_09/README.md)
- **Audio aux loss (0.2) is load-bearing** — reducing it collapses the proposer. Source: [exp 12](experiment_12/README.md)
- **AR augmentations improve robustness** — recency-scaled jitter + insertions/deletions give +8-10% on corruption benchmarks. Source: [exp 13](experiment_13/README.md)
- **NaN from all-masked attention** is silent and devastating. Source: [exp 10](experiment_10/README.md) (bug), [exp 11](experiment_11/README.md) (fix)
- **Context path is currently dormant** — no_events ≈ full accuracy at E8. The ~50% ceiling is audio-only; breaking it requires activating context + density. Source: [exp 14](experiment_14/README.md)
- **Density conditioning is load-bearing** — zero_density drops accuracy by ~25pp, random_density by ~8pp. FiLM conditioning is the second most important signal after audio. Source: [exp 15](experiment_15/README.md)
- **Can't aux-loss out of rubber-stamping** — 0.1 context aux CE had zero effect on context engagement over 4 epochs. The local minimum of "agree with audio" is stable under standard CE. Source: [exp 15](experiment_15/README.md)
- **Wrong opinions worse than no opinions** — rank-weighted context loss forced context to have strong opinions, which corrupted audio's correct rankings and dropped top-K by 3-5pp. Loss-function approaches can't solve a structural problem. Source: [exp 16](experiment_16/README.md)
- **Activation ≠ value** — top-K reranking forced context to engage (50% override rate, first ever), but override accuracy plateaued at 51% (coin flip). Shared encoder gradients degraded audio quality, netting -7.5pp accuracy vs audio-only. Source: [exp 17](experiment_17/README.md)
- **Stop-gradient works, two-stage doesn't help** — gradient isolation protects audio HIT (67.5% at E1), proven reliable. But two-stage architecture with shared encoder features still produced harmful overrides (35% accuracy, -0.94pp). Source: [exp 18](experiment_18/README.md)
- **Gap representation is the right framing** — inter-onset intervals + local audio snippets + own encoders produced the first context that beat audio during training. Delta reached -0.18pp at E2 (best ever). Source: [exp 19](experiment_19/README.md)
- **Warm-start + freeze are proven infrastructure** — audio at 69.5% from step 1, 2x training speed, no degradation risk. Context overrides more (11%) with best F1 ever (22%), but delta -1.18pp. Source: [exp 20](experiment_20/README.md)
- **Relative quality loss works but framing is wrong** — soft targets weighted by closeness doubled override F1 (22%→46%) and pushed accuracy above coin flip (61%) for the first time. But any loss referencing audio's baseline creates conservatism. Source: [exp 21](experiment_21/README.md)
- **Audio confidence is informative, not just bias** — fully blind context (no scores, shuffled) overrides 50%+ uniformly and can't distinguish high-confidence from low-confidence audio picks. Delta -4.6pp despite reasonable per-override quality. Source: [exp 22](experiment_22/README.md)
- **Reranking is fundamentally limited** — 9 experiments across 4 loss functions, 3 info levels, 4 architectures. Best result: 64% override accuracy, 4:1 good:bad ratio, still -3.2pp delta. The discrete override interface can't break even against a 70%-correct base model. Source: [exp 15](experiment_15/README.md)–[exp 23](experiment_23/README.md)
- **Separate paths cannot break even** — 10 experiments across reranking AND additive logits. Additive was safer (best delta -0.64pp vs reranking's -0.77pp) but still negative. Context without audio access caps at ~53% HIT. The architecture must be unified. Source: [exp 15](experiment_15/README.md)–[exp 24](experiment_24/README.md)
- **Unification doesn't guarantee fusion** — putting audio and gap tokens in the same self-attention doesn't mean the model learns to use both. Audio is a shortcut. Context delta collapses from ~7% to ~1.5% in every unified experiment. Source: [exp 25](experiment_25/README.md)–[exp 27](experiment_27/README.md)
- **Audio augmentation is orthogonal to context usage** — heavier augmentation delays overfitting (~3 extra epochs) but doesn't increase context contribution. The model adapts to noise within the audio pathway. Source: [exp 26](experiment_26/README.md)
- **Data diversity raises the ceiling modestly** — full dataset (4x) pushed 68.9%→69.8% HIT and delayed overfitting, but context delta still collapsed. Source: [exp 27](experiment_27/README.md)
- **Context has the answer** — 95% of misses with sufficient history have the target gap value in context. Top-10 accuracy is 96%. The model narrows to the right candidates but can't pick between them. Source: [exp 27-B](experiment_27b/README.md). See also [THE_CONTEXT_ISSUE.md](../THE_CONTEXT_ISSUE.md)
- **Mel-embedded event ramps break the context barrier** — exponential decay spikes at event positions + amplitude jitter produced 71.6% HIT with sustained 4.5-5.7% context delta. First to break 70% AND keep context. Source: [exp 35-C](experiment_35c/README.md)
- **Event embeddings are the optimal context mechanism** — learned embeddings scatter-added to audio tokens at event positions. 73.2% HIT (ATH), deepest context dependency. Source: [exp 42](experiment_42/README.md)
- **Per-sample accuracy ≠ AR quality** — blind human evaluation: exp 14 (68.9% HIT, no context) beat exp 42 (73.2% HIT, deepest context) decisively. Metronome regression from context dependency is the #1 quality problem. Source: [exp 42-AR](experiment_42ar/README.md)
- **Metronome behavior is statistically optimal** — 43.9% of training targets are "continue the pattern." At streak 8+, 83% correct to continue. The model isn't broken, it's learning the dominant pattern. Source: [exp 44-B](experiment_44b/README.md)
- **The model knows the answer but can't pick it** — Top-U 3 oracle = 91.8% HIT. The model considers 2-4 real options per prediction. The 18pp gap between "considered" and "selected" is the key opportunity. Source: [exp 44-C](experiment_44c/README.md)
- **14.2% of failures are structurally unsolvable** — all architectures fail on the same samples with the same 2x/0.5x metric confusion. The model can't distinguish beat from sub-beat. Source: [exp 48](experiment_48/README.md)
- **Virtual tokens solve metronome collapse** — 100% AR survival at step 30 (unprecedented). But 52% hallucination rate — the model over-predicts at sub-beat positions. Source: [exp 49](experiment_49/README.md)
- **Anti-entropy is a robustness tool, not accuracy tool** — entropy penalty improves corruption resilience (+2.6pp metronome) but caps HIT at 73.2%. Eliminating the disambiguation zone doesn't improve accuracy. Source: [exp 50](experiment_50/README.md), [exp 50-B](experiment_50b/README.md)
- **Smaller prediction windows converge faster** — 33-bin window: HIT 74.2% (ATH) at eval 2. But model needs future audio for AR quality — without it, degrades to transient-spamming. Source: [exp 52-L](experiment_52/README.md)
- **Two-stage propose-select breaks the HIT ceiling** — Stage 1 (pure audio) proposes onset candidates, Stage 2 (full context) selects. 74.6% HIT, 7 consecutive improvements, zero oscillations. Proposals are load-bearing (50pp delta when zeroed). Source: [exp 58](experiment_58/README.md)
- **Pattern variety predicts human preference — but non-linearly** — gap_std (+0.30), gap_cv (+0.29), dominant_gap_pct (-0.27), max_metro_streak (-0.27) correlate with human rankings. But the relationship is not linear: too little variety is metronomic (bad), too much variety indicates under-prediction/noise (also bad). Synthetic evaluators built from these metrics are useful as screening tools alongside GT matching and human data, not as standalone judges. Source: [exp 59-B](experiment_59b/README.md), [exp 59-HB](experiment_59hb/README.md)
