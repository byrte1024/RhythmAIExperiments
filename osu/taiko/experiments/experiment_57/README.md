# Experiment 57 - 1:1 Virtual Context Tokens

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

All previous event embedding experiments scatter-add event embeddings onto audio tokens. This means event information and audio information share the same token — the transformer must disentangle them. With scatter collisions (multiple events mapping to the same audio token), information is lost.

[Exp 49](../experiment_49/README.md) showed virtual tokens work (100% AR survival, zero metronome collapse) but used only 32 tokens for out-of-window events, causing collisions and 52% hallucination.

**This experiment gives each out-of-window event its own dedicated virtual token.** 128 virtual tokens (one per event slot) are prepended to the 250 audio tokens, making 378 total. In-window events are scatter-added to audio tokens as usual. Out-of-window events are placed directly into their dedicated virtual token (slot i → vtoken i) — no linear mapping, no scatter collisions.

[Exp 49](../experiment_49/README.md) used 32 virtual tokens with linear mapping, causing collisions when many out-of-window events existed. With 128 slots (matching max_events), each event has a guaranteed home.

### Configuration — combining best findings

This experiment cherry-picks from multiple experiments:

| Feature | Source | Value |
|---|---|---|
| A_BINS / B_BINS | [53-B](../experiment_53b/README.md) | 500 / 500 |
| B_PRED | [53](../experiment_53/README.md) | 250 (N_CLASSES=251) |
| Gap ratios | [45](../experiment_45/README.md) | ON |
| Density jitter | [44](../experiment_44/README.md) | ±10% at 30% (loose) |
| Virtual tokens | [49](../experiment_49/README.md) (improved) | 128 (1:1 mode) |
| Context augmentation | [44](../experiment_44/README.md) | ~14% corruption rate |

**Why loose density jitter**: [Exp 56-B](../experiment_56b/README.md) showed the model is density-sensitive (1.53x ratio) but systematically under-predicts by ~20%. Loose jitter (from exp 44) gives the model more room to learn its own calibration rather than being forced into tight adherence that may not match AR reality. Exp 44 also won the volunteer vote in [53-AR](../experiment_53ar/README.md).

**Why B_PRED=250**: Proven in [exp 53](../experiment_53/README.md) — easier classification (251 classes vs 501), healthier density dependence, and the model sees 500 bins of future audio for context while only committing to 250.

### Architecture

```
Virtual: 128 tokens (1:1 with event slots, out-of-window events placed directly)
Audio:   250 tokens from conv stem (in-window events scatter-added as usual)
Total:   378 tokens through transformer
Cursor:  token 128 + 125 = 253
```

### Launch

```bash
python detection_train.py taiko_v2 --run-name detect_experiment_57 --model-type event_embed --n-virtual-tokens 128 --gap-ratios --b-bins 500 --b-pred 250 --density-jitter-rate 0.30 --density-jitter-pct 0.10 --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3
```

### New benchmarks (added for this experiment)

8 new ablation benchmarks added alongside the existing 10 + 2 AR:

| Short | Name | What it tests |
|---|---|---|
| NA_A | `no_audio_a` | Zero past audio only. Expect worse accuracy but NOT high stop |
| NA_B | `no_audio_b` | Zero future audio only. Expect similar stop rate to full no_audio |
| NA_E | `no_audio_e` | Zero ±10 frames around each event. Tests audio-at-event reliance |
| SO | `swap_out` | Replace past audio with different track. Mismatched context |
| NEP | `no_events_past` | Kill out-of-window events only. **Key for virtual token testing** |
| ERB | `event_ratio_blank` | Make events evenly spaced (ratios = 1.0). Tests gap ratio contribution |
| EGB | `event_gap_blank` | All events at same position (gaps = 0). Tests gap encoding contribution |
| ERGB | `event_ratio_gap_blank` | Both above combined. Tests total embedding beyond presence |

Also added `last_repeat` metric to all benchmarks and main val: % of predictions within 5% of the previous gap. Baseline ~40-45% (from [44-B](../experiment_44b/README.md) data). On metronome benchmark should be ~100%.

### What to watch

**Primary**: `NE`, `NEP`, and baseline deltas:
- `Baseline - NEP` = how much the model relies on out-of-window context (vtokens). **Large = vtokens are load-bearing.**
- `NEP - NE` = how much in-window scatter-added events contribute on top of nothing.
- If `Baseline - NEP ≈ 0`, vtokens are decorative (model doesn't use them). If large, vtokens carry critical context.

**Secondary**: `ERB`, `EGB`, `ERGB` — how much the model relies on gap/ratio features vs just event presence. If ERGB accuracy drops significantly below NE, the gap embeddings are load-bearing. If ERB ~ baseline, gap ratios are decorative.

**Tertiary**: `last_repeat` across benchmarks. Should be highest on metronome (~100%), moderate on baseline (~40%), low on no_events/random_events. Compare regular `last_repeat` vs metronome `last_repeat` to quantify metronome dependency.

## Result

Stopped at eval 4 (epoch 1.0). Virtual tokens not contributing.

### Eval 4 metrics:

| Metric | Value |
|---|---|
| HIT% | 72.5% |
| Accuracy | 53.3% |
| Stop F1 | 0.565 |
| Score | 0.373 |
| Val loss | 2.486 |
| last_repeat | 52.0% |

### Virtual token analysis:

| Benchmark | Accuracy | Delta from baseline |
|---|---|---|
| Baseline | 53.3% | — |
| NEP (kill OOW events) | 54.2% | **-0.9%** (removing vtokens helps) |
| NE (kill ALL events) | 50.0% | -3.3% |

| Eval | Base-NEP (vtoken contribution) | NEP-NE (in-window contribution) |
|------|-------------------------------|-------------------------------|
| 1 | -0.1% | +7.6% |
| 2 | +0.1% | +5.7% |
| 3 | -1.2% | +4.7% |
| 4 | -0.9% | +4.2% |

Vtokens flat at ~0% contribution across all evals. No trend of activation. In-window event contribution shrinking as audio learning improves (same pattern as all prior experiments).

### New benchmark highlights (eval 4):

| Benchmark | Accuracy | Insight |
|---|---|---|
| NA_A (kill past audio) | 46.3% | Model functions without past audio |
| NA_B (kill future audio) | 1.3% | **Model almost entirely depends on future audio** |
| NA_E (kill audio near events) | 18.7% | Heavy reliance on audio at event positions (-34pp) |
| SwapOut (random past audio) | 53.3% | Past audio irrelevant — same as baseline |
| ERB (ratios blanked) | 45.3% | Gap ratios contribute ~8pp |
| EGB (gaps blanked) | 50.0% | = NE. Without gap info, event presence is worthless |
| ERGB (both blanked) | 50.0% | = NE. Confirms gap is foundation |

## Lesson

**128 virtual tokens for out-of-window context do not help.** The model does not learn to use distant event history through virtual tokens. At eval 4, removing all vtokens (NEP) performs identically or slightly better than baseline. The 128 extra tokens dilute attention without contributing information.

**Key findings from new benchmarks:**

1. **Future audio is everything.** NA_B (kill future) drops accuracy to 1.3%. NA_A (kill past) only drops to 46.3%. SwapOut (wrong past audio) has zero impact. The model predicts almost entirely from future audio context.

2. **Gap encoding is the foundation of event embeddings.** EGB (gaps zeroed) = NE (no events) = 50.0%. Event presence without gap information is worthless. Gap ratios add ~8pp on top of gap encoding.

3. **Audio at event positions is critical.** NA_E drops 34pp, more than NE drops from baseline (3.3pp). The model uses audio-at-event-onset more than it uses the event embeddings themselves.

4. **last_repeat at 52%** — higher than the 43.9% dataset baseline, confirming the model is somewhat metronomic.

**Virtual tokens failed for the same reason as exp 49**: the model can't learn to use sparse context tokens when dense audio tokens already dominate attention. The architecture needs a mechanism that forces context usage, not just provides an optional pathway.
