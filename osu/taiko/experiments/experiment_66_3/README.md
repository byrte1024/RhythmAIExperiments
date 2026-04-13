# Experiment 66-3 — LLM-as-Judge for Chart Quality

## Hypothesis

Our neural evaluator (66-1/66-2) is fundamentally limited: it learns corruption detection, not musical quality. LLMs have strong language intuition and emerging music understanding — they might evaluate chart quality from structured descriptions of the gap sequence and/or audio.

**The key question:** Can a general-purpose LLM, given a description of a rhythm game chart (and optionally the audio), rank charts in agreement with human preference?

No one has tried this for rhythm game charts. This is exploratory.

## Background

### What we know from literature
- LLMs can process symbolic music as text (ChatMusician, ACL 2024) but struggle with deep musical reasoning
- Raw number sequences are poorly understood by LLMs — tokenization destroys structure (LLMTime, NeurIPS 2023; "Are LMs Actually Useful?", NeurIPS 2024 Spotlight)
- LLM-as-judge achieves ~85% agreement with humans on text tasks when criteria are well-specified
- Gemini 2.5 Pro handles up to 11 hours of native audio input
- GPT-4o has native audio understanding
- No prior work on LLM evaluation of rhythm game charts

### What our neural evaluator achieved
- 66-2 on 42-AR: **perfect Spearman +1.0** (large quality gaps)
- 66-2 on 53-AR: **Spearman -0.8** (subtle differences — worse than random)
- The evaluator can't distinguish models that are close in quality

## Method

### Test data

Same songs and charts from human evaluation experiments:

**42-AR** (5 songs, 3 models: exp14, exp35c, exp42)
- Human winner: exp14 > exp42 > exp35c
- Large quality differences between models

**53-AR** (5 songs, 4 models: exp14, exp44, exp45, exp53)
- Human winner: exp45 > exp44 > exp53 > exp14
- Subtle differences — the hard test

### Chart encodings (16 total)

Charts are presented to the LLM in different formats to test which representation it understands best:

| # | Encoding | Description | Tests |
|---|----------|-------------|-------|
| 01 | Raw gaps (ms) | Comma-separated gap values | Can LLM read numbers? |
| 02 | Onset times (s) | Absolute timestamps | Alternative numeric format |
| 03 | Gap ratios | Each gap / previous gap | Tempo-invariant rhythm |
| 04 | Beat fractions | quarter, eighth, sixteenth | Musical language the LLM knows |
| 05 | Run-length encoded | "160ms x12, 80ms x4, ..." | Highlights repetition |
| 06 | Visual rhythm | X=note, .=silence text art | Visual pattern recognition |
| 07 | **Stats only** | Summary statistics (density, CV, metro streak, etc.) | Can LLM rank from stats alone? |
| 08 | Density curve | Events/sec in 5s windows | Energy profile |
| 09 | Pattern-annotated | Labels METRONOMIC/varied sections | Pre-analyzed for the LLM |
| 10 | Beat-aligned grid | Quantized to BPM sixteenths | Standard music grid |
| 11 | Section summary | Per-10s quality breakdown | Temporal quality variation |
| 12 | Gap histogram | Distribution of gap values | Statistical shape |
| 13 | Acceleration | Tempo changes over time | Speeding up / slowing down |
| 14 | Musical shorthand | ♩♪♬ notation | Closest to symbolic music |
| 15 | Grid alignment | % of notes on-beat vs off-beat | Timing accuracy |
| 16 | **Combined report** | Stats + density + sections + histogram + grid | Maximum information |

### LLM models tested

| Model | Audio input | Notes |
|---|---|---|
| GPT-4o (o4) | Yes | Native audio understanding |
| Claude Opus 4.6 | No | Text-only, strong reasoning |
| Gemini 2.5 Pro | Yes | 11 hours of audio, native understanding |

### Prompt design

Each test uses:
1. **System prompt**: Explains osu! taiko, defines what makes a good chart (rhythm alignment, pattern variety, density matching, no metronomic behavior, no hallucinations)
2. **Task**: Rank charts A/B/C/D for the same song from best to worst
3. **Chart data**: One of 16 encodings for each model's chart
4. **Audio** (for audio models): Compressed MP3 (<7MB) of the song

Charts are labeled A/B/C/D with randomized assignment (seed 42). Answer key stored separately.

### Test protocol

1. **Quick test**: `07_stats_only` + audio on 1 song — sanity check
2. **If promising**: `16_combined_report` + audio — maximum info
3. **Audio ablation**: Compare with-audio vs without-audio prompts
4. **Encoding comparison**: Which representation gives best human agreement?
5. **Cross-model**: Does GPT-4o (audio) beat Claude (no audio)?

## Launch

```bash
python generate_llm_eval.py --songs both --output llm_eval
```

Then manually upload audio + paste prompts to each LLM.

## Success criteria

- **Per-song #1 match > 25%** (random for 4 models) or > 33% (random for 3 models)
- **Pairwise accuracy > 50%** (random baseline)
- **At least one encoding achieves positive Spearman on 53-AR** (our neural evaluator got -0.8)
- **Audio models outperform text-only** on alignment-dependent songs
- Identify which encoding(s) LLMs understand best

## Expected results

- Audio models (Gemini, GPT-4o) should do better than text-only (Claude) on songs where chart-audio alignment matters
- Stats-based encodings (07, 16) should outperform raw numbers (01, 02) — LLMs reason better from summaries
- Pattern-annotated (09) and section summary (11) might do well since they pre-digest the metronomic detection
- 42-AR (large gaps) should be easier than 53-AR (subtle differences), same pattern as our neural evaluator
- Raw gap sequences (01) will probably fail — LLMs can't parse 500+ numbers meaningfully

## Results

### Encoding 01: Raw gap sequence (ms), no audio

The hardest test — just comma-separated millisecond gap values, no audio, no preprocessing.

| Model | #1 match | Pairwise | Notes |
|---|---|---|---|
| **GPT-4o** | **4/10 (40%)** | **51.1%** | Above baseline on both. Perfect 6/6 on 53-AR song 02 |
| Claude Opus 4.6 | 1/10 (10%) | 42.2% | Below random |
| Gemini 2.5 Pro | 0/7 (0%) | 44.4% | Below random, hit context limits on some songs |

GPT-4o is the clear winner from raw numbers alone — it independently identified metronomic patterns ("539, 539, 1067 repeated dozens of times"), structural contrast, and even estimated BPM correctly on 42-AR songs.

**Per-dataset:**
- 42-AR (3 models, large quality gaps): GPT-4o 2/5 #1, Claude 1/5, Gemini 0/5
- 53-AR (4 models, subtle differences): GPT-4o **2/5 #1** with one perfect match (song 02: 6/6 pairwise). Claude 0/5.

**vs Neural evaluator (66-2):**
- 42-AR: 66-2 got perfect Spearman +1.0 (better than all LLMs)
- 53-AR: 66-2 got Spearman -0.8 (worse than random). GPT-4o did better here — it can extract quality signal from pattern structure that our trained model misses

**Key observations from LLM reasoning:**
- All three LLMs correctly identified B (exp35c) as worst on 42-AR song 1 due to "539, 539, 1067" metronomic loop
- GPT-4o showed strongest ability to reason about pattern variety and structural arcs
- Claude gave detailed musical analysis but ranked incorrectly more often
- Gemini estimated BPM and genre correctly but couldn't translate that into quality ranking

### Encoding 07: Stats only, no audio

Pre-digested summary statistics (density, gap_cv, metro streak, gap histogram, etc.) — no temporal sequence.

| Model | #1 match | Pairwise |
|---|---|---|
| GPT-4o | 1/10 (10%) | 40.0% |
| Claude | 1/10 (10%) | 42.2% |

**Worse than raw gaps for GPT-4o** (40% → 10% #1, 51% → 40% pairwise). Both models give identical rankings — they see the same summary numbers and draw the same wrong conclusions. The stats encoding loses temporal structure. GPT-4o succeeded on raw gaps by detecting *sequential repetition patterns* (the "539, 539, 1067" loop) — invisible in aggregate stats.

Key insight: **sequential pattern is more informative than aggregate statistics.** Our neural evaluator also compresses into aggregates — this may explain why it fails on subtle differences too.

### Encoding 14: Musical shorthand (♩♪♬), no audio

Gaps converted to note duration symbols (quarter, eighth, sixteenth).

| Model | #1 match | Pairwise |
|---|---|---|
| GPT-4o | 1/10 (10%) | 26.7% |
| Claude | 1/10 (10%) | 28.9% |

**Worst encoding — below random.** Both models consistently prefer exp35c (human worst) on 42-AR. The symbolic abstraction equates "more diverse symbols" with quality, when the actual diversity comes from erratic timing. Removing the exact millisecond values destroys the signal.

### Cross-encoding comparison (no audio)

| Encoding | GPT-4o #1 | GPT-4o pw | Claude #1 | Claude pw |
|---|---|---|---|---|
| **01 raw gaps (ms)** | **4/10 (40%)** | **51.1%** | 1/10 (10%) | 42.2% |
| 07 stats only | 1/10 (10%) | 40.0% | 1/10 (10%) | 42.2% |
| 14 musical shorthand | 1/10 (10%) | 26.7% | 1/10 (10%) | 28.9% |
| *Random baseline* | *25-33%* | *50%* | *25-33%* | *50%* |

**More abstraction = worse performance.** GPT-4o needs the actual sequential patterns. Claude is roughly constant across encodings — it may not be able to extract quality signal from gap data regardless of format.

### Emerging pattern

GPT-4o on raw gap sequences is the only combination that reliably beats random. It appears to be pattern-matching on the literal number repetitions in the sequence — a capability that abstractions and summaries destroy.
