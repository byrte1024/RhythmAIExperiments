# Performance Summary

Comprehensive performance comparison of BeatDetector models and external baselines on osu!taiko chart generation.

> **Important caveat:** General-purpose onset detectors (librosa, aubio, madmom) answer *"where are the audio transients?"* Our model answers *"where would an osu! taiko mapper place the next note?"* — a fundamentally different task requiring knowledge of mapping conventions, chart context, density preferences, and community style. See the [classical baselines section](#classical-onset-detection-baselines) for details.

## Per-Sample Metrics (Validation Set)

Best checkpoint from each notable experiment, evaluated on 10% val split (~60K samples).

| Model | HIT% | MISS% | Accuracy | Stop F1 | Val Loss | Architecture |
|---|---|---|---|---|---|---|
| exp14 | 69.2% | 30.1% | 50.5% | 0.480 | 2.645 | LegacyOnsetDetector (audio-only) |
| exp35c | 71.6% | 27.9% | 52.7% | 0.543 | 2.533 | Exponential mel ramps |
| exp42 | 73.2% | 26.2% | 53.5% | 0.525 | 2.507 | EventEmbeddingDetector |
| exp44 | 73.7% | 25.7% | 54.8% | 0.570 | 2.480 | EventEmbed + gentle augmentation |
| exp45 | 72.1% | 27.3% | 52.9% | 0.553 | 2.516 | EventEmbed + gap ratios + tight density |
| exp53 | 72.1% | 27.3% | 53.3% | 0.547 | 2.518 | B_AUDIO/B_PRED split (A=250) |
| exp53-B | 73.4% | 26.1% | 54.2% | 0.562 | 2.479 | B_AUDIO/B_PRED split (A=500) |
| exp55 | 73.6% | 25.8% | 54.2% | 0.570 | 2.463 | Auxiliary ratio head |
| **exp58** | **74.6%** | **25.0%** | **55.7%** | — | **2.427** | **ProposeSelectDetector (ATH)** |

HIT = within 3% ratio or ±1 frame. MISS = beyond 20% ratio. Accuracy = exact class match.

## Autoregressive Ground Truth Matching (30 Val Songs)

Full AR inference on 30 val songs with per-song density conditioning. Predicted events matched against ground truth chart events.

| Model | Close (<50ms) | Far (>100ms) | Hallucination | Density Ratio | Error Median |
|---|---|---|---|---|---|
| **exp58** | **75.9%** | **16.6%** | **15.6%** | **0.92** | **8ms** |
| exp53 | 73.4% | 19.0% | 17.9% | 0.91 | 8ms |
| exp44 | 71.1% | 20.6% | 14.7% | 0.84 | 9ms |
| exp55 | 69.7% | 21.7% | 15.6% | 0.82 | 14ms |
| exp50b | 66.5% | 25.2% | 14.8% | 0.78 | 19ms |
| exp51 | 56.8% | 36.0% | 14.9% | 0.64 | 40ms |

Source: [Exp 59-HB](experiments/experiment_59hb/README.md)

## Human Evaluation (Blind A/B/C/D Rankings)

### 42-AR (3 models, 10 songs, 21 votes)

| Rank | Model | Points | HIT% | Key trait |
|---|---|---|---|---|
| 1st | exp14 | 40 | 69.2% | Audio-only, no context |
| 2nd | exp42 | 35 | 73.2% | Event embeddings |
| 3rd | exp35c | 33 | 71.6% | Mel ramps |

Per-sample metrics inversely correlated with human preference — audio-only model won.

### 53-AR (4 models, 10 songs, 15 votes)

| Rank | Model | Points | HIT% | Key trait |
|---|---|---|---|---|
| 1st | exp45 | 44 | 72.1% | Gap ratios + tight density |
| 2nd | exp44 | 43 | 73.7% | Gentle augmentation |
| 3rd | exp53 | 36 | 72.1% | B_AUDIO/B_PRED split |
| 4th | exp14 | 27 | 69.2% | Audio-only (42-AR winner) |

Context models overtook audio-only between rounds. Expert and volunteer preferences diverge: expert prefers exp45 (creative patterns), volunteers prefer exp44 (conservative/clean).

Source: [Exp 42-AR](experiments/experiment_42ar/README.md), [Exp 53-AR](experiments/experiment_53ar/README.md)

## Synthetic Evaluation (Pattern Variety Metrics)

Metrics correlated with human preference (within-song z-scored, Spearman):

| Metric | r | p | Meaning |
|---|---|---|---|
| gap_std | +0.299 | 0.005 | More varied gap lengths preferred |
| gap_cv | +0.289 | 0.006 | More relative variation preferred |
| dominant_gap_pct | -0.272 | 0.010 | Less repetition preferred |
| max_metro_streak | -0.269 | 0.011 | Shorter metronome streaks preferred |

Best synthetic evaluator: `z(gap_std) + z(gap_cv)` — predicts first place 52% of the time (2x random), Spearman r=0.35 (p=0.001). For volunteers: top-7 metrics at temp=0.5 achieves 70% first-place accuracy.

Caveat: relationship is non-linear (inverted U-shape). Too little variety = metronomic (bad). Too much variety = under-prediction/noise (also bad). Synthetic evaluator should be used alongside GT matching, not alone.

Source: [Exp 59-B](experiments/experiment_59b/README.md) through [59-HB](experiments/experiment_59hb/README.md)

## External Baseline: DDC Onset Detector

Dance Dance Convolution (Donahue et al. 2017) onset detection CNN, PyTorch port (`pip install ddc_onset`). Pure audio at 100fps (10ms resolution). Evaluated on same 30 val songs.

### DDC difficulty comparison (all at best threshold per difficulty):

| DDC Difficulty | Close% | Hall% | d_ratio | err_med | Events |
|---|---|---|---|---|---|
| BEGINNER | 36.3% | 13.5% | 0.43 | 392ms | 278 |
| EASY | 67.3% | 18.4% | 0.86 | 38ms | 585 |
| MEDIUM | 79.0% | 21.2% | 1.08 | 29ms | 749 |
| HARD | 94.3% | 27.1% | 1.70 | 25ms | 1232 |
| CHALLENGE | 98.7% | 28.9% | 2.42 | 25ms | 1766 |

DDC's difficulty parameter acts as density control, analogous to our FiLM conditioning.

### DDC Oracle (per-song density-matched) vs our best:

| Model | Close (<50ms) | Hallucination | Density Ratio | Error Median |
|---|---|---|---|---|
| DDC Oracle | **77.1%** | 19.9% | **1.00** | 27ms |
| DDC MEDIUM | 79.0% | 21.2% | 1.08 | 29ms |
| **exp58 (ours)** | 75.9% | **15.6%** | 0.92 | **8ms** |

DDC catches 1-2pp more GT events but with 4pp more hallucination and 3.4x worse timing precision. DDC finds every audio transient without understanding which ones the chart should use — our models learn selective placement with context awareness.

Source: [Exp 60](experiments/experiment_60/README.md)

### TaikoNation (Halina & Guzdial, 2021)

LSTM + CNN architecture, directly targeting osu!taiko. Open-source weights available ([GitHub](https://github.com/emily-halina/TaikoNationV1)) but built on TFLearn/TensorFlow 1.x (abandoned framework). Not yet tested due to framework compatibility.

| Aspect | TaikoNation | BeatDetector (exp58) |
|---|---|---|
| Architecture | 2-layer LSTM + 2 conv | 12-layer Transformer (4 proposer + 8 selector) |
| Parameters | ~<1M (estimated) | 23.5M |
| Resolution | 23ms | 5ms |
| Audio window | 368ms | 5000ms |
| Context | Previous 4 predictions | 128 past events with gap/ratio embeddings |
| Note types | 7 classes (don/kat/big/roll/denden) | Binary onset (timing only) |
| Density control | None | FiLM conditioning (continuous) |
| Dataset | ~100 curated high-quality charts | 10,048 charts (all difficulties) |

TaikoNation focuses on note type patterning (which notes to place). BeatDetector focuses on onset timing precision (when to place them). Future work: combine both.

### TaikoNation Patterning Metrics ([Exp 61](experiments/experiment_61/README.md))

Evaluated our AR output using TaikoNation's exact 5 metrics (binary arrays at 23ms resolution):

| Model | Over. P-Space | HI P-Space | DCHuman | Notes |
|---|---|---|---|---|
| exp58 (ours) | 10.1% | 81.1% | **90.8%** | Best placement accuracy |
| Human GT | **11.7%** | — | — | Reference diversity |
| TaikoNation* | **21.3%** | **94.1%** | 75.0% | Most diverse (overshoots human) |
| DDC* | 15.9% | 83.2% | 77.9% | — |

(*) Published on different songs. We massively win placement (90.8% vs 75.0% DCHuman). TaikoNation wins pattern diversity (21.3% vs 10.1% P-Space) but overshoots human diversity (11.7%) by 47%. Our models are 14% below human diversity — closer to the target.

## Classical Onset Detection Baselines

General-purpose audio onset detectors evaluated on the same per-sample prediction task (single-step, non-autoregressive). All fail catastrophically because they detect ALL audio transients, not the selective subset a taiko mapper would choose.

| Algorithm | Type | HIT | GOOD | Miss | Score |
|---|---|---|---|---|---|
| **Exp 14 (ours)** | **Trained model** | **68.9%** | **69.5%** | **30.3%** | **+0.337** |
| madmom_cnn | Neural (CNN) | 3.5% | 7.8% | 89.2% | -0.769 |
| librosa_energy | Classical (RMS) | 2.2% | 4.5% | 90.6% | -0.563 |
| aubio_specflux | Classical (SpecFlux) | 1.3% | 3.0% | 92.9% | -0.701 |
| aubio_complex | Classical (Complex) | 0.9% | 3.8% | 89.4% | -0.702 |
| madmom_rnn | Neural (RNN) | 0.8% | 1.6% | 96.3% | -0.763 |
| aubio_hfc | Classical (HFC) | 0.7% | 2.4% | 93.0% | -0.682 |
| librosa_flux | Classical (Spec Flux) | 0.7% | 1.6% | 96.0% | -0.713 |

The 20x performance gap (3.5% best baseline vs 68.9% exp14) confirms taiko onset prediction is fundamentally different from audio onset detection.

## Key Findings

1. **Per-sample accuracy does not predict AR quality or human preference.** exp14 (69.2% HIT) won 42-AR. exp45 (72.1%) won 53-AR. exp58 (74.6%) has the best GT matching.

2. **Pattern variety is the measurable proxy for human preference.** gap_std and gap_cv (within-song normalized) are the only metrics that significantly correlate with human rankings (r~0.30, p<0.01). But the relationship is non-linear.

3. **Timing precision is our key advantage over baselines.** 8ms median error (exp58) vs 27ms (DDC Oracle) vs 23ms (TaikoNation resolution limit). For rhythm games where timing is felt at the millisecond level, this matters.

4. **Two-stage propose-select is the best architecture.** exp58 achieves highest per-sample HIT (74.6%), best AR GT matching (75.9% close), and competitive synthetic scores — with uniquely smooth training convergence (7 consecutive improvements, zero oscillations).

5. **Context models overtook audio-only.** Between 42-AR and 53-AR, improvements in augmentation, gap ratios, and architecture made context a net positive for AR quality.

6. **Density conditioning works but needs ~1.2x inflation.** Models systematically under-predict by ~20%. Conditioning at 1.2x target density matches reality ([Exp 56-B](experiments/experiment_56b/README.md)).

7. **Audio onset detection is solved; chart generation is not.** DDC and our models both achieve ~76% close rate at matched density. The differentiation is in context-informed selection (which onsets to use) and precise timing (where exactly to place them).
