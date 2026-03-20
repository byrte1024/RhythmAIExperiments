# Experiment 42-AR - Human Evaluation of Autoregressive Inference

## Hypothesis

Validation metrics (HIT, miss, entropy) measure per-sample accuracy, but real quality is about how the full autoregressive generation FEELS when played. A chart with 70% per-sample HIT might sound great if errors are spread evenly, or terrible if they cascade in one section.

**Blind human evaluation** comparing three models on full AR-generated charts:
- **Exp 14** (68.9% HIT) — baseline, no context
- **Exp 35-C** (71.6% HIT) — mel ramps, first real context
- **Exp 42** (73.2% HIT) — event embeddings, deepest context

Each evaluator sees 3 videos of one song — one per model, randomized order, no labels. They rank which generation they think is best.

### Method

1. 10 songs selected (all post-training-cutoff, unseen during training):

   **Japanese Electronic Dance (4):**
   | Song | Artist | Source | YouTube |
   |------|--------|--------|---------|
   | Odorouze! | Snow Man | Feb 2026, #1 Japan Hot 100, film "The Specials" theme | [youtube](https://youtu.be/VEiBxLwfU3E) |
   | Cold Night | HANA | Jan 2026, Medalist S2 opening theme | [youtube](https://www.youtube.com/watch?v=7TgxtYgf_Tc) |
   | KIKYU | XAMIYA | Mar 2026, BLUESTAR EP | [youtube](https://www.youtube.com/watch?v=qqznGG164Oc) |
   | DARK GAME | muque | 2026, hyperpop Y2K | [youtube](https://www.youtube.com/watch?v=obyT5_3TMTQ) |

   **Indie Rock (2):**
   | Song | Artist | Source | YouTube |
   |------|--------|--------|---------|
   | Might See You There | Weird Nightmare | Feb 2026, Sub Pop, album *Hoopla* | [youtube](https://www.youtube.com/watch?v=0jGJpAKUNgk) |
   | Watching the Omnibus | The Bug Club | Feb 2026, Sub Pop, album *Every Single Muscle* | [youtube](https://www.youtube.com/watch?v=f8wXBiNsRcg) |

   **Chiptune / Electronic (2):**
   | Song | Artist | Source | Link |
   |------|--------|--------|------|
   | Chipper Choices | RoccoW | 2026 Weeklybeats (electro city-pop) | [FMA](https://freemusicarchive.org/music/RoccoW/weeklybeats-2026/chipper-choices/) |
   | Time Machine | Equiloud | Chiptune/Synth Pop | [FMA](https://freemusicarchive.org/genre/Chiptune/) |

   **Pop (2):**
   | Song | Artist | Source | YouTube |
   |------|--------|--------|---------|
   | Manchild | Sabrina Carpenter | 2025 release, charting 2026, Grammy nominated | [youtube](https://www.youtube.com/watch?v=aSugSGCC12I) |
   | Ordinary | Alex Warren | 2025 release, 10 weeks #1, charting 2026 | [youtube](https://www.youtube.com/watch?v=u2ah9tWTkmk) |

   Audio files stored locally in `audio/` (gitignored — not tracked in repo).
2. Run full AR inference on each song with all 3 model checkpoints
   - Density conditioning: `--density-mean 6.75 --density-peak 12.1` (fixed across all models, chosen randomly)
3. Record gameplay videos of each generated chart
4. Each of ~10 evaluators sees 3 videos of ONE song (their assigned song), ranks them 1st/2nd/3rd
5. No evaluator sees the same song twice or knows which model is which
6. Aggregate: which model gets ranked 1st most often?

### Expected outcome

Exp 42 should rank best most often — the +4.3pp HIT over exp 14 and context dependency should produce more rhythmically coherent charts with fewer skipped/extra notes. But AR error cascading (exp 42's metronome at 25.4%) could hurt if early errors snowball.

### Inference stats

**Aggregate:**

| Metric | Exp 14 | Exp 35-C | Exp 42 |
|--------|--------|----------|--------|
| Total events | 6,394 | 6,725 | **7,182** |
| Mean events/sec | 3.8 | 4.3 | 4.2 |
| **Std events/sec** | 0.6 | **1.8** | **0.4** |
| Total STOPs | 1,172 | 1,108 | 1,251 |
| Total inference time | 161s | 143s | **115s** |

**Key finding: Exp 35-C is wildly inconsistent** (std 1.8 events/sec vs 0.4 for exp 42). The mel ramps cause erratic AR behavior — Weird Nightmare gets 8.6 events/sec (double the norm) while Ordinary gets only 1.9. The mel ramp approach amplifies AR errors because corrupted ramps cascade through the audio signal.

**Exp 42 is the most consistent** (std 0.4) — event embeddings produce stable density across genres. Also fastest at inference (115s total vs 161s for exp 14).

**Per-song events/sec:**

| Song | Exp 14 | Exp 35-C | Exp 42 |
|------|--------|----------|--------|
| Ordinary (pop) | 3.5 | **1.9** | 3.8 |
| Time Machine (chiptune) | 3.4 | 2.6 | **4.2** |
| Cold Night (J-pop) | 3.5 | 3.6 | **4.2** |
| Chipper Choices (electro) | 4.0 | **5.2** | 4.2 |
| Manchild (pop) | 3.9 | 4.1 | 3.9 |
| Odorouze (J-pop dance) | 3.7 | **2.3** | 3.8 |
| Watching the Omnibus (indie) | 5.2 | 5.3 | 5.0 |
| Might See You There (indie) | 4.1 | **8.6** | 4.6 |
| KIKYU (electro) | 2.7 | **4.9** | 4.0 |
| DARK GAME (hyperpop) | 3.8 | 3.9 | **4.6** |

Exp 35-C's outliers (bold): Weird Nightmare at 8.6 eps and Ordinary at 1.9 eps show the mel ramp AR instability. Exp 42 stays in a tight 3.8-5.0 range across all genres.

## Result

*Pending — awaiting human evaluation*

## Lesson

*Pending*
