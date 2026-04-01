# Experiment 53-AR - Human Evaluation of Autoregressive Inference (Round 2)

> **[Full Architecture Specification](ARCHITECTURE.md)** — self-contained reproduction guide with all model, loss, training, and dataset details.

## Hypothesis

[Exp 42-AR](../experiment_42ar/README.md) showed that per-sample accuracy is inversely correlated with human preference — [exp 14](../experiment_14/README.md) (68.9% HIT, no context) beat [exp 42](../experiment_42/README.md) (73.2% HIT, deepest context) decisively. The #1 complaint was metronome regression from context dependency.

Since then, [exp 53](../experiment_53/README.md) introduced the B_AUDIO/B_PRED split (A=250, B_AUDIO=500, B_PRED=250) which achieves:
- Metronome benchmark 52.5% (vs [exp 42](../experiment_42/README.md)'s 25.4%, [exp 44](../experiment_44/README.md)'s 42.9%)
- no_audio_stop 98.6% (vs previous models at 3-18%)
- Healthy density dependence (52% zero_density_stop)
- Random events actually HELP accuracy (+1.1pp vs no events)

**Question:** Does exp 53's improved corruption resilience and anti-metronome behavior translate to better human-perceived AR chart quality? Or does [exp 14](../experiment_14/README.md) still win by avoiding context entirely?

### Models

| Label | Experiment | Checkpoint | HIT | Architecture | Key trait |
|---|---|---|---|---|---|
| exp14 | [Exp 14](../experiment_14/README.md) | best (epoch 8) | 68.9% | LegacyOnsetDetector, no context | Audio-only, won [42-AR](../experiment_42ar/README.md) |
| exp44 | [Exp 44](../experiment_44/README.md) | eval_014 | ~73% | EventEmbeddingDetector | Gentle aug, previous ATH |
| exp45 | [Exp 45](../experiment_45/README.md) | eval_008 | 71.9% | EventEmbed + gap ratios | Best AR density adherence |
| exp53 | [Exp 53](../experiment_53/README.md) | eval_014 | 72.1% | EventEmbed + B_AUDIO/B_PRED split | Best corruption resilience, healthy density |

### Songs (10, all post-training-cutoff)

**Japanese Pop (2):**
| Song | Artist | Source |
|------|--------|--------|
| Five | Arashi | Digital single, March 4 2026 |
| The growing up train | Sakurazaka46 | 14th single, March 11 2026, debuted #1 |

**Japanese Dance/Hardcore (2):**
| Song | Artist | Source |
|------|--------|--------|
| Denkoh Sekka | Camellia | 2026 release, rhythm game composer |
| Xterfusion | REDALiCE × t+pazolite | Arcaea Crossing Collection, Jan 2026 |

**Indie Rock (2):**
| Song | Artist | Source |
|------|--------|--------|
| Stay in Your Lane | Courtney Barnett | March 2026 |
| Heavy Foot | Mon Rovîa | March 2026 |

**Chiptune / Electronic (2):**
| Song | Artist | Source |
|------|--------|--------|
| When the Leaves Leaf | RoccoW | FMA Weeklybeats 2026 |
| One More Time | supernovayuli | Original chiptune |

**Pop (2):**
| Song | Artist | Source |
|------|--------|--------|
| The Best | Conan Gray | March 2026 |
| Younger You | Miley Cyrus | March 2026 |

Audio files stored locally in `audio/` (gitignored).

### Method

1. Run full AR inference on each song with all 4 model checkpoints
   - Density conditioning: TBD (fixed across all models)
2. Render gameplay videos with audio + hit sounds
3. Compile into blind comparisons: Alpha/Beta/Gamma/Delta labels randomly assigned
4. **Self-evaluation**: rank all 10 songs blind
5. **External evaluators**: each assigned 1 song, rank Alpha/Beta/Gamma/Delta
6. Aggregate: 4pts for 1st, 3pts for 2nd, 2pts for 3rd, 1pt for 4th

### Scripts

| Script | Purpose |
|--------|---------|
| `run_inference.py` | Run all 4 models on all 10 songs → CSVs in `charts/` |
| `render_videos.py` | Render chart CSVs to mp4 with audio + hit sounds → `videos/` |
| `compile_videos.py` | Create blind Alpha/Beta/Gamma/Delta comparison videos → `compiled/` |
| `gather_stats.py` | Compare inference stats across models |
| `results/tally_votes.py` | Tally votes and reveal winner |

### Evaluation structure

- `results/votes.json` — all rankings (self + evaluators)
- `compiled/*_mapping.txt` — secret label→model mappings (DO NOT READ until votes are in)
- Scoring: 1st=4pts, 2nd=3pts, 3rd=2pts, 4th=1pt

### Key comparisons vs [42-AR](../experiment_42ar/README.md)

| Question | 42-AR answer | 53-AR hypothesis |
|---|---|---|
| Does context help AR? | No — [exp14](../experiment_14/README.md) (no context) won | Maybe — exp53 has anti-metronome resilience |
| Metronome complaints? | #1 issue for context models | exp53's 52.5% metronome benchmark should reduce this |
| Density adherence? | Not measured | exp53's healthy density dependence should help |
| Pattern variety? | [exp14](../experiment_14/README.md) had most variety | exp53 should compete via resilience, not by lacking context |

### Predictions

- **exp53 should rank higher than exp44/45** — its corruption resilience should prevent metronome lock-in
- **exp14 may still win** — audio-only avoids ALL context problems, and the 42-AR volunteers strongly preferred it
- **The real test**: does exp53's metronome benchmark improvement (52.5% vs 25-43%) translate to perceptible chart quality improvement?

## Result

*Pending*

## Lesson

*Pending*
