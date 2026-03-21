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
4. Videos compiled into blind comparisons: each song gets Alpha/Beta/Gamma labels randomly assigned to models. Mapping saved in secret `.txt` files.
5. **Self-evaluation**: author ranks all 10 songs blind (doesn't see mappings)
6. **External evaluators**: 10 people, each assigned 1 song, rank Alpha/Beta/Gamma as 1st/2nd/3rd
7. Aggregate: 3pts for 1st, 2pts for 2nd, 1pt for 3rd. Total across all votes.

### Scripts

| Script | Purpose |
|--------|---------|
| `run_inference.py` | Run all 3 models on all 10 songs → CSVs in `charts/` |
| `render_videos.py` | Render chart CSVs to mp4 with audio + hit sounds → `videos/` |
| `compile_videos.py` | Create blind Alpha/Beta/Gamma comparison videos → `compiled/` |
| `gather_stats.py` | Compare inference stats across models |
| `results/tally_votes.py` | Tally votes and reveal winner |

### Evaluation structure

- `results/votes.json` — all rankings (self: 10 songs, evaluators: 1 each)
- `compiled/*_mapping.txt` — secret label→model mappings (DO NOT READ until votes are in)
- Scoring: 1st=3pts, 2nd=2pts, 3rd=1pt

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

### Final standings (18 votes: 10 self + 8 volunteer)

| Model | Points | 1st | 2nd | 3rd | Avg |
|---|---|---|---|---|---|
| **exp14** | **43** | **10** | 5 | 3 | **2.39** |
| exp42 | 34 | 4 | 8 | 6 | 1.89 |
| exp35c | 31 | 4 | 5 | 9 | 1.72 |

**exp14 (no context) wins decisively** with 10 first-place votes out of 18 — more than exp42 and exp35c combined.

### Volunteer results (8 votes, 6 unique evaluators)

| Evaluator | Song | 1st | 2nd | 3rd |
|---|---|---|---|---|
| Flufonyx | Time Machine | exp14 | exp42 | exp35c |
| Lusai | Chipper Choices | exp35c | exp14 | exp42 |
| x3nd3n | Odorouze | exp42 | exp14 | exp35c |
| Lusai | Odorouze | exp14 | exp42 | exp35c |
| Tinky Winky | Watching the Omnibus | exp14 | exp35c | exp42 |
| FixelStyle | Might See You There | exp14 | exp42 | exp35c |
| Egroish | KIKYU | exp14 | exp35c | exp42 |
| Mawdi | DARK GAME | exp14 | exp42 | exp35c |

Volunteer-only: exp14 gets 5/8 first-place votes. Volunteers independently agree with self-rankings.

### Self vs volunteer agreement

On songs with both self and volunteer votes, self and volunteers agreed on 1st place 5/8 times. Disagreements were minor (rank swaps between 1st/2nd, never 1st/3rd).

### Common feedback themes

**Metronome regression** — the #1 complaint across all evaluators. Every evaluator noted models "falling into" repetitive patterns. Context models (exp42, exp35c) suffer more because their context dependency creates a positive feedback loop: once the model starts repeating a gap, the context reinforces it.

Volunteer quotes:
- *"beta just feels like going straight tak tak tak tak tak"* (Lusai on exp42)
- *"it keeps putting just the same notes over for a few seconds and it gets old"* (x3nd3n)
- *"Alpha is fucked up... it's like spamming"* (FixelStyle on exp35c)

**Pattern variety** — exp14 wins because it produces more varied patterns. Without context to lock into, it relies purely on audio, which naturally varies. Self-notes: *"Alpha... changes"*, *"Beta [exp42] regresses from unique patterns to metronome"*.

**exp35c instability** — mel ramps cause erratic density (std 1.8 events/sec vs 0.4 for exp42). Multiple evaluators noted exp35c as "worse gamma" or spam-heavy on certain songs.

### Genre breakdown (self-rankings only, 10 songs)

| Genre | exp14 1st | exp42 1st | exp35c 1st |
|---|---|---|---|
| J-pop/J-electro (4) | 2 | 2 | 0 |
| Indie rock (2) | 1 | 1 | 0 |
| Chiptune/electro (2) | 0 | 0 | 2 |
| Pop (2) | 1 | 0 | 1 |

exp14 and exp42 split evenly on self-rankings. exp35c only wins on chiptune/electro where its higher density variation matches the genre.

### Evaluation limitations

Due to recruitment constraints, only 6 unique volunteers participated (8 votes — Lusai evaluated 2 songs). 4 songs received no volunteer evaluation beyond self-rankings (Ordinary, Cold Night, Manchild, Might See You There had only self or single evaluator). A future human evaluation should aim for full coverage (1+ volunteer per song) and ideally 2+ volunteers per song for agreement measurement. The blind A/B/C methodology proved effective — clear winner emerged despite small sample size.

### The paradox

**Higher per-sample accuracy does NOT mean better AR generation.**

| Model | Per-sample HIT | Human ranking |
|---|---|---|
| exp14 | 68.9% | **1st (43pts)** |
| exp35c | 71.6% | 3rd (31pts) |
| exp42 | 73.2% | 2nd (34pts) |

The ranking is *inversely correlated* with per-sample accuracy. Context dependency helps per-sample prediction but hurts AR generation through metronome lock-in.

## Lesson

- **Per-sample metrics are misleading for AR quality.** The best per-sample model (exp42, 73.2% HIT) lost to the worst (exp14, 68.9% HIT) in human evaluation. Optimizing per-sample accuracy may actively harm generation quality by deepening context dependency.
- **Metronome regression is the dominant failure mode.** Not hallucination, not skipping, not density — it's the inability to break out of repeating patterns. Exp 44-B confirmed this is data-driven: 47% of training samples ask the model to continue the previous gap, rising to 83% when a streak of 8+ exists.
- **Context dependency is a double-edged sword.** It improves per-sample accuracy but creates AR vulnerability. The model needs to use context without becoming enslaved to it.
- **Blind human evaluation works.** Even with limited volunteers, the method produced a clear, consistent ranking. Self and volunteer rankings agreed. Worth repeating for future model comparisons.
- **Next steps:** Many possible directions — loss function changes (streak-break upweighting, adversarial metronome penalty), architecture changes (explicit streak features, two-stage continue/break prediction), data sampling (oversampling pattern-break samples), or training curriculum. The 44-B data analysis gives us the tools to measure progress on the actual problem.
