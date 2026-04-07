# Experiment 42-AR — Human Evaluation Specification

## Type

Human evaluation experiment. No model training. Blind A/B/C comparison of autoregressive chart generation quality across three models.

## Purpose

Determine whether higher per-sample validation accuracy translates to better perceived chart quality in full autoregressive generation. Per-sample metrics may be misleading if errors cascade differently across architectures.

## Models Evaluated

| Label | Experiment | Per-sample HIT | Architecture |
|---|---|---|---|
| exp14 | Exp 14 | 68.9% | OnsetDetector, no context (audio-only) |
| exp35c | Exp 35-C | 71.6% | OnsetDetector, mel ramp context |
| exp42 | Exp 42 | 73.2% | EventEmbeddingDetector, event embeddings |

## Evaluation Methodology

### Song Selection

10 songs selected, all post-training-cutoff (unseen during training):
- 4 Japanese Electronic/Dance
- 2 Indie Rock
- 2 Chiptune/Electronic
- 2 Pop

### Chart Generation

Each song was processed through all 3 models using full autoregressive inference:
- Fixed density conditioning: density_mean=6.75, density_peak=12.1 (same across all models)
- Full AR loop: start at bin 0, predict next onset, advance cursor, repeat until end of audio

### Video Production

1. Generated charts rendered to MP4 videos with audio and hit sounds
2. Videos compiled into blind comparisons: each song gets Alpha/Beta/Gamma labels randomly assigned to models
3. Mapping between labels and models saved in secret text files (not revealed until voting complete)

### Evaluation Protocol

- **Self-evaluation**: Author ranked all 10 songs blind (did not see label-to-model mappings)
- **External evaluators**: 6 unique volunteers, 8 total votes (one evaluator rated 2 songs). Each evaluator assigned 1-2 songs, ranked Alpha/Beta/Gamma as 1st/2nd/3rd
- **Scoring**: 3 points for 1st, 2 points for 2nd, 1 point for 3rd. Aggregated across all votes.

### Evaluation Limitations

- Only 6 unique volunteers (8 votes total)
- 4 songs received no volunteer evaluation beyond self-rankings
- Future evaluations should aim for 1+ volunteer per song and ideally 2+ for agreement measurement

## Scripts

| Script | Location | Purpose |
|---|---|---|
| run_inference.py | `experiments/experiment_42ar/run_inference.py` | Run all 3 models on all 10 songs, output CSVs |
| render_videos.py | `experiments/experiment_42ar/render_videos.py` | Render chart CSVs to MP4 with audio + hit sounds |
| compile_videos.py | `experiments/experiment_42ar/compile_videos.py` | Create blind Alpha/Beta/Gamma comparison videos |
| gather_stats.py | `experiments/experiment_42ar/gather_stats.py` | Compare inference stats across models |
| tally_votes.py | `experiments/experiment_42ar/results/tally_votes.py` | Tally votes and reveal winner |

## Output Files

| File | Description |
|---|---|
| results/votes.json | All rankings (self: 10 songs, evaluators: 1 each) |
| compiled/*_mapping.txt | Secret label-to-model mappings |
| charts/*.csv | Generated chart data per model per song |
| videos/*.mp4 | Individual model chart videos |
| compiled/*.mp4 | Blind comparison videos |

## Key Findings

### Final Standings (18 votes: 10 self + 8 volunteer)

| Model | Points | 1st | 2nd | 3rd | Avg |
|---|---|---|---|---|---|
| exp14 | 43 | 10 | 5 | 3 | 2.39 |
| exp42 | 34 | 4 | 8 | 6 | 1.89 |
| exp35c | 31 | 4 | 5 | 9 | 1.72 |

- **exp14 (no context, lowest per-sample HIT) wins decisively** with 10 first-place votes out of 18
- Human ranking is inversely correlated with per-sample accuracy
- Metronome regression is the dominant complaint: context models lock into repeating gaps
- exp42 is the most consistent in density (std 0.4 events/sec vs 1.8 for exp35c) but loses to exp14 on pattern variety
- exp35c shows erratic density due to mel ramp AR instability (1.9 to 8.6 events/sec across songs)

### Inference Statistics

| Metric | Exp 14 | Exp 35-C | Exp 42 |
|---|---|---|---|
| Total events | 6,394 | 6,725 | 7,182 |
| Mean events/sec | 3.8 | 4.3 | 4.2 |
| Std events/sec | 0.6 | 1.8 | 0.4 |
| Total inference time | 161s | 143s | 115s |

## Environment

| Component | Version |
|---|---|
| Python | 3.13.12 |
| PyTorch | 2.12.0.dev20260307+cu128 (nightly) |
| CUDA | 12.8 |
| cuDNN | 9.10.02 |
| GPU | NVIDIA GeForce RTX 5070 (12 GB, compute 12.0) |
| OS | Windows 11 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| librosa | 0.11.0 |
| matplotlib | 3.10.8 |
