# osu!taiko Onset Detection

Autoregressive model that predicts beat onset timings for osu!taiko charts given audio and chart context. Given a position in the audio (the "cursor"), the model predicts the bin offset to the next mapped onset — a classification problem at ~5ms resolution. At inference time, predictions are chained autoregressively to generate a full chart.

## Demos

See [DEMOS.md](DEMOS.md) for generated chart examples

## Architecture

**Current best: ProposeSelectDetector ([Exp 58](experiments/experiment_58))** — 23.5M params, 74.6% HIT (ATH)

Two-stage architecture: Stage 1 (Proposer) detects audio-supported onset positions; Stage 2 (Selector) uses chart context to pick the right ones.

1. **Conv stem**: Mel spectrogram (80 bands, 1000 frames = 500 past + 500 future) → 250 audio tokens (d_model=384)
2. **Stage 1 (Proposer)**: 4 transformer layers over audio tokens only (no events, no density). Each token outputs sigmoid: "audio supports onset here?" Trained recall-focused (focal BCE).
3. **Proposal embedding**: Stage 1 confidences embedded into audio tokens as additive features.
4. **Event embeddings**: For each of 128 past events, a learned embedding (presence + gap_before + gap_after + gap_ratio_before + gap_ratio_after) is scatter-added to the audio token at that event's position.
5. **Stage 2 (Selector)**: 8 transformer layers over enriched audio tokens with FiLM density conditioning. Cursor token at position 125 → 251-class softmax (B_PRED=250 + STOP).

Density conditioning (mean, peak, std) via FiLM modulation controls output density.

**Also notable: EventEmbeddingDetector ([Exp 44](experiments/experiment_44)/[45](experiments/experiment_45))** — 16.1M params, 73.7% HIT. Winner of human evaluation ([53-AR](experiments/experiment_53ar)). Simpler single-stage architecture without the proposer.

### Training

- **Loss**: Mixed soft targets (good_pct=3%, fail_pct=20%, frame_tolerance=2) + hard CE (hard_alpha=0.5). Stage 1: focal BCE (gamma=2, pos_weight=5). Combined: `s2_loss + 0.5 * s1_loss`.
- **Stage 2 freeze**: First 2 evals, Stage 1 trains alone. Then joint training.
- **Augmentation**: Gentle AR augmentation (~14% context corruption) — event jitter, partial metronome, deletion/insertion. "Distort, don't destroy" principle.
- **Data**: ~10K charts from ~2,490 ranked osu!taiko maps, subsample=1 (5.25M training samples)
- **Sampling**: Balanced sqrt-weighted sampling across 251 classes

### Inference features

- Temperature sampling with Top-U (unique cluster) candidate selection
- Metronome detection and suppression (configurable: temperature boost, candidate suppression, or both)
- Near-weight for temporal consistency
- AddAll mode for multi-candidate event generation
- Beat-synced GIF overlay in viewer
- Density inflation (~1.2x) recommended for matching target density ([Exp 56-B](experiments/experiment_56b))

## Data

~10K onset CSVs from ~2,490 ranked osu!taiko maps. Data is not included - see [DATA.md](DATA.md) for how to obtain and prepare it.

## Usage

```bash
# Train (current best config)
python detection_train.py taiko_v2 --run-name my_run --model-type event_embed --epochs 50 --batch-size 48 --subsample 1 --evals-per-epoch 4 --workers 3

# Inference
python detection_inference.py --checkpoint runs/my_run/checkpoints/best.pt --audio song.mp3 --andlaunch

# Inference with density control
python detection_inference.py --checkpoint runs/my_run/checkpoints/best.pt --audio song.mp3 --density-mean 6.75 --density-peak 11.1 --andlaunch

# Inference with metronome suppression
python detection_inference.py --checkpoint runs/my_run/checkpoints/best.pt --audio song.mp3 --andlaunch --metronome-weight 5 --metronome-applymode suppress --metronome-mode pp

# Visualize predictions
python viewer.py predicted.csv --audio song.mp3
```

## Performance

**Current best: [Exp 58](experiments/experiment_58) (ProposeSelectDetector)** — 74.6% HIT, 75.9% close rate in AR

### Per-sample (validation set)

| Model | HIT | MISS | Val Loss | Architecture |
|---|---|---|---|---|
| **Exp 58** (ProposeSelect) | **74.6%** | **25.0%** | **2.427** | Two-stage propose-select (ATH) |
| Exp 44 (EventEmbed) | 73.7% | 25.7% | 2.480 | Gentle augmentation |
| Exp 45 (EventEmbed + gap ratios) | 72.1% | 27.3% | 2.516 | Human eval winner |
| Exp 14 (audio-only) | 69.2% | 30.1% | 2.645 | No context |

### Autoregressive GT matching (30 val songs)

| Model | Close (<50ms) | Hallucination | Error Median |
|---|---|---|---|
| **Exp 58 (ours)** | **75.9%** | **15.6%** | **8ms** |
| DDC Oracle (external) | 77.1% | 19.9% | 27ms |
| madmom_cnn (external) | 3.5% HIT | — | — |

### Human evaluation ([53-AR](experiments/experiment_53ar))

1st: exp45 (44pts) — 2nd: exp44 (43pts) — 3rd: exp53 (36pts) — 4th: exp14 (27pts)

Context models overtook audio-only. Per-sample accuracy does not predict human preference — pattern variety does ([Exp 59-B](experiments/experiment_59b)).

See [PERFORMANCE.md](PERFORMANCE.md) for the full comparison including DDC difficulty analysis, TaikoNation architecture comparison, classical baselines, synthetic evaluator results, and key findings.

## Key Files

| File | Description |
|------|-------------|
| `detection_model.py` | EventEmbeddingDetector and legacy architectures |
| `detection_train.py` | Training loop, loss, benchmarks, graphs |
| `detection_inference.py` | Autoregressive inference with sampling/suppression |
| `viewer.py` | Pygame visualizer with mel/waveform/candidates/GIF |
| `baseline_benchmark.py` | External algorithm benchmark (librosa, aubio, madmom) |
| `parse_osu_taiko.py` | .osz → onset CSV extraction |
| `create_dataset.py` | Audio → mel spectrogram preprocessing |
| `experiments/` | [60+ experiments](experiments/) — hypotheses, results, and graphs |
| `DEMOS.md` | [Generated chart demos](DEMOS.md) |
| `PERFORMANCE.md` | [Full performance comparison](PERFORMANCE.md) — models, baselines, human eval, synthetic eval |
| `requirements.txt` | Pinned dependencies |

## Disclaimer

Commercial use of AI-generated rhythm game content is legally ambiguous. You are solely responsible for your use of this software and any data you obtain for it. See [LICENSE.md](../../LICENSE.md) for details.
