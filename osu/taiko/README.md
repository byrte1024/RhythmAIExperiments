# osu!taiko Onset Detection

Autoregressive model that predicts beat onset timings for osu!taiko charts given audio and chart context. Given a position in the audio (the "cursor"), the model predicts the bin offset to the next mapped onset - a 501-class classification (bins 0-499 at ~5ms resolution + STOP). At inference time, predictions are chained autoregressively to generate a full chart.

## Demos

See [DEMOS.md](DEMOS.md) for generated chart examples

## Architecture

**Current: EventEmbeddingDetector ([Exp 45](experiments/experiment_45))** - ~16.1M params

Audio-only transformer with learned event embeddings injected at onset positions. No separate context encoder, no cross-attention, no mel modification.

1. **Conv stem**: Mel spectrogram (80 bands, 1000 frames) → 250 audio tokens (d_model=384)
2. **Event embeddings**: For each past event, a learned embedding is added to the audio token at that event's position. Each embedding encodes:
   - Learned presence embedding
   - Gap before (sinusoidal) — distance from previous event
   - Gap after (sinusoidal) — distance to next event
   - Gap ratio before (sinusoidal) — rhythm acceleration/deceleration into this event
   - Gap ratio after (sinusoidal) — rhythm acceleration/deceleration out of this event
3. **Self-attention**: 8 transformer layers over 250 enriched audio tokens, with FiLM density conditioning
4. **Output**: Cursor token at position 125 → 501-class logit head

Density conditioning (mean, peak, std) via FiLM modulation controls output density. Tight density jitter (±2% @10%) during training ensures faithful density adherence at inference.

### Training

- **Loss**: Mixed soft targets (good_pct=3%, fail_pct=20%, frame_tolerance=2) + hard CE (hard_alpha=0.5)
- **Augmentation**: Gentle AR augmentation (~14% context corruption) — event jitter, partial metronome, deletion/insertion. "Distort, don't destroy" principle.
- **Data**: ~10K charts from ~2,490 ranked osu!taiko maps, subsample=1 (5.8M training samples)
- **Sampling**: Balanced sqrt-weighted sampling across 501 classes

### Inference features

- Temperature sampling with Top-U (unique cluster) candidate selection
- Metronome detection and suppression (configurable: temperature boost, candidate suppression, or both)
- Near-weight for temporal consistency
- AddAll mode for multi-candidate event generation
- Beat-synced GIF overlay in viewer

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

**Current best: Exp 45** (73.6% HIT, 25.9% MISS on validation set)

| Model | HIT | MISS | Exact | Context delta | Metronome resilience |
|---|---|---|---|---|---|
| **Exp 45** (EventEmbedding + gap ratios) | **73.6%** | **25.9%** | 54.7% | 5.6pp | 44.8% |
| Exp 44 (EventEmbedding) | 73.6% | 25.9% | 54.7% | 4.9pp | 45.7% |
| Exp 42 (EventEmbedding, no aug) | 73.2% | 26.4% | 54.2% | 4.3pp | 25.4% |
| Exp 35-C (mel ramps) | 71.6% | 27.9% | — | — | — |
| Exp 14 (no context) | 68.9% | — | — | 0pp | 50.5% |

Our model vs external onset detection algorithms:

| Algorithm | Type | HIT | GOOD | Miss |
|---|---|---|---|---|
| **Exp 45** | **Trained model** | **73.6%** | **73.9%** | **25.9%** |
| madmom_cnn | Neural (CNN) | 3.5% | 7.8% | 89.2% |
| librosa_energy | Classical (RMS) | 2.2% | 4.5% | 90.6% |
| aubio_specflux | Classical (SpecFlux) | 1.3% | 3.0% | 92.9% |

The 21x gap between the best external algorithm (3.5% HIT) and our model (73.6% HIT) confirms that taiko onset prediction is fundamentally different from audio onset detection.

See [PERFORMANCE.MD](PERFORMANCE.MD) for detailed per-algorithm analysis.

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
| `experiments/` | [45+ experiments](experiments/) - hypotheses, results, and graphs |
| `DEMOS.md` | [Generated chart demos](DEMOS.md) |
| `PERFORMANCE.MD` | [Detailed baseline comparison](PERFORMANCE.MD) |

## Disclaimer

Commercial use of AI-generated rhythm game content is legally ambiguous. You are solely responsible for your use of this software and any data you obtain for it. See [LICENSE.md](../../LICENSE.md) for details.
