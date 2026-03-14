# osu!taiko Onset Detection

Autoregressive model that predicts beat onset timings for osu!taiko charts given audio and chart context. Given a position in the audio (the "cursor"), the model predicts the bin offset to the next mapped onset - a 501-class classification (bins 0-499 at ~5ms resolution + STOP). At inference time, predictions are chained autoregressively to generate a full chart.

## Architecture

**Current: Unified Audio + Gap Fusion ([Exp 25](experiments/experiment_25))** - ~19M params
- **AudioEncoder**: Conv stack + 4 FiLM-conditioned transformer layers over mel spectrogram → 250 audio tokens
- **GapEncoder**: Inter-onset intervals + local mel snippets + 2 transformer layers → C gap tokens
- **FusionTransformer**: Audio + gap tokens concatenated and fused via 4-layer self-attention
- Single cursor token extraction at position 125 → 501-class output head
- Density conditioning (mean, peak, std) via FiLM modulation throughout

## Data

~10K onset CSVs from ~2,490 ranked osu!taiko maps. Data is not included - see [DATA.md](DATA.md) for how to obtain and prepare it.

## Usage

```bash
# Train
python detection_train.py --run-name my_run --epochs 50

# Inference
python detection_inference.py --checkpoint checkpoints/best.pt --audio song.mp3

# Visualize predictions
python viewer.py predicted.csv --audio song.mp3
```

## Performance

Our model vs well-known onset detection algorithms, evaluated on the same validation set.
These external algorithms detect general audio transients - not osu! taiko mapping decisions -
so this comparison demonstrates the gap between "where are the sounds?" and "where would a
taiko mapper place notes?". See [PERFORMANCE.MD](PERFORMANCE.MD) for detailed per-algorithm
analysis with graphs.

| Algorithm | Type | HIT | GOOD | Miss | Score | Frame Err |
|---|---|---|---|---|---|---|
| **[Exp 27](experiments/experiment_27) (best, eval 8)** | **Trained model** | **69.8%** | **70.2%** | **29.8%** | **+0.343** | **11.5** |
| madmom_cnn | Neural (CNN) | 3.5% | 7.8% | 89.2% | -0.769 | 40.8 |
| librosa_energy | Classical (RMS) | 2.2% | 4.5% | 90.6% | -0.563 | 58.8 |
| aubio_specflux | Classical (SpecFlux) | 1.3% | 3.0% | 92.9% | -0.701 | 41.0 |
| aubio_complex | Classical (Complex) | 0.9% | 3.8% | 89.4% | -0.702 | 73.2 |
| madmom_rnn | Neural (RNN) | 0.8% | 1.6% | 96.3% | -0.763 | 40.8 |
| aubio_hfc | Classical (HFC) | 0.7% | 2.4% | 93.0% | -0.682 | 69.4 |
| librosa_flux | Classical (Spec Flux) | 0.7% | 1.6% | 96.0% | -0.713 | 136.8 |

The 20x gap between the best external algorithm (3.5% HIT) and our model (69.8% HIT) confirms that taiko onset prediction is fundamentally different from audio onset detection - it requires learned knowledge of mapping conventions, chart context, and community preferences.

## Key Files

| File | Description |
|------|-------------|
| `detection_model.py` | Unified audio + gap fusion transformer |
| `detection_train.py` | Training loop, loss, benchmarks, graphs |
| `detection_inference.py` | Autoregressive inference |
| `baseline_benchmark.py` | External algorithm benchmark (librosa, aubio, madmom) |
| `viewer.py` | Pygame onset visualizer with audio playback |
| `parse_osu_taiko.py` | .osz → onset CSV extraction |
| `create_dataset.py` | Audio → mel spectrogram preprocessing |
| `experiments/` | [27+ experiments](experiments/) - hypotheses, results, and graphs |
| `PERFORMANCE.MD` | [Detailed baseline comparison](PERFORMANCE.MD) with per-algorithm graphs |


## Disclaimer

Commercial use of AI-generated rhythm game content is legally ambiguous. You are solely responsible for your use of this software and any data you obtain for it. See [LICENSE.md](../../LICENSE.md) for details.
