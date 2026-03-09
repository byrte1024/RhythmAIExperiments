# osu!taiko Onset Detection

Autoregressive model that predicts beat onset timings for osu!taiko charts given audio and density conditioning.

## Architecture

Two-path transformer (~21M params):
- **Audio path** (proposer): Mel spectrogram encoder → candidate onset logits
- **Context path** (selector): Past event encoder → selection logits
- Combined via logit addition (multiplicative gating in probability space)

Input: 5s mel window (80 bins, 5ms resolution) + 128 past events + density conditioning (mean, peak, std).
Output: 501-class classification (bin offset 0–499 + STOP).

## Data

~10K onset CSVs from ~2,490 ranked osu!taiko maps. See [experiments/](experiments/) for training results.

## Usage

```bash
# Train
python detection_train.py --run-name my_run --epochs 50

# Inference
python detection_inference.py --checkpoint checkpoints/best.pt --audio song.mp3

# Visualize predictions
python viewer.py predicted.csv --audio song.mp3
```

## Key Files

| File | Description |
|------|-------------|
| `detection_model.py` | Two-path transformer architecture |
| `detection_train.py` | Training loop, loss, benchmarks, graphs |
| `detection_inference.py` | Autoregressive inference |
| `viewer.py` | Pygame onset visualizer |
| `parse_osu_taiko.py` | .osz → onset CSV extraction |
| `experiments/` | Per-experiment hypotheses, results, and graphs |
