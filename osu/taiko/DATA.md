# Data

## Why data is not included

The training data consists of ranked osu!taiko beatmaps which are copyrighted content (audio files and community-created charts). Redistribution is not permitted. You must obtain the maps yourself from [osu!](https://osu.ppy.sh).

## Obtaining maps

Download osu!taiko beatmap packs from the [osu! beatmap pack listing](https://osu.ppy.sh/beatmaps/packs). The ranked taiko packs are monthly archives (`.rar`) containing `.osz` files (which are ZIP archives with audio + `.osu` chart files).

## Pipeline

### 1. Extract archives

Place `.rar` packs in `charts/`, then extract:

```bash
python extract_rars.py
```

This extracts all `.osz` files into `charts/`. Requires [WinRAR](https://www.win-rar.com/) (path configured in script).

### 2. Parse charts into onset CSVs (optional, legacy)

```bash
python parse_osu_taiko.py
```

Parses `.osz` → per-difficulty onset CSVs in `data/` (columns: `time_ms,type`). This was the original pipeline; `create_dataset.py` below replaces it.

### 3. Create training dataset

```bash
python create_dataset.py my_dataset --workers 6
```

Scans `.osz` files in `charts/`, extracts audio + chart data, and produces a training-ready dataset in `datasets/my_dataset/`:

- `manifest.json` — chart metadata, density stats, audio references, train/val splits
- `mels/*.npy` — mel spectrograms per audio file (80 bins, float16)
- `events/*.npy` — onset bin indices per chart (int32)

This is what `detection_train.py` consumes.
