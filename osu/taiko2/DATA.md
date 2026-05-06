# Data

Datasets are not redistributed. The training data consists of ranked
osu!taiko beatmaps — copyrighted audio + community-created charts —
which you must obtain yourself.

## Obtaining maps

Download osu!taiko beatmap packs from the
[osu! beatmap pack listing](https://osu.ppy.sh/beatmaps/packs) or
any source you choose. The expected on-disk format is `.osz`
archives (zip files containing a `.mp3` / `.ogg` audio file plus
one or more `.osu` chart files per beatmapset).

Place them in any directory; the dataset builder takes that
directory as `--charts-dir`.

## Building a dataset

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.prepare_dataset \
    --name taiko2_v1 \
    --charts-dir <path-to-osz-pack-root>
```

The builder:

1. Walks `--charts-dir` recursively for `.osz` files.
2. For each pack, parses contained `.osu` chart files and decodes
   the audio.
3. Computes log-mel spectrogram features (80 bands, 5 ms / frame
   by default — see [`configs/mel_default.json`](configs/mel_default.json))
   and writes them to `features/{stem}.npy` as float16.
4. Quantizes onsets to a 5 ms-bin grid (`FixedRateEventSampler`,
   `divisor=200`) and writes them to
   `events/{chart_id}.npz` with `bins`, `times_ms`, `kind_ids`.
5. Writes a top-level `manifest.json` recording the audio sampler
   config (so old datasets keep loading after sampler changes), a
   `ChartEntry` per chart with star rating / density / OD / etc.,
   and the train / val split metadata.

Output layout:

```
osu/taiko2/datasets/{name}/
  manifest.json
  features/{stem}.npy
  events/{chart_id}.npz
```

Datasets are gitignored.

## Alternative audio samplers

`prepare_dataset.py` accepts `--audio-sampler` aliases. Built-in:

- `mel` (default) — 80-band log-mel.
- `mel_onset` — 80-band log-mel + 4 sub-band-spectral-flux rows
  appended (range-matched to log-mel dB scale). Output features
  are `(84, T)` float16. Used by
  [#012](experiments/012-onset-channels/).

Custom samplers: pass a fully-qualified
`module:Class|module:ConfigClass` pair as `--audio-sampler`.
The class must be a concrete subclass of
`osu.taiko2.domain.dataset.AudioSampler`.

Override sampler config fields with `--audio-config`, either inline
JSON or a path to a JSON file:

```bash
--audio-config '{"hop_divisor": 400}'
--audio-config osu/taiko2/configs/mel_onset_default.json
```

## Train / val split

Songs (not charts) are bucketed by `beatmapset_id` so all
difficulties of one song land in the same split. Default ratios:
90 % train, 10 % val. Default seed: 42. The split is
deterministic given the manifest contents, ratios, and seed.

To use a different split, pass:

```bash
--split-ratios "train:0.8,val:0.1,test:0.1" --split-seed 7
```

The split is recomputed at sampler-load time, not stored on disk —
so the same manifest can be sliced different ways without rebuild.

## Augmenting an existing dataset's features

If you already have a built `taiko2_v1` and want to use a different
audio sampler that's a function of the cached features (e.g. a
sub-band-flux variant), prefer rebuilding via `prepare_dataset.py`
with the new sampler. It's slower (re-decodes audio) but it's the
canonical pipeline and avoids any chance of feature corruption.

## Star ratings

Star ratings are not in the `.osu` format directly; they're computed
by osu!. Fetch them via the osu! API v2 client:

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.fetch_stars \
    --dataset taiko2_v1
```

Requires `OSU_CLIENT_ID` and `OSU_CLIENT_SECRET` in `.env` or
`secrets.json`. See [`credentials.py`](credentials.py) for the
loader's search order.

## Engagement metadata

Plays / favourites / pass rate from osu! map metadata can be joined
into the manifest:

```bash
osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.fetch_engagement \
    --dataset taiko2_v1
```

Same OAuth credentials.

## Disclaimer

You are responsible for ensuring your use of obtained beatmaps
complies with osu!'s terms of service and copyright in your
jurisdiction. See [`LICENSE.md`](LICENSE.md).
