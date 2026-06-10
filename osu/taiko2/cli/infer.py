"""Abstract inference CLI.

Takes audio + conditioning + a spec JSON listing every component class
(predictor, decoder, input builder, audio sampler, event sampler) plus
a checkpoint path. Loads the Model from the checkpoint, wires the
predictor, synthesizes a stub `Chart` carrying just the audio bytes,
runs `predictor.predict(...)`, and writes the result as an ``.osz``.

Abstract over the `ChartPredictor` interface: this script has no
knowledge of whether the predictor is autoregressive, framewise, or
anything else. All that lives in the spec JSON.

Usage::

    osu/taiko2/.venv/Scripts/python.exe -m osu.taiko2.cli.infer \\
        --config osu/taiko2/experiments/002-exp45-full/config/infer.json \\
        --audio path/to/song.mp3 \\
        --density-mean 6.0 --density-peak 10 --density-std 2.5

Or with inline JSON::

    --config-json '{"checkpoint": "...", "predictor": {...}, ...}'

Spec JSON shape (every component is ``{__class__, config}``; the
predictor itself carries top-level ``__class__`` + ``config`` too):

    {
      "checkpoint": "runs/.../eval_41348/checkpoint.pt",
      "predictor":      {"__class__": "...:AutoregressivePredictor",
                         "config":    {"__class__": "...:AutoregressivePredictorConfig", ...}},
      "decoder":        {"__class__": "...", "config": {...}},
      "input_builder":  {"__class__": "...", "config": {...}},
      "audio_sampler":  {"__class__": "...", "config": {...}},
      "event_sampler":  {"__class__": "...", "config": {...}}
    }

Outputs under ``--out-dir`` (default ``Temp/``):
  - ``{name}.osz``              — playable in osu!
  - ``{name}.steps.jsonl``      — only with ``--debug``
  - ``{name}.metrics.json``     — only with ``--debug``
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from ..inference.spec import assemble_predictor, load_spec

# Sentinel for `--audio` passed without a value → run file picker.
_PICK_SENTINEL = "__pick__"
# Default `--out-dir`. When the user does not override it, output files
# are treated as throwaway and may be cleaned up after `--andopen`.
_DEFAULT_OUT_DIR = Path("Temp")

from ..domain.beatmap import AudioRef, Density, Difficulty, Track
from ..domain.chart import Chart
from ..domain.inference import ChartPredictor, Conditioning
from ..inference.loader import load_model_from_checkpoint


# ─────────────────────────── stub chart from audio ────────────────────

def _load_audio_bytes(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    ext = path.suffix.lower().lstrip(".") or "mp3"
    return raw, ext


def _pick_audio_file() -> Path:
    """Native file dialog for picking an audio file. Raises SystemExit
    on cancel."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    picked = filedialog.askopenfilename(
        title="Select an audio file to infer",
        filetypes=[
            ("Audio", "*.mp3 *.ogg *.wav *.flac *.m4a *.opus"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not picked:
        raise SystemExit("no audio selected; aborting.")
    return Path(picked)


def _is_url(value: str) -> bool:
    """True if `value` looks like an http(s) URL. yt-dlp's extractor
    list is huge — we trust yt-dlp to reject non-supported URLs itself."""
    return value.startswith(("http://", "https://"))


def _download_url_to_temp(url: str) -> tuple[Path, "contextlib.AbstractContextManager[Any]"]:
    """Download audio from `url` via yt-dlp into a temp directory.

    Returns ``(audio_path, cleanup_cm)``. Exit the context manager
    after reading the audio bytes to delete the temp directory.
    """
    # Lazy-imported: yt_dlp is a ~2 MB module tree and we don't want to
    # pay its cost on every infer invocation, only when a URL is passed.
    import yt_dlp

    tmp_dir_cm = tempfile.TemporaryDirectory(prefix="taiko2_infer_ytdlp_")
    tmp_dir = Path(tmp_dir_cm.name)
    outtmpl = str(tmp_dir / "%(id)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }
    print(f"[infer] fetching audio via yt-dlp: {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    downloaded = [p for p in tmp_dir.iterdir() if p.is_file()]
    if not downloaded:
        tmp_dir_cm.cleanup()
        raise SystemExit(f"yt-dlp downloaded nothing for {url!r}")
    # Biggest file is the audio (occasional metadata sidecars are small).
    audio_path = max(downloaded, key=lambda p: p.stat().st_size)
    print(
        f"[infer] got {audio_path.name}  "
        f"({audio_path.stat().st_size:,} bytes, "
        f"source={info.get('extractor', '?')})"
    )
    return audio_path, tmp_dir_cm


def _resolve_audio_arg(value: str | None) -> tuple[Path, contextlib.ExitStack]:
    """Turn the `--audio` argument into a local Path.

    Returns the path plus an `ExitStack` that must be closed after the
    audio bytes have been read (used to delete any URL-download temp
    directory).
    """
    stack = contextlib.ExitStack()
    if value is None:
        stack.close()
        raise SystemExit(
            "--audio is required (pass a path, a URL, or the flag alone "
            "to open a file picker)"
        )
    if value == _PICK_SENTINEL:
        return _pick_audio_file(), stack
    if _is_url(value):
        audio_path, tmp_cm = _download_url_to_temp(value)
        stack.enter_context(tmp_cm)
        return audio_path, stack
    return Path(value), stack


def _stub_chart(audio_path: Path, *, title: str | None = None) -> Chart:
    """Build a minimal `Chart` carrying just audio bytes. Track has no
    onsets (that's what the predictor fills in) and stub metadata."""
    audio_bytes, ext = _load_audio_bytes(audio_path)
    stem = audio_path.stem
    track = Track(
        beatmap_id="0",
        beatmapset_id="0",
        artist="unknown",
        title=title or stem,
        difficulty=Difficulty(
            version="taiko2-infer",
            overall_difficulty=5.0,
            star_rating=None,
        ),
        audio=AudioRef(filename=audio_path.name, format=ext),
        onsets=(),
        density=Density(
            mean=0.0, peak=0, std=0.0, duration_s=0.0, total_events=0,
        ),
    )
    return Chart(track=track, audio=audio_bytes)


# ─────────────────────────── CLI ──────────────────────────────────────

def _dump_metrics(chart: Chart, path: Path) -> None:
    metrics = chart.calculate_metrics()
    path.write_text(
        json.dumps(dataclasses.asdict(metrics), indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="osu.taiko2.cli.infer",
        description="Abstract ChartPredictor inference CLI.",
    )
    p.add_argument("--config", type=Path,
                   help="Path to spec JSON (predictor wiring).")
    p.add_argument("--config-json", type=str,
                   help="Inline spec JSON (alternative to --config).")
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help=(
            "Override the `checkpoint` field in the spec JSON. Useful to "
            "point the same predictor wiring at a different eval's "
            "checkpoint.pt without editing the file."
        ),
    )
    p.add_argument(
        "--audio", type=str, nargs="?",
        const=_PICK_SENTINEL, default=None,
        help=(
            "Input audio source. Can be a local file path, an http(s) "
            "URL (downloaded via yt-dlp), or the flag passed alone "
            "(`--audio`) to open a file picker."
        ),
    )
    p.add_argument("--density-mean", type=float, required=True)
    p.add_argument("--density-peak", type=int, required=True)
    p.add_argument("--density-std", type=float, required=True)
    p.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
        help=(
            "Output directory (default: Temp/). When this is left at the "
            "default AND --andopen is set, the output .osz and its "
            "related debug files are deleted after the viewer closes."
        ),
    )
    p.add_argument("--name", type=str, default=None,
                   help="Output basename (default: audio file stem).")
    p.add_argument("--device", type=str, default="cuda",
                   help="cpu / cuda / cuda:N (default: cuda).")
    p.add_argument("--debug", action="store_true",
                   help="Save per-step AR log + chart metrics JSON.")
    p.add_argument(
        "--typing-config", type=Path, default=None,
        help=(
            "Path to a typing model spec JSON. When provided, a second "
            "pass runs the typing model over the predicted onsets to "
            "assign D/K and normal/big kinds. Without this, all onsets "
            "are DON. Spec shape: {\"checkpoint\": \"...\", \"config\": "
            "{\"strength_threshold\": 0.8, \"bin_ms\": 5.0}}"
        ),
    )
    p.add_argument(
        "--andopen", type=str, nargs="?",
        const="", default=None, metavar="VIEWER_FLAGS",
        help=(
            "After saving, spawn `python -m osu.taiko2.cli.viewer` on the "
            "output .osz. Pass extra viewer flags as a shell-split string "
            "(e.g. --andopen '--index 0'), or pass the flag alone to open "
            "with no extra flags."
        ),
    )
    args = p.parse_args(argv)

    audio_path, audio_cleanup = _resolve_audio_arg(args.audio)
    with audio_cleanup:
        if not audio_path.exists():
            raise SystemExit(f"audio file not found: {audio_path}")

        device = torch.device(args.device)
        spec = load_spec(config=args.config, config_json=args.config_json)
        if args.checkpoint is not None:
            spec["checkpoint"] = str(args.checkpoint)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        name = args.name or audio_path.stem
        osz_path = args.out_dir / f"{name}.osz"
        steps_path = (
            args.out_dir / f"{name}.steps.jsonl" if args.debug else None
        )
        metrics_path = (
            args.out_dir / f"{name}.metrics.json" if args.debug else None
        )

        predictor, _meta = assemble_predictor(
            spec=spec, device=device,
            per_step_log_path=steps_path if args.debug else None,
        )
        chart = _stub_chart(audio_path, title=name)
        conditioning = Conditioning(
            density_mean=args.density_mean,
            density_peak=args.density_peak,
            density_std=args.density_std,
        )

        print(f"[infer] checkpoint: {spec['checkpoint']}")
        print(
            f"[infer] audio:      {audio_path}  "
            f"({len(chart.audio):,} bytes)"
        )
        print(
            f"[infer] conditioning: mean={args.density_mean} "
            f"peak={args.density_peak} std={args.density_std}"
        )
        print(f"[infer] device:     {device}")

        out_chart = predictor.predict(chart, conditioning=conditioning)

    # Optional second pass: typing model assigns D/K + normal/big.
    if args.typing_config is not None:
        from ..inference.typing_pass import load_typing_spec, type_chart

        typing_model, typing_cfg = load_typing_spec(
            args.typing_config, device=device,
        )
        print(
            f"[infer] typing pass: {args.typing_config}  "
            f"(strength_thr={typing_cfg.strength_threshold})"
        )
        typing_features = predictor._extract_features(chart)
        out_chart = type_chart(
            typing_model, out_chart, typing_features,
            device=device, config=typing_cfg,
        )
        from collections import Counter
        kind_counts = Counter(o.kind.value for o in out_chart.track.onsets)
        print(f"[infer] typed: {dict(kind_counts)}")

    # Auto-map difficulty from the generated chart's density.
    from ..domain.chart import estimate_difficulty
    auto_diff = estimate_difficulty(out_chart.track.density.mean)
    out_chart = Chart(
        track=dataclasses.replace(out_chart.track, difficulty=auto_diff),
        audio=out_chart.audio,
    )
    print(
        f"[infer] difficulty: {auto_diff.version} "
        f"(star={auto_diff.star_rating}, OD={auto_diff.overall_difficulty})"
    )

    out_chart.save_osz(osz_path)
    print(
        f"[infer] wrote {osz_path}  "
        f"({len(out_chart.track.onsets):,} onsets)"
    )

    if args.debug:
        if metrics_path is not None:
            _dump_metrics(out_chart, metrics_path)
            print(f"[infer] wrote {metrics_path}")
        if steps_path is not None and steps_path.exists():
            print(f"[infer] step log: {steps_path}")

    if args.andopen is not None:
        viewer_cmd = [
            sys.executable, "-m", "osu.taiko2.cli.viewer", str(osz_path),
            *shlex.split(args.andopen),
        ]
        print(f"[infer] opening viewer: {' '.join(viewer_cmd)}")
        try:
            subprocess.run(viewer_cmd, check=False)
        finally:
            if args.out_dir == _DEFAULT_OUT_DIR:
                _cleanup_outputs(osz_path, steps_path, metrics_path)
    return 0


def _cleanup_outputs(*paths: Path | None) -> None:
    """Delete each path if it exists. Swallows file-system races; the
    rest of the cleanup proceeds regardless of one file missing."""
    for p in paths:
        if p is None:
            continue
        try:
            if p.exists():
                p.unlink()
                print(f"[infer] deleted {p}")
        except OSError as e:
            print(f"[infer] could not delete {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
