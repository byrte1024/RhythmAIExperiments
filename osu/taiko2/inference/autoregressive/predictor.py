"""Concrete `ChartPredictor` for autoregressive bin-offset models.

Composition — takes a `Model`, `ARDecoder`, `ARInputBuilder`,
`AudioSampler`, `EventSampler` at construction. `predict(chart)` runs
the AR loop and returns a new Chart with filled-in onsets.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...domain.beatmap import Onset, OnsetBinned, OnsetKind
from ...domain.chart import Chart
from ...domain.dataset import AudioSampler, EventSampler
from ...domain.inference import ChartPredictor, Conditioning, PredictorConfig
from ...domain.model import Model
from ...parsing.osu import compute_density
from .builders import ARInputBuilder
from .decoders import ARDecoder
from .types import ARContext


@dataclass(frozen=True, slots=True)
class AutoregressivePredictorConfig(PredictorConfig):
    """Knobs for the AR loop itself. Model / decoder / builder configs
    live on their own objects — this one is pure orchestration.
    """
    # Cursor advance when the decoder returns STOP. Matches taiko1's
    # `hop_ms` mechanic; 20 bins ≈ 100 ms at 5 ms/bin.
    hop_bins_on_stop: int = 20
    # Hard cap to prevent runaway AR on degenerate models.
    max_events: int = 10_000
    # Reject onset predictions closer than this to the cursor (treated
    # as if the decoder returned STOP).
    min_onset_gap_bins: int = 1
    # If set, append one JSONL line per AR step here — flushed every
    # write so crashes preserve the trace on disk.
    per_step_log_path: Path | None = None
    # Default onset kind when the model doesn't predict it.
    default_kind: OnsetKind = OnsetKind.DON


class AutoregressivePredictor(ChartPredictor[AutoregressivePredictorConfig]):
    """Full `ChartPredictor` driven by an AR loop.

    Composition:
      - `model`         — a `Model` that emits bin-offset + STOP logits.
      - `decoder`       — interprets each output into an `ARDecision`.
      - `input_builder` — builds the per-step `ModelInput`.
      - `audio_sampler` — one-shot: decode chart audio → features.
      - `event_sampler` — owns bin-to-ms conversion for emitted onsets.

    Requires `conditioning` (raises `ValueError` if not provided).
    """

    def __init__(
        self,
        config: AutoregressivePredictorConfig,
        *,
        model: Model,
        decoder: ARDecoder,
        input_builder: ARInputBuilder,
        audio_sampler: AudioSampler,
        event_sampler: EventSampler,
        device: torch.device | str = torch.device("cpu"),
    ):
        super().__init__(config)
        self._model = model
        self._decoder = decoder
        self._input_builder = input_builder
        self._audio_sampler = audio_sampler
        self._event_sampler = event_sampler
        self._device = torch.device(device) if isinstance(device, str) else device
        self._model.to(self._device).eval()

    # ── public API ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        chart: Chart,
        *,
        conditioning: Conditioning | None = None,
    ) -> Chart:
        if conditioning is None:
            raise ValueError(
                "AutoregressivePredictor requires `conditioning` "
                "(density_mean / density_peak / density_std)."
            )
        if chart.audio is None:
            raise ValueError(
                "AutoregressivePredictor requires chart.audio "
                "(Chart loaded from .osu has no audio; use .osz or a bundle)."
            )

        features = self._extract_features(chart)
        max_bin = int(round(features.shape[1] * self._frames_to_bins_ratio()))

        log_fh = self._open_step_log()
        try:
            past_onsets = self._run_ar_loop(
                features=features,
                max_bin=max_bin,
                conditioning=conditioning,
                log_fh=log_fh,
            )
        finally:
            if log_fh is not None:
                log_fh.close()

        return self._build_output_chart(chart, past_onsets)

    # ── audio decode + feature extraction (one-shot per chart) ────────

    def _extract_features(self, chart: Chart) -> np.ndarray:
        import librosa
        ext = chart.track.audio.format or "mp3"
        fd, name = tempfile.mkstemp(prefix="taiko2_ar_", suffix=f".{ext}")
        os.close(fd)
        tmp = Path(name)
        tmp.write_bytes(chart.audio)  # type: ignore[arg-type]
        try:
            waveform, sr = librosa.load(
                str(tmp), sr=self._audio_sampler.config.sample_rate, mono=True,
            )
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        return self._audio_sampler.sample_waveform(waveform, int(sr))

    def _frames_to_bins_ratio(self) -> float:
        """How many event-sampler bins per audio feature frame. In the
        aligned-defaults setup (sr=22000 hop=110, divisor=200) this is
        exactly 1.0 — one bin per frame."""
        return self._audio_sampler.frame_ms / self._event_sampler.bin_ms

    # ── AR loop ───────────────────────────────────────────────────────

    def _run_ar_loop(
        self,
        *,
        features: np.ndarray,
        max_bin: int,
        conditioning: Conditioning,
        log_fh: Any,
    ) -> list[OnsetBinned]:
        past_onsets: list[OnsetBinned] = []
        cursor_bin = 0
        step = 0
        bin_ms = self._event_sampler.bin_ms

        pbar: Any = None
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(
                total=int(max_bin), unit="bin",
                desc="AR infer", leave=False,
            )
        except ImportError:
            pbar = None

        try:
            while cursor_bin < max_bin and step < self.config.max_events:
                ctx = ARContext(
                    cursor_bin=cursor_bin,
                    step=step,
                    max_bin=max_bin,
                    past_onsets=tuple(past_onsets),
                )
                inp = self._input_builder.build(
                    cursor_bin=cursor_bin,
                    past_onsets=ctx.past_onsets,
                    audio_features=features,
                    conditioning=conditioning,
                    device=self._device,
                )
                out = self._model.predict(inp)
                decision = self._decoder.decode(out, ctx)

                cursor_before = cursor_bin
                placed: list[int] = []

                if decision.is_stop:
                    cursor_bin += self.config.hop_bins_on_stop
                else:
                    # Emit every onset in order. Each offset is cursor-
                    # relative against the *original* cursor (decoder's
                    # contract), so we compute absolute bins from
                    # cursor_before — not from the running cursor_bin.
                    for offset in decision.bin_offsets:
                        if offset < self.config.min_onset_gap_bins:
                            continue
                        new_bin = cursor_before + offset
                        if new_bin <= cursor_bin:
                            # Out of order or duplicate — drop it.
                            continue
                        past_onsets.append(OnsetBinned(
                            time_ms=int(round(new_bin * bin_ms)),
                            kind=self.config.default_kind,
                            bin=new_bin,
                        ))
                        cursor_bin = new_bin
                        placed.append(new_bin)
                    if not placed:
                        # Decoder returned offsets but none survived the
                        # gap/monotonic filter. Advance like STOP so we
                        # don't loop forever.
                        cursor_bin = (
                            cursor_before + self.config.hop_bins_on_stop
                        )

                if log_fh is not None:
                    log_fh.write(json.dumps({
                        "step": step,
                        "cursor_bin": cursor_before,
                        "cursor_bin_after": cursor_bin,
                        "is_stop": decision.is_stop,
                        "bin_offsets": list(decision.bin_offsets),
                        "confidences": list(decision.confidences),
                        "n_placed": len(placed),
                        **decision.extras,
                    }) + "\n")
                    log_fh.flush()

                if pbar is not None:
                    # Clamp the update so a decision past `max_bin`
                    # doesn't overshoot the progress bar total.
                    delta = min(cursor_bin, max_bin) - cursor_before
                    if delta > 0:
                        pbar.update(delta)
                    pbar.set_postfix(
                        onsets=len(past_onsets),
                        step=step,
                        refresh=False,
                    )

                step += 1
        finally:
            if pbar is not None:
                pbar.close()

        return past_onsets

    # ── chart rebuild ─────────────────────────────────────────────────

    @staticmethod
    def _build_output_chart(
        source: Chart, predicted_binned: list[OnsetBinned],
    ) -> Chart:
        # Keep the OnsetBinned instances: their `bin` field is part of
        # the prediction record and is useful for re-inference, viewer,
        # and chart comparison. OnsetBinned subclasses Onset, so this
        # is type-compatible with `Track.onsets: tuple[Onset, ...]`.
        onsets = tuple(predicted_binned)
        density = compute_density(onsets)
        new_track = replace(source.track, onsets=onsets, density=density)
        # Audio preserved on output so the Chart can be saved back to .osz.
        return Chart(track=new_track, audio=source.audio)

    # ── logging helper ────────────────────────────────────────────────

    def _open_step_log(self):
        path = self.config.per_step_log_path
        if path is None:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("a", encoding="utf-8")
