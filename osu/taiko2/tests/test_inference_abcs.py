"""Tests for the inference-side ABCs and the AutoregressivePredictor.

Uses stub Model / Decoder / InputBuilder / AudioSampler / EventSampler
so the AR loop can be driven deterministically without training any
real model. Focus:
  - `ChartPredictor` abstract enforcement.
  - `ARDecision` multi-onset semantics (is_stop, tuple ordering).
  - `AutoregressivePredictor` AR loop — single-onset path, multi-onset
    path, STOP path, cursor-advance semantics.
  - `load_model_from_checkpoint` round-trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from osu.taiko2.domain.beatmap import (
    AudioRef,
    Density,
    Difficulty,
    Onset,
    OnsetBinned,
    OnsetKind,
    Track,
)
from osu.taiko2.domain.chart import Chart
from osu.taiko2.domain.dataset import (
    AudioSampler,
    AudioSamplerConfig,
    EventSampler,
    EventSamplerConfig,
    MelSamplerConfig,
)
from osu.taiko2.domain.inference import (
    ChartPredictor,
    Conditioning,
    PredictorConfig,
)
from osu.taiko2.domain.loss import Loss, LossConfig, LossResult
from osu.taiko2.domain.model import (
    Model,
    ModelConfig,
    ModelInput,
    ModelOutput,
    ModelTarget,
)
from osu.taiko2.domain.training import TrainerConfig, TrainingState
from osu.taiko2.inference.autoregressive import (
    ARContext,
    ARDecision,
    ARDecoder,
    ARDecoderConfig,
    ARInputBuilder,
    ARInputBuilderConfig,
    AutoregressivePredictor,
    AutoregressivePredictorConfig,
)
from osu.taiko2.inference.loader import load_model_from_checkpoint
from osu.taiko2.persistence.checkpoint import Checkpoint
from osu.taiko2.samplers import FixedRateEventSampler


# ─────────────────────────── stub types ───────────────────────────────

@dataclass(frozen=True, slots=True)
class _StubModelConfig(ModelConfig):
    d_model: int = 4


@dataclass(frozen=True, slots=True)
class _StubInput(ModelInput):
    x: torch.Tensor = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class _StubOutput(ModelOutput):
    payload: Any = None


@dataclass(frozen=True, slots=True)
class _StubTarget(ModelTarget):
    y: torch.Tensor = None  # type: ignore[assignment]


class _StubModel(Model[_StubModelConfig, _StubInput, _StubOutput]):
    """Returns a fixed payload handed to the decoder — so tests can
    script a decoder's behavior directly without any tensor math."""

    def __init__(self, config: _StubModelConfig):
        super().__init__(config)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def predict(self, x: _StubInput) -> _StubOutput:
        return _StubOutput(payload=None)


@dataclass(frozen=True, slots=True)
class _StubLossConfig(LossConfig):
    weight: float = 1.0


class _StubLoss(Loss[_StubLossConfig, _StubOutput, _StubTarget]):
    def forward(self, output: _StubOutput, target: _StubTarget) -> LossResult:
        return LossResult(loss=torch.tensor(0.0), metrics={})


@dataclass(frozen=True, slots=True)
class _ScriptedDecoderConfig(ARDecoderConfig):
    decisions: tuple[ARDecision, ...] = ()


class _ScriptedDecoder(ARDecoder[_ScriptedDecoderConfig, _StubOutput]):
    """Returns the next pre-scripted `ARDecision` on every call. Loops
    to `ARDecision()` (== STOP) once the script is exhausted."""

    def __init__(self, config: _ScriptedDecoderConfig):
        super().__init__(config)
        self._i = 0

    def decode(self, output: _StubOutput, context: ARContext) -> ARDecision:
        if self._i < len(self.config.decisions):
            d = self.config.decisions[self._i]
            self._i += 1
            return d
        return ARDecision()  # default STOP


class _NoOpBuilder(ARInputBuilder[ARInputBuilderConfig, _StubInput]):
    def build(self, *, cursor_bin, past_onsets, audio_features,
              conditioning, device):
        return _StubInput(x=torch.zeros(1, device=device))


class _StubAudioSampler(AudioSampler):
    """AudioSampler that returns a fixed (F, T) feature array, bypassing
    librosa/torchaudio."""

    def __init__(self, config: MelSamplerConfig, n_features: int = 4, n_frames: int = 2000):
        self.config = config
        self._n_features = n_features
        self._n_frames = n_frames

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def frame_ms(self) -> float:
        return self.config.effective_hop_length / self.config.sample_rate * 1000.0

    def _transform(self, waveform: np.ndarray) -> np.ndarray:
        return np.zeros((self._n_features, self._n_frames), dtype=np.float32)

    def sample_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.zeros((self._n_features, self._n_frames), dtype=np.float32)


# ─────────────────────────── Chart helpers ────────────────────────────

def _blank_chart() -> Chart:
    track = Track(
        beatmap_id="1", beatmapset_id="1", artist="a", title="t",
        difficulty=Difficulty(version="Oni", overall_difficulty=5.0),
        audio=AudioRef(filename="x.mp3", format="mp3"),
        onsets=tuple(),
        density=Density(mean=0.0, peak=0, std=0.0, duration_s=0.0, total_events=0),
    )
    return Chart(track=track, audio=b"\x00" * 16)


def _make_predictor(
    *,
    decisions: tuple[ARDecision, ...],
    hop_bins_on_stop: int = 20,
    min_onset_gap_bins: int = 1,
    n_frames: int = 2000,
) -> AutoregressivePredictor:
    mel_cfg = MelSamplerConfig()
    evt = FixedRateEventSampler(EventSamplerConfig())
    # Monkey-patch: skip real audio decode. _extract_features always
    # returns the stub sampler's zeros.
    predictor = AutoregressivePredictor(
        config=AutoregressivePredictorConfig(
            hop_bins_on_stop=hop_bins_on_stop,
            min_onset_gap_bins=min_onset_gap_bins,
            max_events=100,
        ),
        model=_StubModel(_StubModelConfig()),
        decoder=_ScriptedDecoder(_ScriptedDecoderConfig(decisions=decisions)),
        input_builder=_NoOpBuilder(ARInputBuilderConfig()),
        audio_sampler=_StubAudioSampler(mel_cfg, n_frames=n_frames),
        event_sampler=evt,
    )
    # Replace the audio-decode path with a no-op that returns our stub
    # features. Keeps the predictor's public API untouched.
    predictor._extract_features = lambda chart: np.zeros(  # type: ignore[method-assign]
        (predictor._audio_sampler.n_features, n_frames),
        dtype=np.float32,
    )
    return predictor


# ─────────────────────────── ChartPredictor ABC ───────────────────────

class TestChartPredictorABC:
    def test_abstract_predict_enforced(self):
        class Broken(ChartPredictor[PredictorConfig]):
            pass
        with pytest.raises(TypeError, match="abstract"):
            Broken(PredictorConfig())  # type: ignore[abstract]

    def test_predict_many_default_loops(self):
        """Default `predict_many` wraps `predict` in a for loop."""
        calls: list[Chart] = []

        class Custom(ChartPredictor[PredictorConfig]):
            def predict(self, chart, *, conditioning=None):
                calls.append(chart)
                return chart

        p = Custom(PredictorConfig())
        a = _blank_chart()
        b = _blank_chart()
        result = p.predict_many([a, b])
        assert len(result) == 2
        assert calls == [a, b]


# ─────────────────────────── ARDecision semantics ─────────────────────

class TestARDecision:
    def test_empty_is_stop(self):
        d = ARDecision()
        assert d.is_stop is True
        assert d.bin_offsets == ()

    def test_nonempty_is_not_stop(self):
        d = ARDecision(bin_offsets=(10, 20, 30))
        assert d.is_stop is False

    def test_extras_default_empty(self):
        d = ARDecision(bin_offsets=(5,))
        assert d.extras == {}


# ─────────────────────────── Autoregressive loop ──────────────────────

class TestAutoregressivePredictor:
    def test_requires_conditioning(self):
        p = _make_predictor(decisions=())
        with pytest.raises(ValueError, match="conditioning"):
            p.predict(_blank_chart(), conditioning=None)

    def test_requires_audio(self):
        p = _make_predictor(decisions=())
        chart_no_audio = Chart(track=_blank_chart().track, audio=None)
        with pytest.raises(ValueError, match="chart.audio"):
            p.predict(chart_no_audio, conditioning=Conditioning(4.0, 8, 1.0))

    def test_single_onset_path(self):
        """Decoder emits one onset per step; cursor jumps to it."""
        p = _make_predictor(decisions=(
            ARDecision(bin_offsets=(100,)),
            ARDecision(bin_offsets=(150,)),
            ARDecision(),  # STOP — advances hop_bins_on_stop then re-asks
        ), hop_bins_on_stop=10, n_frames=600)
        out = p.predict(_blank_chart(), conditioning=Conditioning(4.0, 8, 1.0))
        bins = [o.bin for o in out.track.onsets]
        # Cursor path: 0 → 100 (onset) → 250 (onset=100+150) → hop
        assert bins == [100, 250]

    def test_multi_onset_path(self):
        """Decoder emits N onsets per step; cursor walks through all,
        lands at the last."""
        p = _make_predictor(decisions=(
            ARDecision(bin_offsets=(50, 100, 200, 300)),   # 4 onsets in one step
            ARDecision(),                                   # STOP
        ), hop_bins_on_stop=10, n_frames=600)
        out = p.predict(_blank_chart(), conditioning=Conditioning(4.0, 8, 1.0))
        bins = [o.bin for o in out.track.onsets]
        # Offsets are cursor-relative against the *step's* start cursor
        # (0 here), so absolute positions are [50, 100, 200, 300].
        assert bins == [50, 100, 200, 300]

    def test_stop_advances_cursor(self):
        """STOP without any valid placements advances by hop_bins_on_stop."""
        p = _make_predictor(decisions=(
            ARDecision(),                          # STOP
            ARDecision(bin_offsets=(30,)),         # onset after hop
        ), hop_bins_on_stop=50, n_frames=200)
        out = p.predict(_blank_chart(), conditioning=Conditioning(4.0, 8, 1.0))
        bins = [o.bin for o in out.track.onsets]
        # cursor: 0 → 50 (STOP hop) → 80 (onset = 50+30)
        assert bins == [80]

    def test_duplicate_offsets_dropped(self):
        """Multi-onset decisions that put the second onset on the same
        (or earlier) bin as the first are dropped."""
        p = _make_predictor(decisions=(
            ARDecision(bin_offsets=(100, 100, 50)),  # last two are out-of-order
            ARDecision(),
        ), hop_bins_on_stop=10, n_frames=400)
        out = p.predict(_blank_chart(), conditioning=Conditioning(4.0, 8, 1.0))
        bins = [o.bin for o in out.track.onsets]
        assert bins == [100]  # 100 kept, 100 dropped (dup), 50 dropped (backward)

    def test_too_close_onset_dropped(self):
        """Onsets inside `min_onset_gap_bins` of the cursor are dropped."""
        p = _make_predictor(decisions=(
            ARDecision(bin_offsets=(0, 2, 50)),
            ARDecision(),
        ), hop_bins_on_stop=10, min_onset_gap_bins=5, n_frames=200)
        out = p.predict(_blank_chart(), conditioning=Conditioning(4.0, 8, 1.0))
        bins = [o.bin for o in out.track.onsets]
        # 0 and 2 below min_gap → dropped; 50 kept
        assert bins == [50]

    def test_output_chart_preserves_metadata_and_audio(self):
        p = _make_predictor(decisions=(
            ARDecision(bin_offsets=(50,)),
            ARDecision(),
        ), hop_bins_on_stop=10, n_frames=200)
        source = _blank_chart()
        out = p.predict(source, conditioning=Conditioning(4.0, 8, 1.0))
        assert out.track.artist == source.track.artist
        assert out.track.difficulty.version in (
            "Kantan", "Futsuu", "Muzukashii", "Oni", "Inner Oni",
        )
        assert out.track.difficulty.star_rating is not None
        assert out.audio == source.audio   # audio round-trips

    def test_per_step_log_written(self, tmp_path: Path):
        log_path = tmp_path / "ar.jsonl"
        p = _make_predictor(decisions=(
            ARDecision(bin_offsets=(50, 120), confidences=(0.9, 0.7)),
            ARDecision(),
        ), hop_bins_on_stop=10, n_frames=400)
        p.config = AutoregressivePredictorConfig(
            hop_bins_on_stop=10,
            min_onset_gap_bins=1,
            max_events=100,
            per_step_log_path=log_path,
        )
        p.predict(_blank_chart(), conditioning=Conditioning(4.0, 8, 1.0))

        import json
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 2
        first = json.loads(lines[0])
        assert first["step"] == 0
        assert first["bin_offsets"] == [50, 120]
        assert first["n_placed"] == 2
        assert first["is_stop"] is False


# ─────────────────────────── loader round-trip ────────────────────────

def test_load_model_from_checkpoint(tmp_path: Path):
    """Save a stub model + loss, load them back via the registered
    `module:Class` paths, confirm weights round-trip."""
    model = _StubModel(_StubModelConfig(d_model=4))
    loss = _StubLoss(_StubLossConfig(weight=0.25))
    ckpt = Checkpoint.from_runtime(
        model=model, loss=loss,
        optimizer=None, scheduler=None,
        trainer_config=TrainerConfig(),
        training_state=TrainingState(),
    )
    path = tmp_path / "best.pt"
    ckpt.save(path)

    loaded_model, loaded_loss, meta = load_model_from_checkpoint(path)
    assert isinstance(loaded_model, _StubModel)
    assert isinstance(loaded_loss, _StubLoss)
    assert loaded_loss.config.weight == 0.25
    # Weights match
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1.detach(), p2.detach())
    assert meta.model_class.endswith(":_StubModel")
