"""Data types for AR-style inference.

`ARContext` is what the decoder sees each step — read-only snapshot of
the cursor + history. `ARDecision` is what the decoder returns: zero
or more onsets emitted this step (empty ⇒ STOP). Both are internal to
the AR predictor; the top-level `ChartPredictor` never surfaces them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...domain.beatmap import OnsetBinned


@dataclass(frozen=True, slots=True)
class ARContext:
    """Read-only view of the AR loop's state when the decoder is called."""
    cursor_bin: int
    step: int
    max_bin: int
    past_onsets: tuple[OnsetBinned, ...]


@dataclass(frozen=True, slots=True)
class ARDecision:
    """One AR step's output: 0..N onsets emitted, all cursor-relative.

    `bin_offsets` are ALWAYS cursor-relative, regardless of how the
    model internally encodes multi-onset predictions. A delta-encoded
    model (e.g. exp 64) converts gaps to cumulative offsets inside the
    decoder before returning, so the engine never has to know.

    An empty `bin_offsets` tuple means STOP: the engine advances the
    cursor by `hop_bins_on_stop` instead of placing an onset. Single-
    onset decoders return a length-1 tuple; multi-onset decoders
    return length-N (truncated at the first internal STOP for models
    that emit STOP per slot).
    """
    bin_offsets: tuple[int, ...] = ()
    confidences: tuple[float, ...] = ()
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def is_stop(self) -> bool:
        return not self.bin_offsets


@dataclass(frozen=True, slots=True)
class ARDecoderConfig:
    """Base for AR decoder hyperparameters. Subclass for concrete decoders."""


@dataclass(frozen=True, slots=True)
class ARInputBuilderConfig:
    """Base for AR input-builder hyperparameters. Subclass to declare
    window / context sizes, mel layout, etc."""
