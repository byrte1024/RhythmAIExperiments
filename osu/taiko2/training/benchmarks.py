"""Per-eval benchmark suite — ten scripted distortions of a
`TaikoDetectionSample` that stress-test the model's dependence on
specific input channels (audio vs events, past vs future, legible vs
random) plus metronomic-pattern handling.

Each mode is a small transform applied at the sample level before the
adapter. The eval loop reuses the regular metric pipeline per mode
and emits a metric set per mode — so downstream hooks (JSONL logger,
curves, per-eval JSON) automatically track every benchmark metric
across training.

Contract
========

A `BenchmarkMode` is a `(name, transform, description)` triple.
`transform(sample, rng) -> TaikoDetectionSample | None`:
  - Returns a mutated sample to evaluate under this mode.
  - Returns `None` to skip the sample (used by `advanced_metronome`
    to exclude cases where continuing the metronome pattern IS the
    correct answer — those are uninformative).

Modes
=====

- `normal`              — identity; comparison point for all others.
- `no_audio`            — zero the mel (both past + future).
- `no_future_audio`     — zero the future-B mel.
- `no_past_audio`       — zero the past-A mel.
- `static_audio`        — replace mel with random noise.
- `no_context`          — mark all past events as padded.
- `random_context`      — replace past events with C random offsets.
- `metronome`           — replace past events with C spaced at a
                          random gap.
- `advanced_metronome`  — same, using the dominant gap from the
                          current past events, skipping samples where
                          target is the dominant gap.
- `context_time_shifted` — multiply every past event's cursor offset
                          by a factor drawn from the set
                          `{1/2, 1/3, 1/4, 1/6, 1/8, 2/1, 3/1, 4/1,
                          6/1, 8/1}`. Audio + target untouched — only
                          the past-event context is rescaled.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ..data_samplers.detection import TaikoDetectionSample
from ..domain.beatmap import OnsetKind, RelativeOnset


# ─────────────────────────── mode type ───────────────────────────────

TransformFn = Callable[
    [TaikoDetectionSample, random.Random],
    "TaikoDetectionSample | None",
]


@dataclass(frozen=True, slots=True)
class BenchmarkMode:
    name: str
    transform: TransformFn
    description: str = ""


# ─────────────────────────── helpers ─────────────────────────────────

def _padding_event() -> RelativeOnset:
    return RelativeOnset(time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=0)


def _dominant_past_gap(sample: TaikoDetectionSample) -> int | None:
    """Most common gap (in bins) between consecutive UN-PADDED past
    events, rounded to 5-bin buckets. `None` when fewer than 2 real
    past events exist."""
    real = [
        e for e, padded in zip(sample.past_events, sample.past_events_mask)
        if not bool(padded)
    ]
    if len(real) < 2:
        return None
    gaps = [
        abs(int(real[i].cursor_offset) - int(real[i - 1].cursor_offset))
        for i in range(1, len(real))
        if abs(int(real[i].cursor_offset) - int(real[i - 1].cursor_offset)) > 0
    ]
    if not gaps:
        return None
    bucket = 5
    buckets = Counter((g // bucket) * bucket for g in gaps)
    center, _ = buckets.most_common(1)[0]
    return int(center) + bucket // 2


def _build_metronome_past(
    c_events: int, gap: int,
) -> tuple[tuple[RelativeOnset, ...], np.ndarray]:
    """C events at negative offsets -gap, -2*gap, ..., -C*gap, in
    oldest-first order (most negative at position 0)."""
    offsets = sorted(-i * gap for i in range(1, c_events + 1))
    events = tuple(
        RelativeOnset(
            time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=o,
        )
        for o in offsets
    )
    mask = np.zeros(c_events, dtype=bool)
    return events, mask


# ─────────────────────────── transforms ──────────────────────────────

def _xform_normal(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    return s


def _xform_no_audio(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    return replace(
        s,
        audio_past=np.zeros_like(s.audio_past),
        audio_future=np.zeros_like(s.audio_future),
    )


def _xform_no_future_audio(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    return replace(s, audio_future=np.zeros_like(s.audio_future))


def _xform_no_past_audio(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    return replace(s, audio_past=np.zeros_like(s.audio_past))


def _xform_static_audio(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    return replace(
        s,
        audio_past=np_rng.standard_normal(
            s.audio_past.shape, dtype=np.float32,
        ),
        audio_future=np_rng.standard_normal(
            s.audio_future.shape, dtype=np.float32,
        ),
    )


def _xform_no_context(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    c = len(s.past_events)
    return replace(
        s,
        past_events=tuple(_padding_event() for _ in range(c)),
        past_events_mask=np.ones(c, dtype=bool),
    )


def _xform_random_context(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    c = len(s.past_events)
    # Range roughly covers the sampler's a_bins window; negative =
    # past. Oldest event sits at the front (most negative).
    lo = -max(c * 50, 100)
    offsets = sorted(rng.randint(lo, -1) for _ in range(c))
    events = tuple(
        RelativeOnset(
            time_ms=0, kind=OnsetKind.UNKNOWN, bin=0, cursor_offset=o,
        )
        for o in offsets
    )
    return replace(
        s,
        past_events=events,
        past_events_mask=np.zeros(c, dtype=bool),
    )


def _xform_metronome(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    gap = rng.randint(20, 200)  # bins; ~100 ms to ~1 s at 5 ms/bin
    events, mask = _build_metronome_past(len(s.past_events), gap)
    return replace(s, past_events=events, past_events_mask=mask)


def _xform_advanced_metronome(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    if len(s.future_events) == 0 or bool(s.future_events_mask[0]):
        return None
    dominant = _dominant_past_gap(s)
    if dominant is None or dominant <= 0:
        return None
    target = int(s.future_events[0].cursor_offset)
    # Skip samples where the target IS the dominant past gap —
    # continuing the metronome is the correct answer and would
    # confound the diagnostic.
    if abs(target - dominant) <= 5:
        return None
    events, mask = _build_metronome_past(len(s.past_events), dominant)
    return replace(s, past_events=events, past_events_mask=mask)


TIME_SHIFT_FACTORS: tuple[float, ...] = (
    1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 8,
    2 / 1, 3 / 1, 4 / 1, 6 / 1, 8 / 1,
)


def _xform_context_time_shifted(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    factor = rng.choice(TIME_SHIFT_FACTORS)
    events = tuple(
        replace(
            e,
            cursor_offset=int(round(int(e.cursor_offset) * factor)),
        )
        for e in s.past_events
    )
    # Leave the mask alone — we're rescaling offsets in place, not
    # adding/removing events.
    return replace(s, past_events=events)


# ─────────────────────────── registry ───────────────────────────────

BENCH_NORMAL = BenchmarkMode(
    name="normal", transform=_xform_normal,
    description="Identity — baseline comparison point.",
)
BENCH_NO_AUDIO = BenchmarkMode(
    name="no_audio", transform=_xform_no_audio,
    description="Zero both past and future mel. Expected very high STOP, low accuracy.",
)
BENCH_NO_FUTURE_AUDIO = BenchmarkMode(
    name="no_future_audio", transform=_xform_no_future_audio,
    description="Zero future-B mel; past-A mel intact.",
)
BENCH_NO_PAST_AUDIO = BenchmarkMode(
    name="no_past_audio", transform=_xform_no_past_audio,
    description="Zero past-A mel; future-B mel intact. Measures reliance on past audio.",
)
BENCH_STATIC_AUDIO = BenchmarkMode(
    name="static_audio", transform=_xform_static_audio,
    description="Replace mel with random noise. Worst-case degenerate.",
)
BENCH_NO_CONTEXT = BenchmarkMode(
    name="no_context", transform=_xform_no_context,
    description="All past events padded. Measures audio-only capacity.",
)
BENCH_RANDOM_CONTEXT = BenchmarkMode(
    name="random_context", transform=_xform_random_context,
    description="Past events replaced with C random cursor offsets.",
)
BENCH_METRONOME = BenchmarkMode(
    name="metronome", transform=_xform_metronome,
    description="Past events replaced with C events at a random constant gap.",
)
BENCH_ADVANCED_METRONOME = BenchmarkMode(
    name="advanced_metronome", transform=_xform_advanced_metronome,
    description=(
        "Past events replaced with C events at the DOMINANT past gap, "
        "skipping samples where target==dominant (i.e. only non-trivial "
        "break-from-metronome cases are evaluated)."
    ),
)
BENCH_CONTEXT_TIME_SHIFTED = BenchmarkMode(
    name="context_time_shifted", transform=_xform_context_time_shifted,
    description=(
        "Multiply every past event's cursor offset by a random factor "
        "from {1/2, 1/3, 1/4, 1/6, 1/8, 2/1, 3/1, 4/1, 6/1, 8/1}. "
        "Audio and target untouched — only the past-event context is "
        "rescaled."
    ),
)


# ─────────────────────────── coincidence benchmarks ──────────────────


def _xform_no_coincidence(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    """Zero the last 13 rows of audio (coincidence channels)."""
    n_mel = 80
    if s.audio_past.shape[0] <= n_mel:
        return s
    past = s.audio_past.copy()
    future = s.audio_future.copy()
    past[n_mel:, :] = 0.0
    future[n_mel:, :] = 0.0
    return replace(s, audio_past=past, audio_future=future)


def _xform_no_mel(
    s: TaikoDetectionSample, rng: random.Random,
) -> TaikoDetectionSample | None:
    """Zero the first 80 rows of audio (mel channels), keep coincidence."""
    n_mel = 80
    if s.audio_past.shape[0] <= n_mel:
        return replace(
            s,
            audio_past=np.zeros_like(s.audio_past),
            audio_future=np.zeros_like(s.audio_future),
        )
    past = s.audio_past.copy()
    future = s.audio_future.copy()
    past[:n_mel, :] = 0.0
    future[:n_mel, :] = 0.0
    return replace(s, audio_past=past, audio_future=future)


BENCH_NO_COINCIDENCE = BenchmarkMode(
    name="no_coincidence", transform=_xform_no_coincidence,
    description="Zero coincidence rows (last 13); mel intact.",
)
BENCH_NO_MEL = BenchmarkMode(
    name="no_mel", transform=_xform_no_mel,
    description="Zero mel rows (first 80); coincidence intact.",
)


DEFAULT_BENCHMARKS: tuple[BenchmarkMode, ...] = (
    BENCH_NORMAL,
    BENCH_NO_AUDIO,
    BENCH_NO_FUTURE_AUDIO,
    BENCH_NO_PAST_AUDIO,
    BENCH_STATIC_AUDIO,
    BENCH_NO_CONTEXT,
    BENCH_RANDOM_CONTEXT,
    BENCH_METRONOME,
    BENCH_ADVANCED_METRONOME,
    BENCH_CONTEXT_TIME_SHIFTED,
    BENCH_NO_COINCIDENCE,
    BENCH_NO_MEL,
)


_BY_NAME: dict[str, BenchmarkMode] = {b.name: b for b in DEFAULT_BENCHMARKS}


def benchmarks_by_name(names: "list[str] | tuple[str, ...]") -> list[BenchmarkMode]:
    """Resolve a list of mode names to the shared `BenchmarkMode`
    singletons. Unknown names raise ValueError."""
    out: list[BenchmarkMode] = []
    for n in names:
        if n not in _BY_NAME:
            raise ValueError(
                f"unknown benchmark mode {n!r}; "
                f"known: {sorted(_BY_NAME)!r}"
            )
        out.append(_BY_NAME[n])
    return out
