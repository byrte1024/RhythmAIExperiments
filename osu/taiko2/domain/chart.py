"""`Chart` — the active, analyzable form of a `Track`.

Wraps a parsed `Track` (metadata + onsets) plus optional audio bytes, and
adds analysis methods:

  - `calculate_metrics()` — dense per-chart statistics: type distribution,
    IOI histogram, density timeline, silence/dense regions, same-gap
    streaks, BPM estimate from the mode IOI, self pattern-space diversity.
    Consolidates what `analyze.py`, `analyze_metronome_data.py`,
    and `detection_inference._compute_run_stats` all produced separately.

  - `compare(other)` — two-chart comparison: GT-matching rates (ported
    from `run_comparison.compute_gt_metrics`) and TaikoNation-style
    pattern-space metrics (from `run_comparison.compute_tn_metrics`).

  - `save(path)` / `Chart.load(path)` — single-file container (zip) that
    bundles onset metadata with audio bytes, so a chart can be passed
    around without dragging separate files.
"""
from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import random as _stdlib_random

import numpy as np

from .beatmap import AudioRef, Density, Difficulty, Onset, OnsetKind, Track

# TaikoNation's fixed quantization step for pattern-space metrics.
TN_STEP_MS: int = 23
# 8-bit pattern window used by TaikoNation's over_pspace / hi_pspace.
TN_SCALE: int = 8
# Seed TaikoNation's published results used for the dc_rand baseline.
TN_SEED: int = 2009000042

RESOLUTION_FPS: tuple[int, ...] = (1, 2, 4, 10, 20, 50, 100, 200)


# ─────────────────────────── metrics payload ──────────────────────────

@dataclass(frozen=True, slots=True)
class ChartMetrics:
    """Dense self-statistics for one chart. Purely computed from the
    onset sequence + metadata; no comparison to another chart."""
    # Basic
    total_events: int
    duration_s: float
    events_per_sec: float

    # Type distribution
    count_don: int
    count_ka: int
    count_big_don: int
    count_big_ka: int
    count_drumroll: int
    count_spinner: int
    don_ratio: float          # (don + big_don) / (don + big_don + ka + big_ka)

    # Inter-onset interval (ms)
    ioi_min_ms: float
    ioi_max_ms: float
    ioi_mean_ms: float
    ioi_median_ms: float
    ioi_std_ms: float
    ioi_p95_ms: float
    ioi_p99_ms: float
    short_ioi_count: int      # < 20 ms (potential double-triggers)
    short_ioi_pct: float
    long_gap_count: int       # > 2000 ms (silences/breaks)

    # IOI histogram in 10-ms buckets (only buckets with count >= 2)
    ioi_histogram_10ms: dict[int, int] = field(default_factory=dict)

    # Density timeline (1-second windows) + summary
    density_mean: float = 0.0
    density_peak: int = 0
    density_std: float = 0.0
    density_min: int = 0
    density_cv: float = 0.0
    density_timeline: tuple[int, ...] = ()          # per-second event count
    silence_regions: tuple[tuple[float, float], ...] = ()  # (start_s, end_s) where density=0 for >=2s
    dense_regions: tuple[tuple[float, float], ...] = ()    # regions >2× mean density for >=3s

    # Same-gap streaks (5 % tolerance)
    longest_streak: int = 0
    mean_streak_len: float = 0.0
    streak_event_fraction: float = 0.0

    # BPM estimate
    estimated_bpm: float | None = None
    dominant_ioi_ms: float | None = None

    # Self pattern diversity (TaikoNation): fraction of 2^scale distinct
    # 8-step patterns observed in the 23-ms quantized onset train.
    over_pspace_self: float = 0.0

    # ── Gap-distribution shape ──
    # 200-bucket dense histogram over [0, 2000) ms at 10-ms resolution;
    # gaps above the 2 s cap are excluded (they're already counted in
    # `long_gap_count`).
    #
    # - `gap_peak_count`: peaks (local max ±20 ms, height >= max(5%% of
    #                     global max, 3), 30 ms min separation).
    # - `gap_peak_falloff`: mean `c_{i+1} / c_i` of sorted-desc peak
    #                       counts. 0.0 when <2 peaks.
    # - `gap_random_distance`:    TVD from uniform over the 200 buckets.
    # - `gap_metronome_distance`: TVD from delta-at-mode = 1 - mode_share.
    # - `gap_peaks`: per-peak detail `(bucket_center_ms, count)` sorted
    #                by count descending.
    # - `gap_peak_mass_total`: sum of counts across all kept peaks —
    #                          "how many gaps in this chart live inside
    #                          a recognized peak" (the rest sit in
    #                          noise / tail). Paired with
    #                          `gap_peak_count` as the "raw vs mass"
    #                          summary per chart.
    gap_histogram_dense: tuple[int, ...] = ()
    gap_peak_count: int = 0
    gap_peak_falloff: float = 0.0
    gap_random_distance: float = 0.0
    gap_metronome_distance: float = 0.0
    gap_peaks: tuple[tuple[float, int], ...] = ()
    gap_peak_mass_total: int = 0

    # ── Ratio-distribution shape ──
    # 200-bucket dense histogram of `ratio = gap[i] / gap[i-1]` in log2
    # space over [log2(1/8), log2(8)] = [-3, +3]. Same peak-detection
    # rules as the gap histogram. `ratio_metronome_distance` uses the
    # same formula; a chart whose peak ratio is 2.0 has distance from
    # "all ratios are 2.0". Values outside the log2 range are dropped.
    ratio_histogram_dense: tuple[int, ...] = ()
    ratio_peak_count: int = 0
    ratio_peak_falloff: float = 0.0
    ratio_random_distance: float = 0.0
    ratio_metronome_distance: float = 0.0
    ratio_peaks: tuple[tuple[float, int], ...] = ()
    ratio_peak_mass_total: int = 0


# ─────────────────────────── comparison payload ───────────────────────

@dataclass(frozen=True, slots=True)
class ResolutionComparison:
    """Onset comparison at a single temporal resolution (FPS)."""
    fps: int
    frame_ms: float
    n_frames: int
    # Binary: does frame have >= 1 onset?
    binary_precision: float
    binary_recall: float
    binary_f1: float
    # Integer: onset count per frame
    count_mae: float
    count_corr: float
    count_accuracy: float


@dataclass(frozen=True, slots=True)
class ChartComparison:
    """Pairwise metrics between this chart (self) and another (reference)."""
    # GT-matching (ms tolerances)
    n_self: int
    n_other: int
    matched_rate: float       # self onsets with nearest other within 25 ms
    close_rate: float         # within 50 ms
    far_rate: float           # other onsets with no self within 100 ms
    hallucination_rate: float # self onsets with no other within 100 ms
    error_mean_ms: float
    error_median_ms: float
    density_ratio: float      # self events/s ÷ other events/s

    # Standard precision / recall / F1 at 25 ms tolerance
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # TaikoNation pattern-space (self vs other)
    over_pspace_self: float = 0.0
    over_pspace_other: float = 0.0
    hi_pspace: float = 0.0    # |self ∩ other| / |other| for 8-step patterns
    dc_human: float = 0.0     # direct per-step match rate (%)
    oc_human: float = 0.0     # ±1-step buffered match rate (%)
    dc_rand: float = 0.0      # random baseline vs self (%)

    # Distributional comparison (derived from both charts' ChartMetrics)
    gap_hist_tvd: float = 0.0
    ratio_hist_tvd: float = 0.0
    density_corr: float = 0.0
    density_mae: float = 0.0
    silence_overlap_f1: float = 0.0
    dense_overlap_f1: float = 0.0
    gap_peak_iou: float = 0.0
    ioi_mean_ratio: float = 0.0
    ioi_std_ratio: float = 0.0
    streak_fraction_delta: float = 0.0
    bpm_ratio: float = 0.0

    # Multi-resolution comparison
    fps_comparisons: tuple[ResolutionComparison, ...] = ()


# ─────────────────────────── Chart class ──────────────────────────────

@dataclass(frozen=True, slots=True)
class Chart:
    """A `Track` plus optional audio payload, with analysis methods."""
    track: Track
    audio: bytes | None = None

    # ── Metrics ───────────────────────────────────────────────────────

    def calculate_metrics(self) -> ChartMetrics:
        onsets = self.track.onsets
        times_ms = np.asarray([o.time_ms for o in onsets], dtype=np.int64)

        # Type distribution
        counts = Counter(o.kind for o in onsets)
        c_don = counts.get(OnsetKind.DON, 0)
        c_ka = counts.get(OnsetKind.KA, 0)
        c_bdon = counts.get(OnsetKind.BIG_DON, 0)
        c_bka = counts.get(OnsetKind.BIG_KA, 0)
        c_dr = counts.get(OnsetKind.DRUMROLL, 0)
        c_sp = counts.get(OnsetKind.SPINNER, 0)
        total_hits = c_don + c_ka + c_bdon + c_bka
        don_ratio = (c_don + c_bdon) / total_hits if total_hits else 0.5

        # Duration
        duration_s = self.track.density.duration_s or (
            (float(times_ms[-1] - times_ms[0]) / 1000.0) if len(times_ms) >= 2 else 0.0
        )
        events_per_sec = len(onsets) / duration_s if duration_s > 0 else 0.0

        # IOI stats
        if len(times_ms) >= 2:
            gaps = np.diff(times_ms)
            gaps_pos = gaps[gaps > 0]
        else:
            gaps = np.zeros(0, dtype=np.int64)
            gaps_pos = gaps

        if len(gaps_pos):
            ioi_min = float(gaps_pos.min())
            ioi_max = float(gaps_pos.max())
            ioi_mean = float(gaps_pos.mean())
            ioi_median = float(np.median(gaps_pos))
            ioi_std = float(gaps_pos.std())
            ioi_p95 = float(np.percentile(gaps_pos, 95))
            ioi_p99 = float(np.percentile(gaps_pos, 99))
        else:
            ioi_min = ioi_max = ioi_mean = ioi_median = 0.0
            ioi_std = ioi_p95 = ioi_p99 = 0.0

        short_ioi = int(np.sum(gaps_pos < 20)) if len(gaps_pos) else 0
        short_pct = (short_ioi / len(gaps_pos) * 100.0) if len(gaps_pos) else 0.0
        long_gap = int(np.sum(gaps_pos > 2000)) if len(gaps_pos) else 0

        # IOI histogram (10-ms buckets, drop singletons)
        if len(gaps_pos):
            buckets = Counter((int(g) // 10) * 10 for g in gaps_pos.tolist())
            ioi_hist = {k: v for k, v in sorted(buckets.items()) if v >= 2}
        else:
            ioi_hist = {}

        # Density timeline
        density_timeline: tuple[int, ...] = ()
        silence_regions: list[tuple[float, float]] = []
        dense_regions: list[tuple[float, float]] = []
        d_mean = d_std = d_cv = 0.0
        d_peak = d_min = 0
        if len(times_ms):
            t_end = int(times_ms[-1]) + 1000
            n_secs = max(1, t_end // 1000)
            buckets_arr = np.zeros(n_secs, dtype=np.int64)
            for t in times_ms:
                idx = int(t) // 1000
                if 0 <= idx < n_secs:
                    buckets_arr[idx] += 1
            density_timeline = tuple(int(x) for x in buckets_arr)
            d_mean = float(buckets_arr.mean())
            d_std = float(buckets_arr.std())
            d_peak = int(buckets_arr.max())
            d_min = int(buckets_arr.min())
            d_cv = d_std / d_mean if d_mean > 0 else 0.0

            # Silence regions: contiguous runs of density==0 with length >= 2
            silence_regions = _contiguous_regions(buckets_arr == 0, min_length=2)
            # Dense regions: > 2× mean for >= 3s
            if d_mean > 0:
                dense_regions = _contiguous_regions(
                    buckets_arr > 2 * d_mean, min_length=3,
                )

        # Same-gap streaks
        streaks = _find_streaks(gaps.astype(np.float64), tolerance=0.05) if len(gaps) >= 2 else []
        if streaks:
            longest_streak = max(s[1] for s in streaks)
            mean_streak = float(np.mean([s[1] for s in streaks]))
            events_in_streaks = sum(s[1] + 1 for s in streaks)
            streak_frac = events_in_streaks / len(onsets) if onsets else 0.0
        else:
            longest_streak = 0
            mean_streak = 0.0
            streak_frac = 0.0

        # BPM from mode IOI
        bpm, dominant_ioi = _estimate_bpm_from_ioi(gaps_pos)

        # Gap-distribution + ratio-distribution shape (peak list + counts
        # + TVDs from uniform and delta-at-mode + raw histogram).
        (
            gap_peaks, gap_pc, gap_pf, gap_rd, gap_md, gap_hist_raw,
        ) = _compute_gap_distribution(gaps_pos)
        (
            ratio_peaks, ratio_pc, ratio_pf, ratio_rd, ratio_md, ratio_hist_raw,
        ) = _compute_ratio_distribution(gaps_pos)

        # Self pattern-space
        bin_self = _events_to_binary(times_ms, step_ms=TN_STEP_MS)
        over_self = _over_pspace(bin_self, scale=TN_SCALE)

        return ChartMetrics(
            total_events=len(onsets),
            duration_s=round(duration_s, 2),
            events_per_sec=round(events_per_sec, 3),
            count_don=c_don, count_ka=c_ka,
            count_big_don=c_bdon, count_big_ka=c_bka,
            count_drumroll=c_dr, count_spinner=c_sp,
            don_ratio=round(don_ratio, 4),
            ioi_min_ms=round(ioi_min, 1),
            ioi_max_ms=round(ioi_max, 1),
            ioi_mean_ms=round(ioi_mean, 1),
            ioi_median_ms=round(ioi_median, 1),
            ioi_std_ms=round(ioi_std, 1),
            ioi_p95_ms=round(ioi_p95, 1),
            ioi_p99_ms=round(ioi_p99, 1),
            short_ioi_count=short_ioi,
            short_ioi_pct=round(short_pct, 2),
            long_gap_count=long_gap,
            ioi_histogram_10ms=ioi_hist,
            density_mean=round(d_mean, 3),
            density_peak=d_peak,
            density_std=round(d_std, 3),
            density_min=d_min,
            density_cv=round(d_cv, 3),
            density_timeline=density_timeline,
            silence_regions=tuple(silence_regions),
            dense_regions=tuple(dense_regions),
            longest_streak=longest_streak,
            mean_streak_len=round(mean_streak, 3),
            streak_event_fraction=round(streak_frac, 4),
            estimated_bpm=bpm,
            dominant_ioi_ms=dominant_ioi,
            over_pspace_self=round(over_self, 3),
            gap_histogram_dense=gap_hist_raw,
            gap_peak_count=gap_pc,
            gap_peak_falloff=gap_pf,
            gap_random_distance=gap_rd,
            gap_metronome_distance=gap_md,
            gap_peaks=gap_peaks,
            gap_peak_mass_total=sum(int(c) for _, c in gap_peaks),
            ratio_histogram_dense=ratio_hist_raw,
            ratio_peak_count=ratio_pc,
            ratio_peak_falloff=ratio_pf,
            ratio_random_distance=ratio_rd,
            ratio_metronome_distance=ratio_md,
            ratio_peaks=ratio_peaks,
            ratio_peak_mass_total=sum(int(c) for _, c in ratio_peaks),
        )

    # ── Comparison ────────────────────────────────────────────────────

    def compare(self, other: "Chart", *, seed: int = TN_SEED) -> ChartComparison:
        """Compare self vs `other`. `other` is treated as the reference
        (e.g. ground truth) when the metric is directional."""
        self_ms = np.asarray([o.time_ms for o in self.track.onsets], dtype=np.float64)
        other_ms = np.asarray([o.time_ms for o in other.track.onsets], dtype=np.float64)

        gt = _gt_match_metrics(self_ms, other_ms)
        tn = _tn_pattern_metrics(self_ms, other_ms, rng_seed=seed)
        fps_cmp = _compute_resolution_comparisons(self_ms, other_ms)

        self_metrics = self.calculate_metrics()
        other_metrics = other.calculate_metrics()
        dist = _distributional_comparison(self_metrics, other_metrics)

        return ChartComparison(
            n_self=len(self_ms),
            n_other=len(other_ms),
            matched_rate=gt["matched_rate"],
            close_rate=gt["close_rate"],
            far_rate=gt["far_rate"],
            hallucination_rate=gt["hallucination_rate"],
            error_mean_ms=gt["error_mean_ms"],
            error_median_ms=gt["error_median_ms"],
            density_ratio=gt["density_ratio"],
            precision=gt["precision"],
            recall=gt["recall"],
            f1=gt["f1"],
            over_pspace_self=tn["over_pspace_self"],
            over_pspace_other=tn["over_pspace_other"],
            hi_pspace=tn["hi_pspace"],
            dc_human=tn["dc_human"],
            oc_human=tn["oc_human"],
            dc_rand=tn["dc_rand"],
            gap_hist_tvd=dist["gap_hist_tvd"],
            ratio_hist_tvd=dist["ratio_hist_tvd"],
            density_corr=dist["density_corr"],
            density_mae=dist["density_mae"],
            silence_overlap_f1=dist["silence_overlap_f1"],
            dense_overlap_f1=dist["dense_overlap_f1"],
            gap_peak_iou=dist["gap_peak_iou"],
            ioi_mean_ratio=dist["ioi_mean_ratio"],
            ioi_std_ratio=dist["ioi_std_ratio"],
            streak_fraction_delta=dist["streak_fraction_delta"],
            bpm_ratio=dist["bpm_ratio"],
            fps_comparisons=fps_cmp,
        )

    # ── Save / load ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Write the chart to disk. Dispatches by file extension:

          - ``.osu`` — canonical osu!taiko text file; audio is dropped
            (``.osu`` only references audio by filename).
          - ``.osz`` — osu! archive with the ``.osu`` inside plus the
            attached audio (if present).
          - anything else — the taiko2 bundle format (zip with
            ``track.json`` + ``audio.<ext>``) that round-trips exactly.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".osu":
            return self.save_osu(path)
        if suffix == ".osz":
            return self.save_osz(path)
        return self._save_bundle(path)

    def save_osu(self, path: Path) -> None:
        """Write this chart as a standard osu!taiko ``.osu`` text file.

        Note: drumrolls and spinners are emitted as plain circles because
        the sampler pipeline doesn't preserve their durations. All other
        onset kinds round-trip exactly.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = _track_to_osu_text(self.track)
        path.write_text(text, encoding="utf-8")

    def save_osz(self, path: Path) -> None:
        """Write this chart as an ``.osz`` archive containing one ``.osu``
        and the audio file referenced by it (if ``self.audio`` is set).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        osu_text = _track_to_osu_text(self.track)
        osu_name = _safe_osu_filename(self.track)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(osu_name, osu_text)
            if self.audio is not None:
                z.writestr(self.track.audio.filename, self.audio)

    def _save_bundle(self, path: Path) -> None:
        track_json = json.dumps(
            _track_to_dict(self.track), ensure_ascii=False, indent=2,
        )
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("track.json", track_json)
            if self.audio is not None:
                ext = self.track.audio.format or "bin"
                z.writestr(f"audio.{ext}", self.audio)

    @classmethod
    def load(cls, path: Path, *, index: int | None = None) -> "Chart":
        """Read a Chart from disk. Dispatches by file extension:

          - ``.osu`` — raw osu!taiko chart file; audio is not attached
            (the .osu only references audio by filename). `index` is
            ignored.
          - ``.osz`` — osu! archive containing one or more charts.
            `index` selects which track (required). Audio bytes are read
            from the archive if present.
          - anything else — a Chart bundle previously written by
            :meth:`save` (zip with ``track.json`` and optional audio).

        Raises ValueError / IndexError on malformed inputs or an
        out-of-range `index`.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".osu":
            return cls._load_osu(path)
        if suffix == ".osz":
            if index is None:
                raise ValueError(
                    f".osz archives contain multiple charts; pass index=... "
                    f"to select one ({path})"
                )
            return cls._load_osz(path, index)
        return cls._load_bundle(path)

    # ── Loader helpers ────────────────────────────────────────────────

    @classmethod
    def _load_osu(cls, path: Path) -> "Chart":
        from ..parsing.osu import parse_osu_text
        text = path.read_text(encoding="utf-8", errors="replace")
        track = parse_osu_text(text)
        if track is None:
            raise ValueError(
                f"{path} is not a valid osu!taiko chart (mode=1 with onsets)"
            )
        return cls(track=track, audio=None)

    @classmethod
    def _load_osz(cls, path: Path, index: int) -> "Chart":
        from ..parsing.osz import extract_audio_bytes, load_pack
        pack = load_pack(path)
        if pack is None:
            raise ValueError(f"{path} is not a readable osu! pack")
        if index < 0 or index >= len(pack.tracks):
            raise IndexError(
                f"index {index} out of range for {path} (has {len(pack.tracks)} tracks)"
            )
        track = pack.tracks[index]
        audio: bytes | None
        try:
            audio = extract_audio_bytes(path, track.audio.filename)
        except (FileNotFoundError, zipfile.BadZipFile, KeyError):
            audio = None
        return cls(track=track, audio=audio)

    @classmethod
    def _load_bundle(cls, path: Path) -> "Chart":
        with zipfile.ZipFile(path, "r") as z:
            track = _track_from_dict(json.loads(z.read("track.json").decode("utf-8")))
            audio: bytes | None = None
            # Accept any audio.* entry; prefer the one whose ext matches
            # track.audio.format, fall back to the first audio.* found.
            preferred = f"audio.{track.audio.format}"
            names = z.namelist()
            if preferred in names:
                audio = z.read(preferred)
            else:
                for n in names:
                    if n.startswith("audio.") and n != "audio.":
                        audio = z.read(n)
                        break
        return cls(track=track, audio=audio)


# ─────────────────────────── helpers ──────────────────────────────────

def _find_streaks(
    gaps: np.ndarray, tolerance: float = 0.05,
) -> list[tuple[int, int, float]]:
    """Port of `analyze_metronome_data.find_streaks` (5% tolerance)."""
    if len(gaps) < 2:
        return []
    streaks: list[tuple[int, int, float]] = []
    start = 0
    head = float(gaps[0])
    length = 1
    for i in range(1, len(gaps)):
        g = float(gaps[i])
        if head > 0 and abs(g - head) / head <= tolerance:
            length += 1
            continue
        if length >= 2:
            streaks.append((start, length, head))
        start, head, length = i, g, 1
    if length >= 2:
        streaks.append((start, length, head))
    return streaks


def _contiguous_regions(
    mask: np.ndarray, min_length: int,
) -> list[tuple[float, float]]:
    """Extract contiguous True runs of length ≥ min_length as (start_s, end_s)."""
    regions: list[tuple[float, float]] = []
    in_run = False
    run_start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            run_start = i
        elif not v and in_run:
            if i - run_start >= min_length:
                regions.append((float(run_start), float(i)))
            in_run = False
    if in_run and len(mask) - run_start >= min_length:
        regions.append((float(run_start), float(len(mask))))
    return regions


def _estimate_bpm_from_ioi(
    gaps_ms: np.ndarray,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 240.0,
    ioi_min_ms: float = 100.0,
    ioi_max_ms: float = 1500.0,
) -> tuple[float | None, float | None]:
    """Pick the most common IOI in 5-ms buckets, convert to BPM via
    `60000 / ioi`, then double/halve into the `[min_bpm, max_bpm]` range.

    Matches `_compute_run_stats`' BPM heuristic. Returns
    `(bpm, dominant_ioi_ms)` or `(None, None)` if the input is too short.
    """
    if len(gaps_ms) < 2:
        return None, None
    in_range = gaps_ms[(gaps_ms >= ioi_min_ms) & (gaps_ms <= ioi_max_ms)]
    if not len(in_range):
        return None, None
    buckets = Counter(int(round(float(g) / 5)) * 5 for g in in_range)
    dominant_ioi_ms = buckets.most_common(1)[0][0]
    if dominant_ioi_ms <= 0:
        return None, None
    bpm = 60000.0 / dominant_ioi_ms
    # Normalize into the requested range by doubling / halving.
    while bpm < min_bpm:
        bpm *= 2
    while bpm > max_bpm:
        bpm /= 2
    return round(float(bpm), 2), float(dominant_ioi_ms)


# ─────────────────────────── histogram shape ─────────────────────────

_GAP_HIST_CAP_MS = 2000
_GAP_HIST_BUCKET_MS = 10
_GAP_HIST_N_BUCKETS = _GAP_HIST_CAP_MS // _GAP_HIST_BUCKET_MS     # 200

# Ratio (gap[i] / gap[i-1]) in log2 space. [-3, +3] covers ratios in
# [0.125x, 8x]. 200 buckets at width 0.03 log2 ≈ 2.1% per bucket.
_RATIO_LOG2_MIN = -3.0
_RATIO_LOG2_MAX = +3.0
_RATIO_N_BUCKETS = 200
_RATIO_LOG2_WIDTH = (
    (_RATIO_LOG2_MAX - _RATIO_LOG2_MIN) / _RATIO_N_BUCKETS          # 0.03
)


def _histogram_shape(
    h: np.ndarray,
    *,
    bucket_center_fn,
    height_frac: float = 0.05,
    height_abs: float = 3.0,
    smooth_radius: int = 2,
    min_separation: int = 3,
    metronome_ref_bucket: int | None = None,
) -> "tuple[tuple[tuple[float, int], ...], int, float, float, float]":
    """Peak + distribution-distance analysis on a dense count array.

    Shared by `_compute_gap_distribution` and `_compute_ratio_distribution`
    — same peak-detection rules, same distance formulas, parameterized
    only on how a bucket index maps back to a physical value and which
    bucket "pure metronome" points at.

    ``metronome_ref_bucket``:
      - ``None`` (default, for gaps): use the observed mode → distance
        is ``1 - mode_share``. "Distance from all gaps being [whatever
        the dominant gap is]."
      - fixed int (for ratios): use the bucket at that index → distance
        is ``1 - h[ref]/total``. For ratios this is the ``1.0x`` bucket,
        so "pure metronome" means every ratio is exactly 1.0 regardless
        of what the chart's observed peak is.

    Returns ``(peaks, peak_count, peak_falloff, random_distance,
    metronome_distance)`` where ``peaks`` is a tuple of
    ``(bucket_center, count)`` entries sorted by count descending.
    """
    n = int(h.shape[0])
    total = float(h.sum())
    if total < 2 or n == 0:
        return (), 0, 0.0, 0.0, 0.0

    # Peak detection.
    height_threshold = max(float(h.max()) * height_frac, height_abs)
    candidates: list[int] = []
    for i in range(n):
        if h[i] < height_threshold:
            continue
        lo = max(0, i - smooth_radius)
        hi = min(n, i + smooth_radius + 1)
        window_max = float(h[lo:hi].max())
        if h[i] < window_max:
            continue
        # Plateau: only keep the leftmost bucket in a run of equal max.
        if i > 0 and h[i - 1] == h[i] and h[i - 1] >= window_max:
            continue
        candidates.append(i)

    kept_peaks: list[int] = []
    for i in sorted(candidates, key=lambda k: float(h[k]), reverse=True):
        if any(abs(i - j) < min_separation for j in kept_peaks):
            continue
        kept_peaks.append(i)

    peak_count = len(kept_peaks)

    # Peak detail: (center, count) sorted by count desc.
    peaks_detail = tuple(
        (float(bucket_center_fn(i)), int(h[i]))
        for i in sorted(kept_peaks, key=lambda k: float(h[k]), reverse=True)
    )

    # Falloff: 0 when < 2 peaks (no next peak to decay into).
    if peak_count < 2:
        peak_falloff = 0.0
    else:
        counts_desc = [float(h[i]) for i in kept_peaks]
        counts_desc.sort(reverse=True)
        ratios = [counts_desc[i + 1] / counts_desc[i]
                  for i in range(len(counts_desc) - 1)
                  if counts_desc[i] > 0]
        peak_falloff = float(np.mean(ratios)) if ratios else 0.0

    # TVD from uniform over the full bucket support.
    p = h / total
    q = 1.0 / n
    random_distance = 0.5 * float(np.sum(np.abs(p - q)))

    # TVD from a delta distribution. By default the delta sits at the
    # observed mode (gap convention → "distance from all gaps at the
    # dominant value"). For ratios we pass a fixed reference bucket so
    # "pure metronome" is anchored at the 1.0x bucket instead.
    if metronome_ref_bucket is None:
        metronome_distance = 1.0 - float(h.max()) / total
    else:
        ref = max(0, min(n - 1, int(metronome_ref_bucket)))
        metronome_distance = 1.0 - float(h[ref]) / total

    return (
        peaks_detail,
        peak_count,
        round(peak_falloff, 4),
        round(random_distance, 4),
        round(metronome_distance, 4),
    )


def _compute_gap_distribution(
    gaps_pos: np.ndarray,
) -> "tuple[tuple[tuple[float, int], ...], int, float, float, float, tuple[int, ...]]":
    """``(peaks, peak_count, falloff, random_distance, metronome_distance,
    histogram_dense)`` over a fixed ``[0, 2000) ms`` histogram at 10-ms
    resolution. Peaks are ``(bucket_center_ms, count)`` sorted by count
    desc. ``histogram_dense`` is the raw 200-bucket count array."""
    if len(gaps_pos) < 2:
        return (), 0, 0.0, 0.0, 0.0, ()
    kept = gaps_pos[(gaps_pos >= 0) & (gaps_pos < _GAP_HIST_CAP_MS)]
    if len(kept) < 2:
        return (), 0, 0.0, 0.0, 0.0, ()
    bucket_idx = (kept.astype(np.int64) // _GAP_HIST_BUCKET_MS)
    h = np.bincount(bucket_idx, minlength=_GAP_HIST_N_BUCKETS).astype(np.float64)
    shape = _histogram_shape(
        h,
        bucket_center_fn=lambda i: i * _GAP_HIST_BUCKET_MS + _GAP_HIST_BUCKET_MS / 2,
    )
    return (*shape, tuple(int(x) for x in h))


def _compute_ratio_distribution(
    gaps_pos: np.ndarray,
) -> "tuple[tuple[tuple[float, int], ...], int, float, float, float, tuple[int, ...]]":
    """Same analysis applied to the log2 ratio distribution
    ``ratio[i] = gaps_pos[i] / gaps_pos[i-1]``. Bucketing is log2 so
    doubling (2x) and halving (0.5x) are equidistant from 1x.

    Returns peaks as ``(bucket_center_ratio, count)`` in LINEAR ratio
    units (already exponentiated from log2), sorted by count desc.
    ``histogram_dense`` is the raw 200-bucket count array.
    """
    if len(gaps_pos) < 2:
        return (), 0, 0.0, 0.0, 0.0, ()
    # Pairwise ratios. Need 2 consecutive positive gaps → 3+ onsets.
    denom = gaps_pos[:-1]
    numer = gaps_pos[1:]
    valid = (denom > 0) & (numer > 0)
    if not valid.any():
        return (), 0, 0.0, 0.0, 0.0, ()
    ratios = (numer[valid] / denom[valid]).astype(np.float64)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < 2:
        return (), 0, 0.0, 0.0, 0.0, ()

    log2r = np.log2(ratios)
    in_range = (log2r >= _RATIO_LOG2_MIN) & (log2r < _RATIO_LOG2_MAX)
    log2r = log2r[in_range]
    if len(log2r) < 2:
        return (), 0, 0.0, 0.0, 0.0, ()
    bucket_idx = ((log2r - _RATIO_LOG2_MIN) / _RATIO_LOG2_WIDTH).astype(np.int64)
    bucket_idx = np.clip(bucket_idx, 0, _RATIO_N_BUCKETS - 1)
    h = np.bincount(bucket_idx, minlength=_RATIO_N_BUCKETS).astype(np.float64)

    def _center(i: int) -> float:
        log2_center = _RATIO_LOG2_MIN + (i + 0.5) * _RATIO_LOG2_WIDTH
        return float(2.0 ** log2_center)

    one_bucket = int((0.0 - _RATIO_LOG2_MIN) / _RATIO_LOG2_WIDTH)
    shape = _histogram_shape(
        h,
        bucket_center_fn=_center,
        metronome_ref_bucket=one_bucket,
        smooth_radius=5,
        min_separation=10,
    )
    return (*shape, tuple(int(x) for x in h))


def _events_to_binary(
    times_ms: np.ndarray, *, step_ms: int = TN_STEP_MS,
) -> np.ndarray:
    """TaikoNation-style binarization: one bit per `step_ms` slot."""
    if not len(times_ms):
        return np.zeros(0, dtype=np.int32)
    max_time = int(times_ms.max()) + step_ms
    n = max_time // step_ms + 1
    b = np.zeros(n, dtype=np.int32)
    for t in times_ms:
        idx = int(t) // step_ms
        if 0 <= idx < n:
            b[idx] = 1
    return b


def _over_pspace(binary: np.ndarray, *, scale: int = TN_SCALE) -> float:
    """Fraction of the 2^scale possible scale-bit patterns observed (%)."""
    if len(binary) < scale:
        return 0.0
    patterns: set[tuple[int, ...]] = set()
    for i in range(len(binary) - scale + 1):
        patterns.add(tuple(int(x) for x in binary[i:i + scale]))
    return len(patterns) / (2 ** scale) * 100.0


def _hi_pspace(
    bin_a: np.ndarray, bin_b: np.ndarray, *, scale: int = TN_SCALE,
) -> float:
    """|patterns(a) ∩ patterns(b)| / |patterns(b)| * 100. 0 if `b` empty."""
    def _patterns(arr: np.ndarray) -> set[tuple[int, ...]]:
        s: set[tuple[int, ...]] = set()
        for i in range(len(arr) - scale + 1):
            s.add(tuple(int(x) for x in arr[i:i + scale]))
        return s

    pa = _patterns(bin_a)
    pb = _patterns(bin_b)
    if not pb:
        return 0.0
    return len(pa & pb) / len(pb) * 100.0


def _dc_direct(a: np.ndarray, b: np.ndarray) -> float:
    """Per-step exact-match % starting from `b`'s first 1."""
    limit = min(len(a), len(b))
    if limit == 0:
        return 0.0
    start = 0
    for i in range(limit):
        if b[i] == 1:
            start = i
            break
    total = limit - start
    if total <= 0:
        return 0.0
    return float((a[start:limit] == b[start:limit]).sum() / total * 100.0)


def _oc_buffered(a: np.ndarray, b: np.ndarray, *, buffer: int = 1) -> float:
    """`a`-hit matched if any `b`-hit within ±buffer steps; zeros count as
    matched where `b` is zero. Starts from `b`'s first 1."""
    limit = min(len(a), len(b))
    if limit == 0:
        return 0.0
    start = 0
    for i in range(limit):
        if b[i] == 1:
            start = i
            break
    total = limit - start
    if total <= 0:
        return 0.0
    similarity = 0
    for i in range(start, limit):
        if a[i] == 1:
            matched = False
            for off in range(-buffer, buffer + 1):
                j = i + off
                if 0 <= j < limit and b[j] == 1:
                    matched = True
                    break
            if matched:
                similarity += 1
        elif b[i] == 0:
            similarity += 1
    return float(similarity / total * 100.0)


def _dc_random(binary: np.ndarray, seed: int) -> float:
    """Per-step match of `binary` against a seeded coin-flip stream."""
    if not len(binary):
        return 0.0
    rng = _stdlib_random.Random(seed)
    noise = np.fromiter(
        (rng.getrandbits(1) for _ in range(len(binary))),
        dtype=np.int32, count=len(binary),
    )
    return float((binary == noise).sum() / len(binary) * 100.0)


# ─────────────────────── distributional comparison ─────────────────────


def _hist_tvd(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Total Variation Distance between two count histograms."""
    if not a and not b:
        return 0.0
    n = max(len(a), len(b))
    ha = np.zeros(n, dtype=np.float64)
    hb = np.zeros(n, dtype=np.float64)
    if a:
        ha[: len(a)] = a
    if b:
        hb[: len(b)] = b
    sa, sb = ha.sum(), hb.sum()
    if sa < 1 or sb < 1:
        return 1.0 if (sa >= 1 or sb >= 1) else 0.0
    return round(0.5 * float(np.abs(ha / sa - hb / sb).sum()), 4)


def _density_corr_mae(
    a: tuple[int, ...], b: tuple[int, ...],
) -> tuple[float, float]:
    """Pearson r and MAE of two per-second density timelines."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 0.0
    aa = np.asarray(a[:n], dtype=np.float64)
    ba = np.asarray(b[:n], dtype=np.float64)
    mae = round(float(np.abs(aa - ba).mean()), 4)
    if aa.std() < 1e-12 and ba.std() < 1e-12:
        return 1.0, mae
    if aa.std() < 1e-12 or ba.std() < 1e-12:
        return 0.0, mae
    corr = round(float(np.corrcoef(aa, ba)[0, 1]), 4)
    return corr, mae


def _region_overlap_f1(
    pred: tuple[tuple[float, float], ...],
    gt: tuple[tuple[float, float], ...],
    tolerance_s: float = 1.0,
) -> float:
    """F1 of region overlap. A predicted region matches a GT region if
    their temporal overlap exceeds ``tolerance_s``."""
    if not gt and not pred:
        return 1.0
    if not gt or not pred:
        return 0.0
    matched_gt = 0
    matched_pred: set[int] = set()
    for gs, ge in gt:
        best_overlap = 0.0
        best_idx = -1
        for j, (ps, pe) in enumerate(pred):
            overlap = max(0.0, min(ge, pe) - max(gs, ps))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = j
        if best_overlap >= tolerance_s:
            matched_gt += 1
            matched_pred.add(best_idx)
    rec = matched_gt / len(gt)
    prec = len(matched_pred) / len(pred)
    if prec + rec < 1e-12:
        return 0.0
    return round(2.0 * prec * rec / (prec + rec), 4)


def _gap_peak_iou(
    a: tuple[tuple[float, int], ...],
    b: tuple[tuple[float, int], ...],
    tolerance_ms: float = 20.0,
) -> float:
    """Jaccard overlap of gap-peak positions (within ``tolerance_ms``)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    b_used: set[int] = set()
    matched = 0
    for ac, _ in a:
        for j, (bc, _) in enumerate(b):
            if j not in b_used and abs(ac - bc) <= tolerance_ms:
                matched += 1
                b_used.add(j)
                break
    union = len(a) + len(b) - matched
    return round(matched / max(union, 1), 4)


def _safe_ratio(a: float | None, b: float | None) -> float:
    """``a / b`` clamped to [0, 10], 0.0 when either is missing or zero."""
    if a is None or b is None or b == 0:
        return 0.0
    return round(min(a / b, 10.0), 4)


def _distributional_comparison(
    self_metrics: ChartMetrics, other_metrics: ChartMetrics,
) -> dict[str, float]:
    """Compute distributional / structural comparison fields from
    both charts' ``ChartMetrics``."""
    gap_tvd = _hist_tvd(
        self_metrics.gap_histogram_dense, other_metrics.gap_histogram_dense,
    )
    ratio_tvd = _hist_tvd(
        self_metrics.ratio_histogram_dense, other_metrics.ratio_histogram_dense,
    )
    d_corr, d_mae = _density_corr_mae(
        self_metrics.density_timeline, other_metrics.density_timeline,
    )
    silence_f1 = _region_overlap_f1(
        self_metrics.silence_regions, other_metrics.silence_regions,
    )
    dense_f1 = _region_overlap_f1(
        self_metrics.dense_regions, other_metrics.dense_regions,
    )
    peak_iou = _gap_peak_iou(
        self_metrics.gap_peaks, other_metrics.gap_peaks,
    )
    return dict(
        gap_hist_tvd=gap_tvd,
        ratio_hist_tvd=ratio_tvd,
        density_corr=d_corr,
        density_mae=d_mae,
        silence_overlap_f1=silence_f1,
        dense_overlap_f1=dense_f1,
        gap_peak_iou=peak_iou,
        ioi_mean_ratio=_safe_ratio(
            self_metrics.ioi_mean_ms, other_metrics.ioi_mean_ms,
        ),
        ioi_std_ratio=_safe_ratio(
            self_metrics.ioi_std_ms, other_metrics.ioi_std_ms,
        ),
        streak_fraction_delta=round(
            self_metrics.streak_event_fraction
            - other_metrics.streak_event_fraction, 4,
        ),
        bpm_ratio=_safe_ratio(
            self_metrics.estimated_bpm, other_metrics.estimated_bpm,
        ),
    )


# ─────────────────────── resolution comparison ─────────────────────────


def _resolution_comparison(
    self_ms: np.ndarray, other_ms: np.ndarray, fps: int,
) -> ResolutionComparison:
    """Compare two onset lists binned at a given FPS."""
    frame_ms = 1000.0 / fps
    max_time = 0.0
    if len(self_ms):
        max_time = max(max_time, float(self_ms.max()))
    if len(other_ms):
        max_time = max(max_time, float(other_ms.max()))
    n_frames = int(max_time / frame_ms) + 1 if max_time > 0 else 0
    if n_frames == 0:
        return ResolutionComparison(
            fps=fps, frame_ms=round(frame_ms, 3), n_frames=0,
            binary_precision=0.0, binary_recall=0.0, binary_f1=0.0,
            count_mae=0.0, count_corr=0.0, count_accuracy=1.0,
        )

    pred_counts = np.zeros(n_frames, dtype=np.int64)
    gt_counts = np.zeros(n_frames, dtype=np.int64)
    if len(self_ms):
        idx = np.clip((self_ms / frame_ms).astype(np.int64), 0, n_frames - 1)
        np.add.at(pred_counts, idx, 1)
    if len(other_ms):
        idx = np.clip((other_ms / frame_ms).astype(np.int64), 0, n_frames - 1)
        np.add.at(gt_counts, idx, 1)

    pred_bin = pred_counts > 0
    gt_bin = gt_counts > 0
    tp = int((pred_bin & gt_bin).sum())
    fp = int((pred_bin & ~gt_bin).sum())
    fn = int((~pred_bin & gt_bin).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    count_mae = float(np.abs(pred_counts - gt_counts).mean())
    count_accuracy = float((pred_counts == gt_counts).mean())
    if n_frames >= 2 and pred_counts.std() > 1e-12 and gt_counts.std() > 1e-12:
        count_corr = float(np.corrcoef(
            pred_counts.astype(np.float64),
            gt_counts.astype(np.float64),
        )[0, 1])
    else:
        count_corr = 1.0 if pred_counts.std() < 1e-12 and gt_counts.std() < 1e-12 else 0.0

    return ResolutionComparison(
        fps=fps, frame_ms=round(frame_ms, 3), n_frames=n_frames,
        binary_precision=round(prec, 4),
        binary_recall=round(rec, 4),
        binary_f1=round(f1, 4),
        count_mae=round(count_mae, 4),
        count_corr=round(count_corr, 4),
        count_accuracy=round(count_accuracy, 4),
    )


def _compute_resolution_comparisons(
    self_ms: np.ndarray,
    other_ms: np.ndarray,
    fps_values: tuple[int, ...] = RESOLUTION_FPS,
) -> tuple[ResolutionComparison, ...]:
    return tuple(_resolution_comparison(self_ms, other_ms, fps) for fps in fps_values)


def _gt_match_metrics(
    self_ms: np.ndarray, other_ms: np.ndarray,
) -> dict[str, float]:
    """Port of `run_comparison.compute_gt_metrics`. Returns a flat dict."""
    return gt_match_metrics(self_ms, other_ms)


def gt_match_metrics(
    self_ms: np.ndarray,
    other_ms: np.ndarray,
    tolerances_ms: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 100.0),
) -> dict[str, float]:
    """Onset-matching metrics between two onset lists (in milliseconds).

    ``self_ms`` is the predicted (or "self") onset times; ``other_ms``
    is the reference (ground truth). Both are 1-D float arrays.

    ``tolerances_ms`` controls the threshold sweep. For each tolerance
    ``t``, the result contains ``matched_rate_at_tol_{t}`` (fraction of
    GT onsets with a predicted onset within ``t`` ms) and
    ``halluc_rate_at_tol_{t}`` (fraction of predicted onsets with no GT
    onset within ``t`` ms).

    The canonical keys ``matched_rate`` (tol=25), ``close_rate``
    (tol=50), ``far_rate`` (>100), ``hallucination_rate`` (>100) are
    always present for backward compatibility.
    """
    n_self = len(self_ms)
    n_other = len(other_ms)
    empty: dict[str, float] = dict(
        matched_rate=0.0, close_rate=0.0, far_rate=0.0,
        hallucination_rate=0.0, error_mean_ms=0.0, error_median_ms=0.0,
        density_ratio=0.0,
        precision=0.0, recall=0.0, f1=0.0,
    )
    for t in tolerances_ms:
        t_key = _tol_key(t)
        empty[f"matched_rate_at_tol_{t_key}"] = 0.0
        empty[f"halluc_rate_at_tol_{t_key}"] = 0.0
        empty[f"precision_at_tol_{t_key}"] = 0.0
        empty[f"recall_at_tol_{t_key}"] = 0.0
        empty[f"f1_at_tol_{t_key}"] = 0.0
    if n_self == 0 or n_other == 0:
        return empty

    ps = np.sort(self_ms)
    gs = np.sort(other_ms)

    def _closest(arr: np.ndarray, v: float) -> float:
        i = int(np.searchsorted(arr, v))
        best = float("inf")
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(arr):
                best = min(best, abs(float(arr[j]) - float(v)))
        return best

    gt_err = np.array([_closest(ps, g) for g in gs])
    pe_err = np.array([_closest(gs, p) for p in ps])

    ps_density = n_self / max((ps[-1] - ps[0]) / 1000.0, 0.1) if n_self > 1 else 0.0
    gs_density = n_other / max((gs[-1] - gs[0]) / 1000.0, 0.1) if n_other > 1 else 0.0

    out = dict(
        matched_rate=float((gt_err <= 25).mean()),
        close_rate=float((gt_err <= 50).mean()),
        far_rate=float((gt_err > 100).mean()),
        hallucination_rate=float((pe_err > 100).mean()),
        error_mean_ms=float(gt_err.mean()),
        error_median_ms=float(np.median(gt_err)),
        density_ratio=ps_density / max(gs_density, 0.01),
    )
    # Canonical precision/recall/F1 at tol=25ms (same as matched_rate).
    recall_25 = out["matched_rate"]
    precision_25 = float((pe_err <= 25).mean()) if n_self > 0 else 0.0
    f1_25 = (2 * precision_25 * recall_25 / (precision_25 + recall_25)
             if (precision_25 + recall_25) > 0 else 0.0)
    out["precision"] = precision_25
    out["recall"] = recall_25
    out["f1"] = f1_25

    for t in tolerances_ms:
        t_key = _tol_key(t)
        rec_t = float((gt_err <= t).mean())
        prec_t = float((pe_err <= t).mean()) if n_self > 0 else 0.0
        f1_t = (2 * prec_t * rec_t / (prec_t + rec_t)
                if (prec_t + rec_t) > 0 else 0.0)
        out[f"matched_rate_at_tol_{t_key}"] = rec_t
        out[f"halluc_rate_at_tol_{t_key}"] = float((pe_err > t).mean())
        out[f"precision_at_tol_{t_key}"] = prec_t
        out[f"recall_at_tol_{t_key}"] = rec_t
        out[f"f1_at_tol_{t_key}"] = f1_t
    return out


def _tol_key(tol_ms: float) -> str:
    """Stable string key for a tolerance value (no trailing zeros)."""
    if tol_ms == int(tol_ms):
        return str(int(tol_ms))
    return f"{tol_ms:.1f}"


def _tn_pattern_metrics(
    self_ms: np.ndarray, other_ms: np.ndarray, *, rng_seed: int = TN_SEED,
) -> dict[str, float]:
    """Port of `run_comparison.compute_tn_metrics`. Returns a flat dict."""
    bs = _events_to_binary(self_ms)
    bo = _events_to_binary(other_ms)
    if len(bs) < TN_SCALE * 2 or len(bo) < TN_SCALE * 2:
        return dict(
            over_pspace_self=0.0, over_pspace_other=0.0, hi_pspace=0.0,
            dc_human=0.0, oc_human=0.0, dc_rand=0.0,
        )
    ml = max(len(bs), len(bo))
    ps = np.zeros(ml, dtype=np.int32)
    po = np.zeros(ml, dtype=np.int32)
    ps[:len(bs)] = bs
    po[:len(bo)] = bo
    return dict(
        over_pspace_self=_over_pspace(ps, scale=TN_SCALE),
        over_pspace_other=_over_pspace(po, scale=TN_SCALE),
        hi_pspace=_hi_pspace(ps, po, scale=TN_SCALE),
        dc_human=_dc_direct(ps, po),
        oc_human=_oc_buffered(ps, po, buffer=1),
        dc_rand=_dc_random(ps, rng_seed),
    )


# ─────────────────────────── Track ↔ dict ─────────────────────────────

def _track_to_dict(t: Track) -> dict[str, Any]:
    return {
        "beatmap_id": t.beatmap_id,
        "beatmapset_id": t.beatmapset_id,
        "artist": t.artist,
        "title": t.title,
        "difficulty": {
            "version": t.difficulty.version,
            "overall_difficulty": t.difficulty.overall_difficulty,
            "star_rating": t.difficulty.star_rating,
        },
        "audio": {
            "filename": t.audio.filename,
            "format": t.audio.format,
        },
        "density": {
            "mean": t.density.mean,
            "peak": t.density.peak,
            "std": t.density.std,
            "duration_s": t.density.duration_s,
            "total_events": t.density.total_events,
        },
        "onsets": [[o.time_ms, o.kind.value] for o in t.onsets],
    }


# ─────────────────────────── auto difficulty ─────────────────────────

# Linear fits from taiko2_v1 corpus (R^2=0.91 for star, 0.84 for OD).
_STAR_SLOPE = 0.7238
_STAR_INTERCEPT = 0.1952
_OD_SLOPE = 0.5307
_OD_INTERCEPT = 2.4992

_DIFF_BRACKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 2.0, "Kantan"),
    (2.0, 3.5, "Futsuu"),
    (3.5, 5.0, "Muzukashii"),
    (5.0, 6.5, "Oni"),
    (6.5, 99.0, "Inner Oni"),
)


def estimate_difficulty(density_mean: float) -> Difficulty:
    """Estimate star rating, OD, and version name from density."""
    star = max(0.0, _STAR_SLOPE * density_mean + _STAR_INTERCEPT)
    od = max(1.0, min(10.0, _OD_SLOPE * density_mean + _OD_INTERCEPT))
    version = "Inner Oni"
    for lo, hi, name in _DIFF_BRACKETS:
        if lo <= star < hi:
            version = name
            break
    return Difficulty(
        version=version,
        overall_difficulty=round(od, 1),
        star_rating=round(star, 2),
    )


# ─────────────────────────── Track ↔ .osu text ────────────────────────

# Hit-object encoding tables. These are the INVERSE of
# `parsing.osu._classify_hit_object`'s decode so that parse(emit(x)) == x
# for every onset kind the taiko pipeline preserves.
_CIRCLE_SOUND_BY_KIND: dict[OnsetKind, int] = {
    OnsetKind.DON:      0,   # no whistle, no clap, no finish
    OnsetKind.KA:       2,   # whistle
    OnsetKind.BIG_DON:  4,   # finish
    OnsetKind.BIG_KA:   6,   # finish + whistle
}


def _track_to_osu_text(track: Track) -> str:
    """Serialize a `Track` as an osu! v14 ``.osu`` file string.

    Caveat: drumrolls (type=2) and spinners (type=8) carry durations and
    extra parameters the taiko2 pipeline doesn't store, so they're emitted
    here as plain circles (don). Round-trip fidelity is preserved for the
    four circle kinds (don / ka / big_don / big_ka), which together are
    ~99.7% of events in the taiko2_v1 dataset.
    """
    diff = track.difficulty
    lines: list[str] = [
        "osu file format v14",
        "",
        "[General]",
        f"AudioFilename: {track.audio.filename}",
        "AudioLeadIn: 0",
        "Mode: 1",
        "",
        "[Metadata]",
        f"Title: {track.title}",
        f"TitleUnicode: {track.title}",
        f"Artist: {track.artist}",
        f"ArtistUnicode: {track.artist}",
        "Creator: taiko2",
        f"Version: {diff.version}",
        f"BeatmapID: {track.beatmap_id}",
        f"BeatmapSetID: {track.beatmapset_id}",
        "",
        "[Difficulty]",
        "HPDrainRate: 5",
        "CircleSize: 5",
        f"OverallDifficulty: {diff.overall_difficulty}",
        "ApproachRate: 5",
        "SliderMultiplier: 1.4",
        "SliderTickRate: 1",
        "",
        "[TimingPoints]",
        "",
        "[HitObjects]",
    ]
    for onset in track.onsets:
        # All kinds go out as circles (type=1). Drumrolls/spinners would
        # need durations and slider/spinner parameters we never captured.
        sound = _CIRCLE_SOUND_BY_KIND.get(onset.kind, 0)
        lines.append(f"256,192,{onset.time_ms},1,{sound},0:0:0:0:")
    lines.append("")
    return "\n".join(lines)


def _safe_osu_filename(track: Track) -> str:
    """Canonical .osu filename inside an .osz archive."""
    raw = f"{track.artist} - {track.title} (taiko2) [{track.difficulty.version}].osu"
    for ch in '<>:"/\\|?*\n\r':
        raw = raw.replace(ch, "_")
    raw = raw.strip(". ")
    return raw or "track.osu"


def _track_from_dict(d: dict[str, Any]) -> Track:
    return Track(
        beatmap_id=d["beatmap_id"],
        beatmapset_id=d["beatmapset_id"],
        artist=d["artist"],
        title=d["title"],
        difficulty=Difficulty(
            version=d["difficulty"]["version"],
            overall_difficulty=d["difficulty"]["overall_difficulty"],
            star_rating=d["difficulty"]["star_rating"],
        ),
        audio=AudioRef(
            filename=d["audio"]["filename"],
            format=d["audio"]["format"],
        ),
        density=Density(
            mean=d["density"]["mean"],
            peak=d["density"]["peak"],
            std=d["density"]["std"],
            duration_s=d["density"]["duration_s"],
            total_events=d["density"]["total_events"],
        ),
        onsets=tuple(
            Onset(time_ms=int(t), kind=OnsetKind(k)) for t, k in d["onsets"]
        ),
    )
