"""Training script for chart quality evaluator v2 (exp 66-2).

Bidirectional corruption: both random AND metronomic corruption.
Fixes the regularity bias from 66-1.

Usage:
    python classifier_train_v2.py taiko_v2 --run-name eval_experiment_66_2
"""
import os
import json
import random
import argparse
import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from classifier_model import ChartQualityEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── constants ──
WINDOW_FRAMES = 2000
WINDOW_STRIDE = 4
MAX_EVENTS = 256
BIN_MS = 5.0

# corruption levels
# 0=CLEAN, 1-4=random, 5-8=metro
LEVEL_NAMES = [
    "CLEAN",
    "LIGHT_RAND", "MED_RAND", "HIGH_RAND", "PURE_RAND",
    "LIGHT_METRO", "MED_METRO", "HIGH_METRO", "PURE_METRO",
]

LEVEL_SEVERITY = {
    0: 0,                           # CLEAN
    1: 1, 2: 2, 3: 3, 4: 4,        # random
    5: 1, 6: 2, 7: 3, 8: 4,        # metro
}

LEVEL_TYPE = {
    0: "clean",
    1: "rand", 2: "rand", 3: "rand", 4: "rand",
    5: "metro", 6: "metro", 7: "metro", 8: "metro",
}


# ──────────────────────────────────────────────
#  Global gap distribution (lazy-loaded)
# ──────────────────────────────────────────────

_gap_dist_cache = {"values": None, "probs": None}


def _load_gap_distribution(ds_dir):
    if _gap_dist_cache["values"] is not None:
        return _gap_dist_cache["values"], _gap_dist_cache["probs"]

    cache_path = os.path.join(ds_dir, "gap_distribution.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        _gap_dist_cache["values"] = data["values"]
        _gap_dist_cache["probs"] = data["probs"]
        print(f"  Loaded cached gap distribution: {len(data['values'])} unique gaps")
        return _gap_dist_cache["values"], _gap_dist_cache["probs"]

    print("  Computing global gap distribution...")
    manifest_path = os.path.join(ds_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    evt_dir = os.path.join(ds_dir, "events")
    all_gaps = []
    for c in manifest["charts"]:
        events = np.load(os.path.join(evt_dir, c["event_file"]))
        if len(events) >= 2:
            gaps = np.diff(events)
            all_gaps.extend(gaps[gaps > 0].tolist())
    all_gaps = np.array(all_gaps, dtype=np.int64)
    unique, counts = np.unique(all_gaps, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    np.savez(cache_path, values=unique, probs=probs)
    print(f"  Computed gap distribution: {len(unique)} unique gaps from {len(all_gaps)} total")
    _gap_dist_cache["values"] = unique
    _gap_dist_cache["probs"] = probs
    return unique, probs


# ──────────────────────────────────────────────
#  Post-corruption cleanup
# ──────────────────────────────────────────────

def _cleanup_events(events, mel_frames):
    events = np.sort(events)
    events = events[events >= 0]
    events = np.clip(events, 0, mel_frames - 1)
    if len(events) <= 1:
        return events
    keep = [events[0]]
    for e in events[1:]:
        if e - keep[-1] > 2:
            keep.append(e)
    return np.array(keep, dtype=np.int64)


# ──────────────────────────────────────────────
#  Random corruption (from 66-1)
# ──────────────────────────────────────────────

def corrupt_random(events, level, mel_frames, gap_values, gap_probs, rng):
    """Apply random corruption. level: 1=LIGHT, 2=MED, 3=HIGH, 4=PURE."""
    events = events.copy().astype(np.int64)
    n = len(events)

    if level == 4:
        # PURE_RAND: fully random gaps from global distribution
        if n == 0:
            return events
        gaps = rng.choice(gap_values, size=n, p=gap_probs)
        new_events = np.cumsum(gaps)
        if len(events) > 0:
            new_events = new_events - new_events[0] + events[0]
        return _cleanup_events(new_events, mel_frames)

    if level == 1:
        jitter_bins, all_jitter_bins = 2, 2
        insert_center_p, delete_p, insert_offset_p = 0.01, 0.01, 0.01
    elif level == 2:
        jitter_bins, all_jitter_bins = 6, 6
        insert_center_p, delete_p, insert_offset_p = 0.05, 0.05, 0.05
    else:  # level 3
        jitter_bins, all_jitter_bins = 20, 50
        insert_center_p, delete_p, insert_offset_p = 0.25, 0.15, 0.10

    # 1. deletions
    if delete_p > 0 and n > 1:
        keep_mask = rng.random(n) >= delete_p
        if keep_mask.sum() == 0:
            keep_mask[0] = True
        events = events[keep_mask]
        n = len(events)

    # 2. insertions (center of gap)
    if insert_center_p > 0 and n >= 2:
        new_events = []
        for i in range(n - 1):
            if rng.random() < insert_center_p:
                new_events.append((events[i] + events[i + 1]) // 2)
        if new_events:
            events = np.concatenate([events, new_events])
            n = len(events)

    # 3. insertions (offset)
    if insert_offset_p > 0 and n >= 1:
        new_events = []
        for i in range(n):
            if rng.random() < insert_offset_p:
                offset = rng.choice(gap_values, p=gap_probs)
                new_events.append(events[i] + offset)
        if new_events:
            events = np.concatenate([events, new_events])
            n = len(events)

    # 4. per-event jitter
    if jitter_bins > 0 and n > 0:
        events = events + rng.integers(-jitter_bins, jitter_bins + 1, size=n)

    # 5. all-event jitter
    if all_jitter_bins > 0 and n > 0:
        events = events + rng.integers(-all_jitter_bins, all_jitter_bins + 1)

    return _cleanup_events(events, mel_frames)


# ──────────────────────────────────────────────
#  Metronomic corruption (NEW)
# ──────────────────────────────────────────────

def _local_median_gap(gaps, idx, window=16):
    """Compute median gap in a window around idx."""
    start = max(0, idx - window // 2)
    end = min(len(gaps), idx + window // 2)
    return np.median(gaps[start:end])


def _snap_to_grid(gap, local_med):
    """Snap a gap to the nearest grid value based on local median."""
    grid = local_med * np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    return grid[np.argmin(np.abs(gap - grid))]


ALLOWED_RATIOS = np.array([0.25, 0.5, 1.0, 2.0, 4.0])


def _snap_ratio(ratio):
    """Snap a ratio to the nearest allowed value."""
    return ALLOWED_RATIOS[np.argmin(np.abs(ratio - ALLOWED_RATIOS))]


def corrupt_metro(events, level, mel_frames, rng):
    """Apply metronomic corruption. level: 1=LIGHT, 2=MED, 3=HIGH, 4=PURE."""
    events = events.copy().astype(np.int64)
    n = len(events)
    if n < 3:
        return events

    if level == 4:
        return _corrupt_pure_metro(events, mel_frames, rng)

    gaps = np.diff(events).astype(np.float64)
    gaps = np.maximum(gaps, 1.0)

    if level == 1:
        grid_snap_pct = 0.10
        const_fill_pct = 0.05
        pattern_loop_pct = 0.0
        density_flatten_pct = 0.0
        ratio_snap_pct = 0.10
        ratio_purge = False
    elif level == 2:
        grid_snap_pct = 0.30
        const_fill_pct = 0.15
        pattern_loop_pct = 0.10
        density_flatten_pct = 0.30
        ratio_snap_pct = 0.0
        ratio_purge = False
    else:  # level 3
        grid_snap_pct = 0.60
        const_fill_pct = 0.40
        pattern_loop_pct = 0.30
        density_flatten_pct = 0.60
        ratio_snap_pct = 0.0
        ratio_purge = True

    # ── 1. Grid snap ──
    if grid_snap_pct > 0:
        for i in range(len(gaps)):
            if rng.random() < grid_snap_pct:
                local_med = _local_median_gap(gaps, i)
                if local_med > 0:
                    gaps[i] = _snap_to_grid(gaps[i], local_med)

    # ── 2. Constant-gap fill (per 5-second segment) ──
    if const_fill_pct > 0:
        seg_bins = int(5000 / BIN_MS)  # 5 seconds in bins
        total_bins = events[-1] - events[0]
        n_segs = max(1, int(total_bins / seg_bins))
        for seg_idx in range(n_segs):
            if rng.random() < const_fill_pct:
                seg_start_bin = events[0] + seg_idx * seg_bins
                seg_end_bin = seg_start_bin + seg_bins
                # find events in this segment
                in_seg = (events >= seg_start_bin) & (events < seg_end_bin)
                n_in_seg = in_seg.sum()
                if n_in_seg >= 2:
                    # replace with evenly spaced
                    even_events = np.linspace(seg_start_bin, seg_end_bin, n_in_seg,
                                              endpoint=False, dtype=np.int64)
                    events[in_seg] = even_events

        # recompute gaps after segment fill
        gaps = np.diff(events).astype(np.float64)
        gaps = np.maximum(gaps, 1.0)

    # ── 3. Pattern loop ──
    if pattern_loop_pct > 0 and len(gaps) > 16:
        total_gaps = len(gaps)
        n_to_loop = int(total_gaps * pattern_loop_pct)
        # pick a random start for the looped region
        loop_start = rng.integers(0, max(1, total_gaps - n_to_loop))
        # pick pattern length
        pattern_len = rng.integers(2, min(9, n_to_loop + 1))
        pattern = gaps[loop_start:loop_start + pattern_len].copy()
        # tile the pattern over the region
        for i in range(n_to_loop):
            gaps[loop_start + i] = pattern[i % pattern_len]

    # ── 4. Density flatten ──
    if density_flatten_pct > 0:
        seg_bins = int(5000 / BIN_MS)
        total_bins = events[-1] - events[0]
        n_segs = max(1, int(total_bins / seg_bins))
        # compute per-segment density
        seg_counts = []
        for seg_idx in range(n_segs):
            seg_start_bin = events[0] + seg_idx * seg_bins
            seg_end_bin = seg_start_bin + seg_bins
            seg_counts.append(((events >= seg_start_bin) & (events < seg_end_bin)).sum())
        global_mean = np.mean(seg_counts)

        # rebuild events from gaps, then adjust density per segment
        events_rebuilt = np.concatenate([[events[0]], events[0] + np.cumsum(gaps)]).astype(np.int64)
        for seg_idx in range(n_segs):
            seg_start_bin = events_rebuilt[0] + seg_idx * seg_bins
            seg_end_bin = seg_start_bin + seg_bins
            in_seg = np.where((events_rebuilt >= seg_start_bin) & (events_rebuilt < seg_end_bin))[0]
            current = len(in_seg)
            target = int(current + density_flatten_pct * (global_mean - current))
            if current > target and current > 1:
                # delete random events
                n_delete = current - target
                to_delete = rng.choice(in_seg, size=min(n_delete, len(in_seg) - 1), replace=False)
                events_rebuilt = np.delete(events_rebuilt, to_delete)
            elif target > current and current > 0:
                # insert evenly spaced
                n_insert = target - current
                insert_pos = np.linspace(seg_start_bin, seg_end_bin, n_insert + 2,
                                         dtype=np.int64)[1:-1]
                events_rebuilt = np.concatenate([events_rebuilt, insert_pos])
        events = events_rebuilt
        # don't recompute gaps, we'll use events directly

    # ── 5. Ratio snap/purge ──
    if ratio_snap_pct > 0 or ratio_purge:
        # rebuild events from current gaps
        if density_flatten_pct == 0:
            events = np.concatenate([[events[0]], events[0] + np.cumsum(gaps)]).astype(np.int64)
        gaps_new = np.diff(events).astype(np.float64)
        gaps_new = np.maximum(gaps_new, 1.0)

        for i in range(1, len(gaps_new)):
            ratio = gaps_new[i] / gaps_new[i - 1]
            is_unusual = not any(abs(ratio - r) / r < 0.05 for r in ALLOWED_RATIOS)
            if ratio_purge or (is_unusual and rng.random() < ratio_snap_pct):
                snapped = _snap_ratio(ratio)
                gaps_new[i] = gaps_new[i - 1] * snapped

        events = np.concatenate([[events[0]], events[0] + np.cumsum(gaps_new)]).astype(np.int64)

    return _cleanup_events(events, mel_frames)


def _corrupt_pure_metro(events, mel_frames, rng):
    """PURE_METRO: maximum regularity. Three sub-types chosen randomly."""
    n = len(events)
    if n < 2:
        return events

    gaps = np.diff(events).astype(np.float64)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return events

    subtype = rng.integers(0, 3)

    if subtype == 0:
        # constant gap
        median_gap = np.median(gaps)
        new_events = events[0] + (np.arange(n) * median_gap).astype(np.int64)

    elif subtype == 1:
        # alternating top-2 gaps
        # cluster gaps (5% tolerance)
        gap_clusters = {}
        for g in gaps:
            matched = False
            for center in gap_clusters:
                if abs(g - center) / max(center, 1) <= 0.05:
                    gap_clusters[center] += 1
                    matched = True
                    break
            if not matched:
                gap_clusters[g] = 1
        sorted_gaps = sorted(gap_clusters.keys(), key=lambda g: gap_clusters[g], reverse=True)
        g1 = sorted_gaps[0]
        g2 = sorted_gaps[1] if len(sorted_gaps) > 1 else g1
        alt_gaps = np.array([g1 if i % 2 == 0 else g2 for i in range(n - 1)])
        new_events = np.concatenate([[events[0]], events[0] + np.cumsum(alt_gaps)]).astype(np.int64)

    else:
        # quantized to 1/4 beat grid
        median_gap_ms = np.median(gaps) * BIN_MS
        bpm = 60000.0 / max(median_gap_ms, 50.0)
        beat_bins = 60000.0 / bpm / BIN_MS
        quarter_beat = beat_bins / 4.0
        if quarter_beat < 1:
            quarter_beat = 1.0
        # snap all events
        new_events = np.round(events / quarter_beat) * quarter_beat
        new_events = np.unique(new_events.astype(np.int64))

    return _cleanup_events(new_events, mel_frames)


# ──────────────────────────────────────────────
#  Unified corruption dispatcher
# ──────────────────────────────────────────────

def corrupt_events(events, level, mel_frames, gap_values, gap_probs, rng):
    """Apply corruption at the given level (0-8). Returns corrupted event array."""
    if level == 0:
        return events.copy()
    elif level <= 4:
        return corrupt_random(events, level, mel_frames, gap_values, gap_probs, rng)
    else:
        return corrupt_metro(events, level - 4, mel_frames, rng)


# ──────────────────────────────────────────────
#  Audio augmentation (rating pairs only)
# ──────────────────────────────────────────────

def augment_mel(mel, rng):
    mel = mel.copy()
    mel = mel + rng.uniform(-0.3, 0.3)
    n_freq_mask = rng.integers(2, 5)
    for _ in range(n_freq_mask):
        mel[rng.integers(0, mel.shape[0]), :] = 0.0
    n_time_mask = rng.integers(2, 5)
    T = mel.shape[1]
    for _ in range(n_time_mask):
        width = rng.integers(50, 201)
        start = rng.integers(0, max(1, T - width))
        mel[:, start:start + width] = 0.0
    mel = mel + rng.standard_normal(mel.shape).astype(np.float32) * rng.uniform(0.0, 0.3)
    return mel


# ──────────────────────────────────────────────
#  Pair sampling
# ──────────────────────────────────────────────

# pair type definitions
PAIR_TYPES = {
    "clean_vs_rand": {
        "desc": "CLEAN vs random corruption",
        "sample": lambda rng: (0, rng.integers(1, 5)),
        "is_tie": False,
    },
    "clean_vs_metro": {
        "desc": "CLEAN vs metro corruption",
        "sample": lambda rng: (0, rng.integers(5, 9)),
        "is_tie": False,
    },
    "within_rand": {
        "desc": "Within random (less vs more corrupted)",
        "sample": lambda rng: tuple(sorted(rng.choice([1, 2, 3, 4], size=2, replace=False))),
        "is_tie": False,
    },
    "within_metro": {
        "desc": "Within metro (less vs more corrupted)",
        "sample": lambda rng: tuple(sorted(rng.choice([5, 6, 7, 8], size=2, replace=False))),
        "is_tie": False,
    },
    "cross_tie": {
        "desc": "Cross-type tie (same severity)",
        "sample": lambda rng: (rng.integers(1, 5), rng.integers(1, 5) + 4),  # matched severity
        "is_tie": True,
    },
}


def _sample_pair_type(rng, corruption_only=False):
    """Sample a pair type according to batch composition."""
    if corruption_only:
        weights = [0.25, 0.25, 0.125, 0.125, 0.25]
    else:
        weights = [0.25, 0.25, 0.10, 0.10, 0.15]
    # normalize (rating pairs handled separately)
    w = np.array(weights)
    w = w / w.sum()
    types = list(PAIR_TYPES.keys())
    return types[rng.choice(len(types), p=w)]


def _fix_cross_tie_severity(level_a, level_b):
    """Ensure cross-type ties have matching severity."""
    sev_a = LEVEL_SEVERITY[level_a]
    sev_b = LEVEL_SEVERITY[level_b]
    if sev_a != sev_b:
        # force level_b to match level_a's severity
        if LEVEL_TYPE[level_b] == "metro":
            level_b = sev_a + 4  # metro levels are 5-8
        else:
            level_b = sev_a      # rand levels are 1-4
    return level_a, level_b


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class PairDatasetV2(Dataset):
    """Generates bidirectional corruption pairs + rating pairs."""

    def __init__(self, manifest, ds_dir, chart_indices, mode="corruption",
                 rating_ratio=0.15, augment_audio=True):
        self.ds_dir = ds_dir
        self.mel_dir = os.path.join(ds_dir, "mels")
        self.evt_dir = os.path.join(ds_dir, "events")
        self.mode = mode
        self.rating_ratio = rating_ratio
        self.augment_audio = augment_audio

        self.charts = [manifest["charts"][i] for i in chart_indices]

        self.events = []
        for chart in self.charts:
            evt = np.load(os.path.join(self.evt_dir, chart["event_file"]))
            self.events.append(evt)

        self.rated_charts = []
        for i, chart in enumerate(self.charts):
            if "rating" in chart and "star_rating" in chart:
                self.rated_charts.append((i, chart["rating"], chart["star_rating"]))

        self.gap_values, self.gap_probs = _load_gap_distribution(ds_dir)
        self._mel_cache = {}

    def _get_mel(self, mel_file):
        if mel_file not in self._mel_cache:
            self._mel_cache[mel_file] = np.load(
                os.path.join(self.mel_dir, mel_file), mmap_mode="r"
            )
        return self._mel_cache[mel_file]

    def _extract_window(self, chart_idx, events, rng, do_augment=False):
        chart = self.charts[chart_idx]
        mel = self._get_mel(chart["mel_file"])
        total_frames = mel.shape[1]

        if total_frames <= WINDOW_FRAMES:
            start = 0
        else:
            start = rng.integers(0, total_frames - WINDOW_FRAMES)
        end = start + WINDOW_FRAMES

        mel_window = mel[:, start:min(total_frames, end)].astype(np.float32)
        if mel_window.shape[1] < WINDOW_FRAMES:
            pad = WINDOW_FRAMES - mel_window.shape[1]
            mel_window = np.pad(mel_window, ((0, 0), (0, pad)), mode="constant")

        if do_augment:
            mel_window = augment_mel(mel_window, rng)

        mask = (events >= start) & (events < end)
        evt_window = events[mask].astype(np.int64) - start

        n_evt = min(len(evt_window), MAX_EVENTS)
        event_arr = np.zeros(MAX_EVENTS, dtype=np.int64)
        event_mask = np.ones(MAX_EVENTS, dtype=bool)
        if n_evt > 0:
            event_arr[:n_evt] = evt_window[:n_evt]
            event_mask[:n_evt] = False

        star = chart.get("star_rating", 4.0)
        return mel_window, event_arr, event_mask, np.float32(star)

    def __len__(self):
        return len(self.charts) * 10

    def __getitem__(self, idx):
        rng = np.random.default_rng()

        # decide if this is a rating pair
        if self.mode == "mixed" and self.rating_ratio > 0 and rng.random() < self.rating_ratio:
            return self._rating_pair(rng)

        return self._corruption_pair(rng)

    def _corruption_pair(self, rng):
        """Sample a bidirectional corruption pair."""
        ci = rng.integers(0, len(self.charts))
        chart = self.charts[ci]
        events = self.events[ci]
        mel_frames = self._get_mel(chart["mel_file"]).shape[1]

        # pick pair type
        pair_type = _sample_pair_type(rng, corruption_only=(self.mode == "corruption"))
        pt = PAIR_TYPES[pair_type]
        level_a, level_b = pt["sample"](rng)
        is_tie = pt["is_tie"]

        if is_tie:
            level_a, level_b = _fix_cross_tie_severity(level_a, level_b)

        # for ordered pairs: level_a should be better (lower severity)
        if not is_tie:
            sev_a, sev_b = LEVEL_SEVERITY[level_a], LEVEL_SEVERITY[level_b]
            if sev_a > sev_b:
                level_a, level_b = level_b, level_a
                sev_a, sev_b = sev_b, sev_a
            margin = sev_b - sev_a
        else:
            margin = 0

        events_a = corrupt_events(events, level_a, mel_frames,
                                  self.gap_values, self.gap_probs, rng)
        events_b = corrupt_events(events, level_b, mel_frames,
                                  self.gap_values, self.gap_probs, rng)

        # same window, same mel
        total_frames = self._get_mel(chart["mel_file"]).shape[1]
        if total_frames <= WINDOW_FRAMES:
            start = 0
        else:
            start = rng.integers(0, total_frames - WINDOW_FRAMES)

        mel = self._get_mel(chart["mel_file"])
        end = start + WINDOW_FRAMES
        mel_window = mel[:, start:min(total_frames, end)].astype(np.float32)
        if mel_window.shape[1] < WINDOW_FRAMES:
            pad = WINDOW_FRAMES - mel_window.shape[1]
            mel_window = np.pad(mel_window, ((0, 0), (0, pad)), mode="constant")

        star = np.float32(chart.get("star_rating", 4.0))

        def _window_events(evts):
            mask = (evts >= start) & (evts < end)
            ew = evts[mask].astype(np.int64) - start
            n = min(len(ew), MAX_EVENTS)
            arr = np.zeros(MAX_EVENTS, dtype=np.int64)
            m = np.ones(MAX_EVENTS, dtype=bool)
            if n > 0:
                arr[:n] = ew[:n]
                m[:n] = False
            return arr, m

        evt_a, mask_a = _window_events(events_a)
        evt_b, mask_b = _window_events(events_b)

        # is_tie encoded as negative margin
        margin_val = np.float32(-1.0) if is_tie else np.float32(margin)

        return (mel_window, evt_a, mask_a, star,
                mel_window.copy(), evt_b, mask_b, star,
                margin_val)

    def _rating_pair(self, rng):
        """Sample a cross-set rating pair."""
        if len(self.rated_charts) < 2:
            return self._corruption_pair(rng)

        for _ in range(50):
            i, j = rng.choice(len(self.rated_charts), size=2, replace=False)
            ci_a, rating_a, star_a = self.rated_charts[i]
            ci_b, rating_b, star_b = self.rated_charts[j]

            if self.charts[ci_a].get("beatmapset_id") == self.charts[ci_b].get("beatmapset_id"):
                continue
            if abs(star_a - star_b) > 0.5:
                continue
            gap = abs(rating_a - rating_b)
            if gap < 1.0:
                continue

            if rating_b > rating_a:
                ci_a, ci_b = ci_b, ci_a
                rating_a, rating_b = rating_b, rating_a

            margin = gap

            mel_a, evt_a, mask_a, sr_a = self._extract_window(
                ci_a, self.events[ci_a], rng, do_augment=self.augment_audio)
            mel_b, evt_b, mask_b, sr_b = self._extract_window(
                ci_b, self.events[ci_b], rng, do_augment=self.augment_audio)

            return (mel_a, evt_a, mask_a, sr_a,
                    mel_b, evt_b, mask_b, sr_b,
                    np.float32(margin))

        return self._corruption_pair(rng)


# ──────────────────────────────────────────────
#  Loss
# ──────────────────────────────────────────────

def bidirectional_loss(score_a, score_b, margin, alpha=0.1):
    """Combined loss: Bradley-Terry for ordered pairs, MSE for ties.

    margin > 0: ordered pair (a better than b)
    margin == -1: tie pair (push scores together)
    """
    is_tie = (margin < 0)
    tie_mask = is_tie.float()
    ord_mask = (~is_tie).float()

    # tie loss: (score_a - score_b)^2
    tie_loss = ((score_a - score_b) ** 2) * tie_mask

    # ordered loss: -log(sigmoid(diff - alpha * margin))
    diff = score_a - score_b - alpha * margin.clamp(min=0)
    ord_loss = -F.logsigmoid(diff) * ord_mask

    # combine
    n_tie = tie_mask.sum().clamp(min=1)
    n_ord = ord_mask.sum().clamp(min=1)
    loss = (tie_loss.sum() / n_tie + ord_loss.sum() / n_ord) / 2.0

    return loss


# ──────────────────────────────────────────────
#  Validation
# ──────────────────────────────────────────────

def validate_and_collect(model, val_loader, device, alpha=0.1):
    model.eval()
    total_loss = 0.0
    total = 0
    pairs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            mel_a, evt_a, mask_a, star_a, mel_b, evt_b, mask_b, star_b, margin = [
                x.to(device) for x in batch
            ]

            score_a = model(mel_a, evt_a, mask_a, star_a)
            score_b = model(mel_b, evt_b, mask_b, star_b)

            loss = bidirectional_loss(score_a, score_b, margin, alpha)
            total_loss += loss.item() * mel_a.size(0)
            total += mel_a.size(0)

            sa = score_a.cpu().numpy()
            sb = score_b.cpu().numpy()
            mg = margin.cpu().numpy()
            for i in range(mel_a.size(0)):
                is_tie = mg[i] < 0
                pairs.append({
                    "score_a": float(sa[i]),
                    "score_b": float(sb[i]),
                    "margin": float(mg[i]),
                    "diff": float(sa[i] - sb[i]),
                    "is_tie": bool(is_tie),
                    "correct": bool(sa[i] > sb[i]) if not is_tie else None,
                    "confidence": float(abs(sa[i] - sb[i])),
                })

    # aggregate
    ordered = [p for p in pairs if not p["is_tie"]]
    ties = [p for p in pairs if p["is_tie"]]
    n_ord = max(len(ordered), 1)
    n_tie = max(len(ties), 1)

    correct_ord = sum(1 for p in ordered if p["correct"])

    # per-margin accuracy (ordered only)
    margin_acc = {}
    for m_int in range(1, 5):
        m_pairs = [p for p in ordered if round(p["margin"]) == m_int]
        if m_pairs:
            margin_acc[m_int] = sum(1 for p in m_pairs if p["correct"]) / len(m_pairs)

    # tie quality: mean absolute diff (should be near 0)
    tie_abs_diff = np.mean([abs(p["diff"]) for p in ties]) if ties else 0.0

    metrics = {
        "val_loss": total_loss / max(total, 1),
        "pair_accuracy": correct_ord / n_ord,
        "n_ordered": len(ordered),
        "n_ties": len(ties),
        "tie_abs_diff": float(tie_abs_diff),
        "mean_diff": float(np.mean([p["diff"] for p in ordered])) if ordered else 0.0,
        "mean_confidence": float(np.mean([p["confidence"] for p in ordered])) if ordered else 0.0,
    }
    for m, acc in margin_acc.items():
        metrics[f"acc_margin_{m}"] = acc

    model.train()
    return metrics, pairs


def compute_score_by_level(model, dataset, device, n_samples=100):
    """Score random windows at each of the 9 levels."""
    model.eval()
    rng = np.random.default_rng(42)
    scores_by_level = {i: [] for i in range(9)}

    with torch.no_grad():
        for _ in range(n_samples):
            ci = rng.integers(0, len(dataset.charts))
            chart = dataset.charts[ci]
            events = dataset.events[ci]
            mel_frames = dataset._get_mel(chart["mel_file"]).shape[1]

            for level in range(9):
                corrupted = corrupt_events(events, level, mel_frames,
                                           dataset.gap_values, dataset.gap_probs, rng)
                mel_w, evt_arr, evt_mask, star = dataset._extract_window(ci, corrupted, rng)

                mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)
                evt_t = torch.from_numpy(evt_arr).unsqueeze(0).to(device)
                mask_t = torch.from_numpy(evt_mask).unsqueeze(0).to(device)
                star_t = torch.tensor([star], device=device)

                score = model(mel_t, evt_t, mask_t, star_t).item()
                scores_by_level[level].append(score)

    summary = {}
    raw = {}
    for level in range(9):
        s = np.array(scores_by_level[level])
        name = LEVEL_NAMES[level]
        summary[name] = {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "median": float(np.median(s)),
            "p10": float(np.percentile(s, 10)),
            "p90": float(np.percentile(s, 90)),
            "n": len(s),
        }
        raw[name] = s.tolist()

    model.train()
    return summary, raw


# ──────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────

def save_eval_plots(eval_step, eval_dir, level_summary, level_raw, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = os.path.join(eval_dir, f"eval_{eval_step:03d}")

    # ── 1. Score distributions by level (all 9) ──
    colors_rand = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    colors_metro = ["#2ecc71", "#3498db", "#2980b9", "#1f618d", "#154360"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    # left: random corruption
    rand_names = ["CLEAN", "LIGHT_RAND", "MED_RAND", "HIGH_RAND", "PURE_RAND"]
    for i, name in enumerate(rand_names):
        if name in level_raw:
            parts = axes[0].violinplot([level_raw[name]], positions=[i], showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(colors_rand[i])
                pc.set_alpha(0.6)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(rand_names, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("Quality Score")
    axes[0].set_title("Random Corruption")
    axes[0].grid(True, alpha=0.3)

    # right: metro corruption
    metro_names = ["CLEAN", "LIGHT_METRO", "MED_METRO", "HIGH_METRO", "PURE_METRO"]
    for i, name in enumerate(metro_names):
        if name in level_raw:
            parts = axes[1].violinplot([level_raw[name]], positions=[i], showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(colors_metro[i])
                pc.set_alpha(0.6)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(metro_names, rotation=20, ha="right", fontsize=8)
    axes[1].set_title("Metro Corruption")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Score Distribution by Corruption Level (eval {eval_step})")
    fig.tight_layout()
    fig.savefig(f"{prefix}_score_dist.png", dpi=150)
    plt.close(fig)

    # ── 2. Score diff histogram ──
    ordered = [p for p in pairs if not p["is_tie"]]
    ties = [p for p in pairs if p["is_tie"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    correct_diffs = [p["diff"] for p in ordered if p["correct"]]
    wrong_diffs = [p["diff"] for p in ordered if not p["correct"]]
    if correct_diffs:
        ax1.hist(correct_diffs, bins=50, alpha=0.6, color="#2ecc71", label=f"Correct ({len(correct_diffs)})")
    if wrong_diffs:
        ax1.hist(wrong_diffs, bins=50, alpha=0.6, color="#e74c3c", label=f"Wrong ({len(wrong_diffs)})")
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("score_a - score_b")
    ax1.set_title("Ordered Pairs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    tie_diffs = [p["diff"] for p in ties]
    if tie_diffs:
        ax2.hist(tie_diffs, bins=50, alpha=0.6, color="#3498db", label=f"Ties ({len(tie_diffs)})")
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("score_a - score_b")
    ax2.set_title(f"Tie Pairs (should be centered at 0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Score Differences (eval {eval_step})")
    fig.tight_layout()
    fig.savefig(f"{prefix}_diff_hist.png", dpi=150)
    plt.close(fig)


def save_training_curves(history, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(history) < 2:
        return

    steps = [h["eval_step"] for h in history]

    # ── loss ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [h["train_loss"] for h in history], label="Train", linewidth=2)
    ax.plot(steps, [h["val_loss"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss.png"), dpi=150)
    plt.close(fig)

    # ── pair accuracy + per-margin ──
    margin_colors = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71"}
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [h["pair_accuracy"] for h in history],
            label="Overall", linewidth=2.5, color="#2c3e50", marker="o", markersize=3)
    for m in range(1, 5):
        key = f"acc_margin_{m}"
        vals = [h.get(key, 0.5) for h in history]
        ax.plot(steps, vals, label=f"Margin {m}", linewidth=1.5,
                color=margin_colors[m], linestyle="--")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pairwise Accuracy (ordered pairs)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "pair_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── score levels (all 9) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    rand_names = ["CLEAN", "LIGHT_RAND", "MED_RAND", "HIGH_RAND", "PURE_RAND"]
    metro_names = ["CLEAN", "LIGHT_METRO", "MED_METRO", "HIGH_METRO", "PURE_METRO"]
    colors_r = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    colors_m = ["#2ecc71", "#3498db", "#2980b9", "#1f618d", "#154360"]

    for i, name in enumerate(rand_names):
        vals = [h.get("level_scores", {}).get(name, {}).get("mean", 0) for h in history]
        ax1.plot(steps, vals, label=name, linewidth=2, color=colors_r[i])
    ax1.set_xlabel("Eval Step")
    ax1.set_ylabel("Mean Score")
    ax1.set_title("Random Corruption Scores")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for i, name in enumerate(metro_names):
        vals = [h.get("level_scores", {}).get(name, {}).get("mean", 0) for h in history]
        ax2.plot(steps, vals, label=name, linewidth=2, color=colors_m[i])
    ax2.set_xlabel("Eval Step")
    ax2.set_title("Metro Corruption Scores")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "level_scores.png"), dpi=150)
    plt.close(fig)

    # ── tie quality ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [h.get("tie_abs_diff", 0) for h in history],
            linewidth=2, color="#3498db", marker="o", markersize=3)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Mean |score_a - score_b| for tie pairs")
    ax.set_title("Tie Quality (should decrease)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "tie_quality.png"), dpi=150)
    plt.close(fig)

    # ── master dashboard ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    ax1.plot(steps, [h["pair_accuracy"] for h in history],
             linewidth=2.5, color="#2c3e50", marker="o", markersize=3, label="Overall")
    for m in range(1, 5):
        vals = [h.get(f"acc_margin_{m}", 0.5) for h in history]
        ax1.plot(steps, vals, linewidth=1, color=margin_colors[m], alpha=0.7, label=f"M{m}")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Master Dashboard")
    ax1.legend(loc="lower right", ncol=5, fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.2)

    # score levels
    for i, name in enumerate(rand_names):
        vals = [h.get("level_scores", {}).get(name, {}).get("mean", 0) for h in history]
        ax2.plot(steps, vals, linewidth=2, color=colors_r[i],
                 label=name if name != "CLEAN" else None)
    for i, name in enumerate(metro_names):
        if name == "CLEAN":
            continue
        vals = [h.get("level_scores", {}).get(name, {}).get("mean", 0) for h in history]
        ax2.plot(steps, vals, linewidth=2, color=colors_m[i], linestyle="--", label=name)
    # CLEAN once
    vals = [h.get("level_scores", {}).get("CLEAN", {}).get("mean", 0) for h in history]
    ax2.plot(steps, vals, linewidth=2.5, color="#2ecc71", label="CLEAN")
    ax2.set_ylabel("Score")
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.2)

    ax3.plot(steps, [h["train_loss"] for h in history], label="Train", linewidth=2, color="#3498db")
    ax3.plot(steps, [h["val_loss"] for h in history], label="Val", linewidth=2, color="#e67e22")
    ax3.set_xlabel("Eval Step")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "master.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────

def train(args):
    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    manifest_path = os.path.join(ds_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    charts = manifest["charts"]
    print(f"Dataset: {args.dataset}, {len(charts)} charts")

    # train/val split
    rng_split = np.random.default_rng(42)
    bset_ids = sorted(set(c.get("beatmapset_id", str(i)) for i, c in enumerate(charts)))
    rng_split.shuffle(bset_ids)
    n_val = max(1, int(len(bset_ids) * 0.1))
    val_bsets = set(bset_ids[:n_val])
    train_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id") not in val_bsets]
    val_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id") in val_bsets]
    print(f"Train: {len(train_idx)} charts, Val: {len(val_idx)} charts ({n_val} beatmapsets)")

    mode = "mixed" if args.rating_ratio > 0 else "corruption"
    print(f"Mode: {mode}, rating_ratio={args.rating_ratio:.0%}")

    train_ds = PairDatasetV2(manifest, ds_dir, train_idx, mode=mode,
                             rating_ratio=args.rating_ratio)
    val_ds = PairDatasetV2(manifest, ds_dir, val_idx, mode=mode,
                           rating_ratio=args.rating_ratio, augment_audio=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    model = ChartQualityEvaluator(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    run_dir = os.path.join(SCRIPT_DIR, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    eval_dir = os.path.join(run_dir, "evals")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    start_epoch = 1
    eval_step = 0
    best_val_loss = float("inf")
    history = []

    if args.warm_start:
        print(f"Warm-starting from {args.warm_start}")
        ckpt = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

    if args.resume:
        ckpt_files = sorted([
            f for f in os.listdir(ckpt_dir)
            if f.startswith("eval_") and f.endswith(".pt") and f != "best.pt"
        ])
        if ckpt_files:
            latest = os.path.join(ckpt_dir, ckpt_files[-1])
            print(f"Resuming from {latest}")
            ckpt = torch.load(latest, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 1) + 1
            eval_step = ckpt.get("eval_step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if os.path.exists(os.path.join(run_dir, "history.json")):
                with open(os.path.join(run_dir, "history.json")) as f:
                    history = json.load(f)

    print(f"\nTraining for {args.epochs} epochs, {args.evals_per_epoch} eval(s)/epoch")
    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_ord_total = 0
        epoch_tie_total = 0
        recent_losses = deque(maxlen=50)
        recent_correct = deque(maxlen=50)
        recent_diffs = deque(maxlen=50)
        recent_tie_diffs = deque(maxlen=50)

        n_batches = len(train_loader)
        eval_interval = max(1, n_batches // args.evals_per_epoch)
        eval_at = set(range(eval_interval - 1, n_batches, eval_interval))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for batch_idx, batch in enumerate(pbar):
            mel_a, evt_a, mask_a, star_a, mel_b, evt_b, mask_b, star_b, margin = [
                x.to(args.device) for x in batch
            ]

            with torch.autocast("cuda", enabled=args.amp):
                score_a = model(mel_a, evt_a, mask_a, star_a)
                score_b = model(mel_b, evt_b, mask_b, star_b)
                loss = bidirectional_loss(score_a, score_b, margin, args.alpha)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # tracking
            bs = mel_a.size(0)
            batch_loss = loss.item()
            with torch.no_grad():
                is_tie = (margin < 0)
                diffs = score_a - score_b
                ord_correct = ((diffs > 0) & ~is_tie).float().sum().item()
                n_ord = (~is_tie).sum().item()
                n_tie = is_tie.sum().item()
                tie_abs = diffs[is_tie].abs().mean().item() if n_tie > 0 else 0.0

            epoch_loss += batch_loss * bs
            epoch_correct += ord_correct
            epoch_ord_total += n_ord
            epoch_tie_total += n_tie

            recent_losses.append(batch_loss)
            if n_ord > 0:
                recent_correct.append(ord_correct / n_ord)
                recent_diffs.append(diffs[~is_tie].mean().item())
            if n_tie > 0:
                recent_tie_diffs.append(tie_abs)

            r_loss = sum(recent_losses) / len(recent_losses)
            r_acc = sum(recent_correct) / max(len(recent_correct), 1)
            r_diff = sum(recent_diffs) / max(len(recent_diffs), 1)
            r_tie = sum(recent_tie_diffs) / max(len(recent_tie_diffs), 1)

            pbar.set_postfix_str(
                f"loss={r_loss:.4f} acc={r_acc:.1%} "
                f"diff={r_diff:+.2f} tie_gap={r_tie:.2f} "
                f"[ep_acc={epoch_correct / max(epoch_ord_total, 1):.1%}]"
            )

            if batch_idx in eval_at:
                eval_step += 1
                epoch_frac = epoch + (batch_idx + 1) / n_batches - 1
                _run_eval(model, val_loader, train_ds, args, eval_step, epoch_frac,
                          epoch_loss / max(epoch_ord_total + epoch_tie_total, 1),
                          run_dir, ckpt_dir, eval_dir, history, scheduler, optimizer,
                          scaler, best_val_loss)
                if history and history[-1]["val_loss"] < best_val_loss:
                    best_val_loss = history[-1]["val_loss"]

        scheduler.step()

        if n_batches - 1 not in eval_at:
            eval_step += 1
            _run_eval(model, val_loader, train_ds, args, eval_step, epoch,
                      epoch_loss / max(epoch_ord_total + epoch_tie_total, 1),
                      run_dir, ckpt_dir, eval_dir, history, scheduler, optimizer,
                      scaler, best_val_loss)
            if history and history[-1]["val_loss"] < best_val_loss:
                best_val_loss = history[-1]["val_loss"]

    print("\nTraining complete.")


def _run_eval(model, val_loader, train_ds, args, eval_step, epoch_frac, train_loss,
              run_dir, ckpt_dir, eval_dir, history, scheduler, optimizer, scaler, best_val_loss):
    val_metrics, pairs = validate_and_collect(model, val_loader, args.device, alpha=args.alpha)
    val_loss = val_metrics["val_loss"]

    level_summary, level_raw = compute_score_by_level(model, train_ds, args.device, n_samples=100)

    # print
    tag = f"{epoch_frac:.1f}" if isinstance(epoch_frac, float) else str(epoch_frac)
    acc_str = " ".join(
        f"m{m}={val_metrics.get(f'acc_margin_{m}', 0):.1%}"
        for m in range(1, 5) if f"acc_margin_{m}" in val_metrics
    )

    # score summary: CLEAN, then rand, then metro
    rand_str = " > ".join(f"{LEVEL_NAMES[i]}={level_summary[LEVEL_NAMES[i]]['mean']:+.1f}" for i in [0, 1, 2, 3, 4])
    metro_str = " > ".join(f"{LEVEL_NAMES[i]}={level_summary[LEVEL_NAMES[i]]['mean']:+.1f}" for i in [5, 6, 7, 8])

    print(f"  Eval {eval_step} (epoch {tag}): "
          f"loss={train_loss:.4f}/{val_loss:.4f} | "
          f"pair_acc={val_metrics['pair_accuracy']:.1%} [{acc_str}] | "
          f"tie_gap={val_metrics['tie_abs_diff']:.2f} | "
          f"lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"    rand: {rand_str}")
    print(f"    metro: {metro_str}")

    # monotonicity checks
    rand_means = [level_summary[LEVEL_NAMES[i]]["mean"] for i in [0, 1, 2, 3, 4]]
    metro_means = [level_summary[LEVEL_NAMES[i]]["mean"] for i in [0, 5, 6, 7, 8]]
    mono_rand = all(rand_means[i] >= rand_means[i + 1] for i in range(len(rand_means) - 1))
    mono_metro = all(metro_means[i] >= metro_means[i + 1] for i in range(len(metro_means) - 1))
    print(f"    mono_rand={mono_rand} mono_metro={mono_metro}")

    eval_data = {
        "eval_step": eval_step,
        "epoch": epoch_frac if isinstance(epoch_frac, float) else float(epoch_frac),
        "train_loss": train_loss,
        **val_metrics,
        "level_scores": level_summary,
        "mono_rand": mono_rand,
        "mono_metro": mono_metro,
    }
    history.append(eval_data)

    with open(os.path.join(eval_dir, f"eval_{eval_step:03d}.json"), "w") as f:
        json.dump(eval_data, f, indent=2)
    with open(os.path.join(eval_dir, f"eval_{eval_step:03d}_pairs.json"), "w") as f:
        json.dump({"pairs": pairs, "level_raw": level_raw}, f)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    save_eval_plots(eval_step, eval_dir, level_summary, level_raw, pairs)
    save_training_curves(history, run_dir)

    ckpt = {
        "eval_step": eval_step,
        "epoch": epoch_frac if isinstance(epoch_frac, float) else float(epoch_frac),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(ckpt_dir, f"eval_{eval_step:03d}.pt"))
    if val_loss < best_val_loss:
        torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
        print(f"    * new best val_loss={val_loss:.4f}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train chart quality evaluator v2 (bidirectional)")
    parser.add_argument("dataset", help="Dataset directory name (e.g. taiko_v2)")
    parser.add_argument("--run-name", required=True, help="Run name for output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1, help="Margin scaling factor")
    parser.add_argument("--rating-ratio", type=float, default=0.15,
                        help="Fraction of rating pairs (0 = corruption only)")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-start", type=str, default=None)
    args = parser.parse_args()

    if args.no_amp:
        args.amp = False

    train(args)


if __name__ == "__main__":
    main()
