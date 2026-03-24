"""Training script for onset detection model."""
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from detection_model import OnsetDetector, DualStreamOnsetDetector, InterleavedOnsetDetector, ContextFiLMDetector, FramewiseOnsetDetector, EventEmbeddingDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── hyperparams ──
A_BINS = 500        # past audio context (2.5s at 5ms)
B_BINS = 500        # future audio context (2.5s at 5ms)
C_EVENTS = 128      # past event context
N_CLASSES = 501     # 0-499 bin offsets + 500=STOP
WINDOW = A_BINS + B_BINS
MIN_CURSOR_BIN = 6000  # only train where cursor >= 30s
MAX_TARGETS = 64        # max onsets per forward window for multi-target training


# ═══════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════

class OnsetDataset(Dataset):
    """Yields (mel_window, event_offsets, event_mask, conditioning, target) samples."""

    def __init__(self, manifest, ds_dir, chart_indices, augment=False, subsample=1,
                 multi_target=False):
        self.mel_dir = os.path.join(ds_dir, "mels")
        self.charts = [manifest["charts"][i] for i in chart_indices]
        self.augment = augment
        self.multi_target = multi_target

        self.events = []
        evt_dir = os.path.join(ds_dir, "events")
        for chart in self.charts:
            evt = np.load(os.path.join(evt_dir, chart["event_file"]))
            self.events.append(evt)

        # build sample index, skip samples where cursor is in first 30s
        self.samples = []
        for ci, evt in enumerate(self.events):
            for ei in range(len(evt)):
                cursor = max(0, int(evt[0]) - B_BINS) if ei == 0 else int(evt[ei - 1])
                if cursor >= MIN_CURSOR_BIN:
                    self.samples.append((ci, ei))
            if len(evt) > 0 and int(evt[-1]) >= MIN_CURSOR_BIN:
                self.samples.append((ci, len(evt)))

        # subsample: keep every Nth sample
        if subsample > 1:
            self.samples = self.samples[::subsample]

        # precompute class distribution
        self.class_counts = np.zeros(N_CLASSES, dtype=np.int64)
        for ci_idx, ei_idx in self.samples:
            self.class_counts[self._get_target(ci_idx, ei_idx)] += 1

        # per-worker mmap cache (lazily populated)
        self._mel_cache = {}

    def _get_target(self, ci, ei):
        evt = self.events[ci]
        if ei == 0:
            cursor = max(0, int(evt[0]) - B_BINS) if len(evt) > 0 else 0
        else:
            cursor = int(evt[ei - 1])
        if ei < len(evt):
            offset = max(0, int(evt[ei]) - cursor)
            return N_CLASSES - 1 if offset >= B_BINS else offset
        return N_CLASSES - 1

    def _get_mel(self, mel_file):
        if mel_file not in self._mel_cache:
            # mmap: each worker gets its own handle, OS page cache shares data
            self._mel_cache[mel_file] = np.load(
                os.path.join(self.mel_dir, mel_file), mmap_mode="r"
            )
        return self._mel_cache[mel_file]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, ei = self.samples[idx]
        chart = self.charts[ci]
        evt = self.events[ci]

        if ei == 0:
            cursor = max(0, evt[0] - B_BINS) if len(evt) > 0 else 0
        else:
            cursor = int(evt[ei - 1])

        # target(s)
        if ei < len(evt):
            offset = max(0, int(evt[ei]) - cursor)
            target = N_CLASSES - 1 if offset >= B_BINS else offset
        else:
            target = N_CLASSES - 1

        # mel window (stored as float16, convert 1000-frame slice to float32)
        mel = self._get_mel(chart["mel_file"])
        total_frames = mel.shape[1]
        start = cursor - A_BINS
        end = cursor + B_BINS
        pad_left = max(0, -start)
        pad_right = max(0, end - total_frames)
        mel_window = mel[:, max(0, start):min(total_frames, end)].astype(np.float32)
        if pad_left > 0 or pad_right > 0:
            mel_window = np.pad(mel_window, ((0, 0), (pad_left, pad_right)), mode="constant")

        # past events
        if ei > 0:
            past_start = max(0, ei - C_EVENTS)
            past_bins = evt[past_start:ei].astype(np.int64) - cursor
        else:
            past_bins = np.array([], dtype=np.int64)

        # augmentations
        if self.augment:
            mel_window, past_bins, cond_jitter = self._augment(mel_window, past_bins)
        else:
            cond_jitter = np.ones(3, dtype=np.float32)

        # pad events to C_EVENTS (trim if augmentation added extras)
        if len(past_bins) > C_EVENTS:
            past_bins = past_bins[-C_EVENTS:]
        n_past = len(past_bins)
        event_offsets = np.zeros(C_EVENTS, dtype=np.int64)
        event_mask = np.ones(C_EVENTS, dtype=bool)
        if n_past > 0:
            event_offsets[-n_past:] = past_bins
            event_mask[-n_past:] = False

        cond = np.array([
            chart.get("density_mean", 4.0),
            chart.get("density_peak", 8),
            chart.get("density_std", 1.5),
        ], dtype=np.float32) * cond_jitter

        # multi-target: all onsets in forward window
        if self.multi_target:
            future_mask = (evt > cursor) & (evt <= cursor + B_BINS)
            future_bins = (evt[future_mask].astype(np.int64) - cursor)
            future_bins = np.clip(future_bins, 0, B_BINS - 1)

            n_targets = len(future_bins)
            targets_padded = np.full(MAX_TARGETS, -1, dtype=np.int64)
            if n_targets > MAX_TARGETS:
                future_bins = future_bins[:MAX_TARGETS]
                n_targets = MAX_TARGETS
            if n_targets > 0:
                targets_padded[:n_targets] = future_bins

            return (
                torch.from_numpy(mel_window),
                torch.from_numpy(event_offsets),
                torch.from_numpy(event_mask),
                torch.from_numpy(cond),
                torch.from_numpy(targets_padded),
                torch.tensor(n_targets, dtype=torch.long),
            )

        return (
            torch.from_numpy(mel_window),
            torch.from_numpy(event_offsets),
            torch.from_numpy(event_mask),
            torch.from_numpy(cond),
            torch.tensor(target, dtype=torch.long),
        )

    def _augment(self, mel_window, past_bins):
        rng = np.random.default_rng()
        cond_jitter = np.ones(3, dtype=np.float32)

        # event jitter: global shift + recency-scaled per-event noise (simulates AR error)
        if len(past_bins) > 0:
            n = len(past_bins)
            # global shift applied to ALL events (±0-3 bins) - systematic drift
            global_shift = rng.integers(-3, 4)
            # per-event jitter scaled by recency: 1x oldest → 2x most recent
            jitter_scale = np.linspace(1, 2, n)
            per_event = (rng.integers(-3, 4, size=n) * jitter_scale).astype(np.int64)
            past_bins = np.sort(past_bins + global_shift + per_event)

        # random event deletion (5%) - drop 1-2 events to simulate skip errors
        if len(past_bins) > 2 and rng.random() < 0.05:
            n_drop = rng.integers(1, max(2, min(3, len(past_bins) // 4)))
            keep = np.ones(len(past_bins), dtype=bool)
            keep[rng.choice(len(past_bins), size=n_drop, replace=False)] = False
            past_bins = past_bins[keep]

        # random event insertion (3%) - add 1 fake event to simulate hallucination
        if len(past_bins) > 0 and rng.random() < 0.03:
            lo, hi = past_bins[0], min(past_bins[-1], -1)
            if lo < hi:
                fake = rng.integers(lo, hi, size=1)
                past_bins = np.sort(np.concatenate([past_bins, fake]))

        # partial metronome corruption (2%) - replace RECENT HALF with metronomic events
        # keeps older context intact so model still has real pattern to reference
        if len(past_bins) > 8 and rng.random() < 0.02:
            half = len(past_bins) // 2
            gap = rng.integers(10, 80)
            metro = np.array([-gap * (half - i) for i in range(half)], dtype=np.int64)
            past_bins[-half:] = metro
            past_bins = np.sort(past_bins)

        # partial advanced metronome (2%) - replace OLDEST HALF with dominant-gap metronome
        # recent context stays real, older history becomes metronomic
        elif len(past_bins) > 8 and rng.random() < 0.02:
            gaps = np.diff(past_bins)
            gaps = gaps[gaps > 0]
            if len(gaps) > 2:
                from collections import Counter
                rounded = (gaps // 3) * 3
                dominant = Counter(rounded).most_common(1)[0][0]
                dominant = max(5, int(dominant))
                half = len(past_bins) // 2
                jitter = rng.integers(-1, 2, size=half)
                # oldest half becomes metronomic, starting from where the real oldest event was
                base = past_bins[0]
                metro = np.array([base + (dominant + jitter[i]) * i for i in range(half)], dtype=np.int64)
                past_bins[:half] = metro
                past_bins = np.sort(past_bins)

        # large time shift (2%) - shift 2-4 recent events by ±50 bins
        if len(past_bins) > 4 and rng.random() < 0.02:
            shift = rng.integers(-50, 51)
            n_shift = rng.integers(2, min(5, len(past_bins)))
            past_bins[-n_shift:] += shift
            past_bins = np.sort(past_bins)

        # context truncation (5%) - remove oldest events
        if len(past_bins) > 1 and rng.random() < 0.05:
            past_bins = past_bins[-rng.integers(1, len(past_bins)):]

        # audio fade-in (10%)
        if rng.random() < 0.10:
            fl = rng.integers(20, 101)
            mel_window[:, :fl] *= np.linspace(0, 1, fl, dtype=np.float32)[np.newaxis, :]
        # audio fade-out (10%)
        if rng.random() < 0.10:
            fl = rng.integers(20, 101)
            mel_window[:, -fl:] *= np.linspace(1, 0, fl, dtype=np.float32)[np.newaxis, :]

        # mel gain ±2dB (30%)
        if rng.random() < 0.30:
            mel_window = mel_window + rng.uniform(-2.0, 2.0)
        # mel noise (15%)
        if rng.random() < 0.15:
            mel_window = mel_window + rng.normal(0, rng.uniform(0.1, 0.3), mel_window.shape).astype(np.float32)

        # frequency jitter: shift all mel bands up/down by ±1-3 bins (15%)
        if rng.random() < 0.15:
            shift = rng.integers(-3, 4)
            if shift != 0:
                mel_window = np.roll(mel_window, shift, axis=0)
                if shift > 0:
                    mel_window[:shift, :] = 0
                else:
                    mel_window[shift:, :] = 0

        # SpecAugment freq mask (20%, 1 mask, up to 10 bands)
        if rng.random() < 0.20:
            n = rng.integers(1, 11)
            f = rng.integers(0, mel_window.shape[0] - n)
            mel_window[f:f + n, :] = 0
        # SpecAugment time mask (20%, 1 mask, up to 30 frames)
        if rng.random() < 0.20:
            n = rng.integers(1, 31)
            t = rng.integers(0, mel_window.shape[1] - n)
            mel_window[:, t:t + n] = 0

        # conditioning jitter ±2% (10%) — tighter jitter improves AR density adherence (exp 45)
        if rng.random() < 0.10:
            cond_jitter = rng.uniform(0.98, 1.02, size=3).astype(np.float32)

        return mel_window, past_bins, cond_jitter


# ═══════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════

class OnsetLoss(nn.Module):
    """Trapezoid soft-target cross-entropy loss.

    Soft targets use a trapezoid shape in log-ratio space:
      - Within `good_pct` (3%): full credit (flat plateau)
      - Between `good_pct` and `fail_pct` (20%): linear ramp to zero
      - Beyond `fail_pct`: zero weight (total failure, same as random guess)

    This is proportional: predicting 10 when target is 20 (50% error) is
    punished the same as predicting 100 when target is 200.

    loss = hard_alpha * hard_CE + (1 - hard_alpha) * soft_CE
    STOP class always gets a hard target.
    """

    def __init__(self, weight=None, gamma=0.0, good_pct=0.03, fail_pct=0.20,
                 hard_alpha=0.5, frame_tolerance=2, stop_weight=3.0):
        super().__init__()
        self.gamma = gamma
        self.good_pct = good_pct
        self.fail_pct = fail_pct
        self.hard_alpha = hard_alpha
        self.frame_tolerance = frame_tolerance  # ±N frames always get some credit
        self.stop_weight = stop_weight  # extra penalty for missing STOP
        # precompute log thresholds
        self.log_good = math.log(1 + good_pct)   # ~0.0296
        self.log_fail = math.log(1 + fail_pct)    # ~0.182
        self.register_buffer("weight", weight)

    def _make_soft_targets(self, targets, n_classes):
        """Convert hard targets to trapezoid soft distributions in log-ratio space."""
        B = targets.size(0)
        soft = torch.zeros(B, n_classes, device=targets.device)

        stop = n_classes - 1
        is_stop = targets == stop
        is_bin = ~is_stop

        # STOP class: hard target
        if is_stop.any():
            soft[is_stop, stop] = 1.0

        # bin classes: trapezoid in log-ratio space over bins 0..(stop-1)
        if is_bin.any():
            bin_targets = targets[is_bin].float()  # (M,)
            bins = torch.arange(stop, device=targets.device, dtype=torch.float32)

            # |log((i+1)/(t+1))| = proportional distance in ratio space
            abs_log_ratio = torch.abs(
                torch.log((bins + 1).unsqueeze(0) / (bin_targets + 1).unsqueeze(1))
            )

            # trapezoid: 1.0 inside good, linear ramp to 0 at fail, 0 beyond
            ramp_width = self.log_fail - self.log_good
            ratio_weights = ((self.log_fail - abs_log_ratio) / ramp_width).clamp(0, 1)

            # frame-distance floor: ±frame_tolerance bins always get some credit
            # (prevents tiny targets like t=2 from having zero-width plateaus)
            frame_dist = torch.abs(bins.unsqueeze(0) - bin_targets.unsqueeze(1))
            frame_weights = ((self.frame_tolerance + 1 - frame_dist) / (self.frame_tolerance + 1)).clamp(0, 1)

            # take the max of ratio-based and frame-based
            weights = torch.max(ratio_weights, frame_weights)

            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            soft[is_bin, :stop] = weights

        return soft

    def forward(self, logits, targets):
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=-1).clamp(min=-100)

        # ── hard CE: exact target class ──
        hard_ce = F.cross_entropy(logits, targets, reduction="none")

        # ── soft CE: trapezoid targets ──
        soft_targets = self._make_soft_targets(targets, n_classes)
        soft_ce = -(soft_targets * log_probs).sum(dim=-1)

        # ── mix ──
        ce = self.hard_alpha * hard_ce + (1 - self.hard_alpha) * soft_ce

        # extra penalty when target is STOP - model must learn to stop
        if self.stop_weight > 1.0:
            is_stop = (targets == n_classes - 1)
            ce = ce * torch.where(is_stop, self.stop_weight, 1.0)

        # per-sample class weight
        if self.weight is not None:
            ce = ce * self.weight[targets]

        # optional focal modulation
        if self.gamma > 0:
            pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
            ce = ((1 - pt) ** self.gamma) * ce

        return ce.mean()


class MultiTargetOnsetLoss(nn.Module):
    """Multi-target trapezoid soft-target loss.

    Same proportional log-ratio trapezoid as OnsetLoss, but targets are ALL onsets
    in the forward window, not just the nearest one. Soft target is the normalized
    sum of trapezoids centered on each real onset.

    When n_targets == 0 (no onsets in window), all mass goes to bin 500 (STOP).
    Hard CE component uses the nearest target (first onset in window).
    """

    def __init__(self, gamma=0.0, good_pct=0.03, fail_pct=0.20,
                 hard_alpha=0.3, frame_tolerance=2, empty_weight=1.5,
                 recall_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.good_pct = good_pct
        self.fail_pct = fail_pct
        self.hard_alpha = hard_alpha
        self.frame_tolerance = frame_tolerance
        self.empty_weight = empty_weight
        self.recall_weight = recall_weight
        self.log_good = math.log(1 + good_pct)
        self.log_fail = math.log(1 + fail_pct)

    def _make_multi_soft_targets(self, targets_padded, n_targets, n_classes):
        """Build summed-trapezoid soft targets for multiple onsets per sample.

        targets_padded: (B, MAX_TARGETS) int64, -1 = padding
        n_targets: (B,) int64
        Returns: (B, n_classes) normalized probability distribution
        """
        B, M = targets_padded.shape
        stop = n_classes - 1
        device = targets_padded.device

        bins = torch.arange(stop, device=device, dtype=torch.float32)  # (500,)
        soft = torch.zeros(B, n_classes, device=device)

        valid_mask = targets_padded >= 0  # (B, M)

        if valid_mask.any():
            t_float = targets_padded.float().clamp(min=0)  # (B, M)

            # |log((bin+1)/(target+1))| → (B, M, 500)
            abs_log_ratio = torch.abs(
                torch.log((bins + 1).view(1, 1, -1) / (t_float + 1).unsqueeze(-1))
            )

            # trapezoid ramp in log-ratio space
            ramp_width = self.log_fail - self.log_good
            ratio_weights = ((self.log_fail - abs_log_ratio) / ramp_width).clamp(0, 1)

            # frame distance floor for small targets
            frame_dist = torch.abs(bins.view(1, 1, -1) - t_float.unsqueeze(-1))
            frame_weights = ((self.frame_tolerance + 1 - frame_dist) /
                             (self.frame_tolerance + 1)).clamp(0, 1)

            weights = torch.max(ratio_weights, frame_weights)  # (B, M, 500)
            weights = weights * valid_mask.unsqueeze(-1).float()

            # sum across all targets per sample
            soft[:, :stop] = weights.sum(dim=1)  # (B, 500)

        # empty windows: all mass on STOP
        empty = (n_targets == 0)
        soft[empty, stop] = 1.0

        # normalize
        total = soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
        soft = soft / total

        return soft

    def forward(self, logits, targets_padded, n_targets):
        """
        logits: (B, 501)
        targets_padded: (B, MAX_TARGETS) int64, -1 = padding
        n_targets: (B,) int64
        """
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=-1).clamp(min=-100)

        # ── hard CE against nearest target (first valid) ──
        nearest = targets_padded[:, 0].clone()
        nearest[n_targets == 0] = n_classes - 1  # empty → STOP
        nearest = nearest.clamp(0, n_classes - 1)
        hard_ce = F.cross_entropy(logits, nearest, reduction="none")

        # ── soft CE against multi-target distribution ──
        soft_targets = self._make_multi_soft_targets(targets_padded, n_targets, n_classes)
        soft_ce = -(soft_targets * log_probs).sum(dim=-1)

        # ── per-onset recall loss: -log(prob[target_bin]) for each real onset ──
        recall_loss = torch.zeros(logits.size(0), device=logits.device)
        if self.recall_weight > 0:
            valid_mask = targets_padded >= 0  # (B, M)
            if valid_mask.any():
                # clamp indices for gather (invalid positions use 0, masked out later)
                safe_targets = targets_padded.clamp(0, n_classes - 1)  # (B, M)
                # -log(prob) at each target bin
                per_onset_nll = -log_probs.gather(1, safe_targets)  # (B, M)
                per_onset_nll = per_onset_nll * valid_mask.float()  # zero out padding
                # mean over valid onsets per sample (avoid /0 for empty windows)
                n_valid = valid_mask.float().sum(dim=1).clamp(min=1.0)
                recall_loss = per_onset_nll.sum(dim=1) / n_valid

        # ── mix ──
        ce = self.hard_alpha * hard_ce + (1 - self.hard_alpha) * soft_ce
        ce = ce + self.recall_weight * recall_loss

        # extra weight on empty windows
        if self.empty_weight > 1.0:
            empty = (n_targets == 0)
            ce = ce * torch.where(empty, self.empty_weight, 1.0)

        # focal modulation
        if self.gamma > 0:
            pt = (soft_targets * torch.softmax(logits, dim=-1)).sum(dim=-1)
            ce = ((1 - pt) ** self.gamma) * ce

        return ce.mean()


class SigmoidMultiTargetLoss(nn.Module):
    """Per-bin sigmoid loss for multi-target onset detection.

    Each of the 500 onset bins is an independent binary classifier:
    P(onset at bin k) = sigmoid(logit_k). No competition between bins.

    Targets are log-ratio trapezoid soft labels (same shape as OnsetLoss)
    but used as per-bin targets for BCE, not as a probability distribution.

    Bin 500 (STOP) uses a separate binary target: 1 if no onsets in window.
    """

    def __init__(self, good_pct=0.03, fail_pct=0.20, frame_tolerance=2,
                 empty_weight=1.5, pos_weight=5.0, focal_gamma=2.0):
        super().__init__()
        self.good_pct = good_pct
        self.fail_pct = fail_pct
        self.frame_tolerance = frame_tolerance
        self.empty_weight = empty_weight
        self.pos_weight = pos_weight  # upweight positive bins (onsets are sparse)
        self.focal_gamma = focal_gamma
        self.log_good = math.log(1 + good_pct)
        self.log_fail = math.log(1 + fail_pct)

    def _make_sigmoid_targets(self, targets_padded, n_targets, n_classes):
        """Build per-bin soft targets for sigmoid BCE.

        For each real onset, bins within good_pct get target=1.0, bins within
        fail_pct get linearly interpolated targets, bins outside get target=0.0.
        Multiple onsets: take max across all onsets per bin.

        Returns: (B, n_classes) targets in [0, 1]
        """
        B, M = targets_padded.shape
        stop = n_classes - 1
        device = targets_padded.device

        bins = torch.arange(stop, device=device, dtype=torch.float32)  # (500,)
        targets = torch.zeros(B, n_classes, device=device)

        valid_mask = targets_padded >= 0  # (B, M)

        if valid_mask.any():
            t_float = targets_padded.float().clamp(min=0)  # (B, M)

            # log-ratio distance: |log((bin+1)/(target+1))|
            abs_log_ratio = torch.abs(
                torch.log((bins + 1).view(1, 1, -1) / (t_float + 1).unsqueeze(-1))
            )  # (B, M, 500)

            # trapezoid: 1.0 within good_pct, linear ramp to 0.0 at fail_pct
            ramp_width = self.log_fail - self.log_good
            ratio_weights = ((self.log_fail - abs_log_ratio) / ramp_width).clamp(0, 1)

            # frame distance floor for small targets
            frame_dist = torch.abs(bins.view(1, 1, -1) - t_float.unsqueeze(-1))
            frame_weights = ((self.frame_tolerance + 1 - frame_dist) /
                             (self.frame_tolerance + 1)).clamp(0, 1)

            weights = torch.max(ratio_weights, frame_weights)  # (B, M, 500)
            weights = weights * valid_mask.unsqueeze(-1).float()

            # max across onsets (not sum — sigmoid targets should be in [0,1])
            targets[:, :stop] = weights.max(dim=1).values  # (B, 500)

        # STOP bin: 1.0 if no onsets
        empty = (n_targets == 0)
        targets[empty, stop] = 1.0

        return targets

    def forward(self, logits, targets_padded, n_targets):
        """
        logits: (B, 501) raw logits (sigmoid applied internally)
        targets_padded: (B, MAX_TARGETS) int64, -1 = padding
        n_targets: (B,) int64
        """
        n_classes = logits.size(1)

        # build per-bin soft targets
        bin_targets = self._make_sigmoid_targets(targets_padded, n_targets, n_classes)

        # weighted BCE: upweight positive bins since onsets are sparse
        # (most of 500 bins are negative for any given sample)
        pos_w = torch.where(bin_targets > 0.5,
                            torch.full_like(bin_targets, self.pos_weight),
                            torch.ones_like(bin_targets))

        bce = F.binary_cross_entropy_with_logits(
            logits, bin_targets, weight=pos_w, reduction="none"
        )  # (B, 501)

        # focal modulation: downweight easy negatives, focus on hard cases
        if self.focal_gamma > 0:
            p = torch.sigmoid(logits)
            # pt = p for positive targets, 1-p for negative targets
            pt = bin_targets * p + (1 - bin_targets) * (1 - p)
            focal_weight = (1 - pt) ** self.focal_gamma
            bce = bce * focal_weight

        # extra weight on STOP bin for empty windows
        if self.empty_weight > 1.0:
            empty = (n_targets == 0)
            bce[:, -1] = bce[:, -1] * torch.where(empty, self.empty_weight, 1.0)

        return bce.mean()


class FocalDiceMultiTargetLoss(nn.Module):
    """Focal dice loss for multi-target onset detection.

    Dice loss measures set overlap between predicted and target onset distributions.
    Unlike BCE, it naturally handles class imbalance — predicting everything gives
    low dice (huge denominator), predicting nothing gives zero dice (zero numerator).

    Loss = 1 - (2 * sum(pred * target) + smooth) / (sum(pred) + sum(target) + smooth)

    Combined with a small BCE component for stable gradients at onset positions.
    """

    def __init__(self, good_pct=0.03, fail_pct=0.20, frame_tolerance=2,
                 empty_weight=1.5, bce_weight=0.1, dice_weight=1.0):
        super().__init__()
        self.good_pct = good_pct
        self.fail_pct = fail_pct
        self.frame_tolerance = frame_tolerance
        self.empty_weight = empty_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.log_good = math.log(1 + good_pct)
        self.log_fail = math.log(1 + fail_pct)

    def _make_sigmoid_targets(self, targets_padded, n_targets, n_classes):
        """Same soft trapezoid targets as SigmoidMultiTargetLoss."""
        B, M = targets_padded.shape
        stop = n_classes - 1
        device = targets_padded.device

        bins = torch.arange(stop, device=device, dtype=torch.float32)
        targets = torch.zeros(B, n_classes, device=device)

        valid_mask = targets_padded >= 0

        if valid_mask.any():
            t_float = targets_padded.float().clamp(min=0)

            abs_log_ratio = torch.abs(
                torch.log((bins + 1).view(1, 1, -1) / (t_float + 1).unsqueeze(-1))
            )

            ramp_width = self.log_fail - self.log_good
            ratio_weights = ((self.log_fail - abs_log_ratio) / ramp_width).clamp(0, 1)

            frame_dist = torch.abs(bins.view(1, 1, -1) - t_float.unsqueeze(-1))
            frame_weights = ((self.frame_tolerance + 1 - frame_dist) /
                             (self.frame_tolerance + 1)).clamp(0, 1)

            weights = torch.max(ratio_weights, frame_weights)
            weights = weights * valid_mask.unsqueeze(-1).float()

            targets[:, :stop] = weights.max(dim=1).values

        empty = (n_targets == 0)
        targets[empty, stop] = 1.0

        return targets

    def forward(self, logits, targets_padded, n_targets):
        """
        logits: (B, 501) raw logits
        targets_padded: (B, MAX_TARGETS) int64, -1 = padding
        n_targets: (B,) int64
        """
        n_classes = logits.size(1)
        bin_targets = self._make_sigmoid_targets(targets_padded, n_targets, n_classes)
        pred = torch.sigmoid(logits)

        # dice loss per sample
        smooth = 1.0
        intersection = (pred * bin_targets).sum(dim=1)
        union = pred.sum(dim=1) + bin_targets.sum(dim=1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice  # (B,)

        # small BCE component for stable per-bin gradients
        bce = F.binary_cross_entropy_with_logits(
            logits, bin_targets, reduction="none"
        ).mean(dim=1)  # (B,)

        # extra weight on empty windows
        if self.empty_weight > 1.0:
            empty = (n_targets == 0)
            weight = torch.where(empty, self.empty_weight, 1.0)
            dice_loss = dice_loss * weight
            bce = bce * weight

        loss = self.dice_weight * dice_loss + self.bce_weight * bce
        return loss.mean()


# ═══════════════════════════════════════════════════════════════
#  Metrics & Graphs
# ═══════════════════════════════════════════════════════════════

def print_class_distribution(dataset):
    """Print class distribution stats."""
    counts = torch.from_numpy(dataset.class_counts).float()
    total = counts.sum().item()
    nonzero = (counts > 0).sum().item()

    print(f"\nClass distribution ({total:.0f} total samples):")
    print(f"  Non-empty classes: {nonzero}/{N_CLASSES}")
    print(f"  STOP class (500): {counts[-1]:.0f} ({counts[-1]/total*100:.1f}%)")
    for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 500)]:
        c = counts[lo:hi].sum().item()
        print(f"  Offset {lo:>3}-{hi:>3}: {c:>10.0f} ({c/total*100:5.1f}%)")
    top_k = torch.topk(counts[:500], 10)
    print(f"  Top 10 offsets: {', '.join(f'{i}={int(c)}' for i, c in zip(top_k.indices.tolist(), top_k.values.tolist()))}\n")


def compute_class_weights(dataset, mode="log"):
    """Compute per-class loss weights (used when balanced sampling is off)."""
    counts = torch.from_numpy(dataset.class_counts).float()

    if mode == "log":
        max_count = counts.max()
        weights = torch.log(max_count / (counts + 1.0) + 1.0)
        weights[counts == 0] = 0.0
    elif mode == "sqrt":
        weights = 1.0 / torch.sqrt(counts + 1.0)
    elif mode == "none":
        weights = torch.ones_like(counts)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    weights = weights / weights[weights > 0].mean()
    print(f"  Loss weight mode: {mode}, range: [{weights.min():.4f}, {weights.max():.4f}]")
    return weights


def _top3_gap_peaks(gaps, tolerance=0.05):
    """Find top 3 gap peaks from an array of gaps, merging within tolerance.

    Returns array of 3 values (NaN-padded if fewer than 3 peaks).
    Peaks are sorted by frequency (most common first).
    """
    result = np.full(3, np.nan, dtype=np.float64)
    if len(gaps) == 0:
        return result

    # sort gaps and cluster within tolerance
    sorted_gaps = np.sort(gaps)
    clusters = []  # list of (centroid, count)
    cluster_vals = [sorted_gaps[0]]
    for i in range(1, len(sorted_gaps)):
        centroid = np.mean(cluster_vals)
        if centroid > 0 and abs(sorted_gaps[i] - centroid) / centroid <= tolerance:
            cluster_vals.append(sorted_gaps[i])
        else:
            clusters.append((np.mean(cluster_vals), len(cluster_vals)))
            cluster_vals = [sorted_gaps[i]]
    clusters.append((np.mean(cluster_vals), len(cluster_vals)))

    # sort by count descending
    clusters.sort(key=lambda x: x[1], reverse=True)

    # pick top 3, skipping peaks too close to already-picked ones
    picked = []
    for centroid, count in clusters:
        if len(picked) >= 3:
            break
        too_close = False
        for p in picked:
            if p > 0 and abs(centroid - p) / p <= tolerance:
                too_close = True
                break
        if not too_close:
            picked.append(centroid)

    for i, p in enumerate(picked):
        result[i] = p
    return result


@torch.no_grad()
def validate_and_collect(model, loader, criterion, device, amp_enabled=False, sigmoid_mode=False, framewise=False,
                         multi_target=False):
    """Single pass: compute val loss AND collect predictions + extra data for graphs."""
    model.eval()
    all_targets = []       # single-target: (N,), multi-target: not used
    all_preds = []
    all_cond = []
    all_prev_gap = []
    all_top3_gaps = []
    all_ctx_len = []
    all_topk = []          # top-10 predictions per sample (legacy mode only)
    all_entropy = []       # logit entropy per sample
    all_probs = []         # full softmax (multi-target only)
    all_targets_padded = []  # (N, MAX_TARGETS) (multi-target only)
    all_n_targets = []     # (N,) (multi-target only)
    total_loss = 0.0
    total_n = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        if multi_target:
            mel, evt_off, evt_mask, cond, targets_padded, n_tgt = batch
            targets_padded = targets_padded.to(device, non_blocking=True)
            n_tgt = n_tgt.to(device, non_blocking=True)
        else:
            mel, evt_off, evt_mask, cond, target = batch
            target = target.to(device, non_blocking=True)

        mel = mel.to(device, non_blocking=True)
        evt_off = evt_off.to(device, non_blocking=True)
        evt_mask = evt_mask.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=amp_enabled):
            if framewise:
                # framewise: model returns (B, 125) onset probs
                onset_probs = model(mel, evt_off, evt_mask, cond)
                # build target for loss
                B_fw = onset_probs.size(0)
                fw_target = torch.zeros(B_fw, 125, device=device)
                safe_bins = targets_padded.clamp(min=0)
                token_idx = (safe_bins // 4).clamp(max=124)
                valid_mask_fw = targets_padded >= 0
                idx = token_idx * valid_mask_fw.long()
                fw_target.scatter_(1, idx, valid_mask_fw.float())
                # BCE (same as training — no pos_weight)
                loss = F.binary_cross_entropy(onset_probs, fw_target)
                # create compat logits for legacy metrics
                logits = torch.zeros(B_fw, N_CLASSES, device=device)
                logits[:, :500:4] = onset_probs * 10
            else:
                output = model(mel, evt_off, evt_mask, cond)
                if isinstance(output, tuple):
                    onset_logits, stop_logit = output
                    is_stop = (target == N_CLASSES - 1)
                    stop_target = is_stop.float()
                    stop_bce = F.binary_cross_entropy_with_logits(
                        stop_logit, stop_target, reduction='none')
                    if is_stop.any():
                        stop_loss_pos = stop_bce[is_stop].mean()
                    else:
                        stop_loss_pos = torch.tensor(0.0, device=device)
                    stop_loss_neg = stop_bce[~is_stop].mean()
                    stop_loss = (stop_loss_pos + stop_loss_neg) / 2.0
                    onset_mask = ~is_stop
                    if onset_mask.any():
                        onset_loss = criterion(onset_logits[onset_mask], target[onset_mask])
                    else:
                        onset_loss = torch.tensor(0.0, device=device)
                    loss = onset_loss + 1.5 * stop_loss
                    logits = F.pad(onset_logits, (0, 1), value=-10.0)
                    logits[:, N_CLASSES - 1] = stop_logit * 5.0
                else:
                    logits = output
                    if multi_target:
                        loss = criterion(logits, targets_padded, n_tgt)
                    else:
                        loss = criterion(logits, target)

        B = mel.size(0)
        total_loss += loss.item() * B
        total_n += B

        if sigmoid_mode or framewise:
            probs = torch.sigmoid(logits.float()) if not framewise else onset_probs
        else:
            probs = torch.softmax(logits.float(), dim=-1)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_cond.append(cond.cpu().numpy())

        # entropy
        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)
        all_entropy.append(ent.cpu().numpy())

        if multi_target:
            all_probs.append(probs.cpu().numpy())
            all_targets_padded.append(targets_padded.cpu().numpy())
            all_n_targets.append(n_tgt.cpu().numpy())
            # nearest target for backward-compat metrics
            nearest = targets_padded[:, 0].clone()
            nearest[n_tgt == 0] = N_CLASSES - 1
            nearest = nearest.clamp(0, N_CLASSES - 1)
            all_targets.append(nearest.cpu().numpy())
        else:
            all_targets.append(target.cpu().numpy())
            all_topk.append(logits.topk(10, dim=1).indices.cpu().numpy())

        # context length + prev_gap from event mask
        em = evt_mask.cpu().numpy()
        ctx_lens = (~em).sum(axis=1)
        all_ctx_len.append(ctx_lens)

        eo = evt_off.cpu().numpy()
        prev_gaps = np.full(B, np.nan, dtype=np.float64)
        top3_gaps = np.full((B, 3), np.nan, dtype=np.float64)
        for b in range(B):
            valid = ~em[b]
            if valid.sum() >= 2:
                valid_offsets = eo[b][valid]
                prev_gaps[b] = abs(int(valid_offsets[-2]))
                # compute all gaps between consecutive context events
                gaps = np.abs(np.diff(valid_offsets)).astype(np.float64)
                gaps = gaps[gaps > 0]
                if len(gaps) >= 1:
                    top3_gaps[b] = _top3_gap_peaks(gaps)
        all_prev_gap.append(prev_gaps)
        all_top3_gaps.append(top3_gaps)

    val_loss = total_loss / total_n
    extra = {
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "conds": np.concatenate(all_cond),
        "prev_gaps": np.concatenate(all_prev_gap),
        "top3_gaps": np.concatenate(all_top3_gaps, axis=0),
        "ctx_len": np.concatenate(all_ctx_len),
        "entropy": np.concatenate(all_entropy),
    }
    if multi_target:
        extra["probs"] = np.concatenate(all_probs)
        extra["targets_padded"] = np.concatenate(all_targets_padded)
        extra["n_targets"] = np.concatenate(all_n_targets)
    else:
        extra["topk"] = np.concatenate(all_topk)
    return val_loss, extra


def compute_metrics(targets, preds):
    """Compute all metrics from target/pred arrays."""
    m = {}

    # accuracy
    m["accuracy"] = (targets == preds).mean().item()
    m["unique_preds"] = int(len(np.unique(preds)))

    # per-class prediction counts (how often each class is predicted)
    pred_counts = {}
    vals, counts = np.unique(preds, return_counts=True)
    for v, c in zip(vals, counts):
        pred_counts[int(v)] = int(c)
    m["pred_class_counts"] = pred_counts

    # STOP class (500) F1/precision/recall
    stop = N_CLASSES - 1
    tp = ((preds == stop) & (targets == stop)).sum()
    fp = ((preds == stop) & (targets != stop)).sum()
    fn = ((preds != stop) & (targets == stop)).sum()
    m["stop_precision"] = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
    m["stop_recall"] = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
    m["stop_f1"] = (2 * tp / (2 * tp + fp + fn)).item() if (2 * tp + fp + fn) > 0 else 0.0

    # non-stop samples only for frame/relative error
    ns = targets < stop
    if ns.sum() > 0:
        t_ns = targets[ns].astype(np.float64)
        p_ns = preds[ns].astype(np.float64)

        # frame error: |pred - target|
        frame_err = np.abs(p_ns - t_ns)
        m["frame_error_mean"] = frame_err.mean().item()
        m["frame_error_median"] = np.median(frame_err).item()
        m["frame_error_p90"] = np.percentile(frame_err, 90).item()
        m["frame_error_p99"] = np.percentile(frame_err, 99).item()

        # ratio: (pred+1)/(target+1), 1.0 = perfect
        ratio = (p_ns + 1) / (t_ns + 1)
        pct_err = np.abs(ratio - 1.0)  # 0.0 = perfect, 0.03 = 3% off

        # relative error: log-ratio
        log_ratio = np.log(p_ns + 1) - np.log(t_ns + 1)
        m["rel_error_mean"] = np.abs(log_ratio).mean().item()
        m["rel_error_median"] = np.median(np.abs(log_ratio)).item()
        m["rel_error_std"] = log_ratio.std().item()
        m["ratio_mean"] = ratio.mean().item()
        m["ratio_median"] = np.median(ratio).item()

        # ── primary metrics: HIT / GOOD / MISS rates ──
        # HIT: within 3% ratio OR within 1 frame (either qualifies)
        hit = (pct_err <= 0.03) | (frame_err <= 1)
        good = (pct_err <= 0.10) | (frame_err <= 2)
        miss = pct_err > 0.20

        m["hit_rate"] = hit.mean().item()       # within 3% or ±1 frame
        m["good_rate"] = good.mean().item()     # within 10% or ±2 frames
        m["miss_rate"] = miss.mean().item()     # above 20% off

        # ── frame accuracy tiers ──
        m["exact_match"] = (frame_err == 0).mean().item()
        m["within_1_frame"] = (frame_err <= 1).mean().item()
        m["within_2_frames"] = (frame_err <= 2).mean().item()
        m["within_4_frames"] = (frame_err <= 4).mean().item()

        # ── ratio accuracy tiers ──
        m["within_3pct"] = (pct_err <= 0.03).mean().item()
        m["within_10pct"] = (pct_err <= 0.10).mean().item()
        m["above_20pct"] = (pct_err > 0.20).mean().item()

        # ── model score: continuous quality metric [-1, +1] ──
        # Continuous function of log-ratio error:
        #   0% error → +1.0 (max reward)
        #   3% error → 0.0 (neutral)
        #   200% error → -1.0 (max penalty = magnitude of max reward)
        #   >200% → capped at -1.0
        #   ±1 frame → always +1.0 (for small targets)
        abs_lr = np.abs(log_ratio)
        threshold = np.log(1.03)   # 3% = neutral point
        max_pen = np.log(5.0)      # 400% = -1.0
        pen_range = max_pen - threshold
        # Reward at 0% = magnitude of penalty at 200%
        reward_at_zero = (np.log(3.0) - threshold) / pen_range

        scores = np.where(
            abs_lr <= threshold,
            (1.0 - abs_lr / threshold) * reward_at_zero,
            -np.minimum((abs_lr - threshold) / pen_range, 1.0),
        )
        # ±1 frame always gets max reward (small targets)
        scores[frame_err <= 1] = reward_at_zero
        m["model_score"] = scores.mean().item()
    else:
        m["frame_error_mean"] = 0.0
        m["hit_rate"] = 0.0
        m["miss_rate"] = 1.0
        m["model_score"] = -1.0

    return m


def _classify_match(r_bin, p_bin):
    """Classify a matched (real, pred) pair as HIT/GOOD/MISS."""
    frame_err = abs(p_bin - r_bin)
    pct_err = abs((p_bin + 1) / (r_bin + 1) - 1.0)
    is_hit = pct_err <= 0.03 or frame_err <= 1
    is_good = pct_err <= 0.10 or frame_err <= 2
    is_miss = pct_err > 0.20
    return is_hit, is_good, is_miss, frame_err, pct_err


def compute_multi_target_metrics(targets_padded, n_targets, probs, threshold=0.05):
    """Bidirectional matching metrics between real onsets and thresholded predictions.

    For each sample: extract predicted peaks above threshold, match greedily to real
    onsets (one-to-one, closest first). Compute event-side (recall) and prediction-side
    (precision) HIT/GOOD/MISS rates.

    Returns dict with metrics + internal arrays for graphs (prefixed with _).
    """
    N = len(n_targets)
    stop = N_CLASSES - 1

    # aggregate counters
    total_real = 0
    total_pred = 0
    event_hit = 0
    event_good = 0
    event_matched = 0
    pred_hit = 0
    pred_good = 0
    pred_matched = 0

    # for graphs: matched pairs (501 = unmatched sentinel)
    all_match_real = []
    all_match_pred = []
    all_match_conf = []
    all_match_rank = []  # peak rank within window (0-based)
    all_match_frame_err = []

    for i in range(N):
        nt = int(n_targets[i])
        real_bins = targets_padded[i, :nt].astype(np.float64) if nt > 0 else np.array([], dtype=np.float64)
        p = probs[i, :stop]
        pred_indices = np.where(p >= threshold)[0]
        # sort by confidence descending for rank assignment
        pred_confs = p[pred_indices]
        rank_order = np.argsort(-pred_confs)
        pred_indices_ranked = pred_indices[rank_order]
        pred_confs_ranked = pred_confs[rank_order]

        n_real = len(real_bins)
        n_pred = len(pred_indices_ranked)
        total_real += n_real
        total_pred += n_pred

        if n_real == 0 and n_pred == 0:
            continue

        # unmatched predictions (hallucinations) when no real onsets
        if n_real == 0:
            for pi in range(n_pred):
                all_match_real.append(stop + 1)  # 501 sentinel
                all_match_pred.append(float(pred_indices_ranked[pi]))
                all_match_conf.append(float(pred_confs_ranked[pi]))
                all_match_rank.append(pi)
                all_match_frame_err.append(np.nan)
            continue

        # unmatched real onsets when no predictions
        if n_pred == 0:
            for ri in range(n_real):
                all_match_real.append(float(real_bins[ri]))
                all_match_pred.append(stop + 1)  # 501 sentinel
                all_match_conf.append(0.0)
                all_match_rank.append(-1)
                all_match_frame_err.append(np.nan)
            continue

        # greedy nearest-neighbor matching (one-to-one)
        pred_bins_f = pred_indices_ranked.astype(np.float64)
        dist = np.abs(real_bins.reshape(-1, 1) - pred_bins_f.reshape(1, -1))
        used_real = np.zeros(n_real, dtype=bool)
        used_pred = np.zeros(n_pred, dtype=bool)

        flat_order = np.argsort(dist.ravel())
        for flat_idx in flat_order:
            ri = flat_idx // n_pred
            pi = flat_idx % n_pred
            if used_real[ri] or used_pred[pi]:
                continue
            used_real[ri] = True
            used_pred[pi] = True

            r_bin = float(real_bins[ri])
            p_bin = float(pred_bins_f[pi])
            is_hit, is_good, is_miss, fe, pe = _classify_match(r_bin, p_bin)

            event_matched += 1
            pred_matched += 1
            if is_hit:
                event_hit += 1
                pred_hit += 1
            if is_good:
                event_good += 1
                pred_good += 1

            all_match_real.append(r_bin)
            all_match_pred.append(p_bin)
            all_match_conf.append(float(pred_confs_ranked[pi]))
            all_match_rank.append(int(pi))  # rank by confidence
            all_match_frame_err.append(fe)

            if used_real.sum() == n_real or used_pred.sum() == n_pred:
                break

        # unmatched real → event misses
        for ri in range(n_real):
            if not used_real[ri]:
                all_match_real.append(float(real_bins[ri]))
                all_match_pred.append(stop + 1)
                all_match_conf.append(0.0)
                all_match_rank.append(-1)
                all_match_frame_err.append(np.nan)

        # unmatched pred → hallucinations
        for pi in range(n_pred):
            if not used_pred[pi]:
                all_match_real.append(stop + 1)
                all_match_pred.append(float(pred_bins_f[pi]))
                all_match_conf.append(float(pred_confs_ranked[pi]))
                all_match_rank.append(int(pi))
                all_match_frame_err.append(np.nan)

    m = {}
    m["threshold"] = threshold
    m["total_real_onsets"] = total_real
    m["total_predictions"] = total_pred

    # event-side (recall)
    m["event_recall_hit"] = event_hit / max(total_real, 1)
    m["event_recall_good"] = event_good / max(total_real, 1)
    m["event_recall_matched"] = event_matched / max(total_real, 1)
    m["event_miss_rate"] = 1.0 - event_matched / max(total_real, 1)

    # prediction-side (precision)
    m["pred_precision_hit"] = pred_hit / max(total_pred, 1)
    m["pred_precision_good"] = pred_good / max(total_pred, 1)
    m["hallucination_rate"] = 1.0 - pred_matched / max(total_pred, 1)

    # F1
    rh = m["event_recall_hit"]
    ph = m["pred_precision_hit"]
    m["f1_hit"] = 2 * rh * ph / (rh + ph) if (rh + ph) > 0 else 0.0

    m["avg_preds_per_window"] = total_pred / max(N, 1)
    m["avg_real_per_window"] = total_real / max(N, 1)

    # arrays for graphs (not JSON-serializable — strip before saving)
    m["_matched_real"] = np.array(all_match_real)
    m["_matched_pred"] = np.array(all_match_pred)
    m["_matched_conf"] = np.array(all_match_conf)
    m["_matched_rank"] = np.array(all_match_rank)
    m["_matched_frame_err"] = np.array(all_match_frame_err)

    return m


def _fast_threshold_metrics(targets_padded, n_targets, probs, threshold):
    """Fast aggregate-only multi-target metrics (no per-match details).

    Uses vectorized ops where possible, falling back to minimal per-sample
    loop only for greedy matching.
    """
    N = len(n_targets)
    stop = N_CLASSES - 1
    p_onset = probs[:, :stop]  # (N, 500)

    # predictions above threshold per sample
    above = p_onset >= threshold  # (N, 500)
    n_preds = above.sum(axis=1)  # (N,)
    total_pred = int(n_preds.sum())
    total_real = int(n_targets.sum())

    if total_real == 0 or total_pred == 0:
        return {
            "event_recall_hit": 0.0, "pred_precision_hit": 0.0,
            "f1_hit": 0.0, "avg_preds_per_window": total_pred / max(N, 1),
        }

    event_hit = 0
    pred_hit = 0
    event_matched = 0
    pred_matched = 0

    for i in range(N):
        nt = int(n_targets[i])
        if nt == 0 and n_preds[i] == 0:
            continue

        pred_idx = np.where(above[i])[0]
        np_i = len(pred_idx)

        if nt == 0 or np_i == 0:
            continue

        real_bins = targets_padded[i, :nt].astype(np.float64)
        pred_bins = pred_idx.astype(np.float64)

        # greedy matching: closest pairs first
        dist = np.abs(real_bins.reshape(-1, 1) - pred_bins.reshape(1, -1))
        used_r = np.zeros(nt, dtype=bool)
        used_p = np.zeros(np_i, dtype=bool)

        for flat_idx in np.argsort(dist.ravel()):
            ri = flat_idx // np_i
            pi = flat_idx % np_i
            if used_r[ri] or used_p[pi]:
                continue
            used_r[ri] = True
            used_p[pi] = True

            event_matched += 1
            pred_matched += 1

            r, p = real_bins[ri], pred_bins[pi]
            fe = abs(r - p)
            pe = abs((p + 1) / (r + 1) - 1.0)
            if pe <= 0.03 or fe <= 1:
                event_hit += 1
                pred_hit += 1

            if used_r.sum() == nt or used_p.sum() == np_i:
                break

    rh = event_hit / max(total_real, 1)
    ph = pred_hit / max(total_pred, 1)
    return {
        "event_recall_hit": rh,
        "pred_precision_hit": ph,
        "f1_hit": 2 * rh * ph / (rh + ph) if (rh + ph) > 0 else 0.0,
        "avg_preds_per_window": total_pred / max(N, 1),
    }


def threshold_sweep(targets_padded, n_targets, probs,
                    thresholds=None, subsample=4):
    """Sweep thresholds on a subsample for speed."""
    if thresholds is None:
        thresholds = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40])

    # subsample for speed
    N = len(n_targets)
    if subsample > 1 and N > 1000:
        idx = np.arange(0, N, subsample)
        targets_padded = targets_padded[idx]
        n_targets = n_targets[idx]
        probs = probs[idx]

    results = []
    for t in tqdm(thresholds, desc="Threshold sweep", leave=False):
        m = _fast_threshold_metrics(targets_padded, n_targets, probs, threshold=float(t))
        results.append(m)
    return thresholds, results


# ═══════════════════════════════════════════════════════════════
#  Ablation Benchmarks
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_benchmarks(model, val_loader, device, amp_enabled=False, multi_target=False):
    """Run ablation benchmarks on corrupted validation data.

    Returns dict of benchmark_name -> {stop_rate, mean_pred, pred_std, n_samples}.
    Each benchmark corrupts inputs in a specific way to probe model behavior.
    """
    model.eval()
    rng = np.random.default_rng(42)

    # collect a subset of batches (10% of val set)
    all_batches = []
    total_samples = 0
    target_samples = len(val_loader.dataset) // 10
    for batch in val_loader:
        all_batches.append(batch)
        total_samples += batch[0].size(0)
        if total_samples >= target_samples:
            break

    stop = N_CLASSES - 1

    def run_corrupted(batches, corrupt_fn, name):
        """Run model on corrupted batches, return full arrays + summary stats."""
        all_preds = []
        all_targets = []
        for batch in batches:
            if multi_target:
                mel, evt_off, evt_mask, cond, targets_padded, n_tgt = batch
                # nearest target for benchmark metrics
                target = targets_padded[:, 0].clone()
                target[n_tgt == 0] = stop
                target = target.clamp(0, stop)
            else:
                mel, evt_off, evt_mask, cond, target = batch

            mel, evt_off, evt_mask, cond = corrupt_fn(
                mel.clone(), evt_off.clone(), evt_mask.clone(), cond.clone(), target
            )
            mel = mel.to(device, non_blocking=True)
            evt_off = evt_off.to(device, non_blocking=True)
            evt_mask = evt_mask.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp_enabled):
                output = model(mel, evt_off, evt_mask, cond)
                if isinstance(output, tuple):
                    onset_logits, stop_logit = output
                    # pad to 501 with stop logit
                    logits = F.pad(onset_logits, (0, 1), value=-10.0)
                    logits[:, N_CLASSES - 1] = stop_logit * 5.0
                else:
                    logits = output
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_targets.append(target.numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        non_stop_preds = preds[preds != stop]
        return {
            "preds": preds,
            "targets": targets,
            "n_samples": len(preds),
            "stop_rate": float((preds == stop).mean()),
            "mean_pred": float(non_stop_preds.mean()) if len(non_stop_preds) > 0 else 0.0,
            "pred_std": float(non_stop_preds.std()) if len(non_stop_preds) > 0 else 0.0,
            "unique_preds": int(len(np.unique(preds))),
            "accuracy": float((preds == targets).mean()),
        }

    results = {}
    bench_bar = tqdm(total=12, desc="Benchmarks", leave=False)

    # 1) No events - all past context deleted
    def no_events(mel, evt_off, evt_mask, cond, target):
        evt_off.zero_()
        evt_mask.fill_(True)  # all masked = no events
        return mel, evt_off, evt_mask, cond
    results["no_events"] = run_corrupted(all_batches, no_events, "no_events")
    bench_bar.set_postfix_str("no_events"); bench_bar.update(1)

    # 2) No audio - silent spectrogram
    def no_audio(mel, evt_off, evt_mask, cond, target):
        mel.zero_()
        return mel, evt_off, evt_mask, cond
    results["no_audio"] = run_corrupted(all_batches, no_audio, "no_audio")
    bench_bar.set_postfix_str("no_audio"); bench_bar.update(1)

    # 3) Random events - completely random timestamps
    def random_events(mel, evt_off, evt_mask, cond, target):
        B, C = evt_off.shape
        for b in range(B):
            n_events = rng.integers(4, C)
            offsets = np.sort(rng.integers(-A_BINS, 0, size=n_events))
            evt_off[b].zero_()
            evt_mask[b].fill_(True)
            evt_off[b, -n_events:] = torch.from_numpy(offsets)
            evt_mask[b, -n_events:] = False
        return mel, evt_off, evt_mask, cond
    results["random_events"] = run_corrupted(all_batches, random_events, "random_events")
    bench_bar.set_postfix_str("random_events"); bench_bar.update(1)

    # 4) Static audio - white noise spectrogram
    def static_audio(mel, evt_off, evt_mask, cond, target):
        mel.normal_(mean=mel.mean().item(), std=mel.std().item())
        return mel, evt_off, evt_mask, cond
    results["static_audio"] = run_corrupted(all_batches, static_audio, "static_audio")
    bench_bar.set_postfix_str("static_audio"); bench_bar.update(1)

    # 5) No events + muted audio - everything zeroed
    def no_events_no_audio(mel, evt_off, evt_mask, cond, target):
        mel.zero_()
        evt_off.zero_()
        evt_mask.fill_(True)
        return mel, evt_off, evt_mask, cond
    results["no_events_no_audio"] = run_corrupted(all_batches, no_events_no_audio, "no_events_no_audio")
    bench_bar.set_postfix_str("no_events_no_audio"); bench_bar.update(1)

    # 6) Metronome - fill events with fixed gap far from target
    def metronome(mel, evt_off, evt_mask, cond, target):
        B, C = evt_off.shape
        for b in range(B):
            t = target[b].item()
            # pick a gap that's far from the actual target (2x or 0.5x, whichever is farther)
            if t > 0 and t < stop:
                gap = t * 2 if t < 100 else max(5, t // 3)
                # ensure gap != target
                if abs(gap - t) < 3:
                    gap = max(5, t * 3)
            else:
                gap = rng.integers(10, 80)
            # fill C events backwards from cursor (offset 0)
            offsets = np.array([-gap * (i + 1) for i in range(C)], dtype=np.int64)[::-1]
            offsets = np.clip(offsets, -A_BINS, -1)
            evt_off[b] = torch.from_numpy(offsets.copy())
            evt_mask[b] = False  # all valid
        return mel, evt_off, evt_mask, cond
    results["metronome"] = run_corrupted(all_batches, metronome, "metronome")
    bench_bar.set_postfix_str("metronome"); bench_bar.update(1)

    # 7) Time-shifted context - scale all event offsets by a musical fraction
    fractions = [0.5, 2.0, 1.0 / 3, 0.75, 5.0 / 3]
    def time_shifted(mel, evt_off, evt_mask, cond, target):
        B, C = evt_off.shape
        for b in range(B):
            frac = fractions[rng.integers(0, len(fractions))]
            valid = ~evt_mask[b].bool()
            if valid.any():
                scaled = (evt_off[b].float() * frac).long()
                scaled = scaled.clamp(-A_BINS, -1)
                evt_off[b] = scaled
        return mel, evt_off, evt_mask, cond
    results["time_shifted"] = run_corrupted(all_batches, time_shifted, "time_shifted")
    bench_bar.set_postfix_str("time_shifted"); bench_bar.update(1)

    # 8) Advanced metronome - quantize events to detected BPM
    def advanced_metronome(mel, evt_off, evt_mask, cond, target):
        B, C = evt_off.shape
        for b in range(B):
            valid = ~evt_mask[b].bool()
            n_valid = valid.sum().item()
            if n_valid < 3:
                continue
            offsets = evt_off[b][valid].numpy().astype(np.float64)
            # compute gaps between consecutive events
            gaps = np.diff(offsets)
            gaps = gaps[gaps > 0]
            if len(gaps) < 2:
                continue
            # find dominant gap via histogram (BPM detection)
            gap_min, gap_max = max(3, int(gaps.min())), min(200, int(gaps.max()) + 1)
            if gap_min >= gap_max:
                continue
            hist, edges = np.histogram(gaps, bins=max(1, gap_max - gap_min),
                                       range=(gap_min, gap_max))
            dominant_gap = int(edges[hist.argmax()] + 0.5 * (edges[1] - edges[0]))
            if dominant_gap < 3:
                dominant_gap = 3

            # skip if target is already 1/1 with the dominant gap
            t = target[b].item()
            if t < stop and abs(t - dominant_gap) <= 2:
                continue

            # rebuild events quantized to dominant_gap
            last_offset = int(offsets[-1])  # most recent event offset
            new_offsets = np.array(
                [last_offset - dominant_gap * (i) for i in range(C)],
                dtype=np.int64,
            )[::-1]
            new_offsets = np.clip(new_offsets, -A_BINS, -1)
            evt_off[b].zero_()
            evt_mask[b].fill_(True)
            # fill from the right
            n_fill = min(C, len(new_offsets))
            evt_off[b, -n_fill:] = torch.from_numpy(new_offsets[-n_fill:].copy())
            evt_mask[b, -n_fill:] = False
        return mel, evt_off, evt_mask, cond
    results["advanced_metronome"] = run_corrupted(all_batches, advanced_metronome, "advanced_metronome")
    bench_bar.set_postfix_str("advanced_metronome"); bench_bar.update(1)

    # 9) Zero density - conditioning set to [0, 0, 0]
    def zero_density(mel, evt_off, evt_mask, cond, target):
        cond.zero_()
        return mel, evt_off, evt_mask, cond
    results["zero_density"] = run_corrupted(all_batches, zero_density, "zero_density")
    bench_bar.set_postfix_str("zero_density"); bench_bar.update(1)

    # 10) Random density - conditioning randomized
    def random_density(mel, evt_off, evt_mask, cond, target):
        B = cond.shape[0]
        cond[:, 0] = torch.FloatTensor(B).uniform_(1.0, 12.0)  # mean density
        cond[:, 1] = torch.FloatTensor(B).uniform_(2.0, 20.0)   # peak density
        cond[:, 2] = torch.FloatTensor(B).uniform_(0.5, 4.0)    # density std
        return mel, evt_off, evt_mask, cond
    results["random_density"] = run_corrupted(all_batches, random_density, "random_density")
    bench_bar.set_postfix_str("random_density"); bench_bar.update(1)

    # ── 11) Autoregressive benchmarks ──
    # Run like real inference: predict, move cursor, feed prediction back.
    # Compare predicted onset positions against ground truth onset positions.
    AR_STEPS = 32
    AR_MAX_SAMPLES = 1000

    def _is_hit_val(pred, target):
        if target >= N_CLASSES - 1:
            return pred == target
        fe = abs(pred - target)
        pe = abs((pred + 1) / (target + 1) - 1.0)
        return pe <= 0.03 or fe <= 1

    def _run_ar_inference(batches):
        """Run AR inference loop on val samples.

        For each sample: run 32 predictions, feeding each back as context.
        Collect all predicted absolute positions, then match against all
        ground truth positions using greedy matching.

        Ground truth comes from the dataset's raw event arrays — we look up
        ALL future onsets from each sample's cursor position, not just the
        single target from the batch.

        Returns two result dicts: 'autoregress' (set matching) and
        'lightautoregress' (1:1 index matching).
        """
        # access the underlying dataset for ground truth events
        dataset = val_loader.dataset

        # collect samples: use batch data for model input, dataset for GT
        samples = []
        sample_idx = 0
        for batch in batches:
            if multi_target:
                mel, evt_off, evt_mask, cond, targets_padded, n_tgt = batch
            else:
                mel, evt_off, evt_mask, cond, target = batch

            B = mel.size(0)
            for b in range(B):
                if sample_idx >= len(dataset.samples):
                    break
                valid = ~evt_mask[b]
                if valid.sum() < 4:
                    sample_idx += 1
                    continue

                # get ground truth: all future onsets from cursor
                ci, ei = dataset.samples[sample_idx]
                evt = dataset.events[ci]
                # cursor position
                if ei == 0:
                    cursor_bin = max(0, int(evt[0]) - B_BINS) if len(evt) > 0 else 0
                else:
                    cursor_bin = int(evt[ei - 1])
                # all onsets AFTER cursor
                future_bins = evt[evt > cursor_bin] - cursor_bin
                future_bins = future_bins[future_bins < B_BINS].astype(np.int64)

                if len(future_bins) < 2:
                    sample_idx += 1
                    continue

                samples.append({
                    "mel": mel[b],
                    "evt_off": evt_off[b],
                    "evt_mask": evt_mask[b],
                    "cond": cond[b],
                    "gt_abs": future_bins,  # absolute bin offsets from cursor
                    "density_cond": cond[b, 0].item(),
                })
                sample_idx += 1
                if len(samples) >= AR_MAX_SAMPLES:
                    break
            if len(samples) >= AR_MAX_SAMPLES:
                break

        if len(samples) < 5:
            return None, None

        # per-step tracking
        per_step_preds = [[] for _ in range(AR_STEPS)]
        per_step_entropy = [[] for _ in range(AR_STEPS)]
        per_step_survived = np.zeros(AR_STEPS)

        # light AR: per-step HIT against ground truth onset at same index
        light_hit = np.zeros(AR_STEPS)
        light_total = np.zeros(AR_STEPS)
        light_preds = [[] for _ in range(AR_STEPS)]
        light_targets = [[] for _ in range(AR_STEPS)]

        # set matching accumulators
        all_predicted_sets = []  # list of predicted position arrays
        all_gt_sets = []  # list of ground truth position arrays
        density_conds = []
        density_actuals = []

        for sample in tqdm(samples, desc="  AR bench", leave=False):
            mel_s = sample["mel"].unsqueeze(0).to(device)
            evt_off_s = sample["evt_off"].unsqueeze(0).clone().to(device)
            evt_mask_s = sample["evt_mask"].unsqueeze(0).clone().to(device)
            cond_s = sample["cond"].unsqueeze(0).to(device)
            gt_abs = sample["gt_abs"]  # absolute positions of ground truth onsets

            density_conds.append(sample["density_cond"])
            cursor = 0  # absolute cursor position
            predicted_positions = []

            for step in range(AR_STEPS):
                with torch.no_grad(), torch.autocast("cuda", enabled=amp_enabled):
                    output = model(mel_s, evt_off_s, evt_mask_s, cond_s)
                    if isinstance(output, tuple):
                        onset_logits, stop_logit = output
                        logits = F.pad(onset_logits, (0, 1), value=-10.0)
                        logits[:, N_CLASSES - 1] = stop_logit * 5.0
                    else:
                        logits = output
                    probs = torch.softmax(logits.float(), dim=1)
                    pred = logits.argmax(dim=1).item()
                    ent = -(probs * (probs + 1e-10).log()).sum(dim=1).item()

                if pred >= N_CLASSES - 1:  # STOP
                    break

                per_step_survived[step] += 1
                per_step_preds[step].append(pred)
                per_step_entropy[step].append(ent)

                # absolute position of this prediction
                abs_pos = cursor + pred
                predicted_positions.append(abs_pos)

                # light AR: compare to ground truth onset at same index
                if step < len(gt_abs):
                    gt_target = int(gt_abs[step]) - cursor  # expected gap from current cursor
                    if gt_target > 0:
                        light_total[step] += 1
                        light_preds[step].append(pred)
                        light_targets[step].append(gt_target)
                        if _is_hit_val(pred, gt_target):
                            light_hit[step] += 1

                # move cursor and update context
                cursor = abs_pos
                evt_off_s = evt_off_s - pred
                evt_off_np = evt_off_s[0].cpu().numpy()
                evt_mask_np = evt_mask_s[0].cpu().numpy()
                evt_off_np = np.roll(evt_off_np, -1)
                evt_mask_np = np.roll(evt_mask_np, -1)
                evt_off_np[-1] = 0
                evt_mask_np[-1] = False
                evt_off_s = torch.from_numpy(evt_off_np).unsqueeze(0).to(device)
                evt_mask_s = torch.from_numpy(evt_mask_np).unsqueeze(0).to(device)

            all_predicted_sets.append(np.array(predicted_positions))
            all_gt_sets.append(gt_abs)

            # density
            if len(predicted_positions) >= 2:
                total_bins = predicted_positions[-1] - 0  # total span
                total_s = total_bins * 4.989 / 1000
                if total_s > 0:
                    density_actuals.append(len(predicted_positions) / total_s)

        # ── Build autoregress result (set matching) ──
        # greedy match predicted positions to ground truth positions
        total_gt = 0
        total_pred = 0
        event_hit = 0    # GT onsets matched by a HIT prediction
        event_good = 0   # GT onsets matched by a GOOD prediction
        event_matched = 0  # GT onsets matched at all
        pred_hit = 0     # predictions that HIT a GT onset
        pred_good = 0    # predictions within GOOD of a GT onset
        pred_matched = 0  # predictions matched to any GT onset
        hallucinations = 0  # predictions with no GT match

        for pred_set, gt_set in zip(all_predicted_sets, all_gt_sets):
            total_gt += len(gt_set)
            total_pred += len(pred_set)
            if len(pred_set) == 0 or len(gt_set) == 0:
                hallucinations += len(pred_set)
                continue

            # greedy nearest matching
            used_gt = set()
            for p in pred_set:
                best_dist = float("inf")
                best_gi = -1
                for gi, g in enumerate(gt_set):
                    if gi in used_gt:
                        continue
                    d = abs(int(p) - int(g))
                    if d < best_dist:
                        best_dist = d
                        best_gi = gi
                if best_gi >= 0 and best_dist <= 500:
                    used_gt.add(best_gi)
                    pred_matched += 1
                    event_matched += 1
                    is_h = _is_hit_val(int(p), int(gt_set[best_gi]))
                    pct = abs((p + 1) / (gt_set[best_gi] + 1) - 1.0)
                    is_g = pct <= 0.10 or best_dist <= 2
                    if is_h:
                        event_hit += 1
                        pred_hit += 1
                    if is_g:
                        event_good += 1
                        pred_good += 1
                else:
                    hallucinations += 1

        event_miss = total_gt - event_matched
        pred_miss = total_pred - pred_matched  # = hallucinations

        ar_result = {
            "n_samples": len(samples),
            "ar_steps": AR_STEPS,
            "total_gt_onsets": int(total_gt),
            "total_predicted": int(total_pred),
            "event_hit": int(event_hit),
            "event_good": int(event_good),
            "event_miss": int(event_miss),
            "event_hit_rate": float(event_hit / max(total_gt, 1)),
            "event_good_rate": float(event_good / max(total_gt, 1)),
            "event_miss_rate": float(event_miss / max(total_gt, 1)),
            "pred_hit": int(pred_hit),
            "pred_good": int(pred_good),
            "pred_miss": int(pred_miss),
            "pred_hit_rate": float(pred_hit / max(total_pred, 1)),
            "pred_good_rate": float(pred_good / max(total_pred, 1)),
            "hallucination_rate": float(hallucinations / max(total_pred, 1)),
            "pred_per_sample": float(total_pred / max(len(samples), 1)),
            "gt_per_sample": float(total_gt / max(len(samples), 1)),
            "survival_per_step": [int(s) for s in per_step_survived],
            "steps": [],
        }
        if per_step_survived[0] > 0:
            ar_result["survival_10"] = float(per_step_survived[min(9, AR_STEPS-1)] / per_step_survived[0])
            ar_result["survival_30"] = float(per_step_survived[min(29, AR_STEPS-1)] / per_step_survived[0])
        if density_conds and density_actuals:
            ar_result["density_conditioned_mean"] = float(np.mean(density_conds))
            ar_result["density_actual_mean"] = float(np.mean(density_actuals))
            ar_result["density_ratio"] = float(np.mean(density_actuals) / max(np.mean(density_conds), 0.01))

        for s in range(AR_STEPS):
            step_info = {"step": s, "n_alive": int(per_step_survived[s])}
            if per_step_preds[s]:
                arr = np.array(per_step_preds[s])
                step_info["pred_mean"] = float(arr.mean())
                step_info["pred_std"] = float(arr.std())
                step_info["unique_preds"] = int(len(np.unique(arr)))
            if per_step_entropy[s]:
                step_info["entropy_mean"] = float(np.mean(per_step_entropy[s]))
            ar_result["steps"].append(step_info)

        # ── Build lightautoregress result (1:1 index matching) ──
        la_result = {
            "n_samples": len(samples),
            "ar_steps": AR_STEPS,
            "steps": [],
        }
        hit_rates = []
        for s in range(AR_STEPS):
            step_info = {
                "step": s,
                "n_total": int(light_total[s]),
                "hit_rate": float(light_hit[s] / max(light_total[s], 1)),
            }
            if light_preds[s] and light_targets[s]:
                p_arr = np.array(light_preds[s])
                t_arr = np.array(light_targets[s])
                step_info["pred_mean"] = float(p_arr.mean())
                step_info["pred_std"] = float(p_arr.std())
                step_info["target_mean"] = float(t_arr.mean())
                step_info["frame_err_mean"] = float(np.abs(p_arr - t_arr).mean())
                step_info["unique_preds"] = int(len(np.unique(p_arr)))
                step_info["pred_min"] = int(p_arr.min())
                step_info["pred_max"] = int(p_arr.max())
                step_info["_preds"] = p_arr
                step_info["_targets"] = t_arr
            la_result["steps"].append(step_info)
            hit_rates.append(step_info["hit_rate"])

        la_result["hit_curve"] = hit_rates
        la_result["step0_hit"] = float(hit_rates[0]) if hit_rates else 0

        return ar_result, la_result

    ar_result, la_result = _run_ar_inference(all_batches)
    if ar_result:
        results["autoregress"] = ar_result
    bench_bar.set_postfix_str("autoregress"); bench_bar.update(1)
    if la_result:
        results["lightautoregress"] = la_result
    bench_bar.set_postfix_str("lightautoregress"); bench_bar.update(1)

    bench_bar.close()

    return results


def print_benchmarks(results):
    """Print benchmark results in a compact table."""
    print(f"\n  +-- Ablation Benchmarks -----------------------------------------------+")
    print(f"  | {'Test':<24} {'STOP%':>6} {'Acc':>6} {'Mean':>6} {'Std':>6} {'Uniq':>5} |")
    print(f"  +{'-' * 73}+")
    for name, r in results.items():
        if name in ("autoregress", "lightautoregress"):
            continue
        print(f"  | {name:<24} {r['stop_rate']:>5.1%} {r['accuracy']:>5.1%} "
              f"{r['mean_pred']:>6.1f} {r['pred_std']:>6.1f} {r['unique_preds']:>5d} |")
    print(f"  +{'-' * 73}+")

    # AR benchmarks
    for ar_name in ("autoregress", "lightautoregress"):
        ar = results.get(ar_name)
        if not ar:
            continue
        print(f"\n  +-- {ar_name} ({ar.get('n_samples', 0)} samples, {ar.get('ar_steps', 32)} steps) --+")
        # autoregress: set matching stats
        if "event_hit_rate" in ar:
            print(f"  | Events: HIT={ar['event_hit_rate']:.1%} GOOD={ar['event_good_rate']:.1%} "
                  f"MISS={ar.get('event_miss_rate', 0):.1%}  "
                  f"({ar.get('event_hit', 0)}/{ar['total_gt_onsets']} found) |")
            print(f"  | Preds:  HIT={ar.get('pred_hit_rate', 0):.1%} GOOD={ar.get('pred_good_rate', 0):.1%} "
                  f"HALL={ar['hallucination_rate']:.1%}  "
                  f"({ar.get('pred_hit', 0)}/{ar['total_predicted']} valid) |")
            print(f"  | Surv@10: {ar.get('survival_10', 0):.1%}  @30: {ar.get('survival_30', 0):.1%}", end="")
            if "density_actual_mean" in ar:
                print(f"  Density: {ar['density_conditioned_mean']:.1f}->{ar['density_actual_mean']:.1f} ({ar['density_ratio']:.2f}x) |")
            else:
                print(" |")
        # lightautoregress: per-step HIT curve
        if "hit_curve" in ar:
            curve = ar["hit_curve"]
            print(f"  | Step0 HIT: {ar.get('step0_hit', 0):.1%}  Curve: ", end="")
            print(" ".join(f"{h:.0%}" for h in curve[:16]), end="")
            if len(curve) > 16:
                print(" ...", end="")
            print(" |")
        print(f"  +{'-' * 73}+")


def _to_json_safe(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _serializable(results):
    """Strip numpy arrays and convert types for JSON serialization."""
    out = {}
    for name, r in results.items():
        clean = {k: v for k, v in r.items() if k not in ("preds", "targets") and not k.startswith("_")}
        out[name] = _to_json_safe(clean)
    return out


def save_benchmark_data(results, eval_step, run_dir):
    """Save per-benchmark eval JSON + per-eval prediction distribution graph.

    Folder structure:
      run_dir/benchmarks/<bench_name>/eval_001.json
      run_dir/benchmarks/<bench_name>/eval_001_pred_dist.png
      run_dir/benchmarks/<bench_name>/eval_001_heatmap.png
      run_dir/benchmarks/<bench_name>/history.png  (updated each eval)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    bench_root = os.path.join(run_dir, "benchmarks")
    stop = N_CLASSES - 1

    for name, r in results.items():
        # skip AR benchmarks — they have different structure, handled separately below
        if name in ("autoregress", "lightautoregress"):
            continue
        bench_dir = os.path.join(bench_root, name)
        os.makedirs(bench_dir, exist_ok=True)
        prefix = os.path.join(bench_dir, f"eval_{eval_step:03d}")

        # save eval JSON (no arrays)
        json_data = {k: v for k, v in r.items() if k not in ("preds", "targets")}
        json_data["eval_step"] = eval_step
        with open(f"{prefix}.json", "w") as f:
            json.dump(json_data, f, indent=2)

        preds = r["preds"]
        targets = r["targets"]
        ns = targets < stop
        t_ns = targets[ns]
        p_ns = preds[ns]

        # ── prediction distribution ──
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].hist(t_ns, bins=200, range=(0, 500), color="#4a90d9", alpha=0.8)
        axes[0].set_title(f"{name} - Eval {eval_step}: Original Targets (non-STOP)")
        axes[0].set_ylabel("Count")
        axes[1].hist(p_ns, bins=200, range=(0, 500), color="#e8834a", alpha=0.8)
        n_stop = (preds == stop).sum()
        axes[1].set_title(
            f"{name} - Eval {eval_step}: Predictions "
            f"({len(np.unique(p_ns))} unique, {n_stop} STOP [{r['stop_rate']:.1%}])"
        )
        axes[1].set_xlabel("Bin offset")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(f"{prefix}_pred_dist.png", dpi=100)
        plt.close(fig)

        # ── target vs predicted heatmap ──
        if len(t_ns) > 0:
            fig, ax = plt.subplots(figsize=(7, 7))
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            nbins = 250
            h, xe, ye = np.histogram2d(t_ns, p_ns, bins=nbins,
                                        range=[[0, 500], [0, 500]])
            h = gaussian_filter(h.astype(np.float64), sigma=1.0)
            h[h < 0.5] = np.nan
            ax.imshow(h.T, origin="lower", aspect="auto", extent=[0, 500, 0, 500],
                      cmap="inferno", interpolation="bilinear")
            ax.plot([0, 500], [0, 500], "r--", alpha=0.4, linewidth=1)
            ax.set_xlabel("Target", color="white")
            ax.set_ylabel("Predicted", color="white")
            ax.set_title(f"{name} - Eval {eval_step}: Heatmap", color="white")
            ax.tick_params(colors="white")
            fig.tight_layout()
            fig.savefig(f"{prefix}_heatmap.png", dpi=100, facecolor="black")
            plt.close(fig)

    # ── AR benchmark graphs ──
    ar_dir = os.path.join(bench_root, "_autoregress")
    os.makedirs(ar_dir, exist_ok=True)

    for ar_name in ("autoregress", "lightautoregress"):
        ar = results.get(ar_name)
        if not ar or not ar.get("steps"):
            continue

        prefix = os.path.join(ar_dir, f"eval_{eval_step:03d}_{ar_name}")

        # save JSON
        ar_save = {k: v for k, v in ar.items() if not any(k2.startswith("_") for k2 in [k])}
        # strip numpy arrays from steps
        ar_save_steps = []
        for s in ar.get("steps", []):
            ar_save_steps.append({k: v for k, v in s.items() if not k.startswith("_")})
        ar_save["steps"] = ar_save_steps
        with open(f"{prefix}.json", "w") as f:
            json.dump(ar_save, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

        steps = ar["steps"]

        if ar_name == "lightautoregress":
            # hit rate curve over steps
            hit_curve = ar.get("hit_curve", [])
            if hit_curve:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(len(hit_curve)), hit_curve, "o-", color="#6bc46d", linewidth=2, markersize=4)
                ax.set_xlabel("AR Step")
                ax.set_ylabel("HIT Rate")
                ax.set_title(f"Eval {eval_step}: Light AR — HIT Rate Over 32 Steps")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=hit_curve[0] if hit_curve else 0, color="gray", linestyle="--", alpha=0.5, label=f"Step 0: {hit_curve[0]:.1%}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(f"{prefix}_hit_curve.png", dpi=120)
                plt.close(fig)

            # scatter for steps 0, 4, 8, 16, 31
            scatter_steps = [0, 4, 8, 16, 31]
            fig, axes = plt.subplots(1, len(scatter_steps), figsize=(4 * len(scatter_steps), 4))
            for idx, ss in enumerate(scatter_steps):
                ax = axes[idx] if len(scatter_steps) > 1 else axes
                if ss < len(steps) and "_preds" in steps[ss] and "_targets" in steps[ss]:
                    p = steps[ss]["_preds"]
                    t = steps[ss]["_targets"]
                    ax.scatter(t, p, s=2, alpha=0.3, color="#4a90d9")
                    ax.plot([0, 500], [0, 500], "r-", linewidth=0.5, alpha=0.5)
                    hit_r = steps[ss].get("hit_rate", 0)
                    ax.set_title(f"Step {ss} (HIT={hit_r:.0%})")
                else:
                    ax.set_title(f"Step {ss} (no data)")
                ax.set_xlim(0, 500)
                ax.set_ylim(0, 500)
                ax.set_xlabel("Target")
                if idx == 0:
                    ax.set_ylabel("Predicted")
                ax.set_aspect("equal")
            fig.suptitle(f"Eval {eval_step}: Light AR — Pred vs Target per Step", fontsize=12)
            fig.tight_layout()
            fig.savefig(f"{prefix}_scatter.png", dpi=120)
            plt.close(fig)

            # frame error over steps
            frame_errs = [s.get("frame_err_mean", 0) for s in steps if s.get("n_total", 0) > 0]
            if frame_errs:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(len(frame_errs)), frame_errs, "o-", color="#eb4528", linewidth=2, markersize=4)
                ax.set_xlabel("AR Step")
                ax.set_ylabel("Mean Frame Error")
                ax.set_title(f"Eval {eval_step}: Light AR — Frame Error Over Steps")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{prefix}_frame_error.png", dpi=120)
                plt.close(fig)

            # metronome detection: unique preds + pred range over steps
            active_steps = [s for s in steps if s.get("n_total", 0) > 0 and "unique_preds" in s]
            if active_steps:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
                x = [s["step"] for s in active_steps]
                ax1.plot(x, [s["unique_preds"] for s in active_steps], "o-", color="#c76dba", linewidth=2, markersize=4)
                ax1.set_ylabel("Unique Predictions")
                ax1.set_title(f"Eval {eval_step}: Light AR — Metronome Detection")
                ax1.grid(True, alpha=0.3)

                ax2.fill_between(x,
                                [s.get("pred_min", 0) for s in active_steps],
                                [s.get("pred_max", 0) for s in active_steps],
                                alpha=0.3, color="#4a90d9", label="Pred range")
                ax2.plot(x, [s.get("pred_mean", 0) for s in active_steps],
                        "o-", color="#4a90d9", linewidth=2, markersize=4, label="Pred mean")
                ax2.plot(x, [s.get("target_mean", 0) for s in active_steps],
                        "o-", color="#6bc46d", linewidth=2, markersize=4, label="Target mean")
                ax2.set_xlabel("AR Step")
                ax2.set_ylabel("Bin Value")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{prefix}_metronome.png", dpi=120)
                plt.close(fig)

        elif ar_name == "autoregress":
            # survival curve
            survival = ar.get("survival_per_step", [])
            if survival:
                fig, ax = plt.subplots(figsize=(10, 4))
                if survival[0] > 0:
                    surv_pct = [s / survival[0] for s in survival]
                else:
                    surv_pct = survival
                ax.plot(range(len(surv_pct)), surv_pct, "o-", color="#4a90d9", linewidth=2, markersize=4)
                ax.set_xlabel("AR Step")
                ax.set_ylabel("Survival Rate")
                ax.set_title(f"Eval {eval_step}: AR — Survival Over 32 Steps")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{prefix}_survival.png", dpi=120)
                plt.close(fig)

            # entropy over steps
            ent_per_step = [s.get("entropy_mean", 0) for s in steps if s.get("n_alive", 0) > 0]
            if ent_per_step:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(len(ent_per_step)), ent_per_step, "o-", color="#c76dba", linewidth=2, markersize=4)
                ax.set_xlabel("AR Step")
                ax.set_ylabel("Mean Entropy")
                ax.set_title(f"Eval {eval_step}: AR — Entropy Over Steps")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{prefix}_entropy.png", dpi=120)
                plt.close(fig)

            # prediction mean/std over steps (gap drift detection)
            pred_means = [s.get("pred_mean", 0) for s in steps if s.get("n_alive", 0) > 0]
            pred_stds = [s.get("pred_std", 0) for s in steps if s.get("n_alive", 0) > 0]
            if pred_means:
                fig, ax = plt.subplots(figsize=(10, 4))
                x = range(len(pred_means))
                ax.plot(x, pred_means, "o-", color="#6bc46d", linewidth=2, markersize=4, label="Mean pred")
                ax.fill_between(x,
                               [m - s for m, s in zip(pred_means, pred_stds)],
                               [m + s for m, s in zip(pred_means, pred_stds)],
                               alpha=0.2, color="#6bc46d")
                ax.set_xlabel("AR Step")
                ax.set_ylabel("Predicted Bin")
                ax.set_title(f"Eval {eval_step}: AR — Prediction Distribution Over Steps")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{prefix}_pred_drift.png", dpi=120)
                plt.close(fig)

            # density comparison bar
            dc = ar.get("density_conditioned_mean", 0)
            da = ar.get("density_actual_mean", 0)
            if dc > 0 or da > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(["Conditioned", "Actual"], [dc, da], color=["#4a90d9", "#6bc46d"])
                ax.set_ylabel("Events per second")
                ax.set_title(f"Eval {eval_step}: AR — Density Comparison")
                ax.grid(True, alpha=0.3, axis="y")
                for i, v in enumerate([dc, da]):
                    ax.text(i, v + 0.1, f"{v:.1f}", ha="center")
                fig.tight_layout()
                fig.savefig(f"{prefix}_density.png", dpi=120)
                plt.close(fig)

    # ── history curves (one per benchmark, updated every eval) ──
    _save_benchmark_history_graphs(bench_root, run_dir)


def _save_benchmark_history_graphs(bench_root, run_dir):
    """Read all eval JSONs per benchmark and generate history line plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.isdir(bench_root):
        return

    bench_names = sorted(d for d in os.listdir(bench_root)
                         if os.path.isdir(os.path.join(bench_root, d))
                         and not d.startswith("_"))  # skip _autoregress
    if not bench_names:
        return

    # collect history per benchmark
    all_history = {}
    for name in bench_names:
        bench_dir = os.path.join(bench_root, name)
        evals_data = []
        for fn in sorted(os.listdir(bench_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(bench_dir, fn)) as f:
                    evals_data.append(json.load(f))
        if evals_data:
            all_history[name] = evals_data

    if not all_history:
        return

    # ── per-benchmark history graph (stop_rate + accuracy over evals) ──
    for name, hist in all_history.items():
        epochs = [d.get("eval_step", i + 1) for i, d in enumerate(hist)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Benchmark: {name}", fontsize=14)

        # stop rate
        ax = axes[0, 0]
        ax.plot(epochs, [d["stop_rate"] for d in hist], "o-", color="#e8834a")
        ax.set_ylabel("STOP rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("STOP rate")
        ax.grid(True, alpha=0.3)

        # accuracy
        ax = axes[0, 1]
        ax.plot(epochs, [d["accuracy"] for d in hist], "o-", color="#4a90d9")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Accuracy (vs original target)")
        ax.grid(True, alpha=0.3)

        # mean prediction
        ax = axes[1, 0]
        ax.plot(epochs, [d["mean_pred"] for d in hist], "o-", color="#50b050")
        ax.set_ylabel("Mean predicted bin")
        ax.set_xlabel("Eval Step")
        ax.set_title("Mean prediction (non-STOP)")
        ax.grid(True, alpha=0.3)

        # unique predictions
        ax = axes[1, 1]
        ax.plot(epochs, [d["unique_preds"] for d in hist], "o-", color="#b050b0")
        ax.set_ylabel("Unique values")
        ax.set_xlabel("Eval Step")
        ax.set_title("Prediction diversity")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(bench_root, name, "history.png"), dpi=120)
        plt.close(fig)

    # ── combined overlay: all benchmarks on one graph ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation Benchmarks - All", fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_history)))
    for i, (name, hist) in enumerate(all_history.items()):
        epochs = [d.get("eval_step", j + 1) for j, d in enumerate(hist)]
        axes[0].plot(epochs, [d["stop_rate"] for d in hist], "o-",
                     color=colors[i], label=name, markersize=4)
        axes[1].plot(epochs, [d["accuracy"] for d in hist], "o-",
                     color=colors[i], label=name, markersize=4)

    axes[0].set_title("STOP rate")
    axes[0].set_ylabel("STOP rate")
    axes[0].set_xlabel("Eval Step")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Eval Step")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(bench_root, "all_benchmarks.png"), dpi=120)
    plt.close(fig)


def save_eval_graphs(targets, preds, metrics, eval_step, run_dir, extra=None, mt_metrics=None):
    """Generate and save all graphs for this eval."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    eval_dir = os.path.join(run_dir, "evals")
    os.makedirs(eval_dir, exist_ok=True)
    prefix = os.path.join(eval_dir, f"eval_{eval_step:03d}")

    stop = N_CLASSES - 1
    ns = targets < stop
    t_ns = targets[ns]
    p_ns = preds[ns]

    # ── 0. Prediction distribution histogram ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    # target distribution
    axes[0].hist(t_ns, bins=250, range=(0, 500), color="#4a90d9", alpha=0.8)
    axes[0].set_title(f"Eval {eval_step}: Target Distribution (non-STOP)")
    axes[0].set_xlabel("Bin offset")
    axes[0].set_ylabel("Count")
    # predicted distribution
    axes[1].hist(p_ns, bins=250, range=(0, 500), color="#e8834a", alpha=0.8)
    axes[1].set_title(f"Eval {eval_step}: Predicted Distribution - {len(np.unique(p_ns))} unique values")
    axes[1].set_xlabel("Bin offset")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{prefix}_pred_dist.png", dpi=120)
    plt.close(fig)

    # ── 1. Scatter: target vs predicted ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(t_ns, p_ns, alpha=0.02, s=1, color="#4a90d9")
    ax.plot([0, 500], [0, 500], "r--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Target bin offset")
    ax.set_ylabel("Predicted bin offset")
    ax.set_title(f"Eval {eval_step}: Target vs Predicted")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    fig.tight_layout()
    fig.savefig(f"{prefix}_scatter.png", dpi=120)
    plt.close(fig)

    # ── 2. Heatmap: target vs predicted ──
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    h, xedges, yedges = np.histogram2d(t_ns, p_ns, bins=250, range=[[0, 500], [0, 500]])
    h = gaussian_filter(h.astype(np.float64), sigma=1.0)
    h[h < 0.5] = np.nan
    ax.imshow(h.T, origin="lower", aspect="auto", extent=[0, 500, 0, 500],
              norm=LogNorm(vmin=1), cmap="viridis")
    ax.plot([0, 500], [0, 500], "r--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Target bin offset", color="white")
    ax.set_ylabel("Predicted bin offset", color="white")
    ax.set_title(f"Eval {eval_step}: Prediction Density", color="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(f"{prefix}_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 2b. Entropy heatmap: RGB = (entropy, count, inverse-entropy) ──
    entropy = extra.get("entropy") if extra else None
    if entropy is not None and len(t_ns) > 0:
        ent_ns = entropy[ns]
        n_bins = 250
        bin_range = [[0, 500], [0, 500]]

        # Accumulate entropy sums and counts per (target, pred) bin
        ent_sum = np.zeros((n_bins, n_bins), dtype=np.float64)
        counts = np.zeros((n_bins, n_bins), dtype=np.float64)

        # Digitize targets and preds into bin indices
        t_idx = np.clip(((t_ns.astype(np.float64)) / 500 * n_bins).astype(int), 0, n_bins - 1)
        p_idx = np.clip(((p_ns.astype(np.float64)) / 500 * n_bins).astype(int), 0, n_bins - 1)

        np.add.at(ent_sum, (t_idx, p_idx), ent_ns)
        np.add.at(counts, (t_idx, p_idx), 1)

        # Build RGB image (n_bins x n_bins x 3)
        # Normalize entropy to [0, 1] - use mean entropy per bin
        mask = counts > 0
        mean_ent = np.zeros_like(ent_sum)
        mean_ent[mask] = ent_sum[mask] / counts[mask]

        # Normalize: entropy range from data
        if mean_ent[mask].size > 0:
            ent_max = np.percentile(mean_ent[mask], 99)
            ent_min = np.percentile(mean_ent[mask], 1)
        else:
            ent_min, ent_max = 0, 1
        if ent_max <= ent_min:
            ent_max = ent_min + 1
        ent_norm = np.clip((mean_ent - ent_min) / (ent_max - ent_min), 0, 1)

        # Normalize counts (log scale for visibility)
        log_counts = np.zeros_like(counts)
        log_counts[mask] = np.log1p(counts[mask])
        count_max = log_counts.max() if log_counts.max() > 0 else 1
        count_norm = log_counts / count_max

        # R = entropy, G = count, B = inverse entropy
        rgb = np.zeros((n_bins, n_bins, 3))
        rgb[:, :, 0] = ent_norm       # R: high entropy = red
        rgb[:, :, 1] = count_norm     # G: many predictions = green
        rgb[:, :, 2] = 1 - ent_norm   # B: low entropy = blue

        # Multiply by presence mask so empty bins stay black
        for c in range(3):
            rgb[:, :, c] *= (counts > 0).astype(float)

        # Light gaussian blur for smoothness
        for c in range(3):
            rgb[:, :, c] = gaussian_filter(rgb[:, :, c], sigma=0.8)

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.imshow(rgb.transpose(1, 0, 2), origin="lower", aspect="auto", extent=[0, 500, 0, 500])
        ax.plot([0, 500], [0, 500], "w--", alpha=0.4, linewidth=1)
        ax.set_xlabel("Target bin offset", color="white")
        ax.set_ylabel("Predicted bin offset", color="white")
        ax.set_title(f"Eval {eval_step}: Entropy Heatmap (R=entropy, G=count, B=confident)", color="white")
        ax.tick_params(colors="white")
        fig.tight_layout()
        fig.savefig(f"{prefix}_entropy_heatmap.png", dpi=150)
        plt.close(fig)

    # ── 3. Ratio scatter: target vs relative error ──
    if len(t_ns) > 0:
        ratio = (p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)
        ratio_clipped = np.clip(ratio, 0.1, 10.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(t_ns, ratio_clipped, alpha=0.02, s=1, color="#e8834a")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Target bin offset")
        ax.set_ylabel("Ratio (predicted+1)/(target+1)")
        ax.set_title(f"Eval {eval_step}: Relative Error")
        ax.set_ylim(0.1, 10.0)
        ax.set_yscale("log")
        ax.set_xlim(0, 500)
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_scatter.png", dpi=120)
        plt.close(fig)

        # ── 4. Ratio heatmap ──
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        # bin ratio in log space
        log_ratio = np.log10(ratio_clipped)
        h, xedges, yedges = np.histogram2d(
            t_ns.astype(float), log_ratio, bins=[250, 100],
            range=[[0, 500], [-1, 1]]  # ratio 0.1x to 10x
        )
        h = gaussian_filter(h.astype(np.float64), sigma=1.0)
        h[h < 0.5] = np.nan
        ax.imshow(h.T, origin="lower", aspect="auto",
                  extent=[0, 500, -1, 1], norm=LogNorm(vmin=1), cmap="inferno")
        ax.axhline(0.0, color="white", linestyle="--", alpha=0.5)
        ax.set_xlabel("Target bin offset", color="white")
        ax.set_ylabel("log10(ratio)", color="white")
        ax.set_title(f"Eval {eval_step}: Relative Error Density", color="white")
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["0.1x", "0.3x", "1.0x", "3.2x", "10x"])
        ax.tick_params(colors="white")
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_heatmap.png", dpi=150)
        plt.close(fig)

    # ── 5. Frame vs Ratio error (scatter + heatmap) ──
    if len(t_ns) > 0:
        frame_err = np.abs(p_ns.astype(np.float64) - t_ns.astype(np.float64))
        ratio = (p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)
        ratio_err = np.abs(ratio - 1.0)

        fe_clip = np.clip(frame_err, 0, 200)
        re_clip = np.clip(ratio_err, 0, 5.0)

        # scatter
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(fe_clip, re_clip, alpha=0.02, s=1, color="#4a90d9")
        ax.set_xlabel("Absolute frame error |pred - target|")
        ax.set_ylabel("Absolute ratio error |ratio - 1|")
        ax.set_title(f"Eval {eval_step}: Frame Error vs Ratio Error")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 5.0)
        fig.tight_layout()
        fig.savefig(f"{prefix}_frame_vs_ratio_scatter.png", dpi=120)
        plt.close(fig)

        # heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        h, xe, ye = np.histogram2d(fe_clip, re_clip, bins=[100, 250],
                                    range=[[0, 200], [0, 5.0]])
        h = gaussian_filter(h.astype(np.float64), sigma=1.0)
        h[h < 0.5] = np.nan
        ax.imshow(h.T, origin="lower", aspect="auto", extent=[0, 200, 0, 5.0],
                  norm=LogNorm(vmin=1), cmap="magma")
        ax.set_xlabel("Absolute frame error", color="white")
        ax.set_ylabel("Absolute ratio error", color="white")
        ax.set_title(f"Eval {eval_step}: Frame vs Ratio Error Density", color="white")
        ax.tick_params(colors="white")
        fig.tight_layout()
        fig.savefig(f"{prefix}_frame_vs_ratio_heatmap.png", dpi=150)
        plt.close(fig)

    # ── 6. Ratio error in density space (scatter + heatmap) ──
    conds = extra.get("conds") if extra else None
    prev_gaps = extra.get("prev_gaps") if extra else None
    top3_gaps = extra.get("top3_gaps") if extra else None
    ctx_len = extra.get("ctx_len") if extra else None
    topk = extra.get("topk") if extra else None
    entropy = extra.get("entropy") if extra else None

    if conds is not None and len(t_ns) > 0:
        # conds: (N, 3) = [mean_density, peak_density, density_std]
        # filter to non-stop samples
        conds_ns = conds[ns]
        mean_dens = conds_ns[:, 0]
        peak_dens = conds_ns[:, 1]
        ratio_err_log = np.abs(np.log((p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)))
        re_clip2 = np.clip(ratio_err_log, 0, 2.0)

        # scatter - color by error
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(mean_dens, peak_dens, c=re_clip2, s=1, alpha=0.1,
                        cmap="RdYlGn_r", vmin=0, vmax=1.5)
        fig.colorbar(sc, ax=ax, label="|log-ratio| error")
        ax.set_xlabel("Mean density (events/sec)")
        ax.set_ylabel("Peak density (events/sec)")
        ax.set_title(f"Eval {eval_step}: Error by Chart Density")
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_in_density_scatter.png", dpi=120)
        plt.close(fig)

        # heatmap - mean error per density cell
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        d_range = [[0, max(20, np.percentile(mean_dens, 99))],
                    [0, max(40, np.percentile(peak_dens, 99))]]
        h_sum, xe, ye = np.histogram2d(mean_dens, peak_dens, bins=[80, 80],
                                        range=d_range, weights=re_clip2)
        h_cnt, _, _ = np.histogram2d(mean_dens, peak_dens, bins=[80, 80],
                                      range=d_range)
        h_mean = np.divide(h_sum, h_cnt, where=h_cnt > 5, out=np.full_like(h_sum, np.nan))
        ax.imshow(h_mean.T, origin="lower", aspect="auto",
                  extent=[d_range[0][0], d_range[0][1], d_range[1][0], d_range[1][1]],
                  vmin=0, vmax=1.0, cmap="RdYlGn_r")
        ax.set_xlabel("Mean density (events/sec)", color="white")
        ax.set_ylabel("Peak density (events/sec)", color="white")
        ax.set_title(f"Eval {eval_step}: Mean |log-ratio| Error by Density", color="white")
        ax.tick_params(colors="white")
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_in_density_heatmap.png", dpi=150)
        plt.close(fig)

    # ── 6b. Density-colored scatter: target vs predicted, R=high density, B=low ──
    if conds is not None and len(t_ns) > 0:
        conds_ns_ds = conds[ns]
        d_mean = conds_ns_ds[:, 0]
        r_ch = np.clip(d_mean / 12.0, 0, 1)
        b_ch = np.clip((20 - d_mean) / 20.0, 0, 1)
        g_ch = np.zeros_like(r_ch)
        d_colors = np.stack([r_ch, g_ch, b_ch], axis=1)

        max_pts = 5000
        if len(t_ns) > max_pts:
            ds_idx = np.random.default_rng(42).choice(len(t_ns), max_pts, replace=False)
        else:
            ds_idx = np.arange(len(t_ns))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor("black")
        ax.scatter(t_ns[ds_idx], p_ns[ds_idx], c=d_colors[ds_idx], s=2, alpha=0.5, rasterized=True)
        ax.plot([0, 500], [0, 500], color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Target bin offset")
        ax.set_ylabel("Predicted bin offset")
        ax.set_title(f"Eval {eval_step}: Density Scatter (Red=dense chart, Blue=sparse chart)")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="red", label="High density"),
            Patch(facecolor="purple", label="Medium density"),
            Patch(facecolor="blue", label="Low density"),
        ], fontsize=8, loc="upper left", framealpha=0.7)
        fig.tight_layout()
        fig.savefig(f"{prefix}_density_scatter.png", dpi=120)
        plt.close(fig)

    # ── 7. Forward error: gap ratio continuity (scatter + heatmap) ──
    # X = predicted_gap / prev_gap, Y = target_gap / prev_gap
    if prev_gaps is not None and len(t_ns) > 0:
        pg = prev_gaps[ns]  # prev_gap for non-stop samples
        valid = np.isfinite(pg) & (pg > 0)
        if valid.sum() > 100:
            tg = t_ns[valid].astype(np.float64)
            pg_v = pg[valid].astype(np.float64)
            pr = p_ns[valid].astype(np.float64)

            target_ratio = tg / pg_v    # how big is this gap vs prev gap
            pred_ratio = pr / pg_v      # what model predicted vs prev gap

            # clip to reasonable range
            tr_clip = np.clip(target_ratio, 0, 8)
            pr_clip = np.clip(pred_ratio, 0, 8)

            # scatter
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(pr_clip, tr_clip, alpha=0.02, s=1, color="#6bc46d")
            ax.plot([0, 8], [0, 8], "r--", alpha=0.5, linewidth=1)
            # reference lines for common ratios
            for r, lbl in [(0.5, "½"), (1.0, "1"), (2.0, "2"), (4.0, "4")]:
                ax.axhline(r, color="white", alpha=0.15, linewidth=0.5)
                ax.axvline(r, color="white", alpha=0.15, linewidth=0.5)
            ax.set_xlabel("Predicted gap / prev gap")
            ax.set_ylabel("Target gap / prev gap")
            ax.set_title(f"Eval {eval_step}: Gap Ratio Continuity")
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            fig.tight_layout()
            fig.savefig(f"{prefix}_forward_error_scatter.png", dpi=120)
            plt.close(fig)

            # heatmap
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            h, xe, ye = np.histogram2d(pr_clip, tr_clip, bins=[200, 200],
                                        range=[[0, 8], [0, 8]])
            h = gaussian_filter(h.astype(np.float64), sigma=1.0)
            h[h < 0.5] = np.nan
            ax.imshow(h.T, origin="lower", aspect="auto", extent=[0, 8, 0, 8],
                      norm=LogNorm(vmin=1), cmap="viridis")
            ax.plot([0, 8], [0, 8], "r--", alpha=0.5, linewidth=1)
            for r in [0.5, 1.0, 2.0, 4.0]:
                ax.axhline(r, color="white", alpha=0.2, linewidth=0.5)
                ax.axvline(r, color="white", alpha=0.2, linewidth=0.5)
            ax.set_xlabel("Predicted gap / prev gap", color="white")
            ax.set_ylabel("Target gap / prev gap", color="white")
            ax.set_title(f"Eval {eval_step}: Gap Ratio Continuity Density", color="white")
            ax.tick_params(colors="white")
            fig.tight_layout()
            fig.savefig(f"{prefix}_forward_error_heatmap.png", dpi=150)
            plt.close(fig)

    # ── 7b. Forward metronome error: pred/target vs top context gaps (scatter + heatmap) ──
    # For each sample, normalize pred and target by the top 3 gap peaks in context.
    # R = top1, G = top2, B = top3. Points at (1, 1) = exact metronome continuation.
    if top3_gaps is not None and len(t_ns) > 0:
        tg3 = top3_gaps[ns]  # (N_ns, 3)
        metronome_stats = {}  # for JSON export

        # collect all points for each peak
        all_rx, all_ry = [], []   # top1
        all_gx, all_gy = [], []   # top2
        all_bx, all_by = [], []   # top3
        colors_scatter = []
        xs_scatter = []
        ys_scatter = []

        for k, (label, color, ax_list, ay_list) in enumerate([
            ("top1", "#ff4444", all_rx, all_ry),
            ("top2", "#44cc44", all_gx, all_gy),
            ("top3", "#4488ff", all_bx, all_by),
        ]):
            valid = np.isfinite(tg3[:, k]) & (tg3[:, k] > 0)
            if valid.sum() < 10:
                continue
            gap_k = tg3[valid, k]
            pred_ratio = p_ns[valid].astype(np.float64) / gap_k
            target_ratio = t_ns[valid].astype(np.float64) / gap_k

            ax_list.extend(pred_ratio.tolist())
            ay_list.extend(target_ratio.tolist())

            # stats: what % of predictions land within 5% of this gap peak?
            pred_near = np.abs(pred_ratio - 1.0) <= 0.05
            target_near = np.abs(target_ratio - 1.0) <= 0.05
            both_near = pred_near & target_near  # correct metronome continuation
            pred_near_target_not = pred_near & ~target_near  # model metronomes but shouldn't

            metronome_stats[label] = {
                "n_samples": int(valid.sum()),
                "gap_mean_bins": float(gap_k.mean()),
                "target_continues_pct": float(target_near.mean() * 100),
                "target_breaks_pct": float((~target_near).mean() * 100),
                "pred_continues_pct": float(pred_near.mean() * 100),
                "pred_breaks_pct": float((~pred_near).mean() * 100),
                "both_continue_pct": float(both_near.mean() * 100),
                "pred_continues_target_breaks_pct": float(pred_near_target_not.mean() * 100),
                "pred_breaks_target_continues_pct": float((~pred_near & target_near).mean() * 100),
                "both_break_pct": float((~pred_near & ~target_near).mean() * 100),
            }

        # scatter with all 3 colors
        fig, ax = plt.subplots(figsize=(8, 8))
        for label, color, rxl, ryl in [
            ("top3", "#4488ff", all_bx, all_by),
            ("top2", "#44cc44", all_gx, all_gy),
            ("top1", "#ff4444", all_rx, all_ry),
        ]:
            if len(rxl) > 0:
                rx_clip = np.clip(rxl, 0, 4)
                ry_clip = np.clip(ryl, 0, 4)
                ax.scatter(rx_clip, ry_clip, alpha=0.015, s=1, color=color, label=label)
        ax.plot([0, 4], [0, 4], "w--", alpha=0.4, linewidth=1)
        ax.axhline(1.0, color="white", alpha=0.3, linewidth=0.5)
        ax.axvline(1.0, color="white", alpha=0.3, linewidth=0.5)
        for r in [0.5, 2.0]:
            ax.axhline(r, color="white", alpha=0.15, linewidth=0.5)
            ax.axvline(r, color="white", alpha=0.15, linewidth=0.5)
        ax.set_xlabel("Predicted gap / top_gap")
        ax.set_ylabel("Target gap / top_gap")
        ax.set_title(f"Eval {eval_step}: Forward Metronome Error")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        fig.tight_layout()
        fig.savefig(f"{prefix}_metronome_scatter.png", dpi=120)
        plt.close(fig)

        # heatmap — RGB channels: R=top1, G=top2, B=top3
        if len(all_rx) > 100:
            bins = 200
            rgb = np.zeros((bins, bins, 3), dtype=np.float64)
            for ch, (rxl, ryl) in enumerate([
                (all_rx, all_ry), (all_gx, all_gy), (all_bx, all_by)
            ]):
                if len(rxl) > 0:
                    h, _, _ = np.histogram2d(
                        np.clip(rxl, 0, 4), np.clip(ryl, 0, 4),
                        bins=[bins, bins], range=[[0, 4], [0, 4]])
                    rgb[:, :, ch] = gaussian_filter(h.astype(np.float64), sigma=1.0)
            # log-scale each channel independently, normalize to [0, 1]
            for ch in range(3):
                c = rgb[:, :, ch]
                c[c < 0.5] = 0
                if c.max() > 0:
                    c = np.log1p(c)
                    c /= c.max()
                rgb[:, :, ch] = c

            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            ax.imshow(rgb.transpose(1, 0, 2), origin="lower", aspect="auto",
                      extent=[0, 4, 0, 4])
            ax.plot([0, 4], [0, 4], "w--", alpha=0.4, linewidth=1)
            ax.axhline(1.0, color="white", alpha=0.4, linewidth=0.5)
            ax.axvline(1.0, color="white", alpha=0.4, linewidth=0.5)
            for r in [0.5, 2.0]:
                ax.axhline(r, color="white", alpha=0.2, linewidth=0.5)
                ax.axvline(r, color="white", alpha=0.2, linewidth=0.5)
            ax.set_xlabel("Predicted gap / top_gap", color="white")
            ax.set_ylabel("Target gap / top_gap", color="white")
            ax.set_title(f"Eval {eval_step}: Metronome Density (R=top1 G=top2 B=top3)", color="white")
            ax.tick_params(colors="white")
            fig.tight_layout()
            fig.savefig(f"{prefix}_metronome_heatmap.png", dpi=150)
            plt.close(fig)

        # save JSON stats
        try:
            with open(f"{prefix}_metronome_stats.json", "w", encoding="utf-8") as f:
                json.dump(metronome_stats, f, indent=2)
        except Exception:
            pass

    # ── 8. Ratio confusion histogram ──
    # When wrong, which ratio multiple did the model pick?
    if len(t_ns) > 0:
        ratio_raw = (p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)
        # only look at misses (>20% off)
        pct_err = np.abs(ratio_raw - 1.0)
        misses = pct_err > 0.20
        if misses.sum() > 50:
            miss_ratios = ratio_raw[misses]
            log_miss = np.log2(np.clip(miss_ratios, 1/8, 8))

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(log_miss, bins=400, range=[-3, 3], color="#eb4528", alpha=0.8)
            # mark common musical ratios
            for r, lbl in [(1/4, "¼"), (1/3, "⅓"), (1/2, "½"), (1, "1"),
                           (2, "2"), (3, "3"), (4, "4")]:
                ax.axvline(np.log2(r), color="white", alpha=0.5, linewidth=1, linestyle="--")
                ax.text(np.log2(r), ax.get_ylim()[1] * 0.95, lbl,
                        ha="center", va="top", fontsize=10, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
            ax.set_xlabel("log₂(pred/target)  - musical ratio")
            ax.set_ylabel("Count")
            ax.set_title(f"Eval {eval_step}: Ratio Confusion (misses only, n={misses.sum()})")
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_xticklabels(["⅛", "¼", "½", "1", "2", "4", "8"])
            fig.tight_layout()
            fig.savefig(f"{prefix}_ratio_confusion.png", dpi=120)
            plt.close(fig)

    # ── 9. Accuracy by context length ──
    if ctx_len is not None and len(t_ns) > 0:
        cl = ctx_len[ns]  # context lengths for non-stop samples
        frame_err = np.abs(p_ns.astype(np.float64) - t_ns.astype(np.float64))
        ratio_raw = (p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)
        pct_err = np.abs(ratio_raw - 1.0)
        hit = (pct_err <= 0.03) | (frame_err <= 1)

        # bin by context length
        bin_edges = np.arange(0, 129, 4)  # 0-4, 4-8, ..., 124-128
        bin_idx = np.digitize(cl, bin_edges) - 1
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        hit_rates = []
        counts = []
        for b in range(len(bin_centers)):
            mask = bin_idx == b
            if mask.sum() > 20:
                hit_rates.append(hit[mask].mean())
                counts.append(mask.sum())
            else:
                hit_rates.append(np.nan)
                counts.append(0)

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(bin_centers, counts, width=3.5, alpha=0.3, color="#4a90d9", label="Sample count")
        ax1.set_ylabel("Sample count", color="#4a90d9")
        ax1.tick_params(axis="y", labelcolor="#4a90d9")

        ax2 = ax1.twinx()
        valid_rates = [(c, r) for c, r in zip(bin_centers, hit_rates) if not np.isnan(r)]
        if valid_rates:
            ax2.plot([v[0] for v in valid_rates], [v[1] for v in valid_rates],
                     "o-", color="#eb4528", linewidth=2, markersize=4, label="HIT rate")
        ax2.set_ylabel("HIT rate", color="#eb4528")
        ax2.tick_params(axis="y", labelcolor="#eb4528")
        ax2.set_ylim(0, 1)

        ax1.set_xlabel("Number of past events in context")
        ax1.set_title(f"Eval {eval_step}: Accuracy by Context Length")
        fig.tight_layout()
        fig.savefig(f"{prefix}_accuracy_by_context.png", dpi=120)
        plt.close(fig)

    # ── 10. Top-K accuracy ──
    if topk is not None and len(t_ns) > 0:
        topk_ns = topk[ns]  # (M, 10) - top-10 predictions for non-stop
        targets_ns_col = t_ns.reshape(-1, 1)

        ks = [1, 2, 3, 5, 10]
        # exact match
        exact_topk = [(topk_ns[:, :k] == targets_ns_col).any(axis=1).mean() for k in ks]
        # within ±1 frame
        within1_topk = [np.min(np.abs(topk_ns[:, :k].astype(np.int64) - targets_ns_col), axis=1) <= 1 for k in ks]
        within1_topk = [w.mean() for w in within1_topk]
        # HIT: within 3% ratio or ±1 frame
        hit_topk = []
        for k in ks:
            tk = topk_ns[:, :k].astype(np.float64)
            tgt = targets_ns_col.astype(np.float64)
            ratios = (tk + 1) / (tgt + 1)
            pct_errs = np.abs(ratios - 1.0)
            frame_errs = np.abs(tk - tgt)
            is_hit = ((pct_errs <= 0.03) | (frame_errs <= 1)).any(axis=1)
            hit_topk.append(is_hit.mean())

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(ks))
        w = 0.25
        ax.bar(x - w, exact_topk, w, label="Exact match", color="#eb4528")
        ax.bar(x, within1_topk, w, label="Within ±1 frame", color="#e8834a")
        ax.bar(x + w, hit_topk, w, label="HIT (≤3% or ±1f)", color="#6bc46d")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Top-{k}" for k in ks])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(f"Eval {eval_step}: Top-K Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        for i, (e, w1, h) in enumerate(zip(exact_topk, within1_topk, hit_topk)):
            ax.text(i + w, h + 0.02, f"{h:.1%}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{prefix}_topk_accuracy.png", dpi=120)
        plt.close(fig)

    # ── 11. Logit entropy: correct vs wrong ──
    if entropy is not None and len(t_ns) > 0:
        ent_ns = entropy[ns]
        frame_err = np.abs(p_ns.astype(np.float64) - t_ns.astype(np.float64))
        ratio_raw = (p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)
        pct_err = np.abs(ratio_raw - 1.0)
        hit = (pct_err <= 0.03) | (frame_err <= 1)
        miss = pct_err > 0.20

        fig, ax = plt.subplots(figsize=(10, 5))
        ent_max = np.percentile(ent_ns, 99)
        bins = np.linspace(0, ent_max, 100)
        if hit.sum() > 0:
            ax.hist(ent_ns[hit], bins=bins, alpha=0.6, color="#6bc46d", label=f"HIT (n={hit.sum()})", density=True)
        if miss.sum() > 0:
            ax.hist(ent_ns[miss], bins=bins, alpha=0.6, color="#eb4528", label=f"MISS (n={miss.sum()})", density=True)
        # also show "in between" (GOOD but not HIT)
        between = (~hit) & (~miss)
        if between.sum() > 0:
            ax.hist(ent_ns[between], bins=bins, alpha=0.4, color="#fcb71e", label=f"GOOD (n={between.sum()})", density=True)
        ax.set_xlabel("Softmax entropy (nats)")
        ax.set_ylabel("Density")
        ax.set_title(f"Eval {eval_step}: Model Confidence - HIT vs MISS")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # add mean lines
        if hit.sum() > 0:
            ax.axvline(ent_ns[hit].mean(), color="#6bc46d", linestyle="--", linewidth=2)
        if miss.sum() > 0:
            ax.axvline(ent_ns[miss].mean(), color="#eb4528", linestyle="--", linewidth=2)
        fig.tight_layout()
        fig.savefig(f"{prefix}_entropy_hit_vs_miss.png", dpi=120)
        plt.close(fig)

    # ── 12. Top-K proposal quality ──
    if topk is not None and len(t_ns) > 0:
        topk_ns_data = topk[ns]
        targets_ns_col = t_ns.reshape(-1, 1)

        ks = [1, 2, 3, 5, 10]

        def _topk_hit_rates(topk_data):
            rates = []
            for k in ks:
                tk = topk_data[:, :k].astype(np.float64)
                tgt = targets_ns_col.astype(np.float64)
                ratios = (tk + 1) / (tgt + 1)
                pct_errs = np.abs(ratios - 1.0)
                frame_errs = np.abs(tk - tgt)
                is_hit = ((pct_errs <= 0.03) | (frame_errs <= 1)).any(axis=1)
                rates.append(is_hit.mean())
            return rates

        hit_topk = _topk_hit_rates(topk_ns_data)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(ks))
        ax.bar(x, hit_topk, 0.5, color="#6bc46d")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Top-{k}" for k in ks])
        ax.set_ylabel("HIT Rate")
        ax.set_ylim(0, 1)
        ax.set_title(f"Eval {eval_step}: Top-K HIT Rate")
        ax.grid(True, alpha=0.3, axis="y")
        for i, h in enumerate(hit_topk):
            ax.text(i, h + 0.02, f"{h:.1%}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{prefix}_topk_quality.png", dpi=120)
        plt.close(fig)

        # ── 13. Top-K scatter: each prediction as a group of K candidates ──
        # subsample to avoid overcrowding
        max_points = 2000
        if len(t_ns) > max_points:
            idx = np.random.default_rng(42).choice(len(t_ns), max_points, replace=False)
        else:
            idx = np.arange(len(t_ns))
        t_sub = t_ns[idx]
        topk_sub = topk_ns_data[idx]  # (M, 10)

        k_colors = plt.cm.viridis(np.linspace(0, 1, 10))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor("black")

        # plot thin lines connecting top-K candidates for each sample
        for j in range(len(t_sub)):
            target_val = t_sub[j]
            candidates = topk_sub[j]  # (10,)
            # vertical line from lowest to highest candidate
            xs = np.full(10, target_val)
            ax.plot(xs, candidates, color="white", alpha=0.08, linewidth=0.3)

        # plot each K rank as its own scatter layer
        for k_idx in range(9, -1, -1):  # plot K=10 first (background), K=1 last (foreground)
            ax.scatter(
                t_sub, topk_sub[:, k_idx],
                c=[k_colors[k_idx]], s=1.0 if k_idx > 0 else 3.0,
                alpha=0.4 if k_idx > 0 else 0.8,
                label=f"K={k_idx+1}" if k_idx in [0, 2, 4, 9] else None,
                rasterized=True,
            )

        # diagonal
        ax.plot([0, 500], [0, 500], color="gray", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Target bin offset")
        ax.set_ylabel("Predicted bin offset (top-K)")
        ax.set_title(f"Eval {eval_step}: Top-K Candidate Scatter (n={len(t_sub)})")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)
        fig.tight_layout()
        fig.savefig(f"{prefix}_topk_scatter.png", dpi=120)
        plt.close(fig)

    # ── 14. Ratio pointer field: grid of average error directions ──
    if len(t_ns) > 0:
        log_ratio = np.log2((p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1))
        dx_all = p_ns.astype(np.float64) - t_ns.astype(np.float64)
        dy_all = -log_ratio

        x_bins_rpf = np.linspace(0, 500, 76)  # 75 bins
        y_bins_rpf = np.linspace(-3, 3, 73)   # 72 bins
        x_centers_rpf = (x_bins_rpf[:-1] + x_bins_rpf[1:]) / 2
        y_centers_rpf = (y_bins_rpf[:-1] + y_bins_rpf[1:]) / 2

        x_idx_rpf = np.clip(np.digitize(t_ns, x_bins_rpf) - 1, 0, len(x_centers_rpf) - 1)
        y_idx_rpf = np.clip(np.digitize(log_ratio, y_bins_rpf) - 1, 0, len(y_centers_rpf) - 1)

        grid_dx = np.zeros((len(y_centers_rpf), len(x_centers_rpf)))
        grid_dy = np.zeros((len(y_centers_rpf), len(x_centers_rpf)))
        grid_count = np.zeros((len(y_centers_rpf), len(x_centers_rpf)))

        col_dx = np.zeros(len(x_centers_rpf))
        col_dy = np.zeros(len(x_centers_rpf))
        col_count = np.zeros(len(x_centers_rpf))

        for i in range(len(t_ns)):
            xi, yi = x_idx_rpf[i], y_idx_rpf[i]
            grid_dx[yi, xi] += dx_all[i]
            grid_dy[yi, xi] += dy_all[i]
            grid_count[yi, xi] += 1
            col_dx[xi] += dx_all[i]
            col_dy[xi] += dy_all[i]
            col_count[xi] += 1

        gmask = grid_count > 0
        grid_dx[gmask] /= grid_count[gmask]
        grid_dy[gmask] /= grid_count[gmask]
        cmask = col_count > 0
        col_dx[cmask] /= col_count[cmask]
        col_dy[cmask] /= col_count[cmask]

        magnitude = np.sqrt(grid_dx**2 + grid_dy**2)
        max_mag = np.percentile(magnitude[grid_count >= 3], 95) if (grid_count >= 3).sum() > 0 else 1.0
        cell_w = x_bins_rpf[1] - x_bins_rpf[0]
        cell_h = y_bins_rpf[1] - y_bins_rpf[0]

        def _rpf_color(lr_val):
            if abs(lr_val) <= 0.3:
                return "#6bc46d"
            return "#eb4528" if lr_val > 0 else "#4a90d9"

        # ── 14a. Column-wise average direction (top bar only) ──
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.set_facecolor("#111111")

        col_max_dy = np.abs(col_dy[cmask]).max() if cmask.sum() > 0 else 1.0
        for xi in range(len(x_centers_rpf)):
            if col_count[xi] < 3:
                continue
            x0 = x_centers_rpf[xi]
            adx = col_dx[xi]
            ady = col_dy[xi]
            if abs(adx) < 0.01 and abs(ady) < 0.01:
                continue

            avg_lr = -col_dy[xi]
            c = _rpf_color(avg_lr)
            alpha = min(0.9, 0.3 + 0.6 * min(col_count[xi] / 100, 1.0))
            scale = 2.0 / max(col_max_dy, 1e-6)
            ax.annotate("",
                xy=(x0 + adx * 0.15, ady * scale),
                xytext=(x0, 0),
                arrowprops=dict(arrowstyle="->", color=c, alpha=alpha, linewidth=2.0),
            )

        ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Target bin offset")
        ax.set_ylabel("Avg dir")
        ax.set_title(f"Eval {eval_step}: Column-wise Average Error Direction (red=over, blue=under, green=correct)")
        ax.set_xlim(0, 500)
        ax.set_ylim(-3, 3)
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_pointer_bar.png", dpi=120)
        plt.close(fig)

        # ── 14b. Full ratio pointer field ──
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_facecolor("#111111")

        count_plot = np.where(grid_count > 0, np.log10(grid_count + 1), np.nan)
        ax.imshow(count_plot, aspect="auto", origin="lower", cmap="magma", alpha=0.5,
                  extent=[0, 500, -3, 3], interpolation="nearest")

        arrow_scale = min(cell_w * 0.4, cell_h * 0.4) / max(max_mag, 1e-6)
        for yi_idx in range(len(y_centers_rpf)):
            for xi_idx in range(len(x_centers_rpf)):
                if grid_count[yi_idx, xi_idx] < 3:
                    continue
                x0 = x_centers_rpf[xi_idx]
                y0 = y_centers_rpf[yi_idx]
                adx = grid_dx[yi_idx, xi_idx]
                ady = grid_dy[yi_idx, xi_idx]
                mag = np.sqrt(adx**2 + ady**2)
                if mag < 0.01:
                    continue

                c = _rpf_color(y_centers_rpf[yi_idx])
                n = int(grid_count[yi_idx, xi_idx])
                alpha = min(0.9, 0.2 + 0.7 * min(n / 30, 1.0))

                ax.annotate("",
                    xy=(x0 + adx * arrow_scale, y0 + ady * arrow_scale),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=c, alpha=alpha, linewidth=0.8),
                )

        ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.axhline(1, color="gray", linewidth=0.3, alpha=0.3, linestyle="--")
        ax.axhline(-1, color="gray", linewidth=0.3, alpha=0.3, linestyle="--")
        ax.set_xlabel("Target bin offset")
        ax.set_ylabel("log2(prediction / target)")
        ax.set_title(f"Eval {eval_step}: Ratio Pointer Field (avg direction per cell, min 3 samples)")
        ax.set_xlim(0, 500)
        ax.set_ylim(-3, 3)
        ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_yticklabels(["0.125x", "0.25x", "0.5x", "1x", "2x", "4x", "8x"])
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_pointer_field.png", dpi=120)
        plt.close(fig)

    # ── Multi-target graphs ──
    if mt_metrics is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mr = mt_metrics.get("_matched_real", np.array([]))
        mp = mt_metrics.get("_matched_pred", np.array([]))
        mc = mt_metrics.get("_matched_conf", np.array([]))
        mfe = mt_metrics.get("_matched_frame_err", np.array([]))

        if len(mr) > 0:
            # ── MT1: Matched-pair heatmap (501 = unmatched) ──
            from scipy.ndimage import gaussian_filter
            from matplotlib.colors import LogNorm
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            # clip to 502 range (0-500 = normal, 501 = unmatched sentinel)
            h, xe, ye = np.histogram2d(
                np.clip(mr, 0, 501), np.clip(mp, 0, 501),
                bins=251, range=[[0, 502], [0, 502]])
            h = gaussian_filter(h.astype(np.float64), sigma=1.0)
            h[h < 0.5] = np.nan
            ax.imshow(h.T, origin="lower", aspect="auto", extent=[0, 502, 0, 502],
                      norm=LogNorm(vmin=1), cmap="viridis")
            ax.plot([0, 500], [0, 500], "r--", alpha=0.5, linewidth=1)
            ax.axhline(501, color="yellow", alpha=0.3, linewidth=0.5)
            ax.axvline(501, color="yellow", alpha=0.3, linewidth=0.5)
            ax.set_xlabel("Real onset bin (501=hallucination)", color="white")
            ax.set_ylabel("Predicted bin (501=missed event)", color="white")
            ax.set_title(f"Eval {eval_step}: Matched-Pair Heatmap", color="white")
            ax.tick_params(colors="white")
            fig.tight_layout()
            fig.savefig(f"{prefix}_mt_heatmap.png", dpi=150)
            plt.close(fig)

            # ── MT2: Confidence histogram by outcome ──
            # matched pairs where both are real (not 501)
            both_valid = (mr < 501) & (mp < 501)
            valid_fe = mfe[both_valid]
            valid_conf = mc[both_valid]
            valid_r = mr[both_valid]
            valid_p = mp[both_valid]

            if len(valid_r) > 0:
                pct_err = np.abs((valid_p + 1) / (valid_r + 1) - 1.0)
                fe = np.abs(valid_p - valid_r)
                is_hit = (pct_err <= 0.03) | (fe <= 1)
                is_miss = pct_err > 0.20

                hallucination_conf = mc[(mr >= 501) & (mp < 501)]

                fig, ax = plt.subplots(figsize=(10, 5))
                bins_h = np.linspace(0, max(0.5, np.percentile(mc[mc > 0], 99) if (mc > 0).any() else 0.5), 80)
                if is_hit.sum() > 0:
                    ax.hist(valid_conf[is_hit], bins=bins_h, alpha=0.6, color="#6bc46d",
                            label=f"Matched HIT (n={is_hit.sum()})", density=True)
                if is_miss.sum() > 0:
                    ax.hist(valid_conf[is_miss], bins=bins_h, alpha=0.6, color="#eb4528",
                            label=f"Matched MISS (n={is_miss.sum()})", density=True)
                if len(hallucination_conf) > 0:
                    ax.hist(hallucination_conf, bins=bins_h, alpha=0.4, color="#fcb71e",
                            label=f"Hallucination (n={len(hallucination_conf)})", density=True)
                ax.set_xlabel("Softmax confidence")
                ax.set_ylabel("Density")
                ax.set_title(f"Eval {eval_step}: Confidence by Outcome")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{prefix}_mt_confidence.png", dpi=120)
                plt.close(fig)

            # ── MT3: Threshold sweep ──
            if "probs" in (extra or {}):
                sweep_t, sweep_r = threshold_sweep(
                    extra["targets_padded"], extra["n_targets"], extra["probs"])
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(sweep_t, [r["event_recall_hit"] for r in sweep_r],
                         color="#6bc46d", linewidth=2, label="Event recall (HIT)")
                ax1.plot(sweep_t, [r["pred_precision_hit"] for r in sweep_r],
                         color="#4a90d9", linewidth=2, label="Pred precision (HIT)")
                ax1.plot(sweep_t, [r["f1_hit"] for r in sweep_r],
                         color="#eb4528", linewidth=2, label="F1 (HIT)")
                ax1.axvline(mt_metrics["threshold"], color="white", linestyle="--", alpha=0.5,
                            label=f"threshold={mt_metrics['threshold']}")
                ax1.set_xlabel("Threshold")
                ax1.set_ylabel("Rate")
                ax1.set_ylim(0, 1)
                ax1.legend(loc="upper right")
                ax1.grid(True, alpha=0.3)

                ax2 = ax1.twinx()
                ax2.plot(sweep_t, [r["avg_preds_per_window"] for r in sweep_r],
                         color="#c76dba", linewidth=1, linestyle=":", label="preds/window")
                ax2.set_ylabel("Preds per window", color="#c76dba")
                ax2.tick_params(axis="y", labelcolor="#c76dba")

                ax1.set_title(f"Eval {eval_step}: Threshold Sweep")
                fig.tight_layout()
                fig.savefig(f"{prefix}_mt_threshold_sweep.png", dpi=120)
                plt.close(fig)


def save_training_curves(history, run_dir):
    """Save loss and metric curves across all evals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["eval_step"] for h in history]

    # ── loss ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="Train", linewidth=2)
    ax.plot(epochs, [h["val_loss"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss.png"), dpi=150)
    plt.close(fig)

    # ── accuracy ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [h["val_metrics"].get("accuracy", 0) for h in history],
            label="Val Accuracy", linewidth=2, color="#4a90d9")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy (argmax == target)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "accuracy.png"), dpi=150)
    plt.close(fig)

    # ── STOP F1/precision/recall ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in [
        ("stop_f1", "F1", "#4a90d9"),
        ("stop_precision", "Precision", "#6bc46d"),
        ("stop_recall", "Recall", "#e8834a"),
    ]:
        vals = [h["val_metrics"].get(key, 0) for h in history]
        ax.plot(epochs, vals, label=label, linewidth=2, color=color)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Score")
    ax.set_title("STOP Class (500): F1 / Precision / Recall")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "stop_f1.png"), dpi=150)
    plt.close(fig)

    # ── frame error ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [h["val_metrics"].get("frame_error_mean", 0) for h in history],
            label="Mean", linewidth=2, color="#4a90d9")
    ax.plot(epochs, [h["val_metrics"].get("frame_error_median", 0) for h in history],
            label="Median", linewidth=2, color="#6bc46d")
    ax.plot(epochs, [h["val_metrics"].get("frame_error_p90", 0) for h in history],
            label="P90", linewidth=2, color="#e8834a", linestyle="--")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Frame Error (bins)")
    ax.set_title("Frame Error Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "frame_error.png"), dpi=150)
    plt.close(fig)

    # ── HIT / GOOD / MISS rates ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in [
        ("hit_rate", "HIT (≤3% or ±1f)", "#6bc46d"),
        ("good_rate", "GOOD (≤10% or ±2f)", "#4a90d9"),
        ("miss_rate", "MISS (>20%)", "#eb4528"),
    ]:
        vals = [h["val_metrics"].get(key, 0) for h in history]
        ax.plot(epochs, vals, label=label, linewidth=2, color=color)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Rate")
    ax.set_title("HIT / GOOD / MISS Rates")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "hit_good_miss.png"), dpi=150)
    plt.close(fig)

    # ── frame accuracy tiers ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in [
        ("exact_match", "Exact", "#eb4528"),
        ("within_1_frame", "±1 frame", "#e8834a"),
        ("within_2_frames", "±2 frames", "#fcb71e"),
        ("within_4_frames", "±4 frames", "#6bc46d"),
    ]:
        vals = [h["val_metrics"].get(key, 0) for h in history]
        ax.plot(epochs, vals, label=label, linewidth=2, color=color)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Rate")
    ax.set_title("Frame Accuracy Tiers")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "frame_tiers.png"), dpi=150)
    plt.close(fig)

    # ── ratio accuracy tiers ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label, color in [
        ("within_3pct", "≤3%", "#6bc46d"),
        ("within_10pct", "≤10%", "#4a90d9"),
        ("above_20pct", ">20% (miss)", "#eb4528"),
    ]:
        vals = [h["val_metrics"].get(key, 0) for h in history]
        ax.plot(epochs, vals, label=label, linewidth=2, color=color)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Rate")
    ax.set_title("Ratio Accuracy Tiers")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "ratio_tiers.png"), dpi=150)
    plt.close(fig)

    # ── relative error ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [h["val_metrics"].get("rel_error_mean", 0) for h in history],
            label="Mean |log-ratio|", linewidth=2, color="#c76dba")
    ax.plot(epochs, [h["val_metrics"].get("rel_error_median", 0) for h in history],
            label="Median |log-ratio|", linewidth=2, color="#4a90d9")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("|log((pred+1)/(target+1))|")
    ax.set_title("Relative Error Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "relative_error.png"), dpi=150)
    plt.close(fig)

    # ── model score ──
    fig, ax = plt.subplots(figsize=(10, 5))
    scores = [h["val_metrics"].get("model_score", 0) for h in history]
    ax.plot(epochs, scores, linewidth=2, color="#4a90d9", marker="o", markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(epochs, scores, 0, alpha=0.2,
                     color="#4a90d9" if all(s >= 0 for s in scores) else "#c76dba")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Model Score")
    ax.set_title("Model Score (0%=+0.68, 3%=0, 200%=-0.68, 400%=-1)")
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "model_score.png"), dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Split
# ═══════════════════════════════════════════════════════════════

def split_by_song(manifest, val_ratio=0.1):
    song_to_indices = {}
    for i, chart in enumerate(manifest["charts"]):
        sid = chart.get("beatmapset_id", str(i))
        song_to_indices.setdefault(sid, []).append(i)

    songs = list(song_to_indices.keys())
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * val_ratio))
    val_songs = set(songs[:n_val])

    train_idx, val_idx = [], []
    for sid, indices in song_to_indices.items():
        (val_idx if sid in val_songs else train_idx).extend(indices)
    return train_idx, val_idx


# ═══════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════

def train(args):
    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # run directory
    run_dir = os.path.join(SCRIPT_DIR, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evals"), exist_ok=True)

    print(f"Dataset: {args.dataset} ({manifest['total_charts']} charts)")
    print(f"Run: {args.run_name} → {run_dir}")

    # perf settings
    torch.backends.cudnn.benchmark = True
    amp_enabled = args.amp and args.device == "cuda"
    print(f"AMP: {'enabled' if amp_enabled else 'disabled'}, cudnn.benchmark: True")

    # save run config (don't overwrite on resume)
    config_path = os.path.join(run_dir, "config.json")
    if not args.resume or not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)

    # split
    random.seed(42)
    train_idx, val_idx = split_by_song(manifest, val_ratio=0.1)
    print(f"Train: {len(train_idx)} charts, Val: {len(val_idx)} charts")

    use_multi_target = args.multi_target or args.model_type == "framewise"
    sub = args.subsample
    train_ds = OnsetDataset(manifest, ds_dir, train_idx, augment=True, subsample=sub,
                            multi_target=use_multi_target)
    val_ds = OnsetDataset(manifest, ds_dir, val_idx, augment=False, subsample=sub,
                          multi_target=use_multi_target)
    if sub > 1:
        print(f"Subsample: 1/{sub}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # print class distribution (always useful)
    print_class_distribution(train_ds)

    # balanced sampling: oversample rare targets so the model sees the full range
    if args.balanced:
        counts = train_ds.class_counts
        sample_weights = np.zeros(len(train_ds), dtype=np.float64)
        stop_boost = 20.0 if getattr(args, 'stop_token', False) else 1.0
        for i, (ci, ei) in enumerate(train_ds.samples):
            target = train_ds._get_target(ci, ei)
            w = 1.0 / (counts[target] + 1) ** args.balance_power
            if target == N_CLASSES - 1 and stop_boost > 1.0:
                w *= stop_boost
            sample_weights[i] = min(w, 1.0)  # cap to prevent extreme weights on empty classes
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        print(f"Balanced sampling: ON (1/count^{args.balance_power} weights, replacement=True)")
        loss_weights = None  # sampling handles imbalance, no need for loss weights
    else:
        sampler = None
        loss_weights = compute_class_weights(train_ds, mode=args.weight_mode).to(args.device)

    # workers use mmap - each gets its own file handle, OS page cache shares data
    nw = args.workers
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None),  # no shuffle when using sampler
        sampler=sampler,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=False, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    # model
    if args.model_type == "event_embed":
        model = EventEmbeddingDetector(
            n_mels=80, d_model=args.d_model,
            n_layers=args.enc_layers + args.fusion_layers,
            n_heads=args.n_heads,
            n_classes=N_CLASSES, max_events=C_EVENTS, dropout=args.dropout,
            gap_ratios=args.gap_ratios,
            stop_token=getattr(args, 'stop_token', False),
            n_virtual_tokens=getattr(args, 'n_virtual_tokens', 0),
        ).to(args.device)
    elif args.model_type == "framewise":
        model = FramewiseOnsetDetector(
            n_mels=80, d_model=args.d_model,
            n_layers=args.enc_layers + args.fusion_layers,  # total transformer depth
            n_heads=args.n_heads,
            dropout=args.dropout,
        ).to(args.device)
    elif args.model_type == "context_film":
        model = ContextFiLMDetector(
            n_mels=80, d_model=args.d_model,
            enc_layers=args.enc_layers,
            gap_enc_layers=args.gap_enc_layers,
            fusion_layers=args.fusion_layers,
            n_heads=args.n_heads,
            n_classes=N_CLASSES, max_events=C_EVENTS, dropout=args.dropout,
            snippet_frames=args.snippet_frames,
        ).to(args.device)
    elif args.model_type == "interleaved":
        model = InterleavedOnsetDetector(
            n_mels=80, d_model=args.d_model,
            n_blocks=args.n_blocks,
            n_heads=args.n_heads,
            n_classes=N_CLASSES, max_events=C_EVENTS, dropout=args.dropout,
            snippet_frames=args.snippet_frames,
        ).to(args.device)
    elif args.model_type == "dual_stream":
        model = DualStreamOnsetDetector(
            n_mels=80, d_model=args.d_model,
            enc_layers=args.enc_layers,
            gap_enc_layers=args.gap_enc_layers,
            cross_attn_layers=args.cross_attn_layers,
            n_heads=args.n_heads,
            n_classes=N_CLASSES, max_events=C_EVENTS, dropout=args.dropout,
            snippet_frames=args.snippet_frames,
        ).to(args.device)
    else:
        model = OnsetDetector(
            n_mels=80, d_model=args.d_model,
            enc_layers=args.enc_layers,
            gap_enc_layers=args.gap_enc_layers,
            fusion_layers=args.fusion_layers,
            n_heads=args.n_heads,
            n_classes=N_CLASSES, max_events=C_EVENTS, dropout=args.dropout,
            snippet_frames=args.snippet_frames,
        ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters")

    # torch.compile (disabled on Windows - unstable CUDA cleanup at exit)
    import sys
    if hasattr(torch, "compile") and args.device == "cuda" and not args.no_compile and sys.platform != "win32":
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: failed ({e}), running eager")
    elif sys.platform == "win32":
        print("torch.compile: skipped (Windows)")

    # ── warm-start: load matching weights from a previous checkpoint ──
    if args.warm_start:
        print(f"Warm-start: loading weights from {args.warm_start}")
        ws_ckpt = torch.load(args.warm_start, map_location=args.device, weights_only=False)
        ws_state = ws_ckpt["model"]
        model_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in ws_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f"  Loaded {loaded} params, skipped {skipped}")
        ws_epoch = ws_ckpt.get("epoch", "?")
        ws_metrics = ws_ckpt.get("val_metrics", {})
        print(f"  Source: epoch {ws_epoch}, HIT={ws_metrics.get('hit_rate', 0):.1%}")

    use_framewise = args.model_type == "framewise"

    if use_framewise:
        # framewise uses simple BCE — loss computed inline in training loop
        criterion = None
        print("Loss: Framewise BCE (computed inline)")
    elif use_multi_target and args.dice_loss:
        criterion = FocalDiceMultiTargetLoss(
            good_pct=args.good_pct, fail_pct=args.fail_pct,
            frame_tolerance=args.frame_tolerance,
            empty_weight=args.empty_weight,
        ).to(args.device)
        print(f"Loss: FocalDiceMultiTargetLoss (empty_weight={args.empty_weight})")
    elif use_multi_target and args.sigmoid_loss:
        criterion = SigmoidMultiTargetLoss(
            good_pct=args.good_pct, fail_pct=args.fail_pct,
            frame_tolerance=args.frame_tolerance,
            empty_weight=args.empty_weight,
            pos_weight=args.pos_weight,
            focal_gamma=args.focal_gamma,
        ).to(args.device)
        print(f"Loss: SigmoidMultiTargetLoss (pos_weight={args.pos_weight}, empty_weight={args.empty_weight}, focal_gamma={args.focal_gamma})")
    elif use_multi_target:
        criterion = MultiTargetOnsetLoss(
            gamma=args.focal_gamma,
            good_pct=args.good_pct, fail_pct=args.fail_pct,
            hard_alpha=args.hard_alpha, frame_tolerance=args.frame_tolerance,
            empty_weight=args.empty_weight,
            recall_weight=args.recall_weight,
        ).to(args.device)
        print(f"Loss: MultiTargetOnsetLoss (hard_alpha={args.hard_alpha}, empty_weight={args.empty_weight}, recall_weight={args.recall_weight})")
    else:
        criterion = OnsetLoss(
            weight=loss_weights, gamma=args.focal_gamma,
            good_pct=args.good_pct, fail_pct=args.fail_pct,
            hard_alpha=args.hard_alpha, frame_tolerance=args.frame_tolerance,
            stop_weight=args.stop_weight,
        ).to(args.device)
    # Only pass trainable params to optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_loss = float("inf")
    start_epoch = 1

    # ── resume from checkpoint ──
    if args.resume:
        # find latest checkpoint in the run
        ckpt_files = sorted(
            [f for f in os.listdir(ckpt_dir) if (f.startswith("eval_") or f.startswith("epoch_")) and f.endswith(".pt") and f != "best.pt"]
        )
        if ckpt_files:
            resume_path = os.path.join(ckpt_dir, ckpt_files[-1])
            print(f"Resuming from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])

            # epoch may be fractional (mid-epoch eval); resume from next full epoch
            ckpt_epoch = ckpt["epoch"]
            start_epoch = int(ckpt_epoch) + (0 if ckpt_epoch == int(ckpt_epoch) else 1)
            if start_epoch <= int(ckpt_epoch):
                start_epoch = int(ckpt_epoch) + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))

            # reload history
            history_path = os.path.join(run_dir, "history.json")
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history = json.load(f)
                # trim to resumed eval_step (in case of partial writes)
                ckpt_eval = ckpt.get("eval_step", ckpt.get("epoch"))
                history = [h for h in history
                           if h.get("eval_step", h["epoch"]) <= ckpt_eval]

            print(f"  Resumed at epoch {ckpt_epoch}, eval_step={ckpt.get('eval_step', '?')}, "
                  f"best_val_loss={best_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
            print(f"  History: {len(history)} evals loaded, starting epoch {start_epoch}")
        else:
            print(f"WARNING: --resume set but no checkpoints found in {ckpt_dir}, starting fresh")

    # how many evals per epoch (1 = end of epoch only, 2 = halfway + end, etc.)
    evals_per_epoch = args.evals_per_epoch
    n_batches = len(train_loader)
    eval_step = 0  # global eval counter (for file naming)

    # if resuming, advance eval_step past existing evals
    if args.resume and history:
        eval_step = len(history)

    for epoch in range(start_epoch, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        train_total = 0
        # running metric accumulators (non-stop samples only)
        train_ns_total = 0
        train_miss_sum = 0
        train_w10_sum = 0
        train_w3_sum = 0
        train_hit_sum = 0
        train_score_sum = 0.0

        # compute batch indices where we trigger eval
        if evals_per_epoch > 1:
            eval_at_list = sorted(int(n_batches * (k + 1) / evals_per_epoch) - 1
                                  for k in range(evals_per_epoch - 1))
            eval_at = set(eval_at_list)
        else:
            eval_at_list = []
            eval_at = set()

        # boundaries for eval segment progress bar
        seg_boundaries = [0] + [b + 1 for b in eval_at_list] + [n_batches]
        seg_idx = 0  # which segment we're in
        seg_label = f"eval {eval_step + 1}" if evals_per_epoch > 1 else ""

        epoch_bar = tqdm(total=n_batches, desc=f"Epoch {epoch}/{args.epochs} (eval {eval_step+1}+)",
                         position=0, leave=True)
        seg_size = seg_boundaries[seg_idx + 1] - seg_boundaries[seg_idx]
        seg_bar = tqdm(total=seg_size,
                       desc=f"  → {seg_label} [{seg_idx+1}/{evals_per_epoch}]",
                       position=1, leave=False) if evals_per_epoch > 1 else None

        # rolling window for recent-batch stats
        RECENT_N = 50
        recent_buf = deque(maxlen=RECENT_N)  # (loss*bs, bs, ns, hit, miss, score_sum)

        for batch_idx, batch in enumerate(train_loader):
            if use_multi_target:
                mel, evt_off, evt_mask, cond, targets_padded, n_tgt = batch
                targets_padded = targets_padded.to(args.device, non_blocking=True)
                n_tgt = n_tgt.to(args.device, non_blocking=True)
                # nearest target for batch metrics
                target = targets_padded[:, 0].clone()
                target[n_tgt == 0] = N_CLASSES - 1
                target = target.clamp(0, N_CLASSES - 1)
            else:
                mel, evt_off, evt_mask, cond, target = batch
                target = target.to(args.device, non_blocking=True)

            mel = mel.to(args.device, non_blocking=True)
            evt_off = evt_off.to(args.device, non_blocking=True)
            evt_mask = evt_mask.to(args.device, non_blocking=True)
            cond = cond.to(args.device, non_blocking=True)

            with torch.autocast("cuda", enabled=amp_enabled):
                if use_framewise:
                    # build teacher-forcing target first
                    B_fw = mel.size(0)
                    fw_target = torch.zeros(B_fw, 125, device=mel.device)
                    safe_bins = targets_padded.clamp(min=0)
                    token_idx = (safe_bins // 4).clamp(max=124)
                    valid_mask_fw = targets_padded >= 0
                    idx = token_idx * valid_mask_fw.long()
                    fw_target.scatter_(1, idx, valid_mask_fw.float())

                    # framewise: model returns (B, 125) onset probs
                    # no teacher forcing — model must learn from audio + ramps
                    onset_probs = model(mel, evt_off, evt_mask, cond)
                    # BCE loss — no pos_weight, let natural 13% ratio guide learning
                    loss = F.binary_cross_entropy(onset_probs, fw_target)
                    # for tqdm metrics, use argmax-based nearest target
                    logits = torch.zeros(mel.size(0), N_CLASSES, device=mel.device)
                    # put onset_probs into first 125 positions scaled up
                    logits[:, :500:4] = onset_probs * 10  # rough proxy for compat
                else:
                    output = model(mel, evt_off, evt_mask, cond)
                    if isinstance(output, tuple):
                        onset_logits, stop_logit = output
                        # stop token mode: separate onset + stop losses
                        is_stop = (target == N_CLASSES - 1)
                        stop_target = is_stop.float()  # 1=stop, 0=onset

                        # stop loss: BCE averaged separately for STOP and onset samples
                        # prevents 47/48 easy onsets from drowning the 1/48 STOP signal
                        stop_bce = F.binary_cross_entropy_with_logits(
                            stop_logit, stop_target, reduction='none')
                        if is_stop.any():
                            stop_loss_pos = stop_bce[is_stop].mean()
                        else:
                            stop_loss_pos = torch.tensor(0.0, device=mel.device)
                        stop_loss_neg = stop_bce[~is_stop].mean()
                        stop_loss = (stop_loss_pos + stop_loss_neg) / 2.0

                        # onset loss: CE only on non-STOP samples
                        onset_mask = ~is_stop
                        if onset_mask.any():
                            onset_loss = criterion(onset_logits[onset_mask], target[onset_mask])
                        else:
                            onset_loss = torch.tensor(0.0, device=mel.device)

                        stop_w = getattr(args, 'stop_weight', 1.5)
                        loss = onset_loss + stop_w * stop_loss

                        # pad logits to N_CLASSES for metric compatibility
                        logits = F.pad(onset_logits, (0, 1), value=-10.0)
                        logits[:, N_CLASSES - 1] = stop_logit * 5.0
                    else:
                        logits = output
                        if use_multi_target:
                            loss = criterion(logits, targets_padded, n_tgt)
                        else:
                            loss = criterion(logits, target)

            # NaN safety: skip batch if loss or logits are NaN
            bs = mel.size(0)
            if loss.isnan().item() or logits.isnan().any().item():
                nan_logits = logits.isnan().any(dim=1)
                n_nan = nan_logits.sum().item()
                print(f"\n  WARNING: NaN at batch {batch_idx} ({n_nan}/{bs} samples)")
                print(f"    evt_mask valid counts: {(~evt_mask).sum(dim=1).tolist()}")
                print(f"    evt_off range: [{evt_off.min().item()}, {evt_off.max().item()}]")
                print(f"    mel range: [{mel.min().item():.2f}, {mel.max().item():.2f}]")
                torch.save({
                    "mel": mel.cpu(), "evt_off": evt_off.cpu(),
                    "evt_mask": evt_mask.cpu(), "cond": cond.cpu(),
                    "target": target.cpu(), "batch_idx": batch_idx,
                }, os.path.join(run_dir, "nan_batch_debug.pt"))
                print(f"    Saved debug batch to {run_dir}/nan_batch_debug.pt")
                optimizer.zero_grad(set_to_none=True)
                epoch_bar.update(1)
                if seg_bar is not None:
                    seg_bar.update(1)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * bs
            train_total += bs

            # batch metrics for tqdm (uses nearest target as proxy)
            b_loss_x_bs = loss.item() * bs
            b_bs = bs
            b_ns = 0; b_hit = 0; b_miss = 0; b_score = 0.0
            with torch.no_grad():
                pred = logits.argmax(1)
                ns = target < (N_CLASSES - 1)
                ns_count = ns.sum()
                if ns_count > 0:
                    t_ns = target[ns].float()
                    p_ns = pred[ns].float()
                    frame_err = (p_ns - t_ns).abs()
                    pct_err = ((p_ns + 1) / (t_ns + 1) - 1.0).abs()
                    b_ns = ns_count.item()
                    b_hit = ((pct_err <= 0.10) | (frame_err <= 2)).sum().item()
                    b_miss = (pct_err > 0.20).sum().item()
                    train_ns_total += b_ns
                    train_miss_sum += b_miss
                    train_w10_sum += b_hit
                    train_w3_sum += ((pct_err <= 0.03) | (frame_err <= 1)).sum().item()
                    train_hit_sum += b_hit

                    abs_lr = ((p_ns + 1).log() - (t_ns + 1).log()).abs()
                    thr = math.log(1.03)
                    max_p = math.log(5.0)
                    pen_range = max_p - thr
                    r_at_zero = (math.log(3.0) - thr) / pen_range
                    batch_scores = torch.where(
                        abs_lr <= thr,
                        (1.0 - abs_lr / thr) * r_at_zero,
                        -(((abs_lr - thr) / pen_range).clamp(max=1.0)),
                    )
                    batch_scores[frame_err <= 1] = r_at_zero
                    b_score = batch_scores.sum().item()
                    train_score_sum += b_score

            # stop metrics for tqdm
            b_stop_loss = 0.0
            b_stop_f1 = 0.0
            b_stop_rate = 0.0
            if isinstance(output, tuple):
                b_stop_loss = stop_loss.item()
                with torch.no_grad():
                    sp = (torch.sigmoid(stop_logit) > 0.5)
                    is_s = (target == N_CLASSES - 1)
                    tp = (sp & is_s).sum().item()
                    fp = (sp & ~is_s).sum().item()
                    fn = (~sp & is_s).sum().item()
                    b_stop_f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
                    b_stop_rate = sp.float().mean().item()
            recent_buf.append((b_loss_x_bs, b_bs, b_ns, b_hit, b_miss, b_score, b_stop_loss, b_stop_f1, b_stop_rate))

            # update bars
            epoch_bar.update(1)
            if seg_bar is not None:
                seg_bar.update(1)

            if (batch_idx + 1) % 50 == 0 or batch_idx in eval_at:
                avg_loss = train_loss / train_total if train_total > 0 else 0
                # recent window stats
                r_loss_sum = sum(r[0] for r in recent_buf)
                r_bs_sum = sum(r[1] for r in recent_buf)
                r_ns_sum = sum(r[2] for r in recent_buf)
                r_loss = r_loss_sum / r_bs_sum if r_bs_sum > 0 else 0
                if train_ns_total > 0 and r_ns_sum > 0:
                    hit = train_hit_sum / train_ns_total
                    miss = train_miss_sum / train_ns_total
                    score = train_score_sum / train_ns_total
                    r_hit = sum(r[3] for r in recent_buf) / r_ns_sum
                    r_miss = sum(r[4] for r in recent_buf) / r_ns_sum
                    r_score = sum(r[5] for r in recent_buf) / r_ns_sum
                    stats = (f"L={avg_loss:.3f}|{r_loss:.3f} "
                             f"HIT={hit:.1%}|{r_hit:.1%} "
                             f"miss={miss:.1%}|{r_miss:.1%} "
                             f"score={score:+.3f}|{r_score:+.3f}")
                    # stop token stats
                    r_sL = sum(r[6] for r in recent_buf) / len(recent_buf) if recent_buf else 0
                    r_sF1 = sum(r[7] for r in recent_buf) / len(recent_buf) if recent_buf else 0
                    r_sR = sum(r[8] for r in recent_buf) / len(recent_buf) if recent_buf else 0
                    if r_sL > 0:
                        stats += f" sL={r_sL:.3f} sF1={r_sF1:.2f} sR={r_sR:.3f}"
                else:
                    stats = f"L={avg_loss:.3f}|{r_loss:.3f}"
                epoch_bar.set_postfix_str(stats)

            # ── mid-epoch eval checkpoint ──
            if batch_idx in eval_at:
                if seg_bar is not None:
                    seg_bar.close()
                sub_frac = (batch_idx + 1) / n_batches
                sub_train_loss = train_loss / train_total
                eval_step += 1
                _run_eval(
                    model, val_loader, criterion, args, amp_enabled,
                    eval_step, epoch + sub_frac, sub_train_loss,
                    run_dir, ckpt_dir, history, scheduler, optimizer,
                    scaler, best_val_loss,
                )
                if history and history[-1]["val_loss"] < best_val_loss:
                    best_val_loss = history[-1]["val_loss"]
                model.train()  # back to training mode

                # start next segment bar
                seg_idx += 1
                if seg_idx < len(seg_boundaries) - 1:
                    seg_size = seg_boundaries[seg_idx + 1] - seg_boundaries[seg_idx]
                    seg_bar = tqdm(total=seg_size,
                                   desc=f"  → eval {eval_step + 1} [{seg_idx+1}/{evals_per_epoch}]",
                                   position=1, leave=False)

        epoch_bar.close()
        if seg_bar is not None:
            seg_bar.close()

        scheduler.step()
        train_loss /= train_total

        # ── end-of-epoch eval ──
        eval_step += 1
        _run_eval(
            model, val_loader, criterion, args, amp_enabled,
            eval_step, float(epoch), train_loss,
            run_dir, ckpt_dir, history, scheduler, optimizer,
            scaler, best_val_loss,
        )
        if history and history[-1]["val_loss"] < best_val_loss:
            best_val_loss = history[-1]["val_loss"]

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Run dir: {run_dir}")


def _run_eval(model, val_loader, criterion, args, amp_enabled,
              eval_step, epoch_frac, train_loss,
              run_dir, ckpt_dir, history, scheduler, optimizer,
              scaler, best_val_loss):
    """Run validation, benchmarks, save graphs/checkpoints/history."""
    use_mt = args.multi_target or args.model_type == "framewise"
    use_sigmoid = getattr(args, 'sigmoid_loss', False) or getattr(args, 'dice_loss', False) or args.model_type == "framewise"
    is_framewise = args.model_type == "framewise"
    val_loss, val_extra = validate_and_collect(
        model, val_loader, criterion, args.device, amp_enabled=amp_enabled,
        sigmoid_mode=use_sigmoid, multi_target=use_mt, framewise=is_framewise,
    )
    # backward-compat: nearest-target metrics (always computed)
    val_targets = val_extra["targets"]
    val_preds = val_extra["preds"]
    val_metrics = compute_metrics(val_targets, val_preds)

    # multi-target metrics
    mt_metrics = None
    if use_mt and "probs" in val_extra:
        mt_metrics = compute_multi_target_metrics(
            val_extra["targets_padded"], val_extra["n_targets"],
            val_extra["probs"], threshold=args.threshold,
        )
        # merge key mt metrics into val_metrics for history
        val_metrics["event_recall_hit"] = mt_metrics["event_recall_hit"]
        val_metrics["event_miss_rate"] = mt_metrics["event_miss_rate"]
        val_metrics["pred_precision_hit"] = mt_metrics["pred_precision_hit"]
        val_metrics["hallucination_rate"] = mt_metrics["hallucination_rate"]
        val_metrics["f1_hit"] = mt_metrics["f1_hit"]
        val_metrics["avg_preds_per_window"] = mt_metrics["avg_preds_per_window"]
        val_metrics["threshold"] = mt_metrics["threshold"]

    audio_metrics = None  # unified model has no separate audio path

    tag = f"{epoch_frac:.2f}" if epoch_frac != int(epoch_frac) else f"{int(epoch_frac)}"
    if mt_metrics:
        print(f"  Eval {eval_step} (epoch {tag}): loss={train_loss:.4f}/{val_loss:.4f} | "
              f"eHIT={mt_metrics['event_recall_hit']:.1%} "
              f"eMISS={mt_metrics['event_miss_rate']:.1%} | "
              f"pHIT={mt_metrics['pred_precision_hit']:.1%} "
              f"pHALL={mt_metrics['hallucination_rate']:.1%} | "
              f"F1={mt_metrics['f1_hit']:.3f} "
              f"preds/win={mt_metrics['avg_preds_per_window']:.1f} | "
              f"(nearest: HIT={val_metrics.get('hit_rate', 0):.1%} "
              f"MISS={val_metrics.get('miss_rate', 1):.1%}) | "
              f"score={val_metrics.get('model_score', 0):+.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
    else:
        audio_hit_str = f" audio_HIT={audio_metrics['hit_rate']:.1%}" if audio_metrics else ""
        print(f"  Eval {eval_step} (epoch {tag}): loss={train_loss:.4f}/{val_loss:.4f} | "
              f"HIT={val_metrics.get('hit_rate', 0):.1%}{audio_hit_str} "
              f"GOOD={val_metrics.get('good_rate', 0):.1%} "
              f"MISS={val_metrics.get('miss_rate', 1):.1%} | "
              f"exact={val_metrics.get('exact_match', 0):.1%} ±1f={val_metrics.get('within_1_frame', 0):.1%} "
              f"±2f={val_metrics.get('within_2_frames', 0):.1%} | "
              f"≤3%={val_metrics.get('within_3pct', 0):.1%} ≤10%={val_metrics.get('within_10pct', 0):.1%} | "
              f"stop_f1={val_metrics['stop_f1']:.3f} | "
              f"score={val_metrics.get('model_score', 0):+.3f} | "
              f"uniq={val_metrics.get('unique_preds', 0)} | lr={scheduler.get_last_lr()[0]:.2e}")

    # ── ablation benchmarks ──
    torch.cuda.empty_cache()
    bench_results = run_benchmarks(model, val_loader, args.device, amp_enabled=amp_enabled,
                                   multi_target=use_mt)
    print_benchmarks(bench_results)
    save_benchmark_data(bench_results, eval_step, run_dir)

    # save eval graphs
    save_eval_graphs(val_targets, val_preds, val_metrics, eval_step, run_dir,
                     extra=val_extra, mt_metrics=mt_metrics)

    # ── save eval data ──
    # strip non-serializable arrays from mt_metrics before saving
    mt_save = None
    if mt_metrics:
        mt_save = {k: v for k, v in mt_metrics.items() if not k.startswith("_")}
    epoch_data = {
        "eval_step": eval_step,
        "epoch": epoch_frac,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "lr": scheduler.get_last_lr()[0],
        "val_metrics": val_metrics,
        "benchmarks": _serializable(bench_results),
    }
    if mt_save is not None:
        epoch_data["multi_target_metrics"] = mt_save
    if audio_metrics is not None:
        epoch_data["audio_metrics"] = audio_metrics
    history.append(epoch_data)

    # save eval JSON
    with open(os.path.join(run_dir, "evals", f"eval_{eval_step:03d}.json"), "w") as f:
        json.dump(epoch_data, f, indent=2)

    # save model checkpoint
    ckpt = {
        "eval_step": eval_step,
        "epoch": epoch_frac,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_metrics.get("accuracy", 0),
        "val_metrics": val_metrics,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(ckpt_dir, f"eval_{eval_step:03d}.pt"))
    if val_loss < best_val_loss:
        torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
        print(f"  → New best val_loss: {val_loss:.4f}")

    # save full history + training curves
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_training_curves(history, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train onset detection model")
    parser.add_argument("dataset", help="Dataset name (under datasets/)")
    parser.add_argument("--run-name", required=True, help="Run name for saving outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--model-type", default="unified", choices=["unified", "dual_stream", "interleaved", "context_film", "framewise", "event_embed"],
                        help="Model architecture: unified, dual_stream, interleaved, context_film, framewise, or event_embed (exp 42+)")
    parser.add_argument("--enc-layers", type=int, default=4, help="AudioEncoder transformer layers")
    parser.add_argument("--gap-enc-layers", type=int, default=2, help="GapEncoder self-attention layers")
    parser.add_argument("--cross-attn-layers", type=int, default=2, help="Cross-attention fusion layers (dual_stream only)")
    parser.add_argument("--n-blocks", type=int, default=4, help="Interleaved blocks of [self+cross] (interleaved only)")
    parser.add_argument("--fusion-layers", type=int, default=4, help="Fusion self-attention layers over [audio+gap] tokens")
    parser.add_argument("--snippet-frames", type=int, default=10, help="Mel frames per audio snippet (~5ms each, default 10 = ~50ms)")
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gap-ratios", action="store_true", default=True, help="Add gap ratio features to event embeddings (exp 45+)")
    parser.add_argument("--no-gap-ratios", dest="gap_ratios", action="store_false", help="Disable gap ratio features")
    parser.add_argument("--stop-token", action="store_true", default=False, help="Use learned STOP query token (exp 47e+)")
    parser.add_argument("--n-virtual-tokens", type=int, default=0, help="Virtual tokens for out-of-window context (exp 49+, 0=off)")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0=disabled, default 0)")
    parser.add_argument("--good-pct", type=float, default=0.03, help="Soft target plateau threshold (ratio, default 3%%)")
    parser.add_argument("--fail-pct", type=float, default=0.20, help="Soft target hard cutoff (ratio, default 20%%)")
    parser.add_argument("--hard-alpha", type=float, default=0.5, help="Weight of hard CE in mixed loss (0=pure soft, 1=pure hard)")
    parser.add_argument("--frame-tolerance", type=int, default=2, help="±N frame floor for soft targets (default 2 = ±10ms)")
    parser.add_argument("--stop-weight", type=float, default=1.5, help="Extra loss multiplier when target is STOP (default 1.5)")
    parser.add_argument("--ctx-loss-weight", type=float, default=0.0, help="Auxiliary context loss weight (0=disabled, default 0)")
    parser.add_argument("--balanced", action="store_true", default=True, help="Balanced sampling (default on)")
    parser.add_argument("--no-balanced", dest="balanced", action="store_false", help="Disable balanced sampling")
    parser.add_argument("--balance-power", type=float, default=0.5, help="Balanced sampling exponent: 1/count^power (default 0.5=sqrt, 0.2=fifth root, 1.0=inverse)")
    parser.add_argument("--weight-mode", default="log", choices=["log", "sqrt", "none"], help="Class weight mode (only when --no-balanced)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--subsample", type=int, default=1, help="Train on every Nth sample (e.g. 4 = 4x less data)")
    parser.add_argument("--evals-per-epoch", type=int, default=1, help="Run eval N times per epoch (default 1 = end only)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in the run")
    parser.add_argument("--warm-start", type=str, default=None, help="Path to checkpoint to load matching weights from")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (experimental on Windows)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # multi-target training (exp 36+)
    parser.add_argument("--multi-target", action="store_true", default=False,
                        help="Multi-target training: predict all onsets in forward window")
    parser.add_argument("--empty-weight", type=float, default=1.5,
                        help="Loss multiplier for empty windows (multi-target, default 1.5)")
    parser.add_argument("--recall-weight", type=float, default=1.0,
                        help="Per-onset recall loss weight (multi-target, default 1.0)")
    parser.add_argument("--sigmoid-loss", action="store_true",
                        help="Use per-bin sigmoid BCE instead of softmax CE (multi-target)")
    parser.add_argument("--dice-loss", action="store_true",
                        help="Use focal dice loss instead of softmax CE (multi-target)")
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="Positive class weight for sigmoid BCE (default 5.0)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Threshold for peak extraction (multi-target, default 0.05)")
    args = parser.parse_args()
    train(args)
