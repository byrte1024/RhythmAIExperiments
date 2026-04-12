"""Training script for chart quality evaluator (exp 66-1).

Pairwise Bradley-Terry training on corruption pairs + cross-set rating pairs.

Usage:
    python classifier_train.py taiko_v2 --run-name eval_experiment_66_1
"""
import os
import json
import random
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from classifier_model import ChartQualityEvaluator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── constants ──
WINDOW_FRAMES = 2000   # 10s at 5ms/frame
WINDOW_STRIDE = 4      # conv stem downsamples 4x → 500 tokens
MAX_EVENTS = 256       # max events per window (covers p99)
BIN_MS = 5.0           # ms per mel frame

# corruption levels (0=clean, 1=light, 2=med, 3=high, 4=garbage)
CORRUPTION_NAMES = ["CLEAN", "LIGHT", "MED", "HIGH", "GARBAGE"]


# ──────────────────────────────────────────────
#  Global gap distribution (lazy-loaded)
# ──────────────────────────────────────────────

_gap_dist_cache = {"values": None, "probs": None}


def _load_gap_distribution(ds_dir):
    """Compute and cache the global gap distribution from all charts."""
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
#  Corruption functions
# ──────────────────────────────────────────────

def _cleanup_events(events, mel_frames):
    """Sort, merge within 2 bins, remove negatives, clamp."""
    events = np.sort(events)
    events = events[events >= 0]
    events = np.clip(events, 0, mel_frames - 1)
    # merge within 2 bins
    if len(events) <= 1:
        return events
    keep = [events[0]]
    for e in events[1:]:
        if e - keep[-1] > 2:
            keep.append(e)
    return np.array(keep, dtype=np.int64)


def corrupt_events(events, level, mel_frames, gap_values, gap_probs, rng):
    """Apply corruption at the given level. Returns corrupted event array.

    level: 0=clean, 1=light, 2=med, 3=high, 4=garbage
    """
    if level == 0:
        return events.copy()

    events = events.copy().astype(np.int64)
    n = len(events)

    if level == 4:
        # GARBAGE: fully random gaps from global distribution
        if n == 0:
            return events
        n_events = n
        gaps = rng.choice(gap_values, size=n_events, p=gap_probs)
        new_events = np.cumsum(gaps)
        # shift to start near original chart start
        if len(events) > 0:
            new_events = new_events - new_events[0] + events[0]
        return _cleanup_events(new_events, mel_frames)

    # levels 1-3: parameterized corruption
    if level == 1:
        jitter_bins = 2        # ±10ms
        all_jitter_bins = 2    # ±10ms
        insert_center_p = 0.01
        delete_p = 0.01
        insert_offset_p = 0.01
    elif level == 2:
        jitter_bins = 6        # ±30ms
        all_jitter_bins = 6    # ±30ms
        insert_center_p = 0.05
        delete_p = 0.05
        insert_offset_p = 0.05
    else:  # level 3
        jitter_bins = 20       # ±100ms
        all_jitter_bins = 50   # ±250ms
        insert_center_p = 0.25
        delete_p = 0.15
        insert_offset_p = 0.10

    # 1. deletions
    if delete_p > 0 and n > 1:
        keep_mask = rng.random(n) >= delete_p
        if keep_mask.sum() == 0:
            keep_mask[0] = True  # keep at least one
        events = events[keep_mask]
        n = len(events)

    # 2. insertions (center of gap)
    if insert_center_p > 0 and n >= 2:
        new_events = []
        for i in range(n - 1):
            if rng.random() < insert_center_p:
                mid = (events[i] + events[i + 1]) // 2
                new_events.append(mid)
        if new_events:
            events = np.concatenate([events, new_events])
            n = len(events)

    # 3. insertions (offset from existing event)
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
        jitter = rng.integers(-jitter_bins, jitter_bins + 1, size=n)
        events = events + jitter

    # 5. all-event jitter
    if all_jitter_bins > 0 and n > 0:
        shift = rng.integers(-all_jitter_bins, all_jitter_bins + 1)
        events = events + shift

    return _cleanup_events(events, mel_frames)


# ──────────────────────────────────────────────
#  Audio augmentation (rating pairs only)
# ──────────────────────────────────────────────

def augment_mel(mel, rng):
    """Apply random audio augmentation to mel spectrogram (in log-mel space).

    mel: (80, T) numpy array
    """
    mel = mel.copy()

    # random gain: ±6dB → ±0.3 in log-mel
    gain = rng.uniform(-0.3, 0.3)
    mel = mel + gain

    # freq masking: zero out 2-4 random mel bins
    n_freq_mask = rng.integers(2, 5)
    for _ in range(n_freq_mask):
        bin_idx = rng.integers(0, mel.shape[0])
        mel[bin_idx, :] = 0.0

    # time masking: zero out 2-4 random time segments
    n_time_mask = rng.integers(2, 5)
    T = mel.shape[1]
    for _ in range(n_time_mask):
        width = rng.integers(50, 201)  # 250ms-1s
        start = rng.integers(0, max(1, T - width))
        mel[:, start:start + width] = 0.0

    # additive noise
    noise_std = rng.uniform(0.0, 0.3)
    mel = mel + rng.standard_normal(mel.shape).astype(np.float32) * noise_std

    return mel


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class PairDataset(Dataset):
    """Generates pairs for pairwise quality training.

    Each sample is a pair: (mel_a, events_a, mask_a, star_a, mel_b, events_b, mask_b, star_b, margin)
    where a is better than b, and margin is the level gap.
    """

    def __init__(self, manifest, ds_dir, chart_indices, mode="corruption",
                 corruption_ratio=0.6, augment_audio=True):
        """
        mode: "corruption" (phase 1) or "mixed" (phase 2)
        chart_indices: indices into manifest["charts"] to use
        """
        self.ds_dir = ds_dir
        self.mel_dir = os.path.join(ds_dir, "mels")
        self.evt_dir = os.path.join(ds_dir, "events")
        self.mode = mode
        self.corruption_ratio = corruption_ratio
        self.augment_audio = augment_audio

        self.charts = [manifest["charts"][i] for i in chart_indices]

        # preload events
        self.events = []
        for chart in self.charts:
            evt = np.load(os.path.join(self.evt_dir, chart["event_file"]))
            self.events.append(evt)

        # for rating pairs: group by beatmapset_id, index charts with ratings
        self.rated_charts = []  # (chart_idx, rating, star_rating)
        for i, chart in enumerate(self.charts):
            if "rating" in chart:
                self.rated_charts.append((i, chart["rating"], chart["star_rating"]))

        # load global gap distribution
        self.gap_values, self.gap_probs = _load_gap_distribution(ds_dir)

        # mel cache (per-worker, mmap)
        self._mel_cache = {}

    def _get_mel(self, mel_file):
        if mel_file not in self._mel_cache:
            self._mel_cache[mel_file] = np.load(
                os.path.join(self.mel_dir, mel_file), mmap_mode="r"
            )
        return self._mel_cache[mel_file]

    def _extract_window(self, chart_idx, events, rng, do_augment=False):
        """Extract a random 10s window from a chart.

        Returns: (mel_window, events_in_window, event_mask, star_rating)
        """
        chart = self.charts[chart_idx]
        mel = self._get_mel(chart["mel_file"])
        total_frames = mel.shape[1]

        # sample random window start
        if total_frames <= WINDOW_FRAMES:
            start = 0
        else:
            start = rng.integers(0, total_frames - WINDOW_FRAMES)
        end = start + WINDOW_FRAMES

        mel_window = mel[:, start:min(total_frames, end)].astype(np.float32)
        # pad if needed
        if mel_window.shape[1] < WINDOW_FRAMES:
            pad = WINDOW_FRAMES - mel_window.shape[1]
            mel_window = np.pad(mel_window, ((0, 0), (0, pad)), mode="constant")

        if do_augment:
            mel_window = augment_mel(mel_window, rng)

        # events in window (shifted to window-relative)
        mask = (events >= start) & (events < end)
        evt_window = events[mask].astype(np.int64) - start

        # pad to MAX_EVENTS
        n_evt = min(len(evt_window), MAX_EVENTS)
        event_arr = np.zeros(MAX_EVENTS, dtype=np.int64)
        event_mask = np.ones(MAX_EVENTS, dtype=bool)
        if n_evt > 0:
            event_arr[:n_evt] = evt_window[:n_evt]
            event_mask[:n_evt] = False

        star = chart.get("star_rating", 4.0)
        return mel_window, event_arr, event_mask, np.float32(star)

    def __len__(self):
        # virtual size: each chart can generate many pairs
        return len(self.charts) * 10

    def __getitem__(self, idx):
        rng = np.random.default_rng()

        if self.mode == "corruption" or (self.mode == "mixed" and rng.random() < self.corruption_ratio):
            return self._corruption_pair(rng)
        else:
            return self._rating_pair(rng)

    def _corruption_pair(self, rng):
        """Sample a corruption pair from a single chart."""
        ci = rng.integers(0, len(self.charts))
        events = self.events[ci]
        chart = self.charts[ci]
        mel_frames = self._get_mel(chart["mel_file"]).shape[1]

        # pick two different levels
        level_a, level_b = sorted(rng.choice(5, size=2, replace=False))
        # level_a < level_b → level_a is less corrupted (better)
        margin = level_b - level_a

        events_a = corrupt_events(events, level_a, mel_frames,
                                  self.gap_values, self.gap_probs, rng)
        events_b = corrupt_events(events, level_b, mel_frames,
                                  self.gap_values, self.gap_probs, rng)

        # extract same window from both (same audio)
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

        # same mel for both (no augmentation on corruption pairs)
        return (mel_window, evt_a, mask_a, star,
                mel_window.copy(), evt_b, mask_b, star,
                np.float32(margin))

    def _rating_pair(self, rng):
        """Sample a cross-set rating pair."""
        if len(self.rated_charts) < 2:
            return self._corruption_pair(rng)

        # try to find a valid pair (similar star_rating, different rating)
        for _ in range(50):
            i, j = rng.choice(len(self.rated_charts), size=2, replace=False)
            ci_a, rating_a, star_a = self.rated_charts[i]
            ci_b, rating_b, star_b = self.rated_charts[j]

            # same beatmapset → skip (same rating guaranteed)
            if self.charts[ci_a].get("beatmapset_id") == self.charts[ci_b].get("beatmapset_id"):
                continue

            # star rating within ±0.5
            if abs(star_a - star_b) > 0.5:
                continue

            # rating gap >= 1.0
            gap = abs(rating_a - rating_b)
            if gap < 1.0:
                continue

            # a is higher-rated (better)
            if rating_b > rating_a:
                ci_a, ci_b = ci_b, ci_a
                rating_a, rating_b = rating_b, rating_a

            margin = gap  # continuous margin for rating pairs

            mel_a, evt_a, mask_a, sr_a = self._extract_window(
                ci_a, self.events[ci_a], rng, do_augment=self.augment_audio)
            mel_b, evt_b, mask_b, sr_b = self._extract_window(
                ci_b, self.events[ci_b], rng, do_augment=self.augment_audio)

            return (mel_a, evt_a, mask_a, sr_a,
                    mel_b, evt_b, mask_b, sr_b,
                    np.float32(margin))

        # fallback: corruption pair if no valid rating pair found
        return self._corruption_pair(rng)


# ──────────────────────────────────────────────
#  Loss
# ──────────────────────────────────────────────

def bradley_terry_loss(score_a, score_b, margin, alpha=0.1):
    """Bradley-Terry pairwise loss with adaptive margin.

    score_a should be higher (a is better).
    margin: level gap (1-4 for corruption, continuous for ratings).
    """
    diff = score_a - score_b - alpha * margin
    loss = -F.logsigmoid(diff)
    return loss.mean()


# ──────────────────────────────────────────────
#  Validation (heavy data collection)
# ──────────────────────────────────────────────

def validate_and_collect(model, val_loader, device, alpha=0.1):
    """Run validation, collect per-pair data. Returns (metrics_dict, pairs_list)."""
    model.eval()
    total_loss = 0.0
    total = 0
    pairs = []  # per-pair records

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            mel_a, evt_a, mask_a, star_a, mel_b, evt_b, mask_b, star_b, margin = [
                x.to(device) for x in batch
            ]

            score_a = model(mel_a, evt_a, mask_a, star_a)
            score_b = model(mel_b, evt_b, mask_b, star_b)

            loss = bradley_terry_loss(score_a, score_b, margin, alpha)
            total_loss += loss.item() * mel_a.size(0)
            total += mel_a.size(0)

            # collect per-pair data
            sa = score_a.cpu().numpy()
            sb = score_b.cpu().numpy()
            mg = margin.cpu().numpy()
            for i in range(mel_a.size(0)):
                pairs.append({
                    "score_a": float(sa[i]),
                    "score_b": float(sb[i]),
                    "margin": float(mg[i]),
                    "diff": float(sa[i] - sb[i]),
                    "correct": bool(sa[i] > sb[i]),
                    "confidence": float(abs(sa[i] - sb[i])),
                    "star_a": float(star_a[i].cpu()),
                    "star_b": float(star_b[i].cpu()),
                })

    # aggregate metrics
    n = max(total, 1)
    correct = sum(1 for p in pairs if p["correct"])
    diffs = np.array([p["diff"] for p in pairs])
    margins = np.array([p["margin"] for p in pairs])
    confidences = np.array([p["confidence"] for p in pairs])

    # per-margin accuracy
    margin_acc = {}
    for m_int in range(1, 5):
        mask = (np.round(margins) == m_int)
        if mask.sum() > 0:
            margin_acc[m_int] = float(np.mean([p["correct"] for p, m in zip(pairs, mask) if m]))

    # confidence stats for correct vs incorrect
    correct_conf = confidences[[p["correct"] for p in pairs]]
    wrong_conf = confidences[[not p["correct"] for p in pairs]]

    metrics = {
        "val_loss": total_loss / n,
        "pair_accuracy": correct / n,
        "n_pairs": total,
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "mean_confidence": float(np.mean(confidences)),
        "correct_confidence": float(np.mean(correct_conf)) if len(correct_conf) > 0 else 0.0,
        "wrong_confidence": float(np.mean(wrong_conf)) if len(wrong_conf) > 0 else 0.0,
        "pct_high_conf_correct": float(np.mean([p["correct"] for p in pairs
                                                  if p["confidence"] > np.median(confidences)]))
        if len(pairs) > 0 else 0.0,
    }
    for m, acc in margin_acc.items():
        metrics[f"acc_margin_{m}"] = acc

    model.train()
    return metrics, pairs


def compute_score_by_level(model, dataset, device, n_samples=200):
    """Score random windows at each corruption level.

    Returns: (summary_dict, raw_scores_dict)
    summary_dict: level_name → {mean, std, median, p10, p90, n}
    raw_scores_dict: level_name → list of floats
    """
    model.eval()
    rng = np.random.default_rng(42)
    scores_by_level = {i: [] for i in range(5)}

    with torch.no_grad():
        for _ in range(n_samples):
            ci = rng.integers(0, len(dataset.charts))
            chart = dataset.charts[ci]
            events = dataset.events[ci]
            mel_frames = dataset._get_mel(chart["mel_file"]).shape[1]

            for level in range(5):
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
    for level in range(5):
        s = np.array(scores_by_level[level])
        name = CORRUPTION_NAMES[level]
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


def compute_pair_matrix(model, dataset, device, n_samples=100):
    """Compute accuracy for all 10 corruption level pair types.

    Returns: dict mapping "LEVEL_A_vs_LEVEL_B" → {accuracy, mean_diff, n}
    """
    model.eval()
    rng = np.random.default_rng(42)
    pair_results = {}

    for la in range(5):
        for lb in range(la + 1, 5):
            key = f"{CORRUPTION_NAMES[la]}_vs_{CORRUPTION_NAMES[lb]}"
            correct = 0
            diffs = []

            with torch.no_grad():
                for _ in range(n_samples):
                    ci = rng.integers(0, len(dataset.charts))
                    chart = dataset.charts[ci]
                    events = dataset.events[ci]
                    mel_frames = dataset._get_mel(chart["mel_file"]).shape[1]

                    events_a = corrupt_events(events, la, mel_frames,
                                              dataset.gap_values, dataset.gap_probs, rng)
                    events_b = corrupt_events(events, lb, mel_frames,
                                              dataset.gap_values, dataset.gap_probs, rng)

                    mel_w, _, _, star = dataset._extract_window(ci, events, rng)
                    # use same window position for both
                    def _to_window(evts, mel_w_shape):
                        # events already in dataset coords, extract_window picked a start
                        # simpler: just use the full chart extract
                        n = min(len(evts), MAX_EVENTS)
                        arr = np.zeros(MAX_EVENTS, dtype=np.int64)
                        m = np.ones(MAX_EVENTS, dtype=bool)
                        if n > 0:
                            arr[:n] = evts[:n]
                            m[:n] = False
                        return arr, m

                    # re-extract to get consistent window
                    mel_w, evt_a_arr, mask_a, star = dataset._extract_window(ci, events_a, rng)
                    _, evt_b_arr, mask_b, _ = dataset._extract_window(ci, events_b, rng)

                    mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)
                    evt_a_t = torch.from_numpy(evt_a_arr).unsqueeze(0).to(device)
                    mask_a_t = torch.from_numpy(mask_a).unsqueeze(0).to(device)
                    evt_b_t = torch.from_numpy(evt_b_arr).unsqueeze(0).to(device)
                    mask_b_t = torch.from_numpy(mask_b).unsqueeze(0).to(device)
                    star_t = torch.tensor([star], device=device)

                    sa = model(mel_t, evt_a_t, mask_a_t, star_t).item()
                    sb = model(mel_t, evt_b_t, mask_b_t, star_t).item()

                    if sa > sb:
                        correct += 1
                    diffs.append(sa - sb)

            pair_results[key] = {
                "accuracy": correct / n_samples,
                "mean_diff": float(np.mean(diffs)),
                "std_diff": float(np.std(diffs)),
                "margin": lb - la,
                "n": n_samples,
            }

    model.train()
    return pair_results


# ──────────────────────────────────────────────
#  Visualization
# ──────────────────────────────────────────────

def save_eval_plots(eval_step, eval_dir, level_summary, level_raw, pair_matrix, pairs):
    """Save per-eval diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = os.path.join(eval_dir, f"eval_{eval_step:03d}")

    # ── 1. Score distributions by corruption level ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    positions = []
    for i, name in enumerate(CORRUPTION_NAMES):
        scores = level_raw[name]
        parts = ax.violinplot([scores], positions=[i], showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
    ax.set_xticks(range(5))
    ax.set_xticklabels(CORRUPTION_NAMES)
    ax.set_ylabel("Quality Score")
    ax.set_title(f"Score Distribution by Corruption Level (eval {eval_step})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{prefix}_score_dist.png", dpi=150)
    plt.close(fig)

    # ── 2. Pair matrix heatmap ──
    if pair_matrix:
        labels = []
        accs = []
        for key in sorted(pair_matrix.keys(), key=lambda k: pair_matrix[k]["margin"]):
            labels.append(key.replace("_vs_", "\nvs\n"))
            accs.append(pair_matrix[key]["accuracy"])

        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(range(len(labels)), accs, color=[
            colors[pair_matrix[k]["margin"] - 1] for k in sorted(pair_matrix.keys(),
                                                                   key=lambda k: pair_matrix[k]["margin"])
        ])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([k.replace("_vs_", " vs ") for k in sorted(
            pair_matrix.keys(), key=lambda k: pair_matrix[k]["margin"])],
            rotation=45, ha="right", fontsize=8)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
        ax.set_ylabel("Pairwise Accuracy")
        ax.set_title(f"Accuracy per Pair Type (eval {eval_step})")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(f"{prefix}_pair_matrix.png", dpi=150)
        plt.close(fig)

    # ── 3. Score diff histogram (correct vs incorrect) ──
    if pairs:
        correct_diffs = [p["diff"] for p in pairs if p["correct"]]
        wrong_diffs = [p["diff"] for p in pairs if not p["correct"]]

        fig, ax = plt.subplots(figsize=(10, 5))
        if correct_diffs:
            ax.hist(correct_diffs, bins=50, alpha=0.6, color="#2ecc71", label=f"Correct ({len(correct_diffs)})")
        if wrong_diffs:
            ax.hist(wrong_diffs, bins=50, alpha=0.6, color="#e74c3c", label=f"Wrong ({len(wrong_diffs)})")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("score_a - score_b")
        ax.set_ylabel("Count")
        ax.set_title(f"Score Difference Distribution (eval {eval_step})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_diff_hist.png", dpi=150)
        plt.close(fig)

    # ── 4. Margin vs accuracy scatter ──
    if pairs:
        margins_arr = np.array([p["margin"] for p in pairs])
        correct_arr = np.array([float(p["correct"]) for p in pairs])
        unique_margins = sorted(set(np.round(margins_arr, 1)))

        fig, ax = plt.subplots(figsize=(10, 5))
        margin_accs = []
        for m in unique_margins:
            mask = np.abs(margins_arr - m) < 0.05
            if mask.sum() > 5:
                margin_accs.append((m, correct_arr[mask].mean(), mask.sum()))
        if margin_accs:
            ms, accs, counts = zip(*margin_accs)
            ax.scatter(ms, accs, s=[c * 2 for c in counts], alpha=0.7, color="#3498db")
            for m, a, c in zip(ms, accs, counts):
                ax.annotate(f"n={c}", (m, a), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, alpha=0.7)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Margin (level gap)")
        ax.set_ylabel("Pairwise Accuracy")
        ax.set_title(f"Accuracy vs Margin (eval {eval_step})")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{prefix}_margin_vs_acc.png", dpi=150)
        plt.close(fig)

    # ── 5. Confidence calibration: binned accuracy vs confidence ──
    if pairs:
        confs = np.array([p["confidence"] for p in pairs])
        corrects = np.array([float(p["correct"]) for p in pairs])

        # bin by confidence decile
        n_bins = 10
        bin_edges = np.percentile(confs, np.linspace(0, 100, n_bins + 1))
        bin_accs = []
        bin_centers = []
        bin_counts = []
        for i in range(n_bins):
            mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1] + 1e-9)
            if mask.sum() > 0:
                bin_accs.append(corrects[mask].mean())
                bin_centers.append(confs[mask].mean())
                bin_counts.append(mask.sum())

        fig, ax = plt.subplots(figsize=(10, 5))
        if bin_accs:
            ax.bar(range(len(bin_accs)), bin_accs,
                   color=["#2ecc71" if a > 0.5 else "#e74c3c" for a in bin_accs])
            ax.set_xticks(range(len(bin_centers)))
            ax.set_xticklabels([f"{c:.2f}\n(n={n})" for c, n in zip(bin_centers, bin_counts)],
                              fontsize=7)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Confidence (|score_a - score_b|)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Confidence Calibration (eval {eval_step})")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(f"{prefix}_confidence_cal.png", dpi=150)
        plt.close(fig)


def save_training_curves(history, run_dir):
    """Save training curves across all evals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(history) < 2:
        return

    steps = [h["eval_step"] for h in history]

    # ── 1. Loss ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [h["train_loss"] for h in history], label="Train", linewidth=2)
    ax.plot(steps, [h["val_loss"] for h in history], label="Val", linewidth=2)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Loss")
    ax.set_title("Bradley-Terry Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss.png"), dpi=150)
    plt.close(fig)

    # ── 2. Pair accuracy (overall + per-margin) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [h["pair_accuracy"] for h in history],
            label="Overall", linewidth=2.5, color="#2c3e50", marker="o", markersize=3)
    margin_colors = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#2ecc71"}
    for m in range(1, 5):
        key = f"acc_margin_{m}"
        vals = [h.get(key, None) for h in history]
        if any(v is not None for v in vals):
            vals = [v if v is not None else 0.5 for v in vals]
            ax.plot(steps, vals, label=f"Margin {m}", linewidth=1.5,
                    color=margin_colors[m], linestyle="--")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Pairwise Accuracy")
    ax.set_title("Pairwise Accuracy Over Training")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "pair_accuracy.png"), dpi=150)
    plt.close(fig)

    # ── 3. Score means by corruption level ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    for i, name in enumerate(CORRUPTION_NAMES):
        vals = [h.get("level_scores", {}).get(name, {}).get("mean", 0) for h in history]
        ax.plot(steps, vals, label=name, linewidth=2, color=colors[i])
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Mean Quality Score")
    ax.set_title("Score by Corruption Level Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "level_scores.png"), dpi=150)
    plt.close(fig)

    # ── 4. Score separation (CLEAN - GARBAGE gap) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    separations = {}
    for la, lb in [(0, 4), (0, 1), (1, 2), (2, 3), (3, 4)]:
        na, nb = CORRUPTION_NAMES[la], CORRUPTION_NAMES[lb]
        key = f"{na}-{nb}"
        vals = []
        for h in history:
            ls = h.get("level_scores", {})
            if na in ls and nb in ls:
                vals.append(ls[na]["mean"] - ls[nb]["mean"])
            else:
                vals.append(0)
        separations[key] = vals

    for key, vals in separations.items():
        ax.plot(steps, vals, label=key, linewidth=2 if "GARBAGE" in key else 1.5,
                linestyle="-" if "GARBAGE" in key else "--")
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Score Gap (better - worse)")
    ax.set_title("Score Separation Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "score_separation.png"), dpi=150)
    plt.close(fig)

    # ── 5. Confidence stats ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, [h.get("correct_confidence", 0) for h in history],
            label="Correct pairs", linewidth=2, color="#2ecc71")
    ax.plot(steps, [h.get("wrong_confidence", 0) for h in history],
            label="Wrong pairs", linewidth=2, color="#e74c3c")
    ax.plot(steps, [h.get("mean_confidence", 0) for h in history],
            label="All pairs", linewidth=1.5, color="#3498db", linestyle="--")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Mean |score_a - score_b|")
    ax.set_title("Confidence Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "confidence.png"), dpi=150)
    plt.close(fig)

    # ── 6. Monotonicity ──
    fig, ax = plt.subplots(figsize=(10, 3))
    mono = [1.0 if h.get("monotonic", False) else 0.0 for h in history]
    ax.bar(steps, mono, color=["#2ecc71" if m else "#e74c3c" for m in mono], width=0.8)
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Monotonic")
    ax.set_title("Score Monotonicity (CLEAN > LIGHT > MED > HIGH > GARBAGE)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "monotonicity.png"), dpi=150)
    plt.close(fig)

    # ── 7. Master dashboard ──
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # top: pair accuracy
    ax1.plot(steps, [h["pair_accuracy"] for h in history],
             linewidth=2.5, color="#2c3e50", marker="o", markersize=3, label="Overall")
    for m in range(1, 5):
        key = f"acc_margin_{m}"
        vals = [h.get(key, 0.5) for h in history]
        ax1.plot(steps, vals, linewidth=1, color=margin_colors[m], alpha=0.7, label=f"M{m}")
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Master Dashboard")
    ax1.legend(loc="lower right", ncol=5, fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.2)

    # middle: score levels
    for i, name in enumerate(CORRUPTION_NAMES):
        vals = [h.get("level_scores", {}).get(name, {}).get("mean", 0) for h in history]
        ax2.plot(steps, vals, label=name, linewidth=2, color=colors[i])
    ax2.set_ylabel("Score")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.2)

    # bottom: loss
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

    # train/val split by beatmapset (seed 42)
    rng_split = np.random.default_rng(42)
    bset_ids = sorted(set(c.get("beatmapset_id", str(i)) for i, c in enumerate(charts)))
    rng_split.shuffle(bset_ids)
    n_val = max(1, int(len(bset_ids) * 0.1))
    val_bsets = set(bset_ids[:n_val])
    train_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id") not in val_bsets]
    val_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id") in val_bsets]
    print(f"Train: {len(train_idx)} charts, Val: {len(val_idx)} charts ({n_val} beatmapsets)")

    # determine phase
    phase = args.phase
    if phase == 1:
        mode = "corruption"
        lr = args.lr
        print(f"Phase 1: corruption pretraining, lr={lr}")
    else:
        mode = "mixed"
        lr = args.lr
        print(f"Phase 2: mixed training (corruption {args.corruption_ratio:.0%} + rating), lr={lr}")

    train_ds = PairDataset(manifest, ds_dir, train_idx, mode=mode,
                           corruption_ratio=args.corruption_ratio)
    val_ds = PairDataset(manifest, ds_dir, val_idx, mode=mode,
                         corruption_ratio=args.corruption_ratio, augment_audio=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    # model
    model = ChartQualityEvaluator(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # run directory
    run_dir = os.path.join(SCRIPT_DIR, "runs", args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    eval_dir = os.path.join(run_dir, "evals")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # save args
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # resume or warm-start
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

    # ── training loop ──
    print(f"\nTraining for {args.epochs} epochs, {args.evals_per_epoch} eval(s)/epoch")
    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

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
                loss = bradley_terry_loss(score_a, score_b, margin, args.alpha)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # tracking
            bs = mel_a.size(0)
            epoch_loss += loss.item() * bs
            epoch_correct += (score_a > score_b).float().sum().item()
            epoch_total += bs

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{epoch_correct / max(epoch_total, 1):.1%}",
            )

            # mid-epoch eval
            if batch_idx in eval_at:
                eval_step += 1
                epoch_frac = epoch + (batch_idx + 1) / n_batches - 1
                _run_eval(model, val_loader, train_ds, args, eval_step, epoch_frac,
                          epoch_loss / max(epoch_total, 1),
                          run_dir, ckpt_dir, eval_dir, history, scheduler, optimizer,
                          scaler, best_val_loss)
                if history and history[-1]["val_loss"] < best_val_loss:
                    best_val_loss = history[-1]["val_loss"]

        scheduler.step()

        # end-of-epoch eval (if not already done)
        if n_batches - 1 not in eval_at:
            eval_step += 1
            _run_eval(model, val_loader, train_ds, args, eval_step, epoch,
                      epoch_loss / max(epoch_total, 1),
                      run_dir, ckpt_dir, eval_dir, history, scheduler, optimizer,
                      scaler, best_val_loss)
            if history and history[-1]["val_loss"] < best_val_loss:
                best_val_loss = history[-1]["val_loss"]

    print("\nTraining complete.")


def _run_eval(model, val_loader, train_ds, args, eval_step, epoch_frac, train_loss,
              run_dir, ckpt_dir, eval_dir, history, scheduler, optimizer, scaler, best_val_loss):
    """Run validation with heavy data collection, save plots + JSON + checkpoint."""

    # 1. validate and collect per-pair data
    val_metrics, pairs = validate_and_collect(model, val_loader, args.device, alpha=args.alpha)
    val_loss = val_metrics["val_loss"]

    # 2. score by corruption level (with raw scores for distributions)
    level_summary, level_raw = compute_score_by_level(model, train_ds, args.device, n_samples=100)

    # 3. per-pair-type accuracy matrix (all 10 corruption combinations)
    pair_matrix = compute_pair_matrix(model, train_ds, args.device, n_samples=50)

    # ── print summary ──
    tag = f"{epoch_frac:.1f}" if isinstance(epoch_frac, float) else str(epoch_frac)
    acc_str = " ".join(
        f"m{m}={val_metrics.get(f'acc_margin_{m}', 0):.1%}"
        for m in range(1, 5) if f"acc_margin_{m}" in val_metrics
    )
    level_str = " > ".join(
        f"{name}={level_summary[name]['mean']:+.2f}"
        for name in CORRUPTION_NAMES
    )
    print(f"  Eval {eval_step} (epoch {tag}): "
          f"loss={train_loss:.4f}/{val_loss:.4f} | "
          f"pair_acc={val_metrics['pair_accuracy']:.1%} [{acc_str}] | "
          f"conf={val_metrics['mean_confidence']:.3f} (✓{val_metrics['correct_confidence']:.3f} ✗{val_metrics['wrong_confidence']:.3f}) | "
          f"lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"    scores: {level_str}")

    # print pair matrix summary (grouped by margin)
    for margin in range(1, 5):
        pairs_at_margin = {k: v for k, v in pair_matrix.items() if v["margin"] == margin}
        if pairs_at_margin:
            pair_strs = [f"{k.split('_vs_')[0][:3]}v{k.split('_vs_')[1][:3]}={v['accuracy']:.0%}"
                         for k, v in sorted(pairs_at_margin.items())]
            print(f"    margin {margin}: {' '.join(pair_strs)}")

    # monotonicity check
    means = [level_summary[name]["mean"] for name in CORRUPTION_NAMES]
    monotonic = all(means[i] >= means[i + 1] for i in range(len(means) - 1))
    print(f"    monotonic={monotonic}")

    # ── save eval JSON (heavy) ──
    eval_data = {
        "eval_step": eval_step,
        "epoch": epoch_frac if isinstance(epoch_frac, float) else float(epoch_frac),
        "train_loss": train_loss,
        **val_metrics,
        "level_scores": level_summary,
        "pair_matrix": pair_matrix,
        "monotonic": monotonic,
    }
    history.append(eval_data)

    # save compact eval JSON (metrics only, no per-pair data)
    with open(os.path.join(eval_dir, f"eval_{eval_step:03d}.json"), "w") as f:
        json.dump(eval_data, f, indent=2)

    # save heavy per-pair data separately
    with open(os.path.join(eval_dir, f"eval_{eval_step:03d}_pairs.json"), "w") as f:
        json.dump({"pairs": pairs, "level_raw": level_raw}, f)

    # save history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── save plots ──
    save_eval_plots(eval_step, eval_dir, level_summary, level_raw, pair_matrix, pairs)
    save_training_curves(history, run_dir)

    # ── save checkpoint ──
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
        print(f"    ★ new best val_loss={val_loss:.4f}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train chart quality evaluator")
    parser.add_argument("dataset", help="Dataset directory name (e.g. taiko_v2)")
    parser.add_argument("--run-name", required=True, help="Run name for output directory")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Training phase: 1=corruption only, 2=mixed")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (pairs)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.1, help="Margin scaling factor")
    parser.add_argument("--corruption-ratio", type=float, default=0.6,
                        help="Fraction of corruption pairs in phase 2")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--warm-start", type=str, default=None,
                        help="Path to checkpoint for warm-starting model weights")
    args = parser.parse_args()

    if args.no_amp:
        args.amp = False

    train(args)


if __name__ == "__main__":
    main()
