"""Benchmark external onset detection algorithms against our validation set.

Runs librosa, madmom, and aubio onset detectors on the same val samples
as our model, producing the same per-epoch graphs for comparison.

Usage:
    python baseline_benchmark.py taiko_v2 [--subsample 40]
"""
import os
import sys
import json
import random
import zipfile
import tempfile
import argparse
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── same constants as detection_train.py ──
A_BINS = 500
B_BINS = 500
C_EVENTS = 128
N_CLASSES = 501
WINDOW = A_BINS + B_BINS
MIN_CURSOR_BIN = 6000
SAMPLE_RATE = 22050
HOP_LENGTH = 110
BIN_MS = HOP_LENGTH / SAMPLE_RATE * 1000  # ~4.9887ms


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


def build_val_samples(manifest, ds_dir, val_idx, subsample=40):
    """Build list of (chart_idx, event_idx, cursor, target) for val set."""
    charts = manifest["charts"]
    evt_dir = os.path.join(ds_dir, "events")

    # load events for val charts
    events = {}
    for ci in val_idx:
        chart = charts[ci]
        evt = np.load(os.path.join(evt_dir, chart["event_file"]))
        events[ci] = evt

    samples = []
    for ci in val_idx:
        evt = events[ci]
        for ei in range(len(evt)):
            cursor = max(0, int(evt[0]) - B_BINS) if ei == 0 else int(evt[ei - 1])
            if cursor >= MIN_CURSOR_BIN:
                # target
                if ei < len(evt):
                    offset = max(0, int(evt[ei]) - cursor)
                    target = N_CLASSES - 1 if offset >= B_BINS else offset
                else:
                    target = N_CLASSES - 1
                samples.append((ci, ei, cursor, target))
        # STOP sample
        if len(evt) > 0 and int(evt[-1]) >= MIN_CURSOR_BIN:
            samples.append((ci, len(evt), int(evt[-1]), N_CLASSES - 1))

    # subsample
    if subsample > 1:
        samples = samples[::subsample]

    return samples, events


def get_osz_path(chart):
    """Derive .osz path from chart metadata."""
    # mel_file format: "{basename}__{audio_name}.npy"
    mel_file = chart["mel_file"]
    basename = mel_file.split("__")[0]
    osz_path = os.path.join(SCRIPT_DIR, "charts", basename + ".osz")
    return osz_path, chart["audio_name"]


def load_audio_from_osz(osz_path, audio_name):
    """Extract and load audio from .osz file."""
    import librosa
    with zipfile.ZipFile(osz_path) as z:
        if audio_name not in z.namelist():
            return None
        audio_bytes = z.read(audio_name)

    ext = os.path.splitext(audio_name)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)
    return y


def bin_to_sample(b):
    """Convert bin index to audio sample index."""
    return int(b * HOP_LENGTH)


def sample_to_bin(s):
    """Convert audio sample index to bin index."""
    return s / HOP_LENGTH


# ═══════════════════════════════════════════════════════════════
#  Onset detection algorithms
# ═══════════════════════════════════════════════════════════════

def detect_librosa_flux(y, sr):
    """librosa spectral flux onset detection → onset bins."""
    import librosa
    frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH,
                                         backtrack=False, units="frames")
    return frames.astype(np.int64)


def detect_librosa_energy(y, sr):
    """librosa energy-based onset detection."""
    import librosa
    # compute RMS manually, then use as onset strength
    S = np.abs(librosa.stft(y, hop_length=HOP_LENGTH)) ** 2
    rms = librosa.feature.rms(S=S, hop_length=HOP_LENGTH)[0]
    # simple peak-picking on RMS envelope
    frames = librosa.onset.onset_detect(onset_envelope=rms, sr=sr,
                                         hop_length=HOP_LENGTH,
                                         backtrack=False, units="frames")
    return frames.astype(np.int64)


def detect_aubio(y, sr, method="hfc"):
    """aubio onset detection with given method → onset bins."""
    import aubio
    win_s = 1024
    hop_s = HOP_LENGTH

    # aubio expects float32
    y32 = y.astype(np.float32)
    o = aubio.onset(method, win_s, hop_s, sr)
    onsets = []
    i = 0
    while i + hop_s <= len(y32):
        chunk = y32[i:i + hop_s]
        if o(chunk):
            # get_last returns sample position
            onsets.append(int(o.get_last() / hop_s))
        i += hop_s
    return np.array(onsets, dtype=np.int64)


def detect_madmom_rnn(y, sr):
    """madmom RNN onset detection → onset bins."""
    from madmom.features.onsets import RNNOnsetProcessor, OnsetPeakPickingProcessor
    # madmom processors work on file paths, but we can use the signal processor
    # RNNOnsetProcessor expects a file or signal at specific sr
    # Write to temp file
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        tmp_path = tmp.name
    try:
        proc = RNNOnsetProcessor()
        act = proc(tmp_path)  # activation at 100 fps
        picker = OnsetPeakPickingProcessor(fps=100, threshold=0.3)
        onset_times = picker(act)  # seconds
    finally:
        os.unlink(tmp_path)
    # convert seconds to bins
    onset_bins = (onset_times * 1000 / BIN_MS).astype(np.int64)
    return onset_bins


def detect_madmom_cnn(y, sr):
    """madmom CNN onset detection → onset bins."""
    from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        tmp_path = tmp.name
    try:
        proc = CNNOnsetProcessor()
        act = proc(tmp_path)
        picker = OnsetPeakPickingProcessor(fps=100, threshold=0.3)
        onset_times = picker(act)
    finally:
        os.unlink(tmp_path)
    onset_bins = (onset_times * 1000 / BIN_MS).astype(np.int64)
    return onset_bins


ALGORITHMS = {
    "librosa_flux": detect_librosa_flux,
    "librosa_energy": detect_librosa_energy,
    "aubio_hfc": lambda y, sr: detect_aubio(y, sr, "hfc"),
    "aubio_complex": lambda y, sr: detect_aubio(y, sr, "complex"),
    "aubio_specflux": lambda y, sr: detect_aubio(y, sr, "specflux"),
    "madmom_rnn": detect_madmom_rnn,
    "madmom_cnn": detect_madmom_cnn,
}


# ═══════════════════════════════════════════════════════════════
#  Prediction: given detected onsets, predict for each sample
# ═══════════════════════════════════════════════════════════════

def predict_from_onsets(onset_bins, cursor, total_bins):
    """Given global onset bins and a cursor, predict the next onset offset.

    Returns bin offset (0-499) or 500 (STOP) if no onset found after cursor.
    """
    if len(onset_bins) == 0:
        return N_CLASSES - 1  # STOP

    # find first onset strictly after cursor
    idx = np.searchsorted(onset_bins, cursor, side="right")
    if idx >= len(onset_bins):
        return N_CLASSES - 1  # STOP - no more onsets

    offset = int(onset_bins[idx]) - cursor
    if offset < 0:
        offset = 0
    if offset >= B_BINS:
        return N_CLASSES - 1  # STOP - too far away
    return offset


# ═══════════════════════════════════════════════════════════════
#  Metrics (copied from detection_train.py)
# ═══════════════════════════════════════════════════════════════

def compute_metrics(targets, preds):
    m = {}
    m["accuracy"] = (targets == preds).mean().item()

    stop = N_CLASSES - 1
    tp = ((preds == stop) & (targets == stop)).sum()
    fp = ((preds == stop) & (targets != stop)).sum()
    fn = ((preds != stop) & (targets == stop)).sum()
    m["stop_precision"] = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
    m["stop_recall"] = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
    m["stop_f1"] = (2 * tp / (2 * tp + fp + fn)).item() if (2 * tp + fp + fn) > 0 else 0.0

    ns = targets < stop
    if ns.sum() > 0:
        t_ns = targets[ns].astype(np.float64)
        p_ns = preds[ns].astype(np.float64)

        frame_err = np.abs(p_ns - t_ns)
        m["frame_error_mean"] = frame_err.mean().item()
        m["frame_error_median"] = np.median(frame_err).item()
        m["frame_error_p90"] = np.percentile(frame_err, 90).item()
        m["frame_error_p99"] = np.percentile(frame_err, 99).item()

        ratio = (p_ns + 1) / (t_ns + 1)
        pct_err = np.abs(ratio - 1.0)
        log_ratio = np.log(p_ns + 1) - np.log(t_ns + 1)
        m["rel_error_mean"] = np.abs(log_ratio).mean().item()
        m["rel_error_median"] = np.median(np.abs(log_ratio)).item()

        hit = (pct_err <= 0.03) | (frame_err <= 1)
        good = (pct_err <= 0.10) | (frame_err <= 2)
        miss = pct_err > 0.20

        m["hit_rate"] = hit.mean().item()
        m["good_rate"] = good.mean().item()
        m["miss_rate"] = miss.mean().item()
        m["exact_match"] = (frame_err == 0).mean().item()
        m["within_1_frame"] = (frame_err <= 1).mean().item()
        m["within_3pct"] = (pct_err <= 0.03).mean().item()

        # model score
        abs_lr = np.abs(log_ratio)
        threshold = np.log(1.03)
        max_pen = np.log(5.0)
        pen_range = max_pen - threshold
        reward_at_zero = (np.log(3.0) - threshold) / pen_range
        scores = np.where(
            abs_lr <= threshold,
            (1.0 - abs_lr / threshold) * reward_at_zero,
            -np.minimum((abs_lr - threshold) / pen_range, 1.0),
        )
        scores[frame_err <= 1] = reward_at_zero
        m["model_score"] = scores.mean().item()

    return m


# ═══════════════════════════════════════════════════════════════
#  Graphs (adapted from detection_train.py save_epoch_graphs)
# ═══════════════════════════════════════════════════════════════

def save_graphs(targets, preds, metrics, algo_name, out_dir, conds=None, prev_gaps=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, algo_name)

    stop = N_CLASSES - 1
    ns = targets < stop
    t_ns = targets[ns]
    p_ns = preds[ns]

    # ── 0. Prediction distribution ──
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].hist(t_ns, bins=250, range=(0, 500), color="#4a90d9", alpha=0.8)
    axes[0].set_title(f"{algo_name}: Target Distribution (non-STOP)")
    axes[0].set_xlabel("Bin offset")
    axes[0].set_ylabel("Count")
    axes[1].hist(p_ns, bins=250, range=(0, 500), color="#e8834a", alpha=0.8)
    axes[1].set_title(f"{algo_name}: Predicted Distribution - {len(np.unique(p_ns))} unique values")
    axes[1].set_xlabel("Bin offset")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(f"{prefix}_pred_dist.png", dpi=120)
    plt.close(fig)

    if len(t_ns) == 0:
        return

    # ── 1. Scatter ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(t_ns, p_ns, alpha=0.02, s=1, color="#4a90d9")
    ax.plot([0, 500], [0, 500], "r--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Target bin offset")
    ax.set_ylabel("Predicted bin offset")
    ax.set_title(f"{algo_name}: Target vs Predicted")
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    fig.tight_layout()
    fig.savefig(f"{prefix}_scatter.png", dpi=120)
    plt.close(fig)

    # ── 2. Heatmap ──
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
    ax.set_title(f"{algo_name}: Prediction Density", color="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(f"{prefix}_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 3. Ratio scatter ──
    ratio = (p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)
    ratio_clipped = np.clip(ratio, 0.1, 10.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(t_ns, ratio_clipped, alpha=0.02, s=1, color="#e8834a")
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Target bin offset")
    ax.set_ylabel("Ratio (predicted+1)/(target+1)")
    ax.set_title(f"{algo_name}: Relative Error")
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
    log_ratio = np.log10(ratio_clipped)
    h, xedges, yedges = np.histogram2d(
        t_ns.astype(float), log_ratio, bins=[250, 100],
        range=[[0, 500], [-1, 1]]
    )
    h = gaussian_filter(h.astype(np.float64), sigma=1.0)
    h[h < 0.5] = np.nan
    ax.imshow(h.T, origin="lower", aspect="auto",
              extent=[0, 500, -1, 1], norm=LogNorm(vmin=1), cmap="inferno")
    ax.axhline(0.0, color="white", linestyle="--", alpha=0.5)
    ax.set_xlabel("Target bin offset", color="white")
    ax.set_ylabel("log10(ratio)", color="white")
    ax.set_title(f"{algo_name}: Relative Error Density", color="white")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(["0.1x", "0.3x", "1.0x", "3.2x", "10x"])
    ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(f"{prefix}_ratio_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 5. Frame vs Ratio error ──
    frame_err = np.abs(p_ns.astype(np.float64) - t_ns.astype(np.float64))
    ratio_err = np.abs(ratio - 1.0)
    fe_clip = np.clip(frame_err, 0, 200)
    re_clip = np.clip(ratio_err, 0, 5.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(fe_clip, re_clip, alpha=0.02, s=1, color="#4a90d9")
    ax.set_xlabel("Absolute frame error |pred - target|")
    ax.set_ylabel("Absolute ratio error |ratio - 1|")
    ax.set_title(f"{algo_name}: Frame Error vs Ratio Error")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 5.0)
    fig.tight_layout()
    fig.savefig(f"{prefix}_frame_vs_ratio_scatter.png", dpi=120)
    plt.close(fig)

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
    ax.set_title(f"{algo_name}: Frame vs Ratio Error Density", color="white")
    ax.tick_params(colors="white")
    fig.tight_layout()
    fig.savefig(f"{prefix}_frame_vs_ratio_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 6. Ratio in density space ──
    if conds is not None:
        conds_ns = conds[ns]
        mean_dens = conds_ns[:, 0]
        peak_dens = conds_ns[:, 1]
        ratio_err_log = np.abs(np.log((p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)))
        re_clip2 = np.clip(ratio_err_log, 0, 2.0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(mean_dens, peak_dens, c=re_clip2, s=1, alpha=0.1,
                        cmap="RdYlGn_r", vmin=0, vmax=1.5)
        fig.colorbar(sc, ax=ax, label="|log-ratio| error")
        ax.set_xlabel("Mean density (events/sec)")
        ax.set_ylabel("Peak density (events/sec)")
        ax.set_title(f"{algo_name}: Error by Chart Density")
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_in_density_scatter.png", dpi=120)
        plt.close(fig)

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
        ax.set_title(f"{algo_name}: Mean |log-ratio| Error by Density", color="white")
        ax.tick_params(colors="white")
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_in_density_heatmap.png", dpi=150)
        plt.close(fig)

    # ── 7. Forward error (gap ratio continuity) ──
    if prev_gaps is not None:
        pg = prev_gaps[ns]
        valid = np.isfinite(pg) & (pg > 0)
        if valid.sum() > 100:
            tg = t_ns[valid].astype(np.float64)
            pg_v = pg[valid].astype(np.float64)
            pr = p_ns[valid].astype(np.float64)
            target_ratio = tg / pg_v
            pred_ratio = pr / pg_v
            tr_clip = np.clip(target_ratio, 0, 8)
            pr_clip = np.clip(pred_ratio, 0, 8)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(pr_clip, tr_clip, alpha=0.02, s=1, color="#6bc46d")
            ax.plot([0, 8], [0, 8], "r--", alpha=0.5, linewidth=1)
            for r in [0.5, 1.0, 2.0, 4.0]:
                ax.axhline(r, color="white", alpha=0.15, linewidth=0.5)
                ax.axvline(r, color="white", alpha=0.15, linewidth=0.5)
            ax.set_xlabel("Predicted gap / prev gap")
            ax.set_ylabel("Target gap / prev gap")
            ax.set_title(f"{algo_name}: Gap Ratio Continuity")
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            fig.tight_layout()
            fig.savefig(f"{prefix}_forward_error_scatter.png", dpi=120)
            plt.close(fig)

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
            ax.set_title(f"{algo_name}: Gap Ratio Continuity Density", color="white")
            ax.tick_params(colors="white")
            fig.tight_layout()
            fig.savefig(f"{prefix}_forward_error_heatmap.png", dpi=150)
            plt.close(fig)

    # ── 8. Ratio confusion ──
    ratio_raw = (p_ns.astype(np.float64) + 1) / (t_ns.astype(np.float64) + 1)
    pct_err = np.abs(ratio_raw - 1.0)
    misses = pct_err > 0.20
    if misses.sum() > 50:
        miss_ratios = ratio_raw[misses]
        log_miss = np.log2(np.clip(miss_ratios, 1/8, 8))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(log_miss, bins=400, range=[-3, 3], color="#eb4528", alpha=0.8)
        for r, lbl in [(1/4, "\u00bc"), (1/3, "\u2153"), (1/2, "\u00bd"), (1, "1"),
                       (2, "2"), (3, "3"), (4, "4")]:
            ax.axvline(np.log2(r), color="white", alpha=0.5, linewidth=1, linestyle="--")
            ax.text(np.log2(r), ax.get_ylim()[1] * 0.95, lbl,
                    ha="center", va="top", fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
        ax.set_xlabel("log2(pred/target) - musical ratio")
        ax.set_ylabel("Count")
        ax.set_title(f"{algo_name}: Ratio Confusion (misses only, n={misses.sum()})")
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_xticklabels(["\u215b", "\u00bc", "\u00bd", "1", "2", "4", "8"])
        fig.tight_layout()
        fig.savefig(f"{prefix}_ratio_confusion.png", dpi=120)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset name (e.g. taiko_v2)")
    parser.add_argument("--subsample", type=int, default=40)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS.keys()),
                        help="Which algorithms to run")
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # same split as training
    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)
    print(f"Val charts: {len(val_idx)}")

    samples, events = build_val_samples(manifest, ds_dir, val_idx, subsample=args.subsample)
    print(f"Val samples (1/{args.subsample}): {len(samples)}")

    charts = manifest["charts"]

    # Group samples by audio file (osz_path, audio_name) so we load each audio once
    audio_to_samples = defaultdict(list)
    for i, (ci, ei, cursor, target) in enumerate(samples):
        chart = charts[ci]
        osz_path, audio_name = get_osz_path(chart)
        audio_key = f"{osz_path}|{audio_name}"
        audio_to_samples[audio_key].append(i)

    print(f"Unique audio files to load: {len(audio_to_samples)}")

    # Build conditioning and prev_gaps arrays for graphs
    all_targets = np.array([s[3] for s in samples], dtype=np.int64)
    all_conds = np.zeros((len(samples), 3), dtype=np.float32)
    all_prev_gaps = np.full(len(samples), np.nan, dtype=np.float64)
    for i, (ci, ei, cursor, target) in enumerate(samples):
        chart = charts[ci]
        all_conds[i] = [chart.get("density_mean", 4.0),
                        chart.get("density_peak", 8),
                        chart.get("density_std", 1.5)]
        evt = events[ci]
        if ei >= 2:
            all_prev_gaps[i] = float(evt[ei - 1] - evt[ei - 2])

    # Run each algorithm
    out_dir = os.path.join(SCRIPT_DIR, "runs", "baseline_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    # Pre-detect onsets per audio per algorithm
    # {algo_name: {audio_key: onset_bins}}
    print(f"\nRunning algorithms: {args.algorithms}")

    # Load all audio first
    from tqdm import tqdm
    audio_cache = {}
    print("\nLoading audio files...")
    for audio_key in tqdm(audio_to_samples.keys(), desc="Loading audio"):
        osz_path, audio_name = audio_key.split("|", 1)
        if not os.path.exists(osz_path):
            print(f"  SKIP (no .osz): {osz_path}")
            continue
        y = load_audio_from_osz(osz_path, audio_name)
        if y is not None:
            audio_cache[audio_key] = y
        else:
            print(f"  SKIP (load failed): {audio_name}")

    print(f"Loaded {len(audio_cache)}/{len(audio_to_samples)} audio files")

    for algo_name in args.algorithms:
        if algo_name not in ALGORITHMS:
            print(f"Unknown algorithm: {algo_name}")
            continue

        detect_fn = ALGORITHMS[algo_name]
        print(f"\n{'='*60}")
        print(f"  {algo_name}")
        print(f"{'='*60}")

        # Detect onsets per audio
        onset_cache = {}
        for audio_key, y in tqdm(audio_cache.items(), desc=f"  Detecting [{algo_name}]"):
            try:
                onset_bins = detect_fn(y, SAMPLE_RATE)
                onset_bins = np.sort(onset_bins)
                onset_cache[audio_key] = onset_bins
            except Exception as e:
                print(f"  ERROR on {audio_key}: {e}")
                onset_cache[audio_key] = np.array([], dtype=np.int64)

        # Predict for each sample
        all_preds = np.full(len(samples), N_CLASSES - 1, dtype=np.int64)
        n_matched = 0
        for audio_key, sample_indices in audio_to_samples.items():
            if audio_key not in onset_cache:
                continue
            onset_bins = onset_cache[audio_key]
            y = audio_cache.get(audio_key)
            total_bins = len(y) // HOP_LENGTH if y is not None else 0
            for idx in sample_indices:
                ci, ei, cursor, target = samples[idx]
                all_preds[idx] = predict_from_onsets(onset_bins, cursor, total_bins)
                n_matched += 1

        print(f"  Predicted {n_matched}/{len(samples)} samples")

        # Compute metrics
        metrics = compute_metrics(all_targets, all_preds)
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  HIT rate:    {metrics.get('hit_rate', 0):.4f}")
        print(f"  GOOD rate:   {metrics.get('good_rate', 0):.4f}")
        print(f"  Miss rate:   {metrics.get('miss_rate', 0):.4f}")
        print(f"  Stop F1:     {metrics['stop_f1']:.4f}")
        print(f"  Frame err:   mean={metrics.get('frame_error_mean', 0):.1f}  "
              f"med={metrics.get('frame_error_median', 0):.1f}  "
              f"p99={metrics.get('frame_error_p99', 0):.1f}")
        print(f"  Model score: {metrics.get('model_score', 0):.4f}")

        # Save metrics
        with open(os.path.join(out_dir, f"{algo_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save graphs
        print(f"  Saving graphs...")
        save_graphs(all_targets, all_preds, metrics, algo_name, out_dir,
                    conds=all_conds, prev_gaps=all_prev_gaps)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Algorithm':<20} {'HIT':>7} {'GOOD':>7} {'Miss':>7} {'Score':>7} {'FrErr':>7} {'StopF1':>7}")
    print("-" * 80)
    for algo_name in args.algorithms:
        mpath = os.path.join(out_dir, f"{algo_name}_metrics.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            m = json.load(f)
        print(f"{algo_name:<20} {m.get('hit_rate',0):>6.1%} {m.get('good_rate',0):>6.1%} "
              f"{m.get('miss_rate',0):>6.1%} {m.get('model_score',0):>+6.3f} "
              f"{m.get('frame_error_mean',0):>6.1f} {m.get('stop_f1',0):>6.3f}")

    print(f"\nGraphs saved to: {out_dir}")


if __name__ == "__main__":
    main()
