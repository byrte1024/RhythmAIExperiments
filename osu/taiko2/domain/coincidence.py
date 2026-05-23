"""Coincidence map — cross-band onset identity encoding.

Ported from ``audix/coincidence_map.py``. Pure-function library for
computing 13-row onset summaries from audio waveforms. No CLI, no I/O.

Generates a coincidence color map from audio. Outputs:
  - Full-resolution coincidence heatmap as .npz (n_bands x n_frames x 3)
  - Compressed summary rows for lightweight downstream consumption

The coincidence map encodes *what kind* of onset is happening at each moment.
Similar onset patterns (same bands co-firing) produce similar RGB colors via
locality-sensitive hashing, so recurring instruments/hits are visually and
numerically consistent across the track.

Usage:
    ./coincidence_map.py song.wav
    ./coincidence_map.py song.wav -o output_name
    ./coincidence_map.py "https://youtube.com/watch?v=..." --n-bands 128
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import librosa


def rolling_median_mad(data, window):
    n_bands, n_frames = data.shape
    half = window // 2
    padded = np.pad(data, ((0, 0), (half, half)), mode="edge")
    windows = sliding_window_view(padded, window, axis=1)
    median = np.median(windows, axis=2)
    mad = np.median(np.abs(windows - median[:, :, np.newaxis]), axis=2)
    mad = np.maximum(mad, 1e-6)
    return median[:, :n_frames], mad[:, :n_frames]


def compute_coincidence_map(y, sr, n_bands=64, hop_length=110,
                            fmin=20.0, fmax=8000.0, threshold=1.5,
                            median_window=51, mu=2, refractory=4,
                            decay=0.85, bleed_halflife=2.0):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bands, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
    )
    n_frames = S.shape[1]

    # log spectrogram + max filter vibrato suppression
    S_log = np.log1p(S * 1e3)
    S_max = np.copy(S_log)
    S_max[1:, :] = np.maximum(S_max[1:, :], S_log[:-1, :])
    S_max[:-1, :] = np.maximum(S_max[:-1, :], S_log[1:, :])

    # spectral flux with frame offset μ
    flux = np.zeros_like(S_log)
    if mu < n_frames:
        flux[:, mu:] = np.maximum(S_log[:, mu:] - S_max[:, :-mu], 0)

    # adaptive threshold
    med, mad = rolling_median_mad(flux, median_window)
    thresh_curve = med + threshold * mad

    # spike confidence with refractory + decay
    spike_raw = (flux - thresh_curve) / mad
    spike_confidence = np.zeros_like(spike_raw)
    refractory_counter = np.zeros(n_bands, dtype=np.int32)
    held = np.zeros(n_bands, dtype=np.float64)
    spikes_binary = np.zeros_like(spike_raw)

    for i in range(n_frames):
        for b in range(n_bands):
            held[b] *= decay
            if refractory_counter[b] > 0:
                refractory_counter[b] -= 1
            elif spike_raw[b, i] > 0:
                val = min(spike_raw[b, i], 1.0)
                spikes_binary[b, i] = 1.0
                held[b] = max(held[b], val)
                refractory_counter[b] = refractory
            spike_confidence[b, i] = held[b]

    # vertical bleed
    if bleed_halflife > 0:
        max_reach = int(np.ceil(bleed_halflife * 8))
        offsets = np.arange(1, max_reach + 1)
        weights = np.power(0.5, offsets / bleed_halflife)
        cutoff = np.searchsorted(-weights, -1e-4)
        bleed_kernel = weights[:cutoff]
        bleed = np.zeros_like(spike_confidence)
        for k, w in enumerate(bleed_kernel):
            d = k + 1
            bleed[d:, :] += spike_confidence[:-d, :] * w
            bleed[:-d, :] += spike_confidence[d:, :] * w
        spike_confidence = np.clip(spike_confidence + bleed, 0, 1)

    # IDF weighting
    firing_rate = spikes_binary.mean(axis=1) + 1e-6
    idf = -np.log2(firing_rate)
    idf /= idf.max() + 1e-8

    # IDF-weighted population count
    weighted_pop = (spikes_binary * idf[:, np.newaxis]).sum(axis=0)
    pop_max = np.percentile(weighted_pop, 99) or 1.0
    pop_norm = np.clip(weighted_pop / pop_max, 0, 1)

    # LSH color projection
    rng = np.random.RandomState(42)
    proj = rng.randn(3, n_bands).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)

    weighted_spikes = spike_confidence * idf[:, np.newaxis]
    color_coords = proj @ weighted_spikes

    for d in range(3):
        lo = np.percentile(color_coords[d], 2)
        hi = np.percentile(color_coords[d], 98)
        if hi > lo:
            color_coords[d] = np.clip((color_coords[d] - lo) / (hi - lo), 0, 1)
        else:
            color_coords[d] = 0.5

    # full heatmap (n_bands x n_frames x 3)
    color_heatmap = np.zeros((n_bands, n_frames, 3), dtype=np.float32)
    for d in range(3):
        color_heatmap[:, :, d] = spike_confidence * color_coords[d][np.newaxis, :]

    # mel norm for reference
    S_db = librosa.power_to_db(S, ref=np.max)
    mel_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)

    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    freqs = librosa.mel_frequencies(n_mels=n_bands + 2, fmin=fmin, fmax=fmax)[1:-1]

    return color_heatmap, color_coords, spike_confidence, pop_norm, mel_norm, times, freqs


def compress_map(color_heatmap, color_coords, spike_confidence, pop_norm, n_summary=8):
    """Produce compressed summary rows for lightweight AI consumption."""
    n_bands, n_frames, _ = color_heatmap.shape

    # Row 1-3: color_coords (R, G, B per frame) — the raw LSH projection
    # already (3, n_frames)

    # Row 4: population count (1, n_frames)

    # Row 5: total spike energy (1, n_frames)
    spike_energy = spike_confidence.sum(axis=0)
    se_max = spike_energy.max() or 1.0
    spike_energy /= se_max

    # Rows 6+: band-group averages of spike confidence
    # split n_bands into n_summary groups, average each
    group_sz = max(1, n_bands // n_summary)
    band_groups = np.zeros((n_summary, n_frames), dtype=np.float32)
    for g in range(n_summary):
        start = g * group_sz
        end = min((g + 1) * group_sz, n_bands)
        if start >= n_bands:
            break
        band_groups[g] = spike_confidence[start:end].mean(axis=0)
    bg_max = band_groups.max() or 1.0
    band_groups /= bg_max

    # Stack all summary rows
    summary = np.vstack([
        color_coords,                          # 3 rows: R, G, B
        pop_norm[np.newaxis, :],               # 1 row: IDF popcount
        spike_energy[np.newaxis, :],           # 1 row: total spike energy
        band_groups,                           # n_summary rows: band group avgs
    ])

    return summary


def compute_summary(
    y: np.ndarray, sr: int,
    hop_length: int = 110,
    n_bands: int = 64,
    n_summary: int = 8,
) -> np.ndarray:
    """Convenience: waveform → 13-row summary in one call."""
    heatmap, coords, spike, pop, _, _, _ = compute_coincidence_map(
        y, sr, n_bands=n_bands, hop_length=hop_length,
    )
    return compress_map(heatmap, coords, spike, pop, n_summary=n_summary)
