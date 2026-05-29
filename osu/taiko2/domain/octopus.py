"""Octopus cell onset representation — pure compute functions.

Biologically-inspired onset detection modeled after octopus cells in
the mammalian cochlear nucleus (Golding et al. 1995). Octopus cells
fire exclusively on sound onsets by detecting cross-frequency
coincidence: they have ultra-low input resistance (~7 MOhm), receive
60-200+ auditory nerve fiber inputs spanning ~1/3 of the tonotopic
array, and produce one spike per onset with submillisecond jitter.

The gradient variant slides the cell window by 1 channel, producing
a dense 2D heatmap that encodes both *when* and *where in frequency*
onsets occur.

Pipeline:
  raw audio -> gammatone filterbank (128 ERB-spaced channels)
  -> per-channel envelope (rectify + max-pool to 1ms frames)
  -> log-domain onset function (log(E[t]) - log(E[t-k]))
  -> group delay compensation (align broadband transients)
  -> cross-channel synchrony detection (coincidence gate)
  -> gradient heatmap (sliding window, nonlinear response)

Ported from twoof/octopus_repr/. No CLI, no visualization, no
external references.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def erb_space(
    low_freq: float, high_freq: float, n_filters: int,
    erb_q: float = 9.26449, erb_min_bw: float = 24.7,
) -> np.ndarray:
    """ERB-spaced center frequencies, low-to-high."""
    t = np.arange(1, n_filters + 1)
    cf = -(erb_q * erb_min_bw) + np.exp(
        t * (-np.log(high_freq + erb_q * erb_min_bw)
             + np.log(low_freq + erb_q * erb_min_bw)) / n_filters
    ) * (high_freq + erb_q * erb_min_bw)
    return cf[::-1].copy()


def gammatone_ir(
    sr: int, cf: float, order: int = 4, ir_duration: float = 0.025,
    erb_q: float = 9.26449, erb_min_bw: float = 24.7,
    erb_bw_coeff: float = 1.019,
) -> np.ndarray:
    """Single-channel gammatone impulse response."""
    t = np.arange(0, int(sr * ir_duration)) / sr
    n = order
    erb = ((cf / erb_q) ** n + erb_min_bw ** n) ** (1.0 / n)
    b = erb_bw_coeff * 2 * np.pi * erb
    gt = (t ** (n - 1)) * np.exp(-b * t) * np.cos(2 * np.pi * cf * t)
    return gt / (np.abs(gt).max() + 1e-12)


def compute_group_delays(
    cfs: np.ndarray, order: int = 4,
    erb_q: float = 9.26449, erb_min_bw: float = 24.7,
    erb_bw_coeff: float = 1.019,
) -> np.ndarray:
    """Analytical group delay per channel in ms: t_peak = (order-1)/b."""
    n = order
    delays_ms = np.zeros(len(cfs))
    for i, cf in enumerate(cfs):
        erb = ((cf / erb_q) ** n + erb_min_bw ** n) ** (1.0 / n)
        b = erb_bw_coeff * 2 * np.pi * erb
        delays_ms[i] = (n - 1) / b * 1000
    return delays_ms


def apply_group_delay_compensation(
    onset_fn: np.ndarray, cfs: np.ndarray, hop_ms: float = 1.0,
    order: int = 4, erb_q: float = 9.26449, erb_min_bw: float = 24.7,
    erb_bw_coeff: float = 1.019,
) -> np.ndarray:
    """Shift each channel forward to align broadband transients."""
    delays_ms = compute_group_delays(cfs, order, erb_q, erb_min_bw, erb_bw_coeff)
    max_delay = delays_ms.max()
    compensated = np.zeros_like(onset_fn)
    for ch in range(onset_fn.shape[0]):
        shift = int(round((max_delay - delays_ms[ch]) / hop_ms))
        if shift > 0:
            compensated[ch, shift:] = onset_fn[ch, :-shift]
        else:
            compensated[ch] = onset_fn[ch]
    return compensated


def apply_gammatone_filterbank(
    audio: np.ndarray, sr: int,
    n_filters: int = 128, low_freq: float = 50.0, high_freq: float = 8000.0,
    order: int = 4, ir_duration: float = 0.025, iir_stable_hz: float = 220.0,
    hop_ms: float = 1.0, max_workers: int = 8,
    erb_q: float = 9.26449, erb_min_bw: float = 24.7,
    erb_bw_coeff: float = 1.019,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Hybrid IIR/FIR gammatone -> rectify -> max-pool downsample.

    Returns (envelopes, center_freqs, n_frames).
    envelopes: (n_filters, n_frames) float32 at hop_ms resolution.
    """
    from scipy.signal import lfilter, gammatone, oaconvolve

    high = min(high_freq, sr // 2 - 100)
    cfs = erb_space(low_freq, high, n_filters, erb_q, erb_min_bw)
    n = len(audio)
    hop_samples = max(1, int(sr * hop_ms / 1000))
    n_frames = n // hop_samples
    trim = n_frames * hop_samples

    env_ds = np.zeros((n_filters, n_frames), dtype=np.float32)

    iir_idx = [i for i, cf in enumerate(cfs) if cf >= iir_stable_hz]
    fir_idx = [i for i, cf in enumerate(cfs) if cf < iir_stable_hz]

    iir_coeffs = {i: gammatone(cfs[i], 'iir', fs=sr) for i in iir_idx}
    fir_irs = {
        i: gammatone_ir(sr, cfs[i], order, ir_duration, erb_q, erb_min_bw, erb_bw_coeff)
        for i in fir_idx
    }

    audio_f64 = audio.astype(np.float64)

    def process_iir(i: int) -> tuple[int, np.ndarray]:
        b, a = iir_coeffs[i]
        filtered = lfilter(b, a, audio_f64)
        return i, np.abs(filtered[:trim]).reshape(n_frames, hop_samples).max(axis=1).astype(np.float32)

    def process_fir(i: int) -> tuple[int, np.ndarray]:
        filtered = oaconvolve(audio, fir_irs[i], mode='same')
        return i, np.abs(filtered[:trim]).reshape(n_frames, hop_samples).max(axis=1).astype(np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, env in pool.map(process_iir, iir_idx):
            env_ds[i] = env
        for i, env in pool.map(process_fir, fir_idx):
            env_ds[i] = env

    return env_ds, cfs, n_frames


def compute_gradient(
    audio: np.ndarray, sr: int,
    n_filters: int = 128, low_freq: float = 50.0, high_freq: float = 8000.0,
    order: int = 4, ir_duration: float = 0.025, iir_stable_hz: float = 220.0,
    hop_ms: float = 1.0, max_workers: int = 8,
    onset_lookback: int = 3,
    sync_window_ms: float = 1.5, peak_percentile: float = 90.0,
    gradient_step: int = 1, gradient_cell_width_frac: float = 0.25,
    gradient_nonlinearity_exp: float = 1.5,
    compensate_group_delay: bool = True, norm_percentile: float = 99.9,
    erb_q: float = 9.26449, erb_min_bw: float = 24.7,
    erb_bw_coeff: float = 1.019,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Full octopus gradient pipeline.

    Returns (gradient, center_freqs, n_cells).
    gradient: (n_cells, n_frames) float32 normalized [0, 1] per cell.
    """
    env_ds, cfs, n_frames = apply_gammatone_filterbank(
        audio, sr, n_filters, low_freq, high_freq, order, ir_duration,
        iir_stable_hz, hop_ms, max_workers, erb_q, erb_min_bw, erb_bw_coeff,
    )

    # Onset function: log-domain derivative.
    eps = 1e-10
    log_env = np.log(env_ds + eps)
    k = onset_lookback
    onset_fn = np.zeros_like(log_env)
    onset_fn[:, k:] = np.maximum(0, log_env[:, k:] - log_env[:, :-k])

    # Group delay compensation.
    if compensate_group_delay:
        onset_fn = apply_group_delay_compensation(
            onset_fn, cfs, hop_ms, order, erb_q, erb_min_bw, erb_bw_coeff,
        )

    # Dense cell ranges.
    w = max(4, int(n_filters * gradient_cell_width_frac))
    step = gradient_step
    cell_ranges: list[tuple[int, int]] = []
    lo = 0
    while lo + w <= n_filters:
        cell_ranges.append((lo, lo + w))
        lo += step
    if cell_ranges and cell_ranges[-1][1] < n_filters:
        cell_ranges.append((n_filters - w, n_filters))
    n_cells = len(cell_ranges)

    # Peak detection: vectorized across all channels.
    sync_half = max(1, int(sync_window_ms / hop_ms))
    thresh = np.percentile(onset_fn, peak_percentile, axis=1, keepdims=True)
    above = onset_fn > thresh
    local_max = np.zeros_like(above)
    local_max[:, 1:-1] = (
        (onset_fn[:, 1:-1] > onset_fn[:, :-2]) &
        (onset_fn[:, 1:-1] >= onset_fn[:, 2:])
    )
    peak_map = (above & local_max).astype(np.float32)

    win = np.ones(2 * sync_half + 1, dtype=np.float32)
    peak_windowed = np.zeros_like(peak_map)
    for ch in range(n_filters):
        peak_windowed[ch] = np.convolve(peak_map[ch], win, mode='same')
    peak_windowed = np.clip(peak_windowed, 0, 1)

    # Cumulative sums for O(1) per-cell computation.
    pw_cumsum = np.vstack([
        np.zeros((1, n_frames), dtype=np.float32),
        np.cumsum(peak_windowed, axis=0),
    ])
    onset_cumsum = np.vstack([
        np.zeros((1, n_frames), dtype=np.float32),
        np.cumsum(onset_fn, axis=0),
    ])

    octopus = np.zeros((n_cells, n_frames), dtype=np.float32)
    for ci, (lo, hi) in enumerate(cell_ranges):
        n_chan = hi - lo
        n_active = pw_cumsum[hi] - pw_cumsum[lo]
        magnitude = onset_cumsum[hi] - onset_cumsum[lo]
        octopus[ci] = (n_active / n_chan) ** gradient_nonlinearity_exp * magnitude

    # Per-cell normalization to [0, 1].
    for ci in range(n_cells):
        mx = np.percentile(octopus[ci], norm_percentile - 0.4)
        if mx > 0:
            octopus[ci] = np.clip(octopus[ci] / mx, 0, 1)

    return octopus, cfs, n_cells
