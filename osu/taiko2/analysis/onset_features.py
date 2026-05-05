"""GPU-accelerated onset detection functions + frame-wise evaluation.

Every algorithm here is a pure function over GPU tensors so the survey
CLI can chain them without Python overhead per chart. Two input
families:

  * **Mel-domain** (``energy``, ``spectral_flux``, ``log_filtered_flux``,
    ``hfc_mel``, ``superflux``, ``subband_flux``) — operate on a
    ``(F_mel, T)`` log-mel spectrogram. The cached features the
    dataset already stores are usable directly; no audio decode.
  * **STFT-domain** (``complex_domain``) — needs phase information,
    so a fresh STFT is required.

Outputs are activation envelopes shaped ``(T,)`` (per-frame energies)
or ``(C, T)`` for sub-band variants, on whatever device the input was
on.

Peak-picking and threshold-sweep evaluation live alongside so the CLI
just composes the pieces.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ─────────────────────────── Mel-domain ODFs ──────────────────────────


def energy(mel: torch.Tensor) -> torch.Tensor:
    """Sum of mel-band magnitudes per frame. ``mel: (F_mel, T)``."""
    return mel.sum(dim=-2)


def spectral_flux(mel: torch.Tensor) -> torch.Tensor:
    """Half-wave-rectified magnitude difference, summed over freq.

    ``flux[t] = sum_b max(0, mel[b, t] - mel[b, t-1])``. Pads frame 0
    with zeros so the output length matches ``mel``.
    """
    diff = mel[..., 1:] - mel[..., :-1]
    diff = diff.clamp(min=0.0)
    flux = diff.sum(dim=-2)
    pad = torch.zeros_like(flux[..., :1])
    return torch.cat([pad, flux], dim=-1)


def log_filtered_flux(
    mel: torch.Tensor, *, lambda_compress: float = 1.0,
) -> torch.Tensor:
    """Spectral flux on log-compressed mel.

    Cached features are already log-compressed (top_db=80 mapping
    in MelSampler), so this is essentially the same as
    ``spectral_flux`` for that input but keeps the compression
    parameter explicit for inputs in linear scale.
    """
    log_mel = torch.log1p(lambda_compress * mel.clamp(min=0.0))
    return spectral_flux(log_mel)


def hfc_mel(mel: torch.Tensor, mel_freqs_hz: torch.Tensor) -> torch.Tensor:
    """High-frequency-content envelope on mel bands.

    Classical HFC weights each linear-frequency bin by its index. We
    approximate that on mel by weighting each band by its center
    frequency in Hz.
    """
    weighted = mel * mel_freqs_hz.view(-1, 1)
    return weighted.sum(dim=-2)


def superflux(
    mel: torch.Tensor, *, mu: int = 1, max_filter_size: int = 3,
) -> torch.Tensor:
    """SuperFlux (Böck 2013).

    ``flux[t] = sum_b max(0, mel[b, t] - max_freq(mel[:, t-mu], k))``.
    The max-filter along the frequency axis suppresses vibrato-like
    oscillations between adjacent bands.
    """
    if mu < 1:
        raise ValueError("mu must be >= 1")
    if max_filter_size < 1:
        raise ValueError("max_filter_size must be >= 1")

    F_mel = mel.shape[-2]
    pad = max_filter_size // 2
    # max-pool along frequency axis (treat as 1-D over freq).
    # mel: (F_mel, T) → reshape to (T, 1, F_mel) for max_pool1d over F_mel.
    mel_t = mel.transpose(-1, -2).contiguous()                        # (T, F_mel)
    mel_t = mel_t.unsqueeze(-2)                                       # (T, 1, F_mel)
    maxfilt = F.max_pool1d(
        mel_t, kernel_size=max_filter_size, stride=1, padding=pad,
    )                                                                 # (T, 1, F_mel)
    maxfilt = maxfilt.squeeze(-2).transpose(-1, -2)                   # (F_mel, T)
    # If kernel is even, max_pool1d's `padding` handling can shift the
    # output; clip to original length.
    maxfilt = maxfilt[..., : mel.shape[-1]]

    prev = torch.cat(
        [maxfilt[..., :1].expand(-1, mu), maxfilt[..., :-mu]], dim=-1,
    )                                                                 # (F_mel, T)
    diff = (mel - prev).clamp(min=0.0)
    return diff.sum(dim=-2)


def subband_flux(mel: torch.Tensor, *, n_bands: int) -> torch.Tensor:
    """Spectral flux split into ``n_bands`` equal-mel-band groups.

    Returns ``(n_bands, T)`` so each row is a per-band activation
    envelope. Useful as a multi-channel input feature: low-band
    fires on kicks/DONs, high-band on hats/KAs.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    F_mel = mel.shape[-2]
    if F_mel < n_bands:
        raise ValueError(f"need >= {n_bands} mel bands, got {F_mel}")
    diff = mel[..., 1:] - mel[..., :-1]
    diff = diff.clamp(min=0.0)
    pad = torch.zeros_like(diff[..., :1])
    diff = torch.cat([pad, diff], dim=-1)                             # (F_mel, T)

    band_size = F_mel // n_bands
    rows = []
    for i in range(n_bands):
        lo = i * band_size
        hi = (i + 1) * band_size if i < n_bands - 1 else F_mel
        rows.append(diff[lo:hi, :].sum(dim=-2))
    return torch.stack(rows, dim=0)                                   # (n_bands, T)


# ─────────────────────────── STFT-domain ODFs ─────────────────────────


def complex_domain(
    mag: torch.Tensor, phase: torch.Tensor,
) -> torch.Tensor:
    """Complex-domain (CD) onset detection function.

    Predicts each bin's complex value at frame ``t`` from frames
    ``t-1`` and ``t-2`` (constant magnitude, linearly extrapolated
    phase), then sums the Euclidean distance between predicted and
    observed values across frequency bins. Standard CD ODF from
    Bello et al. 2004 / Duxbury 2003.

    ``mag``, ``phase``: ``(F_stft, T)`` real tensors. Phase is in
    radians (``torch.angle(stft)``).
    """
    if mag.shape != phase.shape:
        raise ValueError("mag and phase must have identical shapes")
    F_stft, T = mag.shape[-2:]

    # Predicted phase: 2*phase[t-1] - phase[t-2], wrapped to [-π, π].
    phi_t1 = torch.cat([phase[..., :1], phase[..., :-1]], dim=-1)
    phi_t2 = torch.cat([phi_t1[..., :1], phi_t1[..., :-1]], dim=-1)
    phi_pred = 2.0 * phi_t1 - phi_t2
    phi_pred = (phi_pred + torch.pi) % (2.0 * torch.pi) - torch.pi

    mag_t1 = torch.cat([mag[..., :1], mag[..., :-1]], dim=-1)

    # Observed - predicted in complex plane.
    cur_re = mag * torch.cos(phase)
    cur_im = mag * torch.sin(phase)
    pred_re = mag_t1 * torch.cos(phi_pred)
    pred_im = mag_t1 * torch.sin(phi_pred)

    diff = ((cur_re - pred_re) ** 2 + (cur_im - pred_im) ** 2).sqrt()
    return diff.sum(dim=-2)


# ─────────────────────────── Peak picking ─────────────────────────────


def normalize_activation(
    a: torch.Tensor, *, percentile: float = 99.0,
) -> torch.Tensor:
    """Divide by the given percentile so threshold sweeps are scale-
    invariant per chart. Operates per-row on multi-channel inputs.
    """
    if a.dim() == 1:
        scale = torch.quantile(a, percentile / 100.0).clamp(min=1e-9)
        return a / scale
    # multi-row: per-row percentile.
    scale = torch.quantile(a, percentile / 100.0, dim=-1, keepdim=True).clamp(min=1e-9)
    return a / scale


def peak_pick(
    activation: torch.Tensor,
    *,
    threshold: float,
    min_distance: int = 1,
) -> torch.Tensor:
    """Return frame indices where ``activation`` is a local maximum
    above ``threshold``, with a minimum-distance NMS pass.

    Greedy NMS — sort peak candidates by activation value descending,
    keep the highest, suppress everything within ``min_distance``
    frames of it, repeat. Returns sorted ascending frame indices.
    """
    if activation.dim() != 1:
        raise ValueError("peak_pick expects a 1-D activation envelope")
    T = activation.shape[0]
    if T < 3:
        return torch.empty(0, dtype=torch.long, device=activation.device)

    # Strict local max: a[t] > a[t-1] and a[t] > a[t+1].
    left = activation[1:-1] > activation[:-2]
    right = activation[1:-1] > activation[2:]
    above = activation[1:-1] > threshold
    is_peak = left & right & above
    candidates = torch.nonzero(is_peak, as_tuple=False).squeeze(-1) + 1   # (P,)
    if candidates.numel() == 0:
        return candidates

    if min_distance <= 1:
        return candidates.sort().values

    # NMS by value, ascending output.
    vals = activation[candidates]
    order = torch.argsort(vals, descending=True)
    cand_sorted = candidates[order]
    kept_mask = torch.ones_like(cand_sorted, dtype=torch.bool)
    kept: list[int] = []
    for i, idx in enumerate(cand_sorted.tolist()):
        if not bool(kept_mask[i]):
            continue
        kept.append(idx)
        # Suppress later candidates within min_distance.
        kept_mask &= (cand_sorted - idx).abs() >= min_distance
        kept_mask[i] = True  # keep self
    out = torch.tensor(sorted(kept), dtype=torch.long, device=activation.device)
    return out


# ─────────────────────────── Evaluation ───────────────────────────────


@dataclass(frozen=True, slots=True)
class FrameEval:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0


def evaluate_frames(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    *,
    tolerance: int,
) -> FrameEval:
    """Greedy nearest-neighbor matching with a per-frame tolerance.

    Each GT frame matches at most one predicted frame (the closest
    unmatched one within ``tolerance``); each predicted frame matches
    at most one GT frame. Standard MIR onset evaluation.

    Both inputs must be sorted ascending.
    """
    pred = pred_frames.cpu().tolist()
    gt = gt_frames.cpu().tolist()
    n_p, n_g = len(pred), len(gt)
    if n_p == 0 and n_g == 0:
        return FrameEval(tp=0, fp=0, fn=0)
    if n_p == 0:
        return FrameEval(tp=0, fp=0, fn=n_g)
    if n_g == 0:
        return FrameEval(tp=0, fp=n_p, fn=0)

    used_pred = [False] * n_p
    used_gt = [False] * n_g
    j_start = 0
    tp = 0
    for i, g in enumerate(gt):
        # Move j_start forward past predictions that are already too
        # far in the past for this gt.
        while j_start < n_p and pred[j_start] < g - tolerance:
            j_start += 1
        # Find the nearest unmatched prediction in window.
        best_j = -1
        best_d = tolerance + 1
        j = j_start
        while j < n_p and pred[j] <= g + tolerance:
            if not used_pred[j]:
                d = abs(pred[j] - g)
                if d < best_d:
                    best_d = d
                    best_j = j
            j += 1
        if best_j >= 0:
            used_pred[best_j] = True
            used_gt[i] = True
            tp += 1
    fp = sum(1 for u in used_pred if not u)
    fn = sum(1 for u in used_gt if not u)
    return FrameEval(tp=tp, fp=fp, fn=fn)


def sweep_thresholds(
    activation: torch.Tensor,
    gt_frames: torch.Tensor,
    *,
    thresholds: torch.Tensor,
    tolerances: tuple[int, ...],
    min_distance: int = 1,
) -> dict[int, list[FrameEval]]:
    """Evaluate ``activation`` at each ``threshold`` × ``tolerance``.

    Returns ``{tolerance: [FrameEval per threshold]}``. Activation
    should be normalized (e.g. via ``normalize_activation``) so the
    same threshold range works across charts.
    """
    out: dict[int, list[FrameEval]] = {tol: [] for tol in tolerances}
    for thr in thresholds.tolist():
        peaks = peak_pick(
            activation, threshold=float(thr), min_distance=min_distance,
        )
        for tol in tolerances:
            out[tol].append(
                evaluate_frames(peaks, gt_frames, tolerance=tol),
            )
    return out


# ─────────────────────────── Mel-frequency helper ─────────────────────


def mel_band_center_freqs_hz(
    sample_rate: int, n_fft: int, n_mels: int,
    *, f_min: float = 20.0, f_max: float | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the center frequency (Hz) of each mel band.

    Mirrors torchaudio's mel filterbank construction so HFC weighting
    matches the cached feature layout. Uses HTK formula like
    librosa.mel_frequencies.
    """
    if f_max is None:
        f_max = sample_rate / 2.0

    def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + f / 700.0)

    def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min = hz_to_mel(torch.tensor(f_min, device=device, dtype=torch.float32))
    m_max = hz_to_mel(torch.tensor(f_max, device=device, dtype=torch.float32))
    mels = torch.linspace(float(m_min), float(m_max), n_mels + 2, device=device)
    hz = mel_to_hz(mels)
    # Center freq of band i = hz[i + 1] (n_mels + 2 boundary points).
    return hz[1:-1].contiguous()
