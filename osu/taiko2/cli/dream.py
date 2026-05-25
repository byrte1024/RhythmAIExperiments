"""Activation maximization for FramewiseDetector.

Freezes a trained checkpoint and optimizes the input mel spectrogram
to produce a target activation map. Reveals what spectral patterns the
model associates with onsets.

Modes:
  dream           Start from noise, optimize toward a target map.
  counterfactual   Start from a real sample, flip specific bins.
  saliency         Gradient of output w.r.t. input (no optimization).

Axes of variation (all modes except saliency):
  --events         empty | real | ablated
  --cond-sweep     Run the same dream at multiple density_mean values.

Usage::

    python -m osu.taiko2.cli.dream \
        --checkpoint osu/taiko2/runs/exp_017e_.../checkpoints/best.pt \
        --dataset taiko2_v1 --n-charts 5 \
        --mode dream --target gt \
        --out-dir osu/taiko2/experiments/020-activation-maximization/custom/dreams \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..data_samplers.detection import (
    TaikoDetectionSampler,
    TaikoDetectionSamplerConfig,
)
from ..inference.loader import load_model_from_checkpoint
from ..models.event_embedding import EventEmbeddingInput
from ..training.framewise_adapter import (
    FramewiseSampleAdapter,
    FramewiseSampleAdapterConfig,
)


# ─────────────────────────── config ──────────────────────────────────

@dataclass(frozen=True, slots=True)
class DreamConfig:
    iterations: int = 3000
    lr: float = 0.03
    lr_end: float = 0.001
    lambda_tv: float = 0.01
    lambda_l2: float = 0.001
    lambda_realism: float = 0.0
    mel_min: float = -35.0
    mel_max: float = 55.0
    jitter_px: int = 2
    seed: int = 42
    use_lbfgs: bool = False
    realistic_init: bool = False


# Per-band dataset statistics (from 50 charts of taiko2_v1).
# Used for realistic initialization and realism penalty.
_BAND_MEAN = np.array([
    21.79, 26.05, 25.20, 23.75, 23.46, 22.73, 22.16, 21.55, 20.94, 20.42,
    19.88, 19.37, 18.91, 18.48, 18.10, 17.74, 17.44, 17.13, 16.83, 16.53,
    16.28, 16.08, 15.87, 15.65, 15.42, 15.21, 15.05, 14.87, 14.70, 14.53,
    14.37, 14.22, 14.10, 13.97, 13.86, 13.74, 13.63, 13.53, 13.43, 13.34,
    13.25, 13.17, 13.09, 13.00, 12.91, 12.84, 12.76, 12.68, 12.61, 12.53,
    12.45, 12.37, 12.29, 12.22, 12.15, 12.08, 12.02, 11.95, 11.88, 11.82,
    11.75, 11.69, 11.62, 11.56, 11.50, 11.43, 11.37, 11.31, 11.25, 11.18,
    11.12, 11.06, 10.99, 10.93, 10.87, 10.83, 10.71, 10.45, 10.51, 10.17,
], dtype=np.float32)
_BAND_STD = np.array([
    14.09, 14.69, 15.06, 15.22, 15.42, 15.57, 15.76, 15.87, 15.92, 15.95,
    15.93, 15.90, 15.85, 15.79, 15.72, 15.67, 15.59, 15.53, 15.46, 15.39,
    15.31, 15.22, 15.13, 15.06, 14.98, 14.90, 14.82, 14.74, 14.66, 14.58,
    14.50, 14.43, 14.35, 14.27, 14.19, 14.12, 14.04, 13.97, 13.89, 13.82,
    13.74, 13.67, 13.59, 13.52, 13.44, 13.37, 13.30, 13.22, 13.15, 13.07,
    13.00, 12.93, 12.86, 12.79, 12.72, 12.65, 12.58, 12.50, 12.43, 12.37,
    12.30, 12.23, 12.16, 12.10, 12.03, 11.96, 11.89, 11.83, 11.76, 11.69,
    11.63, 11.56, 11.50, 11.43, 11.37, 11.32, 11.28, 11.28, 11.28, 11.28,
], dtype=np.float32)


PRESETS: dict[str, dict] = {
    "default": {},
    "legible": {
        "lambda_tv": 0.001,
        "lambda_l2": 0.0001,
        "lambda_realism": 0.005,
        "use_lbfgs": True,
        "realistic_init": True,
        "iterations": 500,
    },
}


# ─────────────────────────── regularization ──────────────────────────

def _total_variation(x: torch.Tensor) -> torch.Tensor:
    """TV loss over (1, F, T) mel."""
    diff_f = (x[:, 1:, :] - x[:, :-1, :]).abs().mean()
    diff_t = (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
    return diff_f + diff_t


# ─────────────────────────── mel inversion ───────────────────────────

def _mel_to_audio(
    mel_np: np.ndarray, sr: int = 22000, n_fft: int = 2048,
    hop_length: int = 110, fmin: float = 20.0, fmax: float = 8000.0,
    power: float = 2.0, top_db: float = 80.0, n_iter: int = 64,
) -> np.ndarray:
    """Griffin-Lim inversion from log-power mel to waveform."""
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=mel_np.shape[0],
        fmin=fmin, fmax=fmax,
    )
    mel_power = librosa.db_to_power(mel_np, ref=1.0)
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    stft_power = mel_basis_pinv @ mel_power
    stft_power = np.maximum(stft_power, 0.0)
    if power == 2.0:
        stft_mag = np.sqrt(stft_power)
    else:
        stft_mag = np.power(stft_power, 1.0 / power)
    audio = librosa.griffinlim(
        stft_mag, n_iter=n_iter, hop_length=hop_length, n_fft=n_fft,
    )
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


# ─────────────────────────── dream loop ──────────────────────────────

def _run_dream(
    model: torch.nn.Module,
    target_map: torch.Tensor,
    event_offsets: torch.Tensor,
    event_mask: torch.Tensor,
    conditioning: torch.Tensor,
    config: DreamConfig,
    device: torch.device,
    init_mel: torch.Tensor | None = None,
    perturbation_budget: float | None = None,
) -> dict[str, np.ndarray]:
    """Core optimization loop. Returns dreamed mel + trajectory."""
    n_mels = 80
    n_frames = 1000

    if init_mel is not None:
        mel_param = init_mel.clone().detach().to(device).requires_grad_(True)
        mel_anchor = init_mel.clone().detach().to(device)
    elif config.realistic_init:
        band_mean_t = torch.from_numpy(_BAND_MEAN).to(device).view(1, n_mels, 1)
        band_std_t = torch.from_numpy(_BAND_STD).to(device).view(1, n_mels, 1)
        rng = torch.Generator(device="cpu").manual_seed(config.seed)
        mel_param = (
            torch.randn(1, n_mels, n_frames, generator=rng).to(device)
            * band_std_t + band_mean_t
        )
        mel_param = mel_param.requires_grad_(True)
        mel_anchor = None
    else:
        rng = torch.Generator(device="cpu").manual_seed(config.seed)
        mel_param = torch.randn(
            1, n_mels, n_frames, generator=rng,
        ).to(device) * 5.0 + 15.0
        mel_param = mel_param.requires_grad_(True)
        mel_anchor = None

    # Realism reference (per-band mean/std on device).
    realism_mean = torch.from_numpy(_BAND_MEAN).to(device).view(1, n_mels, 1)
    realism_std = torch.from_numpy(_BAND_STD).to(device).view(1, n_mels, 1)

    if config.use_lbfgs:
        optimizer = torch.optim.LBFGS(
            [mel_param], lr=config.lr, max_iter=1, history_size=20,
        )
    else:
        optimizer = torch.optim.Adam([mel_param], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=config.lr_end,
    )

    trajectory: dict[str, list[float]] = {
        "loss_total": [], "loss_bce": [], "loss_tv": [], "loss_l2": [],
        "conf_at_target_mean": [], "conf_at_nontarget_mean": [],
    }
    target_bins = target_map[0] > 0.5

    # Shared forward + loss computation used by both Adam and L-BFGS.
    _last = {"loss_bce": 0.0, "loss_tv": 0.0, "loss_l2": 0.0, "loss": 0.0,
             "logits": None}

    def _compute_loss() -> torch.Tensor:
        mel_input = mel_param
        if config.jitter_px > 0 and _last.get("step", 0) % 2 == 0:
            jt = torch.randint(
                -config.jitter_px, config.jitter_px + 1, (2,),
            ).tolist()
            mel_input = torch.roll(mel_input, shifts=jt, dims=[1, 2])

        inp = EventEmbeddingInput(
            mel=mel_input,
            event_offsets=event_offsets,
            event_mask=event_mask,
            conditioning=conditioning,
        )
        out = model.predict(inp)
        logits = out.logits

        loss_bce = F.binary_cross_entropy_with_logits(
            logits, target_map, reduction="mean",
        )
        loss_tv = config.lambda_tv * _total_variation(mel_param)
        loss_l2 = config.lambda_l2 * mel_param.norm()

        loss = loss_bce + loss_tv + loss_l2

        if config.lambda_realism > 0:
            per_band_mean = mel_param.mean(dim=2, keepdim=True)
            loss_realism = config.lambda_realism * (
                ((per_band_mean - realism_mean) / realism_std.clamp(min=1.0))
                .pow(2).mean()
            )
            loss = loss + loss_realism

        if perturbation_budget is not None and mel_anchor is not None:
            delta = mel_param - mel_anchor
            loss_budget = 0.1 * F.relu(delta.norm() - perturbation_budget)
            loss = loss + loss_budget

        _last["loss_bce"] = float(loss_bce.detach())
        _last["loss_tv"] = float(loss_tv.detach())
        _last["loss_l2"] = float(loss_l2.detach())
        _last["loss"] = float(loss.detach())
        _last["logits"] = logits.detach()
        return loss

    pbar = tqdm(range(config.iterations), desc="Dreaming", leave=False)
    for step in pbar:
        _last["step"] = step

        if config.use_lbfgs:
            def closure():
                optimizer.zero_grad()
                loss = _compute_loss()
                loss.backward()
                return loss
            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            loss = _compute_loss()
            loss.backward()
            optimizer.step()

        scheduler.step()

        with torch.no_grad():
            mel_param.clamp_(config.mel_min, config.mel_max)

        with torch.no_grad():
            conf = torch.sigmoid(_last["logits"][0])
            t_conf = float(conf[target_bins].mean()) if target_bins.any() else 0.0
            trajectory["loss_total"].append(_last["loss"])
            trajectory["loss_bce"].append(_last["loss_bce"])
            trajectory["loss_tv"].append(_last["loss_tv"])
            trajectory["loss_l2"].append(_last["loss_l2"])
            trajectory["conf_at_target_mean"].append(t_conf)
            trajectory["conf_at_nontarget_mean"].append(
                float(conf[~target_bins].mean()) if (~target_bins).any() else 0.0,
            )
            if step % 50 == 0 or step == config.iterations - 1:
                pbar.set_postfix(bce=f"{_last['loss_bce']:.3f}", tgt=f"{t_conf:.3f}")

    with torch.no_grad():
        inp_final = EventEmbeddingInput(
            mel=mel_param,
            event_offsets=event_offsets,
            event_mask=event_mask,
            conditioning=conditioning,
        )
        out_final = model.predict(inp_final)
        conf_final = torch.sigmoid(out_final.logits[0]).cpu().numpy()

    dreamed_mel = mel_param[0].detach().cpu().numpy()
    return {
        "dreamed_mel": dreamed_mel,
        "confidence_map": conf_final,
        "target_map": target_map[0].cpu().numpy(),
        "trajectory": {k: np.array(v) for k, v in trajectory.items()},
    }


# ─────────────────────────── saliency ────────────────────────────────

def _run_saliency(
    model: torch.nn.Module,
    mel: torch.Tensor,
    event_offsets: torch.Tensor,
    event_mask: torch.Tensor,
    conditioning: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Gradient saliency: d(confidence) / d(mel) for each output bin."""
    mel_input = mel.clone().detach().requires_grad_(True)
    inp = EventEmbeddingInput(
        mel=mel_input,
        event_offsets=event_offsets,
        event_mask=event_mask,
        conditioning=conditioning,
    )
    out = model.predict(inp)
    logits = out.logits
    conf = torch.sigmoid(logits)
    conf.sum().backward()

    grad = mel_input.grad[0].cpu().numpy()
    return {
        "saliency": grad,
        "confidence_map": conf[0].detach().cpu().numpy(),
        "mel": mel[0].cpu().numpy(),
    }


# ─────────────────────────── visualization ───────────────────────────

def _annotate_mel_ax(
    ax, mel: np.ndarray, target_map: np.ndarray,
    past_events: np.ndarray | None = None,
    label: str = "",
):
    """Draw mel heatmap with cursor divider, onset markers, and labels.

    Mel is (80, 1000) = 500 past + 500 future frames.
    Target map is (500,) covering future bins only.
    Onset bin N in target_map corresponds to mel frame 500 + N.
    past_events is an array of cursor-relative offsets (negative = past).
    """
    n_past = mel.shape[1] // 2  # 500

    ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    ax.axvline(n_past, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(n_past - 5, mel.shape[0] - 3, "PAST", color="white",
            fontsize=8, ha="right", va="top", alpha=0.7)
    ax.text(n_past + 5, mel.shape[0] - 3, "FUTURE", color="white",
            fontsize=8, ha="left", va="top", alpha=0.7)

    # Future onsets from target map (cyan).
    onset_bins = np.where(target_map > 0.5)[0]
    for b in onset_bins:
        ax.axvline(n_past + b, color="cyan", alpha=0.5, linewidth=0.5)

    # Past events (yellow).
    if past_events is not None:
        for off in past_events:
            frame = n_past + int(off)
            if 0 <= frame < mel.shape[1]:
                ax.axvline(frame, color="yellow", alpha=0.3, linewidth=0.5)

    ax.set_title(label)
    ax.set_ylabel("Mel band")


def _save_mel_comparison(
    real_mel: np.ndarray | None,
    dreamed_mel: np.ndarray,
    target_map: np.ndarray,
    confidence_map: np.ndarray,
    out_path: Path,
    title: str = "",
    past_event_offsets: np.ndarray | None = None,
):
    """Mel comparison figure with past/future layout.

    Mel spectrograms are (80, 1000) = 500 past + 500 future frames.
    The cursor sits at frame 500 (white dashed line).
    Target onsets (cyan) are in the future half only (frames 500-999).
    Past events (yellow) are in the past half (frames 0-499).
    The confidence/target plot uses bin indices (0-499 = future only).
    """
    n_rows = 3 if real_mel is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows), dpi=100)

    row = 0
    if real_mel is not None:
        _annotate_mel_ax(
            axes[row], real_mel, target_map,
            past_events=past_event_offsets,
            label="Real mel (cyan=future GT onsets, yellow=past events)",
        )
        row += 1

    _annotate_mel_ax(
        axes[row], dreamed_mel, target_map,
        past_events=past_event_offsets,
        label="Dreamed mel (cyan=target onsets, yellow=past events)",
    )
    row += 1

    # Confidence plot: future bins only (0-499).
    axes[row].plot(target_map, label="Target", alpha=0.7, linewidth=1)
    axes[row].plot(confidence_map, label="Model output", alpha=0.7, linewidth=1)
    axes[row].set_xlim(0, len(target_map))
    axes[row].set_ylim(-0.05, 1.05)
    axes[row].set_xlabel("Future bin (0-499, each = 5ms)")
    axes[row].set_ylabel("Activation")
    axes[row].legend()
    axes[row].set_title("Target vs dreamed confidence (future bins only)")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_trajectory(trajectory: dict[str, np.ndarray], out_path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=100)

    ax1.plot(trajectory["loss_total"], label="total")
    ax1.plot(trajectory["loss_bce"], label="bce")
    ax1.plot(trajectory["loss_tv"], label="tv", alpha=0.5)
    ax1.plot(trajectory["loss_l2"], label="l2", alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Optimization loss")

    ax2.plot(trajectory["conf_at_target_mean"], label="Target bins (mean conf)")
    ax2.plot(trajectory["conf_at_nontarget_mean"], label="Non-target bins (mean conf)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.set_title("Confidence at target vs non-target bins")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_saliency(
    saliency: np.ndarray, mel: np.ndarray,
    confidence_map: np.ndarray, out_path: Path,
):
    n_past = mel.shape[1] // 2  # 500
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=100)

    axes[0].imshow(mel, aspect="auto", origin="lower", cmap="magma")
    axes[0].axvline(n_past, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
    axes[0].set_title("Input mel (dashed = cursor)")
    axes[0].set_ylabel("Mel band")

    vmax = np.percentile(np.abs(saliency), 99)
    axes[1].imshow(
        saliency, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )
    axes[1].axvline(n_past, color="black", linewidth=1.5, linestyle="--", alpha=0.8)
    axes[1].set_title("Saliency (d(conf_sum) / d(mel), dashed = cursor)")
    axes[1].set_ylabel("Mel band")

    axes[2].plot(confidence_map)
    axes[2].set_xlim(0, len(confidence_map))
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Future bin (0-499, each = 5ms)")
    axes[2].set_ylabel("Confidence")
    axes[2].set_title("Model confidence (future bins only)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_cond_sweep(
    sweep_results: list[dict], out_path: Path,
    past_event_offsets: np.ndarray | None = None,
):
    """Compare dreamed mels across conditioning values."""
    n = len(sweep_results)
    target = sweep_results[0]["target_map"]
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3 * (n + 1)), dpi=100)

    for i, res in enumerate(sweep_results):
        _annotate_mel_ax(
            axes[i], res["dreamed_mel"], target,
            past_events=past_event_offsets,
            label=f"density_mean={res['density_mean']:.1f}",
        )

    ax_conf = axes[n]
    for res in sweep_results:
        ax_conf.plot(
            res["confidence_map"],
            label=f"density={res['density_mean']:.1f}",
            alpha=0.7,
        )
    ax_conf.plot(target, label="Target", color="black", linestyle="--", alpha=0.5)
    ax_conf.set_xlim(0, len(target))
    ax_conf.set_ylim(-0.05, 1.05)
    ax_conf.set_xlabel("Future bin (0-499, each = 5ms)")
    ax_conf.set_ylabel("Confidence")
    ax_conf.legend(fontsize=8)
    ax_conf.set_title("Confidence maps across density conditioning")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_events_sweep(
    sweep_results: list[dict], out_path: Path,
    past_event_offsets: np.ndarray | None = None,
):
    """Compare dreamed mels across event configurations."""
    n = len(sweep_results)
    target = sweep_results[0]["target_map"]
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3 * (n + 1)), dpi=100)

    for i, res in enumerate(sweep_results):
        _annotate_mel_ax(
            axes[i], res["dreamed_mel"], target,
            past_events=past_event_offsets if res["event_mode"] == "real" else None,
            label=f"events={res['event_mode']}",
        )

    ax_conf = axes[n]
    for res in sweep_results:
        ax_conf.plot(
            res["confidence_map"],
            label=f"events={res['event_mode']}",
            alpha=0.7,
        )
    ax_conf.plot(target, label="Target", color="black", linestyle="--", alpha=0.5)
    ax_conf.set_xlim(0, len(target))
    ax_conf.set_ylim(-0.05, 1.05)
    ax_conf.set_xlabel("Future bin (0-499, each = 5ms)")
    ax_conf.set_ylabel("Confidence")
    ax_conf.legend(fontsize=8)
    ax_conf.set_title("Confidence maps across event configurations")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────── mel analysis ─────────────────────────────

def _save_mel_analysis(
    dreamed_mel: np.ndarray,
    real_mel: np.ndarray | None,
    target_map: np.ndarray,
    confidence_map: np.ndarray,
    out_dir: Path,
    slug: str = "",
    past_event_offsets: np.ndarray | None = None,
):
    """Compute and save detailed statistics about what the model optimizes.

    Analyzes mel energy at onset vs non-onset frames, per-band profiles,
    correlation between mel features and confidence, and compares dreamed
    vs real mel distributions. Also analyzes the past half of the mel
    around past event positions.
    """
    n_past = dreamed_mel.shape[1] // 2  # 500
    n_bins = len(target_map)
    onset_bins = np.where(target_map > 0.5)[0]
    notonset_bins = np.where(target_map <= 0.5)[0]

    # Future half of dreamed mel: columns corresponding to prediction bins.
    dream_future = dreamed_mel[:, n_past:]  # (80, 500)

    # ── Per-band mean energy at onset vs non-onset frames ────────────
    onset_band_mean = dream_future[:, onset_bins].mean(axis=1) if len(onset_bins) else np.zeros(80)
    notonset_band_mean = dream_future[:, notonset_bins].mean(axis=1) if len(notonset_bins) else np.zeros(80)
    band_delta = onset_band_mean - notonset_band_mean

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)

    axes[0, 0].barh(range(80), onset_band_mean, alpha=0.7, label="Onset frames")
    axes[0, 0].barh(range(80), notonset_band_mean, alpha=0.5, label="Non-onset frames")
    axes[0, 0].set_ylabel("Mel band")
    axes[0, 0].set_xlabel("Mean energy (dB)")
    axes[0, 0].set_title("Per-band energy: onset vs non-onset frames")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].barh(range(80), band_delta, color=np.where(band_delta > 0, "tab:red", "tab:blue"))
    axes[0, 1].set_ylabel("Mel band")
    axes[0, 1].set_xlabel("Energy delta (onset - non-onset)")
    axes[0, 1].axvline(0, color="black", linewidth=0.5)
    axes[0, 1].set_title("Per-band onset selectivity (red = onset louder)")

    # ── Band-group summary ───────────────────────────────────────────
    groups = [
        ("0-9 (sub-bass)", slice(0, 10)),
        ("10-19 (bass)", slice(10, 20)),
        ("20-29 (low-mid)", slice(20, 30)),
        ("30-39 (mid)", slice(30, 40)),
        ("40-49 (high-mid)", slice(40, 50)),
        ("50-59 (presence)", slice(50, 60)),
        ("60-69 (brilliance)", slice(60, 70)),
        ("70-79 (air)", slice(70, 80)),
    ]
    group_names = [g[0] for g in groups]
    group_onset = [onset_band_mean[g[1]].mean() for g in groups]
    group_notonset = [notonset_band_mean[g[1]].mean() for g in groups]

    x = np.arange(len(groups))
    axes[1, 0].bar(x - 0.2, group_onset, 0.35, label="Onset", alpha=0.7)
    axes[1, 0].bar(x + 0.2, group_notonset, 0.35, label="Non-onset", alpha=0.5)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(group_names, rotation=45, ha="right", fontsize=7)
    axes[1, 0].set_ylabel("Mean energy (dB)")
    axes[1, 0].set_title("Band-group energy: onset vs non-onset")
    axes[1, 0].legend(fontsize=8)

    # ── Confidence vs mel energy correlation ─────────────────────────
    per_frame_energy = dream_future.mean(axis=0)  # (500,)
    low_band_energy = dream_future[:30, :].mean(axis=0)
    high_band_energy = dream_future[40:, :].mean(axis=0)

    corr_all = float(np.corrcoef(per_frame_energy, confidence_map)[0, 1]) if n_bins >= 2 else 0.0
    corr_low = float(np.corrcoef(low_band_energy, confidence_map)[0, 1]) if n_bins >= 2 else 0.0
    corr_high = float(np.corrcoef(high_band_energy, confidence_map)[0, 1]) if n_bins >= 2 else 0.0

    axes[1, 1].scatter(per_frame_energy, confidence_map, s=2, alpha=0.3, label=f"all bands (r={corr_all:.3f})")
    axes[1, 1].scatter(low_band_energy, confidence_map, s=2, alpha=0.3, label=f"bands 0-29 (r={corr_low:.3f})")
    axes[1, 1].scatter(high_band_energy, confidence_map, s=2, alpha=0.3, label=f"bands 40-79 (r={corr_high:.3f})")
    axes[1, 1].set_xlabel("Mean mel energy at frame")
    axes[1, 1].set_ylabel("Model confidence")
    axes[1, 1].set_title("Mel energy vs confidence (per future frame)")
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(f"Mel analysis: {slug}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{slug}_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # ── Real vs dreamed comparison ───────────────────────────────────
    if real_mel is not None:
        real_future = real_mel[:, n_past:]
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), dpi=100)

        real_onset_mean = real_future[:, onset_bins].mean(axis=1) if len(onset_bins) else np.zeros(80)
        real_notonset_mean = real_future[:, notonset_bins].mean(axis=1) if len(notonset_bins) else np.zeros(80)

        axes2[0, 0].plot(onset_band_mean, range(80), label="Dream onset", alpha=0.8)
        axes2[0, 0].plot(real_onset_mean, range(80), label="Real onset", alpha=0.8, linestyle="--")
        axes2[0, 0].set_ylabel("Mel band")
        axes2[0, 0].set_xlabel("Mean energy (dB)")
        axes2[0, 0].set_title("Onset frames: dreamed vs real per-band profile")
        axes2[0, 0].legend(fontsize=8)

        axes2[0, 1].plot(notonset_band_mean, range(80), label="Dream non-onset", alpha=0.8)
        axes2[0, 1].plot(real_notonset_mean, range(80), label="Real non-onset", alpha=0.8, linestyle="--")
        axes2[0, 1].set_ylabel("Mel band")
        axes2[0, 1].set_xlabel("Mean energy (dB)")
        axes2[0, 1].set_title("Non-onset frames: dreamed vs real per-band profile")
        axes2[0, 1].legend(fontsize=8)

        # Per-band correlation between dreamed and real.
        band_corrs = np.array([
            float(np.corrcoef(dreamed_mel[b, :], real_mel[b, :])[0, 1])
            if np.std(dreamed_mel[b, :]) > 1e-8 and np.std(real_mel[b, :]) > 1e-8
            else 0.0
            for b in range(80)
        ])
        axes2[1, 0].barh(range(80), band_corrs)
        axes2[1, 0].set_ylabel("Mel band")
        axes2[1, 0].set_xlabel("Pearson r (dreamed vs real)")
        axes2[1, 0].set_title("Per-band dreamed-real correlation (full 1000 frames)")
        axes2[1, 0].axvline(0, color="black", linewidth=0.5)

        # Energy histogram: dreamed vs real.
        axes2[1, 1].hist(dreamed_mel.ravel(), bins=100, alpha=0.5, label="Dreamed", density=True)
        axes2[1, 1].hist(real_mel.ravel(), bins=100, alpha=0.5, label="Real", density=True)
        axes2[1, 1].set_xlabel("Mel energy (dB)")
        axes2[1, 1].set_ylabel("Density")
        axes2[1, 1].set_title("Mel value distribution: dreamed vs real")
        axes2[1, 1].legend(fontsize=8)

        fig2.suptitle(f"Dream vs real: {slug}", fontsize=11)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"{slug}_vs_real.png", bbox_inches="tight")
        plt.close(fig2)

    # ── Temporal offset analysis ────────────────────────────────────
    # For each onset, gather mel energy and confidence at offsets
    # -10..+10 frames relative to the onset bin. This reveals whether
    # the model places energy (or confidence) before, at, or after the
    # onset position.
    offsets = np.arange(-10, 11)  # -10 to +10
    n_offsets = len(offsets)

    if len(onset_bins) >= 2:
        # Dream mel energy per offset (mean across onsets and bands).
        dream_offset_energy = np.zeros((n_offsets, 80), dtype=np.float64)
        dream_offset_count = np.zeros(n_offsets, dtype=np.int64)
        # Confidence per offset.
        conf_offset_sum = np.zeros(n_offsets, dtype=np.float64)
        conf_offset_count = np.zeros(n_offsets, dtype=np.int64)

        for ob in onset_bins:
            for oi, off in enumerate(offsets):
                frame = ob + off
                if 0 <= frame < n_bins:
                    dream_offset_energy[oi] += dream_future[:, frame]
                    dream_offset_count[oi] += 1
                    conf_offset_sum[oi] += confidence_map[frame]
                    conf_offset_count[oi] += 1

        safe_count = np.maximum(dream_offset_count, 1)
        dream_offset_mean = dream_offset_energy / safe_count[:, None]  # (21, 80)
        dream_offset_total = dream_offset_mean.mean(axis=1)  # (21,) avg across bands
        dream_offset_low = dream_offset_mean[:, :30].mean(axis=1)
        dream_offset_high = dream_offset_mean[:, 40:].mean(axis=1)
        conf_offset_mean = conf_offset_sum / np.maximum(conf_offset_count, 1)

        # Same for real mel if available.
        real_offset_total = None
        if real_mel is not None:
            real_future = real_mel[:, n_past:]
            real_offset_energy = np.zeros((n_offsets, 80), dtype=np.float64)
            for ob in onset_bins:
                for oi, off in enumerate(offsets):
                    frame = ob + off
                    if 0 <= frame < n_bins:
                        real_offset_energy[oi] += real_future[:, frame]
            real_offset_mean = real_offset_energy / safe_count[:, None]
            real_offset_total = real_offset_mean.mean(axis=1)

        # Peak offset: where does the model place the most energy?
        peak_energy_offset = int(offsets[np.argmax(dream_offset_total)])
        peak_conf_offset = int(offsets[np.argmax(conf_offset_mean)])

        fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10), dpi=100)

        axes3[0, 0].plot(offsets, dream_offset_total, "o-", label="Dream (all bands)", linewidth=2)
        axes3[0, 0].plot(offsets, dream_offset_low, "s--", label="Dream (bands 0-29)", alpha=0.7)
        axes3[0, 0].plot(offsets, dream_offset_high, "^--", label="Dream (bands 40-79)", alpha=0.7)
        if real_offset_total is not None:
            axes3[0, 0].plot(offsets, real_offset_total, "o-", label="Real (all bands)", alpha=0.5)
        axes3[0, 0].axvline(0, color="black", linewidth=0.5, linestyle=":")
        axes3[0, 0].axvline(peak_energy_offset, color="red", linewidth=1, linestyle="--", alpha=0.5)
        axes3[0, 0].set_xlabel("Frame offset from onset (negative = before)")
        axes3[0, 0].set_ylabel("Mean mel energy (dB)")
        axes3[0, 0].set_title(f"Energy around onsets (peak at offset {peak_energy_offset})")
        axes3[0, 0].legend(fontsize=7)

        axes3[0, 1].plot(offsets, conf_offset_mean, "o-", color="tab:orange", linewidth=2)
        axes3[0, 1].axvline(0, color="black", linewidth=0.5, linestyle=":")
        axes3[0, 1].axvline(peak_conf_offset, color="red", linewidth=1, linestyle="--", alpha=0.5)
        axes3[0, 1].set_xlabel("Frame offset from onset (negative = before)")
        axes3[0, 1].set_ylabel("Mean confidence")
        axes3[0, 1].set_title(f"Confidence around onsets (peak at offset {peak_conf_offset})")
        axes3[0, 1].set_ylim(-0.05, 1.05)

        # Heatmap: per-band energy at each offset.
        im = axes3[1, 0].imshow(
            dream_offset_mean.T, aspect="auto", origin="lower", cmap="magma",
            extent=[offsets[0] - 0.5, offsets[-1] + 0.5, 0, 80],
        )
        axes3[1, 0].axvline(0, color="white", linewidth=1, linestyle=":")
        axes3[1, 0].set_xlabel("Frame offset from onset")
        axes3[1, 0].set_ylabel("Mel band")
        axes3[1, 0].set_title("Per-band energy around onsets (dreamed)")
        plt.colorbar(im, ax=axes3[1, 0], label="dB")

        # Delta heatmap vs real if available.
        if real_mel is not None:
            delta_offset = dream_offset_mean - real_offset_mean
            vmax_d = np.percentile(np.abs(delta_offset), 95)
            im2 = axes3[1, 1].imshow(
                delta_offset.T, aspect="auto", origin="lower", cmap="RdBu_r",
                extent=[offsets[0] - 0.5, offsets[-1] + 0.5, 0, 80],
                vmin=-vmax_d, vmax=vmax_d,
            )
            axes3[1, 1].axvline(0, color="black", linewidth=1, linestyle=":")
            axes3[1, 1].set_xlabel("Frame offset from onset")
            axes3[1, 1].set_ylabel("Mel band")
            axes3[1, 1].set_title("Energy delta (dreamed - real) around onsets")
            plt.colorbar(im2, ax=axes3[1, 1], label="dB delta")
        else:
            axes3[1, 1].set_visible(False)

        fig3.suptitle(f"Temporal onset analysis: {slug}", fontsize=11)
        fig3.tight_layout()
        fig3.savefig(out_dir / f"{slug}_temporal.png", bbox_inches="tight")
        plt.close(fig3)
    else:
        peak_energy_offset = 0
        peak_conf_offset = 0
        dream_offset_total = np.zeros(n_offsets)
        dream_offset_low = np.zeros(n_offsets)
        dream_offset_high = np.zeros(n_offsets)
        dream_offset_mean = np.zeros((n_offsets, 80))
        conf_offset_mean = np.zeros(n_offsets)

    # ── Past event temporal analysis ─────────────────────────────────
    # Same offset analysis but on the PAST half of the mel around past
    # event positions. Past events sit in mel frames 0..499 (offsets are
    # negative, so mel frame = n_past + offset).
    dream_past = dreamed_mel[:, :n_past]  # (80, 500)
    past_peak_energy_offset = 0
    past_dream_offset_total = np.zeros(n_offsets)
    past_dream_offset_low = np.zeros(n_offsets)
    past_dream_offset_high = np.zeros(n_offsets)
    past_dream_offset_mean = np.zeros((n_offsets, 80))
    past_event_band_mean = np.zeros(80)
    past_nonevent_band_mean = np.zeros(80)
    past_band_delta = np.zeros(80)

    past_frames: list[int] = []
    if past_event_offsets is not None and len(past_event_offsets) > 0:
        for off in past_event_offsets:
            frame = n_past + int(off)
            if 0 <= frame < n_past:
                past_frames.append(frame)

    if len(past_frames) >= 2:
        # Per-band energy at past event frames vs non-event frames.
        past_event_indices = np.array(past_frames, dtype=np.int64)
        past_all_indices = np.arange(n_past)
        past_nonevent_indices = np.setdiff1d(past_all_indices, past_event_indices)

        past_event_band_mean = dream_past[:, past_event_indices].mean(axis=1)
        if len(past_nonevent_indices) > 0:
            past_nonevent_band_mean = dream_past[:, past_nonevent_indices].mean(axis=1)
        past_band_delta = past_event_band_mean - past_nonevent_band_mean

        # Temporal offset analysis.
        past_offset_energy = np.zeros((n_offsets, 80), dtype=np.float64)
        past_offset_count = np.zeros(n_offsets, dtype=np.int64)

        for pf in past_frames:
            for oi, off in enumerate(offsets):
                frame = pf + off
                if 0 <= frame < n_past:
                    past_offset_energy[oi] += dream_past[:, frame]
                    past_offset_count[oi] += 1

        safe_count_p = np.maximum(past_offset_count, 1)
        past_dream_offset_mean = past_offset_energy / safe_count_p[:, None]
        past_dream_offset_total = past_dream_offset_mean.mean(axis=1)
        past_dream_offset_low = past_dream_offset_mean[:, :30].mean(axis=1)
        past_dream_offset_high = past_dream_offset_mean[:, 40:].mean(axis=1)
        past_peak_energy_offset = int(offsets[np.argmax(past_dream_offset_total)])

        real_past_offset_total = None
        if real_mel is not None:
            real_past = real_mel[:, :n_past]
            real_past_offset_energy = np.zeros((n_offsets, 80), dtype=np.float64)
            for pf in past_frames:
                for oi, off in enumerate(offsets):
                    frame = pf + off
                    if 0 <= frame < n_past:
                        real_past_offset_energy[oi] += real_past[:, frame]
            real_past_offset_mean = real_past_offset_energy / safe_count_p[:, None]
            real_past_offset_total = real_past_offset_mean.mean(axis=1)

        fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10), dpi=100)

        # Top row: per-band analysis.
        axes4[0, 0].barh(range(80), past_event_band_mean, alpha=0.7, label="Event frames")
        axes4[0, 0].barh(range(80), past_nonevent_band_mean, alpha=0.5, label="Non-event frames")
        axes4[0, 0].set_ylabel("Mel band")
        axes4[0, 0].set_xlabel("Mean energy (dB)")
        axes4[0, 0].set_title("Past: per-band energy at event vs non-event")
        axes4[0, 0].legend(fontsize=7)

        axes4[0, 1].barh(range(80), past_band_delta,
                         color=np.where(past_band_delta > 0, "tab:red", "tab:blue"))
        axes4[0, 1].set_ylabel("Mel band")
        axes4[0, 1].set_xlabel("Delta (event - non-event)")
        axes4[0, 1].axvline(0, color="black", linewidth=0.5)
        axes4[0, 1].set_title("Past: per-band event selectivity")

        # Band-group summary for past.
        past_group_event = [past_event_band_mean[g[1]].mean() for g in groups]
        past_group_nonevent = [past_nonevent_band_mean[g[1]].mean() for g in groups]
        x_g = np.arange(len(groups))
        axes4[0, 2].bar(x_g - 0.2, past_group_event, 0.35, label="Event", alpha=0.7)
        axes4[0, 2].bar(x_g + 0.2, past_group_nonevent, 0.35, label="Non-event", alpha=0.5)
        axes4[0, 2].set_xticks(x_g)
        axes4[0, 2].set_xticklabels(group_names, rotation=45, ha="right", fontsize=7)
        axes4[0, 2].set_ylabel("Mean energy (dB)")
        axes4[0, 2].set_title("Past: band-group event vs non-event")
        axes4[0, 2].legend(fontsize=7)

        # Bottom row: temporal analysis.
        axes4[1, 0].plot(offsets, past_dream_offset_total, "o-", label="Dream (all bands)", linewidth=2)
        axes4[1, 0].plot(offsets, past_dream_offset_low, "s--", label="Dream (0-29)", alpha=0.7)
        axes4[1, 0].plot(offsets, past_dream_offset_high, "^--", label="Dream (40-79)", alpha=0.7)
        if real_past_offset_total is not None:
            axes4[1, 0].plot(offsets, real_past_offset_total, "o-", label="Real (all)", alpha=0.5)
        axes4[1, 0].axvline(0, color="black", linewidth=0.5, linestyle=":")
        axes4[1, 0].axvline(past_peak_energy_offset, color="red", linewidth=1, linestyle="--", alpha=0.5)
        axes4[1, 0].set_xlabel("Frame offset from past event")
        axes4[1, 0].set_ylabel("Mean mel energy (dB)")
        axes4[1, 0].set_title(f"Past events: energy (peak at {past_peak_energy_offset})")
        axes4[1, 0].legend(fontsize=7)

        im4 = axes4[1, 1].imshow(
            past_dream_offset_mean.T, aspect="auto", origin="lower", cmap="magma",
            extent=[offsets[0] - 0.5, offsets[-1] + 0.5, 0, 80],
        )
        axes4[1, 1].axvline(0, color="white", linewidth=1, linestyle=":")
        axes4[1, 1].set_xlabel("Frame offset from past event")
        axes4[1, 1].set_ylabel("Mel band")
        axes4[1, 1].set_title("Per-band energy around past events (dreamed)")
        plt.colorbar(im4, ax=axes4[1, 1], label="dB")

        if real_mel is not None:
            delta_past = past_dream_offset_mean - real_past_offset_mean
            vmax_dp = max(np.percentile(np.abs(delta_past), 95), 1e-6)
            im5 = axes4[1, 2].imshow(
                delta_past.T, aspect="auto", origin="lower", cmap="RdBu_r",
                extent=[offsets[0] - 0.5, offsets[-1] + 0.5, 0, 80],
                vmin=-vmax_dp, vmax=vmax_dp,
            )
            axes4[1, 2].axvline(0, color="black", linewidth=1, linestyle=":")
            axes4[1, 2].set_xlabel("Frame offset from past event")
            axes4[1, 2].set_ylabel("Mel band")
            axes4[1, 2].set_title("Energy delta (dreamed - real) around past events")
            plt.colorbar(im5, ax=axes4[1, 2], label="dB delta")
        else:
            axes4[1, 2].set_visible(False)

        fig4.suptitle(f"Past event analysis: {slug} ({len(past_frames)} events)", fontsize=11)
        fig4.tight_layout()
        fig4.savefig(out_dir / f"{slug}_past_events.png", bbox_inches="tight")
        plt.close(fig4)

    # ── Save numeric data ────────────────────────────────────────────
    stats = {
        "onset_band_mean": onset_band_mean,
        "notonset_band_mean": notonset_band_mean,
        "band_delta": band_delta,
        "per_frame_energy": per_frame_energy,
        "low_band_energy": low_band_energy,
        "high_band_energy": high_band_energy,
        "confidence_map": confidence_map,
        "corr_all_bands": np.float64(corr_all),
        "corr_low_bands": np.float64(corr_low),
        "corr_high_bands": np.float64(corr_high),
        "low_high_energy_ratio": np.float64(
            low_band_energy.mean() / max(high_band_energy.mean(), 1e-8)
        ),
        "onset_mean_energy": np.float64(per_frame_energy[onset_bins].mean() if len(onset_bins) else 0.0),
        "notonset_mean_energy": np.float64(per_frame_energy[notonset_bins].mean() if len(notonset_bins) else 0.0),
        # Future onset temporal.
        "temporal_offsets": offsets,
        "temporal_energy_mean": dream_offset_total,
        "temporal_energy_low": dream_offset_low,
        "temporal_energy_high": dream_offset_high,
        "temporal_energy_per_band": dream_offset_mean,
        "temporal_conf_mean": conf_offset_mean,
        "peak_energy_offset": np.int64(peak_energy_offset),
        "peak_conf_offset": np.int64(peak_conf_offset),
        # Past event temporal.
        "past_event_band_mean": past_event_band_mean,
        "past_nonevent_band_mean": past_nonevent_band_mean,
        "past_band_delta": past_band_delta,
        "past_temporal_energy_mean": past_dream_offset_total,
        "past_temporal_energy_low": past_dream_offset_low,
        "past_temporal_energy_high": past_dream_offset_high,
        "past_temporal_energy_per_band": past_dream_offset_mean,
        "past_peak_energy_offset": np.int64(past_peak_energy_offset),
        "past_n_events": np.int64(len(past_frames)),
    }
    np.savez(out_dir / f"{slug}_analysis.npz", **stats)


# ─────────────────────────── target builders ─────────────────────────

def _target_single(bin_idx: int, n_bins: int = 500) -> np.ndarray:
    t = np.zeros(n_bins, dtype=np.float32)
    t[bin_idx] = 1.0
    return t


def _target_metronome(gap_bins: int, n_bins: int = 500) -> np.ndarray:
    t = np.zeros(n_bins, dtype=np.float32)
    for i in range(0, n_bins, gap_bins):
        t[i] = 1.0
    return t


def _target_from_sample(sample, b_pred: int = 500) -> np.ndarray:
    t = np.zeros(b_pred, dtype=np.float32)
    for i, ev in enumerate(sample.future_events):
        if bool(sample.future_events_mask[i]):
            continue
        off = int(ev.cursor_offset)
        if 0 <= off < b_pred:
            t[off] = 1.0
    return t


# ─────────────────────────── input builders ──────────────────────────

def _make_empty_events(
    c_events: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.zeros(1, c_events, dtype=torch.int64, device=device)
    mask = torch.ones(1, c_events, dtype=torch.bool, device=device)
    return offsets, mask


def _make_real_events(
    sample, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    c_events = len(sample.past_events)
    offsets = torch.tensor(
        [o.cursor_offset for o in sample.past_events],
        dtype=torch.int64, device=device,
    ).unsqueeze(0)
    mask = torch.tensor(
        sample.past_events_mask, dtype=torch.bool, device=device,
    ).unsqueeze(0)
    return offsets, mask


def _make_conditioning(
    density_mean: float, density_peak: float, density_std: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [[density_mean, density_peak, density_std]],
        dtype=torch.float32, device=device,
    )


# ─────────────────────────── sample selection ────────────────────────

def _load_sampler(dataset: str, split: str = "val") -> TaikoDetectionSampler:
    cfg = TaikoDetectionSamplerConfig(
        batch_size=1,
        dataset_root=str(
            Path(__file__).resolve().parent.parent / "datasets" / dataset
        ),
        split=split,
        a_bins=500, b_bins=500, c_events=128, d_events=100,
        min_cursor_bin=6000,
    )
    sampler = TaikoDetectionSampler(cfg)
    sampler.load_data(progress=True)
    return sampler


def _pick_diverse_samples(
    sampler: TaikoDetectionSampler, n_charts: int,
) -> list[int]:
    """Pick sample indices spread across the density range.

    Sorts charts by density_mean, then picks n_charts evenly spaced
    across the sorted list. Returns one sample index per chart (the
    first sample for that chart).
    """
    n_total = sampler.count_charts()
    if n_charts >= n_total:
        n_charts = n_total

    chart_densities: list[tuple[int, float, str]] = []
    for ci in range(n_total):
        entry = sampler._chart_entries[ci]
        dm = entry.density_mean if entry.density_mean is not None else 0.0
        if dm < 2.0 or dm > 7.0:
            continue
        chart_densities.append((ci, dm, sampler._chart_ids[ci]))

    chart_densities.sort(key=lambda x: x[1])

    step = max(1, len(chart_densities) // n_charts)
    picked_charts = []
    for i in range(0, len(chart_densities), step):
        if len(picked_charts) >= n_charts:
            break
        picked_charts.append(chart_densities[i])

    sample_indices: list[int] = []
    for ci, dm, cid in picked_charts:
        for si, (sc, _ei) in enumerate(sampler._samples):
            if sc == ci:
                sample_indices.append(si)
                break

    return sample_indices


# ─────────────────────────── per-chart runner ────────────────────────

def _run_one_chart(
    model: torch.nn.Module,
    sample,
    chart_dir: Path,
    args: argparse.Namespace,
    dcfg: DreamConfig,
    c_events: int,
    device: torch.device,
):
    """Run all modes for one chart sample. Saves to chart_dir."""
    chart_dir.mkdir(parents=True, exist_ok=True)

    real_mel_np = np.concatenate(
        [sample.audio_past, sample.audio_future], axis=1,
    )

    # Extract past event offsets for visualization.
    past_offsets = np.array([
        o.cursor_offset for i, o in enumerate(sample.past_events)
        if not sample.past_events_mask[i]
    ], dtype=np.int64)

    # Build target map.
    if args.target == "gt":
        target_np = _target_from_sample(sample)
    elif args.target == "model":
        real_mel_t = torch.from_numpy(real_mel_np).unsqueeze(0).to(
            device=device, dtype=torch.float32,
        )
        ev_off, ev_mask = _make_real_events(sample, device)
        cond = _make_conditioning(
            sample.density_mean, sample.density_peak, sample.density_std,
            device,
        )
        with torch.no_grad():
            out = model.predict(EventEmbeddingInput(
                mel=real_mel_t, event_offsets=ev_off,
                event_mask=ev_mask, conditioning=cond,
            ))
            target_np = torch.sigmoid(out.logits[0]).cpu().numpy()
            target_np = (target_np > args.model_threshold).astype(np.float32)
    elif args.target == "single":
        target_np = _target_single(args.target_bin)
    elif args.target.startswith("metro"):
        target_np = _target_metronome(args.metro_gap)
    else:
        target_np = _target_from_sample(sample)

    target_t = torch.from_numpy(target_np).unsqueeze(0).to(device)
    n_onsets = int((target_np > 0.5).sum())

    base_density_mean = sample.density_mean
    base_density_peak = float(sample.density_peak)
    base_density_std = sample.density_std

    if args.mode == "saliency":
        real_mel_t = torch.from_numpy(real_mel_np).unsqueeze(0).to(
            device=device, dtype=torch.float32,
        )
        ev_off, ev_mask = _make_real_events(sample, device)
        cond = _make_conditioning(
            base_density_mean, base_density_peak, base_density_std, device,
        )
        result = _run_saliency(model, real_mel_t, ev_off, ev_mask, cond)
        _save_saliency(
            result["saliency"], result["mel"],
            result["confidence_map"], chart_dir / "saliency.png",
        )
        np.savez(
            chart_dir / "saliency.npz",
            saliency=result["saliency"],
            confidence_map=result["confidence_map"],
            mel=result["mel"],
        )
        return {"chart_id": sample.chart_id, "n_onsets": n_onsets,
                "density_mean": base_density_mean, "mode": "saliency"}

    # ── Dream / Counterfactual ───────────────────────────────────────

    init_mel = None
    perturbation_budget = None
    if args.mode == "counterfactual":
        init_mel = torch.from_numpy(real_mel_np).unsqueeze(0).to(
            device=device, dtype=torch.float32,
        )
        perturbation_budget = args.perturbation_budget

    # ── Event sweep ──────────────────────────────────────────────────

    event_modes = ["empty", "real"]
    event_sweep_results = []

    for event_mode in event_modes:
        if event_mode == "empty":
            ev_off, ev_mask = _make_empty_events(c_events, device)
        else:
            ev_off, ev_mask = _make_real_events(sample, device)

        cond = _make_conditioning(
            base_density_mean, base_density_peak, base_density_std, device,
        )

        result = _run_dream(
            model, target_t, ev_off, ev_mask, cond, dcfg, device,
            init_mel=init_mel, perturbation_budget=perturbation_budget,
        )

        slug = f"{args.mode}_{event_mode}"
        _save_mel_comparison(
            real_mel_np, result["dreamed_mel"],
            result["target_map"], result["confidence_map"],
            chart_dir / f"{slug}_mel.png",
            title=f"{sample.chart_id} | {args.mode} | events={event_mode} | "
                  f"density={base_density_mean:.1f} | {n_onsets} onsets",
            past_event_offsets=past_offsets,
        )
        _save_trajectory(result["trajectory"], chart_dir / f"{slug}_trajectory.png")

        audio = _mel_to_audio(result["dreamed_mel"])
        sf.write(chart_dir / f"{slug}_dreamed.wav", audio, 22000)

        np.savez(
            chart_dir / f"{slug}_data.npz",
            dreamed_mel=result["dreamed_mel"],
            confidence_map=result["confidence_map"],
            target_map=result["target_map"],
            **{f"traj_{k}": v for k, v in result["trajectory"].items()},
        )

        _save_mel_analysis(
            result["dreamed_mel"], real_mel_np,
            result["target_map"], result["confidence_map"],
            chart_dir, slug=slug,
            past_event_offsets=past_offsets if event_mode == "real" else None,
        )

        event_sweep_results.append({
            "event_mode": event_mode,
            "dreamed_mel": result["dreamed_mel"],
            "confidence_map": result["confidence_map"],
            "target_map": result["target_map"],
        })

    # Save real audio once.
    audio_real = _mel_to_audio(real_mel_np)
    sf.write(chart_dir / "real.wav", audio_real, 22000)

    if len(event_sweep_results) > 1:
        _save_events_sweep(
            event_sweep_results, chart_dir / "events_sweep.png",
            past_event_offsets=past_offsets,
        )

    # ── Conditioning sweep ───────────────────────────────────────────

    if args.cond_sweep:
        sweep_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
        sweep_results = []

        ev_off, ev_mask = _make_empty_events(c_events, device)

        for dm in tqdm(sweep_values, desc="Cond sweep", leave=False):
            cond = _make_conditioning(dm, base_density_peak, base_density_std, device)
            result = _run_dream(
                model, target_t, ev_off, ev_mask, cond, dcfg, device,
                init_mel=init_mel, perturbation_budget=perturbation_budget,
            )
            sweep_results.append({
                "density_mean": dm,
                "dreamed_mel": result["dreamed_mel"],
                "confidence_map": result["confidence_map"],
                "target_map": result["target_map"],
            })

            audio = _mel_to_audio(result["dreamed_mel"])
            sf.write(chart_dir / f"cond_dm{dm:.0f}_dreamed.wav", audio, 22000)

            _save_mel_analysis(
                result["dreamed_mel"], real_mel_np,
                result["target_map"], result["confidence_map"],
                chart_dir, slug=f"cond_dm{dm:.0f}",
            )

        _save_cond_sweep(
            sweep_results, chart_dir / "cond_sweep.png",
            past_event_offsets=past_offsets,
        )

        np.savez(
            chart_dir / "cond_sweep_data.npz",
            density_means=np.array(sweep_values),
            **{f"mel_dm{dm:.0f}": r["dreamed_mel"] for dm, r in
               zip(sweep_values, sweep_results)},
            **{f"conf_dm{dm:.0f}": r["confidence_map"] for dm, r in
               zip(sweep_values, sweep_results)},
        )

    return {"chart_id": sample.chart_id, "n_onsets": n_onsets,
            "density_mean": base_density_mean, "mode": args.mode}


# ─────────────────────────── entry point ─────────────────────────────

def _run_experiment(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset_vals: dict = dict(PRESETS.get(args.preset, {}))
    preset_vals["seed"] = args.seed
    if args.iterations is not None:
        preset_vals["iterations"] = args.iterations
    if args.lr is not None:
        preset_vals["lr"] = args.lr
    dcfg = DreamConfig(**preset_vals)
    print(f"Preset: {args.preset} -> iterations={dcfg.iterations}, "
          f"lr={dcfg.lr}, tv={dcfg.lambda_tv}, l2={dcfg.lambda_l2}, "
          f"realism={dcfg.lambda_realism}, lbfgs={dcfg.use_lbfgs}, "
          f"realistic_init={dcfg.realistic_init}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _, meta = load_model_from_checkpoint(
        Path(args.checkpoint), device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    c_events = meta.model_config.c_events

    # ── No-dataset modes (single/metro without real samples) ─────────

    if not args.dataset:
        if args.target in ("gt", "model"):
            print(f"ERROR: --target {args.target} requires --dataset",
                  file=sys.stderr)
            return 1
        if args.target == "single":
            target_np = _target_single(args.target_bin)
        elif args.target.startswith("metro"):
            target_np = _target_metronome(args.metro_gap)
        else:
            print(f"ERROR: unknown target {args.target!r}", file=sys.stderr)
            return 1

        target_t = torch.from_numpy(target_np).unsqueeze(0).to(device)
        ev_off, ev_mask = _make_empty_events(c_events, device)
        cond = _make_conditioning(3.0, 8.0, 2.0, device)

        print(f"Target: {args.target}, {int((target_np > 0.5).sum())} onsets")
        result = _run_dream(
            model, target_t, ev_off, ev_mask, cond, dcfg, device,
        )
        _save_mel_comparison(
            None, result["dreamed_mel"],
            result["target_map"], result["confidence_map"],
            out_dir / "dream_mel.png", title=f"{args.target}",
        )
        _save_trajectory(result["trajectory"], out_dir / "dream_trajectory.png")
        audio = _mel_to_audio(result["dreamed_mel"])
        sf.write(out_dir / "dream.wav", audio, 22000)
        np.savez(
            out_dir / "dream_data.npz",
            dreamed_mel=result["dreamed_mel"],
            confidence_map=result["confidence_map"],
            target_map=result["target_map"],
            **{f"traj_{k}": v for k, v in result["trajectory"].items()},
        )

        if args.cond_sweep:
            sweep_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
            sweep_results = []
            for dm in tqdm(sweep_values, desc="Cond sweep"):
                cond = _make_conditioning(dm, 8.0, 2.0, device)
                result = _run_dream(
                    model, target_t, ev_off, ev_mask, cond, dcfg, device,
                )
                sweep_results.append({
                    "density_mean": dm,
                    "dreamed_mel": result["dreamed_mel"],
                    "confidence_map": result["confidence_map"],
                    "target_map": result["target_map"],
                })
                audio = _mel_to_audio(result["dreamed_mel"])
                sf.write(out_dir / f"cond_dm{dm:.0f}.wav", audio, 22000)
            _save_cond_sweep(sweep_results, out_dir / "cond_sweep.png")

        print(f"All outputs saved to {out_dir}")
        return 0

    # ── Dataset modes (multi-chart) ──────────────────────────────────

    print(f"Loading dataset {args.dataset}, split={args.split}")
    sampler = _load_sampler(args.dataset, args.split)

    sample_indices = _pick_diverse_samples(sampler, args.n_charts)
    print(f"Selected {len(sample_indices)} charts across density range")

    summaries: list[dict] = []
    for i, si in enumerate(tqdm(sample_indices, desc="Charts")):
        sample = sampler.raw_sample(si)
        safe_id = sample.chart_id.replace("/", "_").replace(" ", "_")[:60]
        chart_dir = out_dir / f"chart_{i:02d}_{safe_id}"

        tqdm.write(
            f"  [{i+1}/{len(sample_indices)}] {sample.chart_id} "
            f"(density={sample.density_mean:.2f})"
        )

        summary = _run_one_chart(
            model, sample, chart_dir, args, dcfg, c_events, device,
        )
        summaries.append(summary)

    # Save manifest of all charts processed.
    manifest = {
        "mode": args.mode,
        "target": args.target,
        "n_charts": len(summaries),
        "iterations": dcfg.iterations,
        "charts": summaries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )

    print(f"All outputs saved to {out_dir}")
    print(f"  {len(summaries)} charts processed")
    for s in summaries:
        print(f"    {s['chart_id']}: density={s['density_mean']:.2f}, "
              f"onsets={s['n_onsets']}")
    return 0


# ─────────────────────────── CLI ─────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Activation maximization for FramewiseDetector.",
    )
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--out-dir", required=True, type=str)

    p.add_argument("--dataset", default=None, type=str,
                   help="Dataset name for loading real samples.")
    p.add_argument("--n-charts", default=5, type=int,
                   help="Number of charts to dream on, spread across "
                        "the density range.")
    p.add_argument("--split", default="val", type=str)

    p.add_argument("--mode", default="dream",
                   choices=["dream", "counterfactual", "saliency"])
    p.add_argument("--target", default="gt",
                   help="Target map: gt, single, metro, model")
    p.add_argument("--target-bin", default=250, type=int,
                   help="Bin index for --target single.")
    p.add_argument("--metro-gap", default=40, type=int,
                   help="Gap in bins for --target metro.")
    p.add_argument("--model-threshold", default=0.4, type=float,
                   help="Threshold for --target model.")

    p.add_argument("--iterations", default=None, type=int,
                   help="Override iterations (default from preset).")
    p.add_argument("--lr", default=None, type=float,
                   help="Override learning rate (default from preset).")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--preset", default="default",
                   choices=list(PRESETS.keys()),
                   help="Optimization preset: 'default' (Adam, high reg) "
                        "or 'legible' (L-BFGS, low reg, realistic init).")

    p.add_argument("--cond-sweep", action="store_true",
                   help="Run conditioning density sweep.")
    p.add_argument("--perturbation-budget", default=50.0, type=float,
                   help="L2 budget for counterfactual mode.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return _run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
