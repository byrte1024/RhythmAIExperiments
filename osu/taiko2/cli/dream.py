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
        --dataset taiko2_v1 --sample-idx 0 \
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
    iterations: int = 1000
    lr: float = 0.03
    lr_end: float = 0.001
    lambda_tv: float = 0.01
    lambda_l2: float = 0.001
    mel_min: float = -35.0
    mel_max: float = 55.0
    jitter_px: int = 2
    seed: int = 42


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
    else:
        rng = torch.Generator(device="cpu").manual_seed(config.seed)
        mel_param = torch.randn(
            1, n_mels, n_frames, generator=rng,
        ).to(device) * 5.0 + 15.0
        mel_param = mel_param.requires_grad_(True)
        mel_anchor = None

    optimizer = torch.optim.Adam([mel_param], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.iterations, eta_min=config.lr_end,
    )

    trajectory: dict[str, list[float]] = {
        "loss_total": [], "loss_bce": [], "loss_tv": [], "loss_l2": [],
        "conf_at_target_mean": [], "conf_at_nontarget_mean": [],
    }
    target_bins = target_map[0] > 0.5

    for step in range(config.iterations):
        optimizer.zero_grad()

        mel_input = mel_param
        if config.jitter_px > 0 and step % 2 == 0:
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

        if perturbation_budget is not None and mel_anchor is not None:
            delta = mel_param - mel_anchor
            loss_budget = 0.1 * F.relu(delta.norm() - perturbation_budget)
            loss = loss + loss_budget

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            mel_param.clamp_(config.mel_min, config.mel_max)

        with torch.no_grad():
            conf = torch.sigmoid(logits[0])
            trajectory["loss_total"].append(float(loss))
            trajectory["loss_bce"].append(float(loss_bce))
            trajectory["loss_tv"].append(float(loss_tv))
            trajectory["loss_l2"].append(float(loss_l2))
            trajectory["conf_at_target_mean"].append(
                float(conf[target_bins].mean()) if target_bins.any() else 0.0,
            )
            trajectory["conf_at_nontarget_mean"].append(
                float(conf[~target_bins].mean()) if (~target_bins).any() else 0.0,
            )

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

def _save_mel_comparison(
    real_mel: np.ndarray | None,
    dreamed_mel: np.ndarray,
    target_map: np.ndarray,
    confidence_map: np.ndarray,
    out_path: Path,
    title: str = "",
):
    """Side-by-side mel + activation map figure."""
    n_rows = 3 if real_mel is not None else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows), dpi=100)

    row = 0
    if real_mel is not None:
        axes[row].imshow(
            real_mel, aspect="auto", origin="lower", cmap="magma",
        )
        onset_bins = np.where(target_map > 0.5)[0]
        for b in onset_bins:
            x = b * (real_mel.shape[1] / len(target_map))
            axes[row].axvline(x, color="cyan", alpha=0.5, linewidth=0.5)
        axes[row].set_title("Real mel (GT onsets in cyan)")
        axes[row].set_ylabel("Mel band")
        row += 1

    axes[row].imshow(
        dreamed_mel, aspect="auto", origin="lower", cmap="magma",
    )
    onset_bins = np.where(target_map > 0.5)[0]
    for b in onset_bins:
        x = b * (dreamed_mel.shape[1] / len(target_map))
        axes[row].axvline(x, color="cyan", alpha=0.5, linewidth=0.5)
    axes[row].set_title("Dreamed mel (target onsets in cyan)")
    axes[row].set_ylabel("Mel band")
    row += 1

    axes[row].plot(target_map, label="Target", alpha=0.7, linewidth=1)
    axes[row].plot(confidence_map, label="Model output", alpha=0.7, linewidth=1)
    axes[row].set_xlim(0, len(target_map))
    axes[row].set_ylim(-0.05, 1.05)
    axes[row].set_xlabel("Bin")
    axes[row].set_ylabel("Activation")
    axes[row].legend()
    axes[row].set_title("Target vs dreamed confidence map")

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
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=100)

    axes[0].imshow(mel, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Input mel")
    axes[0].set_ylabel("Mel band")

    vmax = np.percentile(np.abs(saliency), 99)
    axes[1].imshow(
        saliency, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
    )
    axes[1].set_title("Saliency (d(conf_sum) / d(mel))")
    axes[1].set_ylabel("Mel band")

    axes[2].plot(confidence_map)
    axes[2].set_xlim(0, len(confidence_map))
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Bin")
    axes[2].set_ylabel("Confidence")
    axes[2].set_title("Model confidence map")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_cond_sweep(
    sweep_results: list[dict], out_path: Path,
):
    """Compare dreamed mels across conditioning values."""
    n = len(sweep_results)
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3 * (n + 1)), dpi=100)

    for i, res in enumerate(sweep_results):
        axes[i].imshow(
            res["dreamed_mel"], aspect="auto", origin="lower", cmap="magma",
        )
        axes[i].set_title(f"density_mean={res['density_mean']:.1f}")
        axes[i].set_ylabel("Mel band")

    ax_conf = axes[n]
    for res in sweep_results:
        ax_conf.plot(
            res["confidence_map"],
            label=f"density={res['density_mean']:.1f}",
            alpha=0.7,
        )
    ax_conf.plot(
        sweep_results[0]["target_map"],
        label="Target", color="black", linestyle="--", alpha=0.5,
    )
    ax_conf.set_xlim(0, len(sweep_results[0]["confidence_map"]))
    ax_conf.set_ylim(-0.05, 1.05)
    ax_conf.set_xlabel("Bin")
    ax_conf.set_ylabel("Confidence")
    ax_conf.legend(fontsize=8)
    ax_conf.set_title("Confidence maps across density conditioning")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_events_sweep(
    sweep_results: list[dict], out_path: Path,
):
    """Compare dreamed mels across event configurations."""
    n = len(sweep_results)
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3 * (n + 1)), dpi=100)

    for i, res in enumerate(sweep_results):
        axes[i].imshow(
            res["dreamed_mel"], aspect="auto", origin="lower", cmap="magma",
        )
        axes[i].set_title(f"events={res['event_mode']}")
        axes[i].set_ylabel("Mel band")

    ax_conf = axes[n]
    for res in sweep_results:
        ax_conf.plot(
            res["confidence_map"],
            label=f"events={res['event_mode']}",
            alpha=0.7,
        )
    ax_conf.plot(
        sweep_results[0]["target_map"],
        label="Target", color="black", linestyle="--", alpha=0.5,
    )
    ax_conf.set_xlim(0, len(sweep_results[0]["confidence_map"]))
    ax_conf.set_ylim(-0.05, 1.05)
    ax_conf.set_xlabel("Bin")
    ax_conf.set_ylabel("Confidence")
    ax_conf.legend(fontsize=8)
    ax_conf.set_title("Confidence maps across event configurations")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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


# ─────────────────────────── main logic ──────────────────────────────

def _load_sample(dataset: str, sample_idx: int, split: str = "val"):
    cfg = TaikoDetectionSamplerConfig(
        dataset_root=str(
            Path(__file__).resolve().parent.parent / "datasets" / dataset
        ),
        split=split,
        a_bins=500, b_bins=500, c_events=128, d_events=100,
        min_cursor_bin=6000,
    )
    sampler = TaikoDetectionSampler(cfg)
    sampler.load_data(progress=True)
    sample = sampler.raw_sample(sample_idx)
    return sample, sampler


def _run_experiment(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dcfg = DreamConfig(
        iterations=args.iterations,
        lr=args.lr,
        seed=args.seed,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model, _, meta = load_model_from_checkpoint(
        Path(args.checkpoint), device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    c_events = meta.model_config.c_events

    sample = None
    real_mel_np = None
    if args.dataset:
        print(f"Loading sample {args.sample_idx} from {args.dataset}")
        sample, sampler = _load_sample(
            args.dataset, args.sample_idx, args.split,
        )
        real_mel_np = np.concatenate(
            [sample.audio_past, sample.audio_future], axis=1,
        )

    # Build target map.
    if args.target == "gt":
        if sample is None:
            print("ERROR: --target gt requires --dataset", file=sys.stderr)
            return 1
        target_np = _target_from_sample(sample)
    elif args.target == "single":
        target_np = _target_single(args.target_bin)
    elif args.target.startswith("metro"):
        target_np = _target_metronome(args.metro_gap)
    elif args.target == "model":
        if sample is None:
            print("ERROR: --target model requires --dataset", file=sys.stderr)
            return 1
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
    else:
        print(f"ERROR: unknown target {args.target!r}", file=sys.stderr)
        return 1

    target_t = torch.from_numpy(target_np).unsqueeze(0).to(device)
    n_onsets = int((target_np > 0.5).sum())
    print(f"Target: {args.target}, {n_onsets} onsets")

    if args.mode == "saliency":
        if real_mel_np is None:
            print("ERROR: saliency requires --dataset", file=sys.stderr)
            return 1
        real_mel_t = torch.from_numpy(real_mel_np).unsqueeze(0).to(
            device=device, dtype=torch.float32,
        )
        ev_off, ev_mask = _make_real_events(sample, device)
        cond = _make_conditioning(
            sample.density_mean, sample.density_peak, sample.density_std,
            device,
        )
        result = _run_saliency(model, real_mel_t, ev_off, ev_mask, cond)
        _save_saliency(
            result["saliency"], result["mel"],
            result["confidence_map"], out_dir / "saliency.png",
        )
        np.savez(
            out_dir / "saliency.npz",
            saliency=result["saliency"],
            confidence_map=result["confidence_map"],
            mel=result["mel"],
        )
        print(f"Saved saliency to {out_dir}")
        return 0

    # ── Dream / Counterfactual ───────────────────────────────────────

    # Default conditioning from sample or fallback.
    if sample is not None:
        base_density_mean = sample.density_mean
        base_density_peak = float(sample.density_peak)
        base_density_std = sample.density_std
    else:
        base_density_mean = 3.0
        base_density_peak = 8.0
        base_density_std = 2.0

    init_mel = None
    perturbation_budget = None
    if args.mode == "counterfactual":
        if real_mel_np is None:
            print("ERROR: counterfactual requires --dataset", file=sys.stderr)
            return 1
        init_mel = torch.from_numpy(real_mel_np).unsqueeze(0).to(
            device=device, dtype=torch.float32,
        )
        perturbation_budget = args.perturbation_budget

    # ── Event sweep ──────────────────────────────────────────────────

    event_modes = ["empty", "real"] if sample is not None else ["empty"]
    event_sweep_results = []

    for event_mode in event_modes:
        if event_mode == "empty":
            ev_off, ev_mask = _make_empty_events(c_events, device)
        else:
            ev_off, ev_mask = _make_real_events(sample, device)

        cond = _make_conditioning(
            base_density_mean, base_density_peak, base_density_std, device,
        )

        print(f"Running dream: events={event_mode}, "
              f"density_mean={base_density_mean:.1f}")
        result = _run_dream(
            model, target_t, ev_off, ev_mask, cond, dcfg, device,
            init_mel=init_mel, perturbation_budget=perturbation_budget,
        )

        slug = f"{args.mode}_{event_mode}"
        _save_mel_comparison(
            real_mel_np, result["dreamed_mel"],
            result["target_map"], result["confidence_map"],
            out_dir / f"{slug}_mel.png",
            title=f"{args.mode} | events={event_mode} | "
                  f"density={base_density_mean:.1f}",
        )
        _save_trajectory(result["trajectory"], out_dir / f"{slug}_trajectory.png")

        audio = _mel_to_audio(result["dreamed_mel"])
        sf.write(out_dir / f"{slug}_dreamed.wav", audio, 22000)
        if real_mel_np is not None:
            audio_real = _mel_to_audio(real_mel_np)
            sf.write(out_dir / f"{slug}_real.wav", audio_real, 22000)

        np.savez(
            out_dir / f"{slug}_data.npz",
            dreamed_mel=result["dreamed_mel"],
            confidence_map=result["confidence_map"],
            target_map=result["target_map"],
            **{f"traj_{k}": v for k, v in result["trajectory"].items()},
        )

        event_sweep_results.append({
            "event_mode": event_mode,
            "dreamed_mel": result["dreamed_mel"],
            "confidence_map": result["confidence_map"],
            "target_map": result["target_map"],
        })

    if len(event_sweep_results) > 1:
        _save_events_sweep(event_sweep_results, out_dir / "events_sweep.png")
        print("Saved event sweep comparison")

    # ── Conditioning sweep ───────────────────────────────────────────

    if args.cond_sweep:
        sweep_values = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
        sweep_results = []

        ev_off, ev_mask = _make_empty_events(c_events, device)

        for dm in sweep_values:
            cond = _make_conditioning(dm, base_density_peak, base_density_std, device)
            print(f"Conditioning sweep: density_mean={dm:.1f}")
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
            sf.write(out_dir / f"cond_dm{dm:.0f}_dreamed.wav", audio, 22000)

        _save_cond_sweep(sweep_results, out_dir / "cond_sweep.png")

        np.savez(
            out_dir / "cond_sweep_data.npz",
            density_means=np.array(sweep_values),
            **{f"mel_dm{dm:.0f}": r["dreamed_mel"] for dm, r in
               zip(sweep_values, sweep_results)},
            **{f"conf_dm{dm:.0f}": r["confidence_map"] for dm, r in
               zip(sweep_values, sweep_results)},
        )
        print("Saved conditioning sweep")

    print(f"All outputs saved to {out_dir}")
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
    p.add_argument("--sample-idx", default=0, type=int,
                   help="Sample index within the dataset split.")
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

    p.add_argument("--iterations", default=1000, type=int)
    p.add_argument("--lr", default=0.03, type=float)
    p.add_argument("--seed", default=42, type=int)

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
