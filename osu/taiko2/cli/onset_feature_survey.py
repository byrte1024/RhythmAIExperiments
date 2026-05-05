"""Survey classical onset detection algorithms against taiko GT.

Runs eight onset detection functions on every chart in a split,
peak-picks at a sweep of thresholds, and evaluates against the GT
event bins at multiple frame tolerances. Outputs per-chart and
aggregate metrics so we can rank algorithms by recall (the metric
that matters when we want to feed the activation as a free "look
here" channel to the downstream model).

Mel-domain ODFs (energy, spectral flux, log-filtered flux, HFC,
SuperFlux, sub-band flux × 4 / × 8) read the cached log-mel
features the dataset already stores — fast, no audio decode.

The phase-needing ODF (Complex Domain) requires a fresh STFT, so
those charts decode the source `.osz` audio. The script can run
without `--charts-dir` (skipping CD) for a fast mel-only pass.

Usage::

    python -m osu.taiko2.cli.onset_feature_survey \\
        --dataset taiko2_v1 \\
        --output osu/taiko2/experiments/011-onset-feature-survey/results \\
        --split val --device cuda \\
        --charts-dir path/to/osz/packs   # optional, enables CD
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..dataset import _safe_filename
from ..analysis.onset_features import (
    FrameEval,
    complex_domain,
    energy,
    evaluate_frames,
    hfc_mel,
    log_filtered_flux,
    mel_band_center_freqs_hz,
    normalize_activation,
    peak_pick,
    spectral_flux,
    subband_flux,
    superflux,
)
from ..persistence.events import load_event_bins
from ..persistence.manifest import load_manifest
from ..splits import chart_ids_for_split


# ─────────────────────────── Config ───────────────────────────────────


# Mel-domain algorithms read the cached log-mel features.
MEL_ALGOS = (
    "energy",
    "spectral_flux",
    "log_filtered_flux",
    "hfc_mel",
    "superflux",
    "subband_sf_4",
    "subband_sf_8",
)
# STFT-domain algorithms need a fresh STFT (audio decode).
STFT_ALGOS = ("complex_domain",)
# Sub-band flux is reported both as the per-band channels (each row)
# and as the union peak-set across bands.
SUBBAND_ALGOS = {"subband_sf_4": 4, "subband_sf_8": 8}


# ─────────────────────────── Args ─────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Survey classical onset detection functions.",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name under --datasets-dir or absolute path.")
    p.add_argument("--datasets-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "datasets",
                   help="Root directory containing built datasets.")
    p.add_argument("--charts-dir", type=Path, default=None,
                   help="Directory of .osz packs. Required for STFT-domain "
                        "algorithms (Complex Domain). Mel-domain algorithms "
                        "run without it.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory for per-algo CSVs and aggregate "
                        "JSON. Created if missing.")
    p.add_argument("--split", default="val",
                   help="Split name (default 'val').")
    p.add_argument("--split-ratios", default="train:0.9,val:0.1")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Torch device for ODF compute (default cuda if "
                        "available, else cpu).")
    p.add_argument("--max-charts", type=int, default=None,
                   help="Process at most N charts (debug / smoke).")
    p.add_argument("--n-thresholds", type=int, default=51,
                   help="Number of thresholds per algorithm in the sweep.")
    p.add_argument("--threshold-min", type=float, default=0.0,
                   help="Lowest threshold (post-normalization).")
    p.add_argument("--threshold-max", type=float, default=2.0,
                   help="Highest threshold (post-normalization).")
    p.add_argument("--tolerances", default="0,1,2,5,10,20",
                   help="Comma-separated frame tolerances to evaluate.")
    p.add_argument("--min-peak-distance", type=int, default=1,
                   help="Minimum frames between accepted peaks "
                        "(NMS during peak picking).")
    p.add_argument("--norm-percentile", type=float, default=99.0,
                   help="Per-chart percentile used to scale activations "
                        "before threshold sweep.")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_split_ratios(raw: str) -> tuple[tuple[str, float], ...]:
    parts: list[tuple[str, float]] = []
    for frag in raw.split(","):
        name, _, ratio = frag.strip().partition(":")
        if not name or not ratio:
            raise ValueError(f"bad split-ratios fragment {frag!r}")
        parts.append((name.strip(), float(ratio)))
    return tuple(parts)


def _resolve_dataset(name_or_path: str, datasets_dir: Path) -> Path:
    p = Path(name_or_path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (datasets_dir / name_or_path).resolve()


def _parse_tolerances(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(x.strip()) for x in raw.split(",") if x.strip()}))


# ─────────────────────────── .osz lookup ──────────────────────────────


def _build_osz_index(
    charts_dir: Path,
    *,
    progress: bool = True,
) -> dict[tuple[str, str], Path]:
    """Walk ``charts_dir`` and map ``(pack.basename, audio_filename)`` ->
    ``osz_path`` for every pack found.

    Loading every pack just to read its `audio_files` list is the
    cheapest reliable way to recover the path mapping post-build. The
    parsing is fast (zipfile metadata only).
    """
    from ..parsing.osz import load_pack

    osz_paths = sorted(charts_dir.rglob("*.osz"))
    if progress:
        try:
            from tqdm import tqdm
            osz_paths = list(tqdm(
                osz_paths, desc="Indexing .osz packs", unit="pack",
            ))
        except ImportError:
            pass

    index: dict[tuple[str, str], Path] = {}
    for osz_path in osz_paths:
        pack = load_pack(osz_path)
        if pack is None:
            continue
        for audio in pack.audio_files:
            index[(pack.basename, audio.filename)] = osz_path
    return index


# ─────────────────────────── ODF compute ──────────────────────────────


def _compute_mel_activations(
    log_mel: torch.Tensor, freqs_hz: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return a dict of name -> activation tensor on the same device.

    Multi-channel sub-band activations are reduced to a single envelope
    by summing across bands here so they can be peak-picked uniformly.
    The full per-band tensor is also returned for joint-coverage
    analysis under the keys ``subband_sf_4_bands`` / ``subband_sf_8_bands``.
    """
    out: dict[str, torch.Tensor] = {}
    out["energy"] = energy(log_mel)
    out["spectral_flux"] = spectral_flux(log_mel)
    out["log_filtered_flux"] = log_filtered_flux(log_mel)
    out["hfc_mel"] = hfc_mel(log_mel, freqs_hz)
    out["superflux"] = superflux(log_mel)
    for name, n in SUBBAND_ALGOS.items():
        bands = subband_flux(log_mel, n_bands=n)            # (n, T)
        out[f"{name}_bands"] = bands
        out[name] = bands.sum(dim=0)                        # collapse for sweep
    return out


@dataclass(frozen=True, slots=True)
class StftConfig:
    sample_rate: int
    n_fft: int
    hop_length: int


def _compute_stft_activations(
    waveform: torch.Tensor, *, stft_cfg: StftConfig,
) -> dict[str, torch.Tensor]:
    """Compute STFT-domain ODFs from a 1-D waveform on the input device."""
    window = torch.hann_window(stft_cfg.n_fft, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=stft_cfg.n_fft,
        hop_length=stft_cfg.hop_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="constant",
    )                                                       # (F, T)
    mag = stft.abs()
    phase = torch.angle(stft)
    return {"complex_domain": complex_domain(mag, phase)}


# ─────────────────────────── Per-chart pipeline ───────────────────────


def _process_chart(
    *,
    chart_id: str,
    log_mel: torch.Tensor,
    gt_frames: torch.Tensor,
    waveform: torch.Tensor | None,
    freqs_hz: torch.Tensor,
    thresholds: torch.Tensor,
    tolerances: tuple[int, ...],
    min_peak_distance: int,
    norm_percentile: float,
    stft_cfg: StftConfig | None,
) -> tuple[
    dict[str, dict[int, list[FrameEval]]],   # algo -> tolerance -> [FrameEval per threshold]
    dict[str, torch.Tensor],                 # algo -> predicted-frames @ best threshold (for joint coverage)
    int,                                     # n_gt
]:
    n_gt = int(gt_frames.numel())
    activations: dict[str, torch.Tensor] = _compute_mel_activations(log_mel, freqs_hz)
    if waveform is not None and stft_cfg is not None:
        activations.update(_compute_stft_activations(waveform, stft_cfg=stft_cfg))

    out_evals: dict[str, dict[int, list[FrameEval]]] = {}
    # Predicted peaks at the recall-≥-0.95 threshold per algo, for the
    # multi-channel coverage analysis. Picked at tolerance == max in the
    # sweep so we use the most permissive matching.
    out_peaks_high_recall: dict[str, torch.Tensor] = {}
    max_tol = max(tolerances)

    for name, act in activations.items():
        if name.endswith("_bands"):
            continue                        # multi-channel — handled separately
        act_n = normalize_activation(act, percentile=norm_percentile)

        per_tol: dict[int, list[FrameEval]] = {tol: [] for tol in tolerances}
        thresh_peaks: list[torch.Tensor] = []
        for thr in thresholds.tolist():
            peaks = peak_pick(
                act_n, threshold=float(thr), min_distance=min_peak_distance,
            )
            thresh_peaks.append(peaks)
            for tol in tolerances:
                per_tol[tol].append(
                    evaluate_frames(peaks, gt_frames, tolerance=tol),
                )
        out_evals[name] = per_tol

        # High-recall threshold: smallest threshold whose recall at
        # max-tolerance is >= 0.95, fall back to lowest if none cross.
        chosen = 0
        for i, ev in enumerate(per_tol[max_tol]):
            if ev.recall >= 0.95:
                chosen = i
                break
        out_peaks_high_recall[name] = thresh_peaks[chosen]

    return out_evals, out_peaks_high_recall, n_gt


# ─────────────────────────── Aggregation ──────────────────────────────


def _aggregate(
    accumulator: dict[str, dict[int, list[list[FrameEval]]]],
    thresholds: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    """Roll per-chart per-(threshold, tolerance) evals into pooled
    precision / recall / F1.

    Pooled means we sum TP, FP, FN across charts before computing
    P/R/F1 — gives weight by per-chart event count, which is what we
    want (a 1000-onset chart counts more than a 50-onset chart).
    """
    summary: dict[str, dict[str, Any]] = {}
    n_thresh = len(thresholds)
    thr_list = thresholds.tolist()
    for algo, per_tol in accumulator.items():
        algo_out: dict[str, Any] = {"thresholds": thr_list, "by_tolerance": {}}
        for tol, per_chart in per_tol.items():
            tps = [0] * n_thresh
            fps = [0] * n_thresh
            fns = [0] * n_thresh
            for chart_evals in per_chart:
                for i, ev in enumerate(chart_evals):
                    tps[i] += ev.tp
                    fps[i] += ev.fp
                    fns[i] += ev.fn
            precisions: list[float] = []
            recalls: list[float] = []
            f1s: list[float] = []
            for tp, fp, fn in zip(tps, fps, fns):
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
            best_idx = int(np.argmax(f1s)) if f1s else 0
            algo_out["by_tolerance"][str(tol)] = {
                "precision": precisions,
                "recall": recalls,
                "f1": f1s,
                "best_f1": f1s[best_idx] if f1s else 0.0,
                "best_threshold": thr_list[best_idx] if thr_list else 0.0,
                "best_precision": precisions[best_idx] if precisions else 0.0,
                "best_recall": recalls[best_idx] if recalls else 0.0,
                "tp_sum": tps,
                "fp_sum": fps,
                "fn_sum": fns,
            }
        summary[algo] = algo_out
    return summary


def _joint_coverage(
    peaks_per_algo: dict[str, list[torch.Tensor]],
    gts: list[torch.Tensor],
    *,
    tolerances: tuple[int, ...],
) -> dict[str, dict[int, dict[str, float]]]:
    """For every subset of algorithms (size 1..k), what's the union recall
    across charts at each tolerance.

    Truncated to subsets of size <= 4 — past 4 the combinatorics blow up
    and the coverage saturates. Returns
    ``{frozenset(algos)-as-tuple: {tol: {recall, precision, f1}}}``.
    """
    from itertools import combinations

    algos = sorted(peaks_per_algo.keys())
    out: dict[str, dict[int, dict[str, float]]] = {}
    max_subset = min(4, len(algos))

    for size in range(1, max_subset + 1):
        for combo in combinations(algos, size):
            tag = "+".join(combo)
            out[tag] = {}
            for tol in tolerances:
                tp = fp = fn = 0
                for chart_idx, gt in enumerate(gts):
                    union = torch.cat(
                        [peaks_per_algo[a][chart_idx] for a in combo]
                    ).sort().values.unique()
                    ev = evaluate_frames(union, gt, tolerance=tol)
                    tp += ev.tp
                    fp += ev.fp
                    fn += ev.fn
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
                out[tag][tol] = {
                    "precision": p, "recall": r, "f1": f,
                    "tp": tp, "fp": fp, "fn": fn,
                }
    return out


# ─────────────────────────── Output writers ───────────────────────────


def _write_per_chart_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ─────────────────────────── Main ─────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ds_root = _resolve_dataset(args.dataset, args.datasets_dir)
    if not (ds_root / "manifest.json").exists():
        print(f"manifest.json not found under {ds_root}", file=sys.stderr)
        return 2

    manifest = load_manifest(ds_root / "manifest.json")
    sampler_cfg = manifest.sampler_config       # MelSamplerConfig
    sample_rate = int(sampler_cfg.sample_rate)
    n_fft = int(sampler_cfg.n_fft)
    hop_length = int(sampler_cfg.effective_hop_length)
    n_mels = int(sampler_cfg.n_mels)
    f_min = float(sampler_cfg.f_min)
    f_max = float(sampler_cfg.f_max) if sampler_cfg.f_max is not None else sample_rate / 2.0
    bin_ms = hop_length / sample_rate * 1000.0  # 5.000 ms with defaults

    print(f"[dataset] root={ds_root}  charts={len(manifest.charts)}  "
          f"sr={sample_rate}  hop={hop_length}  bin_ms={bin_ms:.4f}")

    split_ratios = _parse_split_ratios(args.split_ratios)
    allowed = chart_ids_for_split(
        manifest, args.split, split_ratios, args.split_seed,
    )
    entries = [e for e in manifest.charts if e.chart_id in allowed]
    if args.max_charts is not None:
        entries = entries[: args.max_charts]
    print(f"[split] {args.split!r} -> {len(entries)} charts")

    use_audio = args.charts_dir is not None
    osz_index: dict[tuple[str, str], Path] = {}
    if use_audio:
        if not args.charts_dir.exists():
            print(f"--charts-dir {args.charts_dir} does not exist", file=sys.stderr)
            return 2
        osz_index = _build_osz_index(args.charts_dir, progress=not args.no_progress)
        print(f"[audio] indexed {len(osz_index)} (pack, audio) entries")

    device = torch.device(args.device)
    freqs_hz = mel_band_center_freqs_hz(
        sample_rate, n_fft, n_mels,
        f_min=f_min, f_max=f_max, device=device,
    )
    thresholds = torch.linspace(
        args.threshold_min, args.threshold_max, args.n_thresholds,
    )
    tolerances = _parse_tolerances(args.tolerances)
    print(f"[sweep] thresholds={args.n_thresholds} ({args.threshold_min}..{args.threshold_max})  "
          f"tolerances={tolerances} frames")

    stft_cfg = StftConfig(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
    )

    # accumulator[algo][tolerance] = list[ list[FrameEval per threshold] per chart ]
    accumulator: dict[str, dict[int, list[list[FrameEval]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    peaks_high_recall: dict[str, list[torch.Tensor]] = defaultdict(list)
    gt_per_chart: list[torch.Tensor] = []
    per_chart_rows: list[dict[str, Any]] = []
    n_charts_processed = 0
    n_charts_skipped = 0
    t_start = time.time()

    iterator: Any = entries
    if not args.no_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(entries, desc="Charts", unit="chart")
        except ImportError:
            pass

    from ..parsing.osz import load_audio_waveform

    for entry in iterator:
        # Load GT event bins.
        events_path = ds_root / "events" / f"{_safe_filename(entry.chart_id)}.npz"
        if not events_path.exists():
            n_charts_skipped += 1
            continue
        try:
            bins = load_event_bins(events_path)
        except Exception:
            n_charts_skipped += 1
            continue
        gt_frames = torch.as_tensor(np.asarray(bins, dtype=np.int64), device=device)

        # Load cached log-mel features.
        feat_path = ds_root / entry.features_path
        if not feat_path.exists():
            n_charts_skipped += 1
            continue
        try:
            mel_np = np.load(feat_path).astype(np.float32, copy=False)
        except Exception:
            n_charts_skipped += 1
            continue
        log_mel = torch.from_numpy(mel_np).to(device)        # (n_mels, T)

        # Optionally load audio for STFT-domain ODFs.
        waveform: torch.Tensor | None = None
        if use_audio:
            # Recover pack basename from chart_id "<basename> [<diff>]".
            basename = entry.chart_id.rsplit(" [", 1)[0]
            osz_path = osz_index.get((basename, entry.audio_filename))
            if osz_path is not None:
                try:
                    wav, sr = load_audio_waveform(
                        osz_path, entry.audio_filename, target_sr=sample_rate,
                    )
                    if sr != sample_rate:
                        # load_audio_waveform should resample; sanity check.
                        n_charts_skipped += 1
                        continue
                    waveform = torch.from_numpy(wav).to(device)
                except Exception:
                    waveform = None  # CD will be skipped for this chart

        evals, peaks_hr, n_gt = _process_chart(
            chart_id=entry.chart_id,
            log_mel=log_mel,
            gt_frames=gt_frames,
            waveform=waveform,
            freqs_hz=freqs_hz,
            thresholds=thresholds,
            tolerances=tolerances,
            min_peak_distance=args.min_peak_distance,
            norm_percentile=args.norm_percentile,
            stft_cfg=stft_cfg,
        )

        for algo, by_tol in evals.items():
            for tol, evs in by_tol.items():
                accumulator[algo][tol].append(evs)
        for algo, peaks in peaks_hr.items():
            peaks_high_recall[algo].append(peaks)
        gt_per_chart.append(gt_frames.cpu())

        # Per-chart row uses best-F1 at the canonical 50ms tolerance
        # (or the largest available tolerance if 10 isn't in the set).
        canonical_tol = 10 if 10 in tolerances else max(tolerances)
        row: dict[str, Any] = {
            "chart_id": entry.chart_id,
            "n_gt": n_gt,
            "tolerance_frames": canonical_tol,
        }
        for algo in evals:
            evs = evals[algo][canonical_tol]
            f1s = [e.f1 for e in evs]
            best_i = int(np.argmax(f1s)) if f1s else 0
            row[f"{algo}_best_f1"] = round(f1s[best_i], 4) if f1s else 0.0
            row[f"{algo}_best_recall"] = round(evs[best_i].recall, 4) if f1s else 0.0
            row[f"{algo}_best_precision"] = round(evs[best_i].precision, 4) if f1s else 0.0
            row[f"{algo}_best_threshold"] = round(thresholds[best_i].item(), 4) if f1s else 0.0
        per_chart_rows.append(row)

        n_charts_processed += 1

    elapsed = time.time() - t_start
    print(f"[done] processed={n_charts_processed}  skipped={n_charts_skipped}  "
          f"elapsed={elapsed:.1f}s")

    # ── Aggregate ──
    summary = _aggregate(accumulator, thresholds)
    if peaks_high_recall:
        coverage = _joint_coverage(
            dict(peaks_high_recall), gt_per_chart, tolerances=tolerances,
        )
    else:
        coverage = {}

    # ── Write outputs ──
    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    # Per-chart CSV.
    if per_chart_rows:
        fieldnames = list(per_chart_rows[0].keys())
        _write_per_chart_csv(
            out_root / "per_chart.csv",
            per_chart_rows, fieldnames=fieldnames,
        )

    # Per-algo summary curves CSV.
    for algo, algo_summary in summary.items():
        algo_dir = out_root / "per_algo" / algo
        algo_dir.mkdir(parents=True, exist_ok=True)
        with (algo_dir / "curves.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "tolerance", "precision", "recall", "f1",
                        "tp", "fp", "fn"])
            for tol_str, data in algo_summary["by_tolerance"].items():
                for i, thr in enumerate(algo_summary["thresholds"]):
                    w.writerow([
                        round(thr, 4), tol_str,
                        round(data["precision"][i], 6),
                        round(data["recall"][i], 6),
                        round(data["f1"][i], 6),
                        data["tp_sum"][i], data["fp_sum"][i], data["fn_sum"][i],
                    ])

    # Aggregate JSON.
    summary_meta = {
        "dataset": str(ds_root),
        "split": args.split,
        "n_charts_processed": n_charts_processed,
        "n_charts_skipped": n_charts_skipped,
        "tolerances": list(tolerances),
        "thresholds": thresholds.tolist(),
        "device": str(device),
        "use_audio": use_audio,
        "elapsed_s": round(elapsed, 2),
        "by_algo": summary,
        "joint_coverage": {
            tag: {str(tol): vals for tol, vals in by_tol.items()}
            for tag, by_tol in coverage.items()
        },
    }
    _write_summary_json(out_root / "summary.json", summary_meta)
    print(f"[output] {out_root.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
