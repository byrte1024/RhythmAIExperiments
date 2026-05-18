"""Regenerate diagnostic PNGs from saved NPZs for all eval dirs.

Usage::

    osu/taiko2/.venv/bin/python -m osu.taiko2.cli.regenerate_diagnostics \
        --run-dir osu/taiko2/runs/exp_017_framewise_bce

Walks ``eval_*`` dirs under the run dir, reads each NPZ, and
overwrites the corresponding PNG with the current rendering code.
Also processes ``eval_*/train_noaug/`` if present.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _regenerate_value_hist_target(d: Path, step: int) -> None:
    npz_path = d / "value_hist_target.npz"
    if not npz_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.load(npz_path)["values"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, yscale in zip(axes, ("linear", "log")):
        ax.hist(
            vals, bins=50, range=(0, 1),
            edgecolor="none", alpha=0.8, color="#e86850",
        )
        ax.set_xlabel("target value")
        ax.set_ylabel("count")
        ax.set_yscale(yscale)
        ax.set_title(f"Target distribution ({yscale}) - step {step:,}")
    fig.tight_layout()
    fig.savefig(d / "value_hist_target.png", dpi=120)
    plt.close(fig)


def _regenerate_value_hist_pred(d: Path, step: int) -> None:
    npz_path = d / "value_hist_pred.npz"
    if not npz_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = np.load(npz_path)["values"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, yscale in zip(axes, ("linear", "log")):
        ax.hist(
            vals, bins=101, range=(0, 1),
            edgecolor="none", alpha=0.8, color="#4a90d9",
        )
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("count")
        ax.set_yscale(yscale)
        ax.set_title(f"Prediction distribution ({yscale}) - step {step:,}")
    fig.tight_layout()
    fig.savefig(d / "value_hist_pred.png", dpi=120)
    plt.close(fig)


def _regenerate_reliability(d: Path, step: int) -> None:
    """Recompute calibration plot.

    Prefers ``reliability.npz`` with exact ``cal_fine_*`` counters
    (saved by the fixed artifact code). Falls back to
    ``confidence_by_outcome.npz`` reservoir data for evals generated
    before the fix — that path is approximate (reservoir caps distort
    class ratios) and the plot title says so.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rel_path = d / "reliability.npz"
    outcome_path = d / "confidence_by_outcome.npz"

    # Prefer exact fine-bin counters if available.
    if rel_path.exists():
        rel_data = np.load(rel_path)
        if "cal_fine_total" in rel_data:
            _render_reliability_from_fine(d, step, rel_data)
            return

    if not outcome_path.exists():
        return

    # Fallback: reservoir-sampled data (approximate).
    data = np.load(outcome_path)
    tp = data["tp"]
    fn = data["fn"]
    fp = data["fp"]
    tn = data["tn"]

    all_conf = np.concatenate([tp, fn, fp, tn])
    all_positive = np.concatenate([
        np.ones(len(tp)),
        np.ones(len(fn)),
        np.zeros(len(fp)),
        np.zeros(len(tn)),
    ])

    n_cal_bins = 10
    cal_conf = np.zeros(n_cal_bins)
    cal_pos_rate = np.zeros(n_cal_bins)
    cal_count = np.zeros(n_cal_bins, dtype=np.int64)
    ece = 0.0
    total = len(all_conf)

    if total > 0:
        for b in range(n_cal_bins):
            lo = b / n_cal_bins
            hi = (b + 1) / n_cal_bins
            mask = (all_conf >= lo) & (all_conf < hi)
            n_in = int(mask.sum())
            cal_count[b] = n_in
            if n_in > 0:
                cal_conf[b] = float(all_conf[mask].mean())
                cal_pos_rate[b] = float(all_positive[mask].mean())
                ece += (n_in / total) * abs(cal_conf[b] - cal_pos_rate[b])

    brier = float(((all_conf - all_positive) ** 2).mean()) if total > 0 else 0.0

    np.savez(
        d / "reliability.npz",
        cal_conf=cal_conf, cal_pos_rate=cal_pos_rate, cal_count=cal_count,
        ece=np.float64(ece), brier=np.float64(brier),
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect calibration")
    valid = cal_count > 0
    if valid.any():
        ax.plot(
            cal_conf[valid], cal_pos_rate[valid], "o-",
            color="#d62728", label="model",
        )
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical positive rate (GT=1)")
    ax.set_title(
        f"Calibration - step {step:,}\n"
        f"ECE={ece:.4f}  Brier={brier:.4f}"
    )
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(d / "reliability.png", dpi=120)
    plt.close(fig)


def _regenerate_per_bin_rate(d: Path, step: int) -> None:
    npz_path = d / "per_bin_rate.npz"
    if not npz_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    pos_rate = data["pos_rate"]
    recall = data["recall"]
    fpr = data["fpr"]

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(pos_rate))
    ax.plot(x, pos_rate, label="P(target=1)", alpha=0.7)
    ax.plot(x, recall, label="recall@bin", alpha=0.7)
    ax.plot(x, fpr, label="FPR@bin", alpha=0.7)
    ax.set_xlabel("bin")
    ax.set_ylabel("rate")
    ax.set_title(f"Per-bin rates - step {step:,}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "per_bin_rate.png", dpi=120)
    plt.close(fig)


def _regenerate_confidence_by_outcome(d: Path, step: int) -> None:
    npz_path = d / "confidence_by_outcome.npz"
    if not npz_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    tp = data["tp"]
    fn = data["fn"]
    fp = data["fp"]
    tn = data["tn"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 51)
    for vals, label, color in [
        (tp, f"TP ({len(tp):,})", "#2ca02c"),
        (fn, f"FN ({len(fn):,})", "#d62728"),
        (fp, f"FP ({len(fp):,})", "#ff7f0e"),
        (tn, f"TN ({len(tn):,})", "#1f77b4"),
    ]:
        if len(vals):
            ax.hist(
                vals, bins=bins, alpha=0.5, label=label,
                color=color, edgecolor="none", density=True,
            )
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title(f"Confidence by outcome - step {step:,}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "confidence_by_outcome.png", dpi=120)
    plt.close(fig)


def _regenerate_value_hist_combined(d: Path, step: int) -> None:
    """Side-by-side target + prediction histograms for direct comparison."""
    tgt_path = d / "value_hist_target.npz"
    pred_path = d / "value_hist_pred.npz"
    if not tgt_path.exists() or not pred_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tgt_vals = np.load(tgt_path)["values"]
    pred_vals = np.load(pred_path)["values"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col, (vals, label, color) in enumerate([
        (tgt_vals, "Target", "#e86850"),
        (pred_vals, "Prediction", "#4a90d9"),
    ]):
        n_hist_bins = 101 if col == 1 else 50
        for row, yscale in enumerate(["linear", "log"]):
            ax = axes[row, col]
            ax.hist(
                vals, bins=n_hist_bins, range=(0, 1),
                edgecolor="none", alpha=0.8, color=color,
            )
            ax.set_xlabel("value")
            ax.set_ylabel("count")
            ax.set_yscale(yscale)
            ax.set_title(f"{label} ({yscale}) - step {step:,}")
    fig.tight_layout()
    fig.savefig(d / "value_hist_combined.png", dpi=120)
    plt.close(fig)


def _regenerate_calibration(d: Path, step: int) -> None:
    """Regenerate calibration plot from calibration.npz."""
    npz_path = d / "calibration.npz"
    if not npz_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    mean_conf = data["mean_conf"]
    pos_rate = data["pos_rate"]
    count = data["count"]
    populated = data["populated"]
    ece = float(data["ece"])
    n = len(mean_conf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect")
    if populated.any():
        ax.plot(
            mean_conf[populated], pos_rate[populated], "o-",
            color="#d62728", label="model", markersize=6,
        )
        for b in range(n):
            if populated[b]:
                ax.annotate(
                    f"{count[b]:,}",
                    (mean_conf[b], pos_rate[b]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="#666666",
                )
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical positive rate (GT=1)")
    ax.set_title(f"Calibration - step {step:,}  ECE={ece:.4f}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax2 = axes[1]
    bucket_edges = np.linspace(0, 1, n + 1)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
    ax2.bar(
        bucket_centers, count,
        width=1.0 / n * 0.85, color="#4a90d9", alpha=0.8,
    )
    ax2.set_xlabel("confidence bucket")
    ax2.set_ylabel("count")
    ax2.set_yscale("log")
    ax2.set_title(f"Bucket population - step {step:,}")
    ax2.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(d / "calibration.png", dpi=120)
    plt.close(fig)


def _delete_reliability(d: Path) -> None:
    rel_png = d / "reliability.png"
    if rel_png.exists():
        rel_png.unlink()


def _process_dir(d: Path, step: int) -> int:
    _delete_reliability(d)
    count = 0
    for fn in (
        _regenerate_value_hist_target,
        _regenerate_value_hist_pred,
        _regenerate_value_hist_combined,
        _regenerate_per_bin_rate,
        _regenerate_confidence_by_outcome,
        _regenerate_calibration,
    ):
        try:
            fn(d, step)
            count += 1
        except Exception as exc:
            print(f"  WARN: {fn.__name__} failed in {d}: {exc}")
    return count


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Regenerate diagnostic PNGs from saved NPZs.",
    )
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="Run directory (e.g. osu/taiko2/runs/exp_017_framewise_bce).",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} not found", file=sys.stderr)
        return 2

    eval_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("eval_")
    )
    if not eval_dirs:
        print(f"No eval_* dirs found in {run_dir}", file=sys.stderr)
        return 1

    total = 0
    for ed in eval_dirs:
        step = int(ed.name.removeprefix("eval_"))
        n = _process_dir(ed, step)
        print(f"  {ed.name}: regenerated {n} plots")
        total += n

        noaug = ed / "train_noaug"
        if noaug.is_dir():
            n2 = _process_dir(noaug, step)
            print(f"  {ed.name}/train_noaug: regenerated {n2} plots")
            total += n2

    print(f"Done. {total} plots regenerated across {len(eval_dirs)} eval(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
