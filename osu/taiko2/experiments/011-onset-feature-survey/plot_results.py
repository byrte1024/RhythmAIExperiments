"""Generate the experiment graphs from a completed survey's summary.json.

Reads ``results/summary.json`` and writes one PNG per planned figure
into ``graphs/``. Intended to be run after the survey CLI completes;
re-runs cheaply.

Usage::

    osu/taiko2/.venv/Scripts/python.exe \
        osu/taiko2/experiments/011-onset-feature-survey/plot_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE / "results" / "summary.json"
DEFAULT_GRAPHS = HERE / "graphs"

# Stable order + colors so graphs stay comparable run-to-run.
ALGO_ORDER = (
    "energy",
    "spectral_flux",
    "log_filtered_flux",
    "hfc_mel",
    "superflux",
    "subband_sf_4",
    "subband_sf_8",
    "complex_domain",
)
ALGO_COLORS = {
    "energy":            "#888888",
    "spectral_flux":     "#1f77b4",
    "log_filtered_flux": "#17becf",
    "hfc_mel":           "#ff7f0e",
    "superflux":         "#d62728",
    "subband_sf_4":      "#2ca02c",
    "subband_sf_8":      "#9467bd",
    "complex_domain":    "#e377c2",
}


def _algos(summary: dict) -> list[str]:
    available = list(summary["by_algo"].keys())
    return [a for a in ALGO_ORDER if a in available] + [
        a for a in available if a not in ALGO_ORDER
    ]


def _color(algo: str) -> str:
    return ALGO_COLORS.get(algo, "#000000")


def plot_pr_curves(summary: dict, out_path: Path, *, tol: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    tol_key = str(tol)
    for algo in _algos(summary):
        data = summary["by_algo"][algo]["by_tolerance"][tol_key]
        ax.plot(
            data["recall"], data["precision"],
            "-", color=_color(algo), linewidth=1.5, label=algo,
        )
        # Mark best-F1 point.
        best_idx = int(np.argmax(data["f1"]))
        ax.plot(
            data["recall"][best_idx], data["precision"][best_idx],
            "o", color=_color(algo), markersize=7, markeredgecolor="white",
            markeredgewidth=1,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Precision-recall @ ±{tol} frames (±{tol*5} ms)\n"
                 f"markers = best-F1 operating point per algorithm")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tolerance_sweep(summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)
    tolerances = summary["tolerances"]
    for algo in _algos(summary):
        f1s = [
            summary["by_algo"][algo]["by_tolerance"][str(t)]["best_f1"]
            for t in tolerances
        ]
        ax.plot(
            tolerances, f1s, "o-", color=_color(algo),
            linewidth=1.6, markersize=5, label=algo,
        )
    ax.set_xlabel("tolerance (frames; 1 frame = 5 ms)")
    ax.set_ylabel("best F1 (pooled)")
    ax.set_title("Best F1 vs tolerance per algorithm")
    ax.grid(alpha=0.3)
    ax.set_xticks(tolerances)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_recall_at_high_threshold(
    summary: dict, out_path: Path, *, target_recall: float = 0.95,
    tol: int = 10,
) -> None:
    """For each algorithm, find the threshold whose recall first
    reaches ``target_recall`` at the given tolerance; plot the
    recall + precision at that threshold.

    Calibration view for the downstream channel design — what
    precision do we pay for guaranteed-high recall?
    """
    algos = _algos(summary)
    tol_key = str(tol)

    chosen_recall: list[float] = []
    chosen_precision: list[float] = []
    chosen_threshold: list[float] = []
    for algo in algos:
        data = summary["by_algo"][algo]["by_tolerance"][tol_key]
        thr = summary["by_algo"][algo]["thresholds"]
        # Smallest threshold (= highest recall) where recall >= target.
        # Thresholds are ascending, so iterate from the start.
        chosen_idx = 0
        for i, r in enumerate(data["recall"]):
            if r >= target_recall:
                chosen_idx = i
                break
        else:
            chosen_idx = int(np.argmax(data["recall"]))
        chosen_recall.append(data["recall"][chosen_idx])
        chosen_precision.append(data["precision"][chosen_idx])
        chosen_threshold.append(thr[chosen_idx])

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    x = np.arange(len(algos))
    w = 0.4
    bars1 = ax.bar(x - w/2, chosen_recall, w, color="#2ca02c",
                   label="recall", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w/2, chosen_precision, w, color="#d62728",
                   label="precision", edgecolor="black", linewidth=0.5)
    for i, t in enumerate(chosen_threshold):
        ax.text(x[i], 1.01, f"thr={t:.2f}",
                ha="center", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.axhline(target_recall, color="#666", linestyle="--", linewidth=1,
               label=f"target recall = {target_recall:.2f}")
    ax.set_ylabel("score")
    ax.set_title(f"Recall + precision at the threshold that hits "
                 f"recall ≥ {target_recall:.2f} (±{tol} frames)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_joint_coverage(
    summary: dict, out_path: Path, *, tol: int = 10, top_per_size: int = 3,
) -> None:
    """For each subset size 1..K, show the top-N subsets by recall."""
    coverage = summary.get("joint_coverage", {})
    if not coverage:
        return

    by_size: dict[int, list[tuple[str, dict]]] = {}
    for tag, by_tol in coverage.items():
        size = tag.count("+") + 1
        by_size.setdefault(size, []).append((tag, by_tol[str(tol)]))
    sizes = sorted(by_size.keys())

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120)

    bar_y: list[float] = []
    bar_recall: list[float] = []
    bar_precision: list[float] = []
    bar_labels: list[str] = []
    bar_groups: list[int] = []
    cur = 0.0

    for size in sizes:
        items = by_size[size]
        items.sort(key=lambda x: -x[1]["recall"])
        for tag, m in items[:top_per_size]:
            bar_y.append(cur)
            bar_recall.append(m["recall"])
            bar_precision.append(m["precision"])
            bar_labels.append(tag)
            bar_groups.append(size)
            cur += 1
        cur += 0.6  # gap between size groups

    h = 0.4
    ys = np.array(bar_y)
    ax.barh(ys - h/2, bar_recall, h, color="#2ca02c",
            label="recall", edgecolor="black", linewidth=0.4)
    ax.barh(ys + h/2, bar_precision, h, color="#d62728",
            label="precision", edgecolor="black", linewidth=0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"size {s}: {lbl}" for s, lbl in zip(bar_groups, bar_labels)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("score (pooled across val charts)")
    ax.set_title(f"Top {top_per_size} subsets by recall, per size, "
                 f"@ ±{tol} frames\nThresholds set per-algorithm to recall ≥ 0.95")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_algo_grid(summary: dict, out_path: Path) -> None:
    """One mini-panel per algorithm — recall, precision, F1 vs threshold
    at ±10 frames. Lets the eye see operating-point shape per algo.
    """
    algos = _algos(summary)
    n = len(algos)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.7 * rows),
                             dpi=120, sharex=True, sharey=True)
    axes_flat = axes.ravel() if n > 1 else [axes]

    for i, algo in enumerate(algos):
        ax = axes_flat[i]
        thr = summary["by_algo"][algo]["thresholds"]
        data = summary["by_algo"][algo]["by_tolerance"]["10"]
        ax.plot(thr, data["recall"], "-", color="#2ca02c", linewidth=1.4,
                label="recall")
        ax.plot(thr, data["precision"], "-", color="#d62728", linewidth=1.4,
                label="precision")
        ax.plot(thr, data["f1"], "-", color="#1f77b4", linewidth=1.4,
                label="F1")
        best_idx = int(np.argmax(data["f1"]))
        ax.axvline(thr[best_idx], color="#444", linestyle=":", linewidth=1)
        ax.text(0.98, 0.98,
                f"best F1={data['f1'][best_idx]:.3f}\n"
                f"@thr={thr[best_idx]:.2f}\n"
                f"P={data['precision'][best_idx]:.2f}  "
                f"R={data['recall'][best_idx]:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#aaa"))
        ax.set_title(algo, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    for j in range(len(algos), len(axes_flat)):
        axes_flat[j].axis("off")

    axes_flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("P / R / F1 vs threshold @ ±10 frames", fontsize=12, y=1.0)
    fig.supxlabel("threshold (post-99th-percentile normalization)", y=-0.01)
    fig.supylabel("score")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_algo_summary_bars(
    summary: dict, out_path: Path, *, tol: int = 10,
) -> None:
    """Best F1 / recall / precision per algorithm at the chosen tolerance,
    side-by-side bars — the at-a-glance ranking.
    """
    algos = _algos(summary)
    tol_key = str(tol)
    f1s = [summary["by_algo"][a]["by_tolerance"][tol_key]["best_f1"] for a in algos]
    recalls = [summary["by_algo"][a]["by_tolerance"][tol_key]["best_recall"] for a in algos]
    precisions = [summary["by_algo"][a]["by_tolerance"][tol_key]["best_precision"] for a in algos]

    x = np.arange(len(algos))
    w = 0.27

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)
    ax.bar(x - w, recalls, w, color="#2ca02c", label="recall",
           edgecolor="black", linewidth=0.4)
    ax.bar(x, precisions, w, color="#d62728", label="precision",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w, f1s, w, color="#1f77b4", label="F1",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (best-F1 operating point)")
    ax.set_title(f"Per-algorithm best-F1 ranking @ ±{tol} frames "
                 f"(±{tol*5} ms)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_GRAPHS)
    p.add_argument("--tol", type=int, default=10,
                   help="Canonical tolerance (frames) for headline plots.")
    args = p.parse_args(argv)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_algo_summary_bars(
        summary, args.out_dir / "01_algo_summary_bars.png", tol=args.tol,
    )
    plot_pr_curves(
        summary, args.out_dir / "02_pr_curves_tol10.png", tol=args.tol,
    )
    plot_tolerance_sweep(
        summary, args.out_dir / "03_tolerance_sweep.png",
    )
    plot_recall_at_high_threshold(
        summary, args.out_dir / "04_recall_at_high_threshold.png",
        target_recall=0.95, tol=args.tol,
    )
    plot_joint_coverage(
        summary, args.out_dir / "05_joint_coverage.png", tol=args.tol,
    )
    plot_per_algo_grid(
        summary, args.out_dir / "06_per_algo_grid.png",
    )

    print(f"[plots] wrote 6 figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
