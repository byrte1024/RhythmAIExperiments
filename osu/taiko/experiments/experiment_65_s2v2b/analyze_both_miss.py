"""Analyze where both S1 and S2v2 miss onsets.

Are failures concentrated in specific charts or spread uniformly?

Usage:
    cd osu/taiko
    python experiments/experiment_65_s2v2b/analyze_both_miss.py \
        --s1-checkpoint runs/s1_experiment_65/checkpoints/best.pt \
        --s2-checkpoint runs/s2v2_experiment_65/checkpoints/best.pt
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, TAIKO_DIR)

from detection_s1_model import ConformerProposer
from detection_s1_train import S1Dataset, A_BINS, B_BINS, B_PRED, N_MELS
from detection_s2v2_model import ContextProposer
from detection_s2v2_train import S2v2Dataset, C_EVENTS


def load_s1(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ConformerProposer(n_mels=N_MELS, d_model=a.get("d_model", 384),
                              n_layers=a.get("n_layers", 8), n_heads=8,
                              conv_kernel=a.get("conv_kernel", 31),
                              a_bins=A_BINS, b_bins=B_BINS, b_pred=B_PRED)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def load_s2v2(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = ContextProposer(d_model=a.get("d_model", 256),
                            n_gru_layers=a.get("n_gru_layers", 4),
                            b_pred=B_PRED, max_events=C_EVENTS)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-checkpoint", required=True)
    parser.add_argument("--s2-checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    ds_dir = os.path.join(TAIKO_DIR, "datasets", "taiko_v2")
    with open(os.path.join(ds_dir, "manifest.json")) as f:
        manifest = json.load(f)

    charts = manifest["charts"]
    song_to_charts = {}
    for i, c in enumerate(charts):
        sid = c.get("beatmapset_id", str(i))
        song_to_charts.setdefault(sid, []).append(i)
    songs = list(song_to_charts.keys())
    random.seed(42)
    random.shuffle(songs)
    n_val = max(1, int(len(songs) * 0.1))
    val_songs = set(songs[:n_val])
    val_idx = [i for i, c in enumerate(charts) if c.get("beatmapset_id", str(i)) in val_songs]

    s1_model = load_s1(args.s1_checkpoint, args.device)
    s2_model = load_s2v2(args.s2_checkpoint, args.device)

    s1_ds = S1Dataset(manifest, ds_dir, val_idx, augment=False)
    s2_ds = S2v2Dataset(manifest, ds_dir, val_idx, augment=False)
    assert len(s1_ds) == len(s2_ds)

    s1_loader = DataLoader(s1_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)
    s2_loader = DataLoader(s2_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, pin_memory=True)

    print(f"Val samples: {len(s1_ds)}")
    print(f"Val charts: {len(val_idx)}")

    # Map sample index → chart index
    # S1Dataset stores (ci, ei, cursor) in self.samples
    sample_to_chart = []
    for ci, ei, cursor in s1_ds.samples:
        sample_to_chart.append(ci)  # ci is index into s1_ds.charts (which is val subset)
    sample_to_chart = np.array(sample_to_chart)

    # Per-chart stats
    n_val_charts = len(s1_ds.charts)
    chart_onset_bins = np.zeros(n_val_charts, dtype=np.int64)
    chart_both_miss = np.zeros(n_val_charts, dtype=np.int64)
    chart_s1_miss = np.zeros(n_val_charts, dtype=np.int64)
    chart_s2_miss = np.zeros(n_val_charts, dtype=np.int64)
    chart_sample_count = np.zeros(n_val_charts, dtype=np.int64)

    sample_idx = 0
    print("Running inference...")
    with torch.no_grad():
        for (mel, s1_tgt), (gaps, ratios, mask, cond, s2_tgt) in tqdm(
                zip(s1_loader, s2_loader), total=len(s1_loader)):
            mel = mel.to(args.device)
            gaps, ratios, mask, cond = gaps.to(args.device), ratios.to(args.device), mask.to(args.device), cond.to(args.device)

            s1_conf = torch.sigmoid(s1_model(mel)).cpu().numpy()
            s2_conf = torch.sigmoid(s2_model(gaps, ratios, mask, cond)).cpu().numpy()
            tgt = (s1_tgt.numpy() >= 0.5).astype(np.float32)

            B = mel.size(0)
            for b in range(B):
                if sample_idx >= len(sample_to_chart):
                    break
                ci = sample_to_chart[sample_idx]

                onset_mask = tgt[b] == 1
                n_onsets = onset_mask.sum()
                if n_onsets > 0:
                    s1_miss = (s1_conf[b][onset_mask] < 0.5)
                    s2_miss = (s2_conf[b][onset_mask] < 0.5)
                    both = s1_miss & s2_miss

                    chart_onset_bins[ci] += n_onsets
                    chart_both_miss[ci] += both.sum()
                    chart_s1_miss[ci] += s1_miss.sum()
                    chart_s2_miss[ci] += s2_miss.sum()

                chart_sample_count[ci] += 1
                sample_idx += 1

    # Compute per-chart both-miss rate
    chart_both_miss_rate = np.zeros(n_val_charts)
    chart_s1_miss_rate = np.zeros(n_val_charts)
    chart_s2_miss_rate = np.zeros(n_val_charts)
    for ci in range(n_val_charts):
        if chart_onset_bins[ci] > 0:
            chart_both_miss_rate[ci] = chart_both_miss[ci] / chart_onset_bins[ci]
            chart_s1_miss_rate[ci] = chart_s1_miss[ci] / chart_onset_bins[ci]
            chart_s2_miss_rate[ci] = chart_s2_miss[ci] / chart_onset_bins[ci]

    # Filter charts with enough data
    valid = chart_onset_bins >= 50
    valid_rates = chart_both_miss_rate[valid]

    print(f"\n{'='*70}")
    print(f"Per-Chart Both-Miss Rate Distribution ({valid.sum()} charts with 50+ onset bins)")
    print(f"{'='*70}")
    print(f"  Mean:   {valid_rates.mean():.1%}")
    print(f"  Median: {np.median(valid_rates):.1%}")
    print(f"  Std:    {valid_rates.std():.1%}")
    print(f"  Min:    {valid_rates.min():.1%}")
    print(f"  P25:    {np.percentile(valid_rates, 25):.1%}")
    print(f"  P75:    {np.percentile(valid_rates, 75):.1%}")
    print(f"  P90:    {np.percentile(valid_rates, 90):.1%}")
    print(f"  P95:    {np.percentile(valid_rates, 95):.1%}")
    print(f"  Max:    {valid_rates.max():.1%}")

    # Concentration: what % of both-miss bins come from worst N% of charts?
    sorted_idx = np.argsort(-chart_both_miss[valid])
    sorted_misses = chart_both_miss[valid][sorted_idx]
    total_misses = sorted_misses.sum()
    cumulative = np.cumsum(sorted_misses) / max(total_misses, 1)

    print(f"\n  Concentration:")
    for pct in [5, 10, 20, 50]:
        n = max(1, int(len(sorted_misses) * pct / 100))
        print(f"    Worst {pct}% of charts ({n} charts) contain {cumulative[n-1]:.1%} of all both-miss bins")

    # Top 20 worst charts
    print(f"\n{'='*70}")
    print(f"Top 20 Worst Charts (highest both-miss rate)")
    print(f"{'='*70}")
    print(f"{'Chart':>5} {'BothMiss%':>10} {'S1Miss%':>9} {'S2Miss%':>9} {'OnsetBins':>10} {'Samples':>8} {'Density':>8} {'Stars':>6} | Artist - Title [Diff]")
    print("-" * 130)

    worst_idx = np.argsort(-chart_both_miss_rate)
    shown = 0
    for ci in worst_idx:
        if chart_onset_bins[ci] < 50:
            continue
        chart = s1_ds.charts[ci]
        name = f"{chart.get('artist','?')[:20]} - {chart.get('title','?')[:25]} [{chart.get('difficulty','?')}]"
        density = chart.get("density_mean", 0)
        stars = chart.get("star_rating", 0)
        print(f"{ci:>5} {chart_both_miss_rate[ci]:>9.1%} {chart_s1_miss_rate[ci]:>8.1%} {chart_s2_miss_rate[ci]:>8.1%} "
              f"{chart_onset_bins[ci]:>10,} {chart_sample_count[ci]:>8,} {density:>7.1f} {stars:>5.1f} | {name}")
        shown += 1
        if shown >= 20:
            break

    # Bottom 20 best charts
    print(f"\nTop 20 Best Charts (lowest both-miss rate)")
    print("-" * 130)
    best_idx = np.argsort(chart_both_miss_rate)
    shown = 0
    for ci in best_idx:
        if chart_onset_bins[ci] < 50:
            continue
        chart = s1_ds.charts[ci]
        name = f"{chart.get('artist','?')[:20]} - {chart.get('title','?')[:25]} [{chart.get('difficulty','?')}]"
        density = chart.get("density_mean", 0)
        stars = chart.get("star_rating", 0)
        print(f"{ci:>5} {chart_both_miss_rate[ci]:>9.1%} {chart_s1_miss_rate[ci]:>8.1%} {chart_s2_miss_rate[ci]:>8.1%} "
              f"{chart_onset_bins[ci]:>10,} {chart_sample_count[ci]:>8,} {density:>7.1f} {stars:>5.1f} | {name}")
        shown += 1
        if shown >= 20:
            break

    # Correlation with chart properties
    print(f"\n{'='*70}")
    print("Correlation of both-miss rate with chart properties:")
    valid_charts = [s1_ds.charts[ci] for ci in range(n_val_charts) if valid[ci]]
    valid_densities = np.array([c.get("density_mean", 0) for c in valid_charts])
    valid_stars = np.array([c.get("star_rating", 0) for c in valid_charts])
    valid_durations = np.array([c.get("duration_s", 0) for c in valid_charts])
    valid_events = np.array([c.get("total_events", 0) for c in valid_charts])

    from scipy import stats as sp_stats
    for name, vals in [("density_mean", valid_densities), ("star_rating", valid_stars),
                       ("duration_s", valid_durations), ("total_events", valid_events)]:
        if len(vals) > 10 and vals.std() > 0:
            r, p = sp_stats.pearsonr(valid_rates, vals)
            print(f"  {name:<15}: r={r:+.3f}  p={p:.4f}")

    # Density bucket analysis
    print(f"\n  Both-miss rate by density bucket:")
    for lo, hi, label in [(0, 3, "sparse 0-3"), (3, 5, "medium 3-5"),
                           (5, 7, "dense 5-7"), (7, 20, "very dense 7+")]:
        bucket = (valid_densities >= lo) & (valid_densities < hi)
        if bucket.sum() > 0:
            print(f"    {label:<16}: {valid_rates[bucket].mean():.1%}  (n={bucket.sum()} charts)")

    # Save
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "n_val_charts": int(valid.sum()),
        "both_miss_rate_mean": float(valid_rates.mean()),
        "both_miss_rate_median": float(np.median(valid_rates)),
        "both_miss_rate_std": float(valid_rates.std()),
        "both_miss_rate_p90": float(np.percentile(valid_rates, 90)),
        "both_miss_rate_p95": float(np.percentile(valid_rates, 95)),
        "concentration_top10pct": float(cumulative[max(1, int(len(sorted_misses) * 0.1)) - 1]),
        "concentration_top20pct": float(cumulative[max(1, int(len(sorted_misses) * 0.2)) - 1]),
    }
    with open(os.path.join(output_dir, "both_miss_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Graph: distribution histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Histogram of per-chart both-miss rate
    ax = axes[0]
    ax.hist(valid_rates * 100, bins=50, color="#eb4528", alpha=0.8)
    ax.axvline(valid_rates.mean() * 100, color="black", linestyle="--", label=f"Mean={valid_rates.mean():.1%}")
    ax.axvline(np.median(valid_rates) * 100, color="blue", linestyle="--", label=f"Median={np.median(valid_rates):.1%}")
    ax.set_xlabel("Both-miss rate (%)")
    ax.set_ylabel("Number of charts")
    ax.set_title("Per-Chart Both-Miss Rate Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Concentration curve
    ax = axes[1]
    x = np.arange(1, len(cumulative) + 1) / len(cumulative) * 100
    ax.plot(x, cumulative * 100, "r-", linewidth=2)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Uniform")
    ax.set_xlabel("% of charts (sorted worst first)")
    ax.set_ylabel("% of total both-miss bins")
    ax.set_title("Both-Miss Concentration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Both-miss rate vs density
    ax = axes[2]
    ax.scatter(valid_densities, valid_rates * 100, alpha=0.3, s=10, color="#4a90d9")
    ax.set_xlabel("Chart density (events/sec)")
    ax.set_ylabel("Both-miss rate (%)")
    ax.set_title("Both-Miss Rate vs Chart Density")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "both_miss_distribution.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
