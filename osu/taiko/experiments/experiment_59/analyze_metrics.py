"""Experiment 59: AR Quality Metric Discovery.

Computes candidate metrics on all AR-generated charts from 42-AR and 53-AR,
then correlates each metric with human preference scores (per-song-model level).

Each vote (self or volunteer) is a separate data point: the model's score on
that song (4/3/2/1) paired with the computed metrics for that model×song chart.

Usage:
    cd osu/taiko
    python experiments/experiment_59/analyze_metrics.py
"""

import json
import math
import os
import sys

import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAIKO_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BIN_MS = 4.9887

# ═══════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════

def load_csv_events_ms(csv_path):
    """Load predicted events from CSV, return array of times in ms."""
    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("time_ms") or not line:
                continue
            parts = line.split(",")
            if parts:
                events.append(int(parts[0]))
    return np.array(events, dtype=np.float64)


def load_mel(mel_path):
    """Load mel spectrogram numpy file."""
    return np.load(mel_path)


def load_42ar_data():
    """Load 42-AR votes and chart paths."""
    ar_dir = os.path.join(TAIKO_DIR, "experiments", "experiment_42ar")
    with open(os.path.join(ar_dir, "results", "votes.json"), "r", encoding="utf-8") as f:
        votes_data = json.load(f)

    # Load mappings
    compiled_dir = os.path.join(ar_dir, "compiled")
    mappings = {}
    for fname in os.listdir(compiled_dir):
        if not fname.endswith("_mapping.txt"):
            continue
        song_stem = fname.replace("_mapping.txt", "")
        m = {}
        for line in open(os.path.join(compiled_dir, fname), encoding="utf-8"):
            if "=" in line and "Song:" not in line:
                label, model = line.strip().split(" = ")
                m[label.strip()] = model.strip()
        mappings[song_stem] = m

    # Build chart paths: charts/{model}/{song}.csv
    charts_dir = os.path.join(ar_dir, "charts")

    data_points = []
    all_votes = votes_data["self_rankings"] + votes_data.get("evaluators", [])

    for vote in all_votes:
        song_name = vote["song"]
        evaluator = vote.get("name", "self")
        # find mapping for this song
        song_mapping = None
        for stem, m in mappings.items():
            # fuzzy match
            if stem.lower().replace(" ", "")[:20] in song_name.lower().replace(" ", "") or \
               song_name.lower().replace(" ", "")[:20] in stem.lower().replace(" ", ""):
                song_mapping = m
                break
        if song_mapping is None:
            continue

        pts = [4, 3, 2, 1]
        for i, rank_key in enumerate(["rank_1", "rank_2", "rank_3"]):
            if rank_key not in vote:
                break
            label = vote[rank_key]
            model = song_mapping.get(label)
            if model is None:
                continue
            # find CSV
            csv_path = os.path.join(charts_dir, model, song_name + ".csv")
            if not os.path.exists(csv_path):
                # try without exact match
                for f in os.listdir(os.path.join(charts_dir, model)):
                    if f.endswith(".csv") and song_name[:20].lower() in f.lower():
                        csv_path = os.path.join(charts_dir, model, f)
                        break
            mel_path = csv_path.replace(".csv", "_mel.npy")

            score = pts[i]
            data_points.append({
                "round": "42-AR",
                "evaluator": evaluator,
                "song": song_name,
                "model": model,
                "score": score,
                "rank": i + 1,
                "csv_path": csv_path,
                "mel_path": mel_path,
            })
        # last place (may be implicit — 3-model round)
        n_models = len(song_mapping)
        if n_models == 3 and len([k for k in vote if k.startswith("rank_")]) >= 3:
            pass  # all 3 accounted for

    return data_points


def load_53ar_data():
    """Load 53-AR votes and chart paths."""
    ar_dir = os.path.join(TAIKO_DIR, "experiments", "experiment_53ar")
    with open(os.path.join(ar_dir, "results", "votes.json"), "r", encoding="utf-8") as f:
        votes_data = json.load(f)

    # Load mappings
    compiled_dir = os.path.join(ar_dir, "compiled")
    mappings = {}
    for fname in os.listdir(compiled_dir):
        if not fname.endswith("_mapping.txt"):
            continue
        song_stem = fname.replace("_mapping.txt", "")
        m = {}
        for line in open(os.path.join(compiled_dir, fname), encoding="utf-8"):
            line = line.strip()
            if "=" in line:
                label, model = line.split(" = ")
                m[label.strip()] = model.strip()
        mappings[song_stem] = m

    charts_dir = os.path.join(ar_dir, "charts")

    # Song name mapping: 53-AR CSVs are named like "01_arashi_five_{model}.csv"
    song_stems_53 = {}
    for fname in os.listdir(compiled_dir):
        if fname.endswith("_mapping.txt"):
            stem = fname.replace("_mapping.txt", "")
            song_stems_53[stem] = stem

    data_points = []
    all_votes = votes_data["self_rankings"] + votes_data.get("evaluators", [])

    for vote in all_votes:
        song_name = vote["song"]
        evaluator = vote.get("name", "self")

        # find mapping
        song_mapping = None
        matched_stem = None
        for stem, m in mappings.items():
            # fuzzy match song name to compiled stem
            song_clean = song_name.lower().replace(" ", "").replace("-", "").replace("_", "")
            stem_clean = stem[3:].lower().replace("_", "")  # skip "01_" prefix
            if stem_clean[:15] in song_clean or song_clean[:15] in stem_clean:
                song_mapping = m
                matched_stem = stem
                break
        if song_mapping is None:
            continue

        pts = [4, 3, 2, 1]
        for i, rank_key in enumerate(["rank_1", "rank_2", "rank_3", "rank_4"]):
            if rank_key not in vote:
                break
            label = vote[rank_key]
            model = song_mapping.get(label)
            if model is None:
                continue

            csv_path = os.path.join(charts_dir, f"{matched_stem}_{model}.csv")
            mel_path = csv_path.replace(".csv", "_mel.npy")

            data_points.append({
                "round": "53-AR",
                "evaluator": evaluator,
                "song": song_name,
                "model": model,
                "score": pts[i],
                "rank": i + 1,
                "csv_path": csv_path,
                "mel_path": mel_path,
            })

    return data_points


# ═══════════════════════════════════════════════════════════════
#  Metric Computation
# ═══════════════════════════════════════════════════════════════

def compute_chart_metrics(events_ms, mel=None):
    """Compute all candidate metrics for a single AR-generated chart."""
    m = {}

    if len(events_ms) < 3:
        return None

    gaps = np.diff(events_ms)
    gaps = gaps[gaps > 0]
    if len(gaps) < 2:
        return None

    # ── Gap distribution metrics ──
    m["n_events"] = len(events_ms)
    m["gap_mean"] = float(gaps.mean())
    m["gap_median"] = float(np.median(gaps))
    m["gap_std"] = float(gaps.std())
    m["gap_cv"] = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0  # coefficient of variation

    # Gap entropy (discretize into 10ms bins)
    bin_width = 10
    gap_bins = (gaps / bin_width).astype(int)
    _, counts = np.unique(gap_bins, return_counts=True)
    probs = counts / counts.sum()
    m["gap_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-10)))

    # Number of unique gaps (within 5% tolerance)
    sorted_gaps = np.sort(gaps)
    unique_gaps = []
    for g in sorted_gaps:
        if not unique_gaps or abs(g - unique_gaps[-1]) / max(unique_gaps[-1], 1) > 0.05:
            unique_gaps.append(g)
    m["n_unique_gaps"] = len(unique_gaps)

    # Dominant gap % (most common gap within 5% tolerance)
    gap_clusters = {}
    for g in gaps:
        matched = False
        for center in gap_clusters:
            if abs(g - center) / max(center, 1) <= 0.05:
                gap_clusters[center] += 1
                matched = True
                break
        if not matched:
            gap_clusters[g] = 1
    sorted_clusters = sorted(gap_clusters.values(), reverse=True)
    m["dominant_gap_pct"] = float(sorted_clusters[0] / len(gaps))
    m["top3_gap_pct"] = float(sum(sorted_clusters[:3]) / len(gaps)) if len(sorted_clusters) >= 3 else m["dominant_gap_pct"]

    # Longest metronome streak (consecutive gaps within 5% of each other)
    max_streak = 1
    streak = 1
    for i in range(1, len(gaps)):
        if abs(gaps[i] - gaps[i-1]) / max(gaps[i-1], 1) <= 0.05:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
    m["max_metro_streak"] = max_streak
    m["max_metro_streak_pct"] = float(max_streak / len(gaps))

    # ── Pattern dynamics ──
    # Gap autocorrelation (lag-1)
    if len(gaps) > 2:
        m["gap_autocorr"] = float(np.corrcoef(gaps[:-1], gaps[1:])[0, 1])
    else:
        m["gap_autocorr"] = 0.0

    # Pattern change rate (8s sliding window, how often dominant gap changes)
    window_ms = 8000
    events_sorted = np.sort(events_ms)
    n_windows = 0
    n_changes = 0
    prev_dominant = None
    for start in range(0, int(events_sorted[-1]), 4000):  # 4s hop
        end = start + window_ms
        window_events = events_sorted[(events_sorted >= start) & (events_sorted < end)]
        if len(window_events) < 3:
            continue
        wgaps = np.diff(window_events)
        wgaps = wgaps[wgaps > 0]
        if len(wgaps) < 2:
            continue
        # dominant gap in this window
        wgap_bins = (wgaps / 10).astype(int)
        vals, cnts = np.unique(wgap_bins, return_counts=True)
        dominant = vals[cnts.argmax()] * 10
        if prev_dominant is not None:
            n_windows += 1
            if abs(dominant - prev_dominant) / max(prev_dominant, 1) > 0.10:
                n_changes += 1
        prev_dominant = dominant
    m["pattern_change_rate"] = float(n_changes / max(n_windows, 1))

    # ── Audio alignment metrics (only if mel available) ──
    if mel is not None and mel.shape[1] > 0:
        # Mel energy per frame
        mel_energy = np.sum(mel ** 2, axis=0)  # (n_frames,)

        # Energy at onset positions
        onset_frames = (events_ms / BIN_MS).astype(int)
        onset_frames = onset_frames[(onset_frames >= 0) & (onset_frames < len(mel_energy))]

        if len(onset_frames) > 5:
            onset_energy = mel_energy[onset_frames]

            # Random positions for comparison
            rng = np.random.RandomState(42)
            random_frames = rng.randint(0, len(mel_energy), size=len(onset_frames))
            random_energy = mel_energy[random_frames]

            m["onset_energy_mean"] = float(onset_energy.mean())
            m["random_energy_mean"] = float(random_energy.mean())
            m["energy_ratio"] = float(onset_energy.mean() / max(random_energy.mean(), 1e-10))

            # Onset-to-peak alignment: for each onset, distance to nearest energy peak
            # Find peaks: frames where energy is higher than both neighbors
            peaks = []
            for i in range(1, len(mel_energy) - 1):
                if mel_energy[i] > mel_energy[i-1] and mel_energy[i] > mel_energy[i+1]:
                    peaks.append(i)
            peaks = np.array(peaks)

            if len(peaks) > 10:
                peak_dists = []
                for of in onset_frames:
                    idx = np.searchsorted(peaks, of)
                    best = float("inf")
                    for j in [idx-1, idx, idx+1]:
                        if 0 <= j < len(peaks):
                            best = min(best, abs(peaks[j] - of))
                    peak_dists.append(best * BIN_MS)  # convert to ms
                m["peak_dist_mean"] = float(np.mean(peak_dists))
                m["peak_dist_median"] = float(np.median(peak_dists))
            else:
                m["peak_dist_mean"] = 0.0
                m["peak_dist_median"] = 0.0

            # Energy change correlation: do gaps change when energy changes?
            if len(gaps) > 10:
                gap_diffs = np.abs(np.diff(gaps))
                # energy diff at each onset position
                onset_f = onset_frames[:len(gaps)]
                if len(onset_f) > 10:
                    e_at_onsets = mel_energy[np.clip(onset_f, 0, len(mel_energy)-1)]
                    e_diffs = np.abs(np.diff(e_at_onsets))
                    min_len = min(len(gap_diffs), len(e_diffs))
                    if min_len > 5:
                        m["energy_gap_corr"] = float(np.corrcoef(gap_diffs[:min_len], e_diffs[:min_len])[0, 1])
                    else:
                        m["energy_gap_corr"] = 0.0
                else:
                    m["energy_gap_corr"] = 0.0
            else:
                m["energy_gap_corr"] = 0.0
        else:
            m["onset_energy_mean"] = 0.0
            m["random_energy_mean"] = 0.0
            m["energy_ratio"] = 0.0
            m["peak_dist_mean"] = 0.0
            m["peak_dist_median"] = 0.0
            m["energy_gap_corr"] = 0.0
    else:
        m["onset_energy_mean"] = 0.0
        m["random_energy_mean"] = 0.0
        m["energy_ratio"] = 0.0
        m["peak_dist_mean"] = 0.0
        m["peak_dist_median"] = 0.0
        m["energy_gap_corr"] = 0.0

    # ── Density metrics ──
    duration_s = (events_ms[-1] - events_ms[0]) / 1000.0
    m["density"] = float(len(events_ms) / max(duration_s, 0.1))

    # Density stability across 8s windows
    densities = []
    for start in range(0, int(events_sorted[-1]), 4000):
        end = start + 8000
        n = ((events_sorted >= start) & (events_sorted < end)).sum()
        densities.append(n / 8.0)
    if densities:
        m["density_std"] = float(np.std(densities))
        m["density_cv"] = float(np.std(densities) / max(np.mean(densities), 0.01))
    else:
        m["density_std"] = 0.0
        m["density_cv"] = 0.0

    return m


# ═══════════════════════════════════════════════════════════════
#  Analysis & Visualization
# ═══════════════════════════════════════════════════════════════

def compute_correlations(data_points):
    """Compute Spearman correlation between each metric and human score."""
    # Get all metric keys from the first valid point
    metric_keys = None
    for dp in data_points:
        if dp.get("metrics"):
            metric_keys = sorted(dp["metrics"].keys())
            break
    if not metric_keys:
        return {}

    correlations = {}
    for key in metric_keys:
        scores = []
        values = []
        for dp in data_points:
            if dp.get("metrics") and key in dp["metrics"]:
                v = dp["metrics"][key]
                if v is not None and not np.isnan(v):
                    scores.append(dp["score"])
                    values.append(v)
        if len(scores) >= 10:
            r, p = sp_stats.spearmanr(values, scores)
            correlations[key] = {"r": float(r), "p": float(p), "n": len(scores)}

    return correlations


def save_graphs(data_points, correlations, output_dir):
    """Generate analysis graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Correlation bar chart (sorted by absolute r) ──
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    names = [k for k, v in sorted_corrs]
    rs = [v["r"] for k, v in sorted_corrs]
    ps = [v["p"] for k, v in sorted_corrs]
    colors = ["#6bc46d" if p < 0.05 else "#ff9900" if p < 0.10 else "#cccccc" for p in ps]
    bars = ax.barh(range(len(names)), rs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Spearman r (with human score)")
    ax.set_title("Metric Correlations with Human Preference\n(green=p<0.05, orange=p<0.10, gray=n.s.)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlations.png"), dpi=150)
    plt.close(fig)

    # ── 2. Top 6 scatter plots ──
    top6 = sorted_corrs[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    model_colors = {
        "exp14": "#4a90d9", "exp35c": "#e6a817", "exp42": "#eb4528",
        "exp44": "#6bc46d", "exp45": "#c76dba", "exp53": "#00cccc",
    }
    for i, (key, corr) in enumerate(top6):
        ax = axes[i]
        for dp in data_points:
            if dp.get("metrics") and key in dp["metrics"]:
                v = dp["metrics"][key]
                if v is not None and not np.isnan(v):
                    c = model_colors.get(dp["model"], "#999999")
                    ax.scatter(v, dp["score"] + np.random.uniform(-0.15, 0.15),
                              c=c, s=40, alpha=0.6, edgecolors="black", linewidths=0.3)
        ax.set_xlabel(key)
        ax.set_ylabel("Human Score")
        ax.set_title(f"{key}\nr={corr['r']:.3f}, p={corr['p']:.3f}")
        ax.set_yticks([1, 2, 3, 4])
    fig.suptitle("Top 6 Correlating Metrics vs Human Score", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "top6_scatter.png"), dpi=150)
    plt.close(fig)

    # ── 3. Per-model metric distributions (box plots for top metrics) ──
    top4_keys = [k for k, v in sorted_corrs[:4]]
    models = sorted(set(dp["model"] for dp in data_points if dp.get("metrics")))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, key in enumerate(top4_keys):
        ax = axes[i]
        model_vals = {m: [] for m in models}
        for dp in data_points:
            if dp.get("metrics") and key in dp["metrics"]:
                v = dp["metrics"][key]
                if v is not None and not np.isnan(v):
                    model_vals[dp["model"]].append(v)
        box_data = [model_vals[m] for m in models if model_vals[m]]
        box_labels = [m for m in models if model_vals[m]]
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for j, m in enumerate(box_labels):
                bp["boxes"][j].set_facecolor(model_colors.get(m, "#cccccc"))
            ax.set_title(f"{key}\n(r={correlations[key]['r']:.3f})")
            ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Top 4 Metrics: Distribution by Model", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "model_distributions.png"), dpi=150)
    plt.close(fig)

    # ── 4. Gap histogram comparison (all models overlaid) ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, model in enumerate(models[:6]):
        ax = axes[i]
        all_gaps = []
        for dp in data_points:
            if dp["model"] == model and dp.get("events_ms") is not None and len(dp["events_ms"]) > 2:
                gaps = np.diff(dp["events_ms"])
                gaps = gaps[(gaps > 0) & (gaps < 2000)]
                all_gaps.extend(gaps)
        if all_gaps:
            ax.hist(all_gaps, bins=100, range=(0, 1000), color=model_colors.get(model, "#999"),
                    alpha=0.7, density=True)
        ax.set_title(f"{model} (n={len(all_gaps):,} gaps)")
        ax.set_xlabel("Gap (ms)")
        ax.set_ylabel("Density")
    for j in range(len(models), 6):
        axes[j].set_visible(False)
    fig.suptitle("Gap Distributions by Model (all songs pooled)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gap_histograms.png"), dpi=150)
    plt.close(fig)

    # ── 5. Metric heatmap (model × metric) ──
    top10_keys = [k for k, v in sorted_corrs[:10]]
    fig, ax = plt.subplots(figsize=(14, 6))
    matrix = []
    for model in models:
        row = []
        for key in top10_keys:
            vals = [dp["metrics"][key] for dp in data_points
                    if dp["model"] == model and dp.get("metrics") and key in dp["metrics"]
                    and dp["metrics"][key] is not None and not np.isnan(dp["metrics"][key])]
            row.append(np.mean(vals) if vals else 0)
        matrix.append(row)
    matrix = np.array(matrix)
    # normalize per column for visualization
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        rng = col.max() - col.min()
        if rng > 0:
            matrix[:, j] = (col - col.min()) / rng
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(top10_keys)))
    ax.set_xticklabels(top10_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Top 10 Metrics: Normalized Heatmap by Model")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metric_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"Graphs saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    output_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Load data from both AR rounds
    print("Loading 42-AR data...")
    data_42 = load_42ar_data()
    print(f"  {len(data_42)} vote-model-song entries")

    print("Loading 53-AR data...")
    data_53 = load_53ar_data()
    print(f"  {len(data_53)} vote-model-song entries")

    all_data = data_42 + data_53
    print(f"Total: {len(all_data)} data points")

    # Compute metrics for each chart
    print("\nComputing metrics...")
    valid = 0
    skipped = 0
    seen_charts = {}  # cache: csv_path → metrics
    for dp in all_data:
        csv_path = dp["csv_path"]
        if csv_path in seen_charts:
            dp["metrics"] = seen_charts[csv_path]
            dp["events_ms"] = dp.get("events_ms", seen_charts.get(csv_path + "_events"))
            valid += 1
            continue

        if not os.path.exists(csv_path):
            skipped += 1
            continue

        events_ms = load_csv_events_ms(csv_path)
        dp["events_ms"] = events_ms

        mel = None
        if os.path.exists(dp["mel_path"]):
            try:
                mel = load_mel(dp["mel_path"])
            except Exception:
                pass

        metrics = compute_chart_metrics(events_ms, mel)
        if metrics is None:
            skipped += 1
            continue

        dp["metrics"] = metrics
        seen_charts[csv_path] = metrics
        seen_charts[csv_path + "_events"] = events_ms
        valid += 1

    print(f"  Valid: {valid}, Skipped: {skipped}")

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(all_data)

    # Print results
    print(f"\n{'='*70}")
    print("METRIC CORRELATIONS WITH HUMAN PREFERENCE (Spearman)")
    print(f"{'='*70}")
    print(f"{'Metric':>25s} {'r':>8s} {'p':>8s} {'n':>5s} {'sig':>5s}")
    for key, corr in sorted(correlations.items(), key=lambda x: abs(x[1]["r"]), reverse=True):
        sig = "***" if corr["p"] < 0.001 else "**" if corr["p"] < 0.01 else "*" if corr["p"] < 0.05 else "." if corr["p"] < 0.10 else ""
        print(f"{key:>25s} {corr['r']:>+8.3f} {corr['p']:>8.4f} {corr['n']:>5d} {sig:>5s}")

    # Save results
    results_path = os.path.join(output_dir, "correlations.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate graphs
    save_graphs(all_data, correlations, output_dir)


if __name__ == "__main__":
    main()
