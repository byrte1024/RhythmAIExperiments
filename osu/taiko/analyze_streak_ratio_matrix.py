"""Analyze the training data: count samples by (context streak length, target/gap ratio).

Builds a matrix of how many samples fall into each (streak, ratio) bucket.
This reveals whether the model sees enough examples of pattern-breaking at each streak length.
"""
import os
import sys
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_train import OnsetDataset, split_by_song, N_CLASSES, C_EVENTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")

STREAK_BINS = [1, 2, 3, 5, 8, 10, 16]
RATIO_BINS = [1/8, 1/6, 1/4, 1/3, 1/2, 1/1, 2/1, 3/1, 4/1, 6/1, 8/1]
RATIO_LABELS = ["1/8", "1/6", "1/4", "1/3", "1/2", "1/1", "2/1", "3/1", "4/1", "6/1", "8/1"]
TOLERANCE = 0.05


def find_streak(gaps):
    """Find length of same-gap streak ending at the last gap (5% tolerance)."""
    if len(gaps) < 1:
        return 0
    recent_gap = gaps[-1]
    if recent_gap <= 0:
        return 0
    streak = 1
    for j in range(len(gaps) - 2, -1, -1):
        if recent_gap > 0 and abs(gaps[j] - recent_gap) / recent_gap <= TOLERANCE:
            streak += 1
        else:
            break
    return streak


def classify_streak(streak_len):
    """Map streak length to the nearest bin."""
    best = STREAK_BINS[0]
    for b in STREAK_BINS:
        if streak_len >= b:
            best = b
    return best


def classify_ratio(ratio):
    """Map ratio to nearest bin (within 10% tolerance)."""
    best_dist = float('inf')
    best_bin = None
    for rb in RATIO_BINS:
        if rb > 0:
            dist = abs(ratio / rb - 1.0)
            if dist < best_dist:
                best_dist = dist
                best_bin = rb
    if best_dist <= 0.15:  # within 15% of a ratio bin
        return best_bin
    return None  # doesn't match any clean ratio


def main():
    with open(os.path.join(DS_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(42)
    train_idx, _ = split_by_song(manifest, val_ratio=0.1)

    train_ds = OnsetDataset(manifest, DS_DIR, train_idx, augment=False, subsample=1,
                            multi_target=False)
    print(f"Train samples: {len(train_ds)}")

    # matrix: streak_bin_idx x ratio_bin_idx
    matrix = np.zeros((len(STREAK_BINS), len(RATIO_BINS)), dtype=np.int64)
    unclassified = 0
    stop_count = 0
    no_context = 0

    from tqdm import tqdm
    for idx in tqdm(range(len(train_ds.samples)), desc="Classifying"):
        ci, ei = train_ds.samples[idx]
        evt = train_ds.events[ci]
        target = train_ds._get_target(ci, ei)

        if target >= N_CLASSES - 1:
            stop_count += 1
            continue

        # need at least 2 past events to have gaps
        if ei < 2:
            no_context += 1
            continue

        # compute past gaps
        past_start = max(0, ei - C_EVENTS)
        past_events = evt[past_start:ei]
        past_gaps = np.diff(past_events).astype(np.float64)
        past_gaps = past_gaps[past_gaps > 0]

        if len(past_gaps) < 1:
            no_context += 1
            continue

        # streak length
        streak = find_streak(past_gaps)
        streak_bin = classify_streak(streak)
        streak_idx = STREAK_BINS.index(streak_bin)

        # ratio: target_gap / last_gap
        last_gap = past_gaps[-1]
        if last_gap <= 0:
            no_context += 1
            continue

        ratio = target / last_gap
        ratio_bin = classify_ratio(ratio)

        if ratio_bin is None:
            unclassified += 1
            continue

        ratio_idx = RATIO_BINS.index(ratio_bin)
        matrix[streak_idx, ratio_idx] += 1

    total_classified = matrix.sum()
    print(f"\nClassified: {total_classified:,d}")
    print(f"STOP: {stop_count:,d}")
    print(f"No context: {no_context:,d}")
    print(f"Unclassified ratio: {unclassified:,d}")

    # print matrix
    print(f"\n{'':>10}", end="")
    for label in RATIO_LABELS:
        print(f"{label:>10}", end="")
    print(f"{'TOTAL':>12}")
    print("-" * (10 + 10 * len(RATIO_LABELS) + 12))

    for si, sb in enumerate(STREAK_BINS):
        print(f"streak {sb:>2}:", end="")
        for ri in range(len(RATIO_BINS)):
            count = matrix[si, ri]
            print(f"{count:>10,d}", end="")
        row_total = matrix[si].sum()
        print(f"{row_total:>12,d}")

    print("-" * (10 + 10 * len(RATIO_LABELS) + 12))
    print(f"{'TOTAL':>10}", end="")
    for ri in range(len(RATIO_BINS)):
        col_total = matrix[:, ri].sum()
        print(f"{col_total:>10,d}", end="")
    print(f"{total_classified:>12,d}")

    # percentage version
    print(f"\n\nPercentage of classified samples:")
    print(f"{'':>10}", end="")
    for label in RATIO_LABELS:
        print(f"{label:>10}", end="")
    print()
    for si, sb in enumerate(STREAK_BINS):
        print(f"streak {sb:>2}:", end="")
        for ri in range(len(RATIO_BINS)):
            pct = matrix[si, ri] / total_classified * 100
            print(f"{pct:>9.2f}%", end="")
        print()

    # continuation rate per streak
    print(f"\n\nContinuation rate (ratio=1/1) per streak:")
    r11_idx = RATIO_BINS.index(1.0)
    for si, sb in enumerate(STREAK_BINS):
        row_total = matrix[si].sum()
        if row_total > 0:
            cont = matrix[si, r11_idx]
            print(f"  streak {sb:>2}: {cont:>10,d} / {row_total:>10,d} = {cont/row_total*100:.1f}% continue")

    # save matrix
    out_path = os.path.join(SCRIPT_DIR, "experiments", "streak_ratio_matrix.json")
    result = {
        "streak_bins": STREAK_BINS,
        "ratio_bins": [float(r) for r in RATIO_BINS],
        "ratio_labels": RATIO_LABELS,
        "matrix": matrix.tolist(),
        "total_classified": int(total_classified),
        "stop_count": int(stop_count),
        "unclassified": int(unclassified),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    # render heatmap visualization
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # 1. Raw counts (log scale)
    ax = axes[0]
    m_plot = matrix.astype(np.float64)
    m_plot[m_plot == 0] = 0.5  # avoid log(0)
    im = ax.imshow(m_plot, aspect="auto", cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=m_plot.max()))
    ax.set_xticks(range(len(RATIO_LABELS)))
    ax.set_xticklabels(RATIO_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(STREAK_BINS)))
    ax.set_yticklabels([f"streak {s}" for s in STREAK_BINS])
    ax.set_title("Raw Counts (log scale)")
    ax.set_xlabel("Target / Previous Gap Ratio")
    ax.set_ylabel("Context Streak Length")
    # annotate cells
    for si in range(len(STREAK_BINS)):
        for ri in range(len(RATIO_BINS)):
            val = matrix[si, ri]
            if val > 0:
                txt = f"{val:,d}" if val < 10000 else f"{val/1000:.0f}k"
                color = "white" if val > m_plot.max() * 0.1 else "black"
                ax.text(ri, si, txt, ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Row-normalized (what % of each streak goes to each ratio)
    ax = axes[1]
    row_sums = matrix.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1
    row_pct = matrix / row_sums * 100
    im2 = ax.imshow(row_pct, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_xticks(range(len(RATIO_LABELS)))
    ax.set_xticklabels(RATIO_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(STREAK_BINS)))
    ax.set_yticklabels([f"streak {s}" for s in STREAK_BINS])
    ax.set_title("Row-Normalized (% of streak going to each ratio)")
    ax.set_xlabel("Target / Previous Gap Ratio")
    for si in range(len(STREAK_BINS)):
        for ri in range(len(RATIO_BINS)):
            val = row_pct[si, ri]
            if val >= 0.5:
                color = "white" if val > 50 else "black"
                ax.text(ri, si, f"{val:.1f}%", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im2, ax=ax, shrink=0.8, label="%")

    # 3. Break rate (1 - continuation rate per streak)
    ax = axes[2]
    r11_idx = RATIO_BINS.index(1.0)
    streak_labels = [f"streak {s}" for s in STREAK_BINS]
    cont_rates = []
    break_rates = []
    for si in range(len(STREAK_BINS)):
        row_total = matrix[si].sum()
        if row_total > 0:
            cont = matrix[si, r11_idx] / row_total * 100
            cont_rates.append(cont)
            break_rates.append(100 - cont)
        else:
            cont_rates.append(0)
            break_rates.append(0)

    bars = ax.barh(range(len(STREAK_BINS)), break_rates, color="#e8834a", label="Break")
    ax.barh(range(len(STREAK_BINS)), cont_rates, left=break_rates, color="#4a90d9", label="Continue")
    ax.set_yticks(range(len(STREAK_BINS)))
    ax.set_yticklabels(streak_labels)
    ax.set_xlabel("%")
    ax.set_title("Continue vs Break Rate by Streak Length")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    for i, (br, cr) in enumerate(zip(break_rates, cont_rates)):
        ax.text(br / 2, i, f"{br:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(br + cr / 2, i, f"{cr:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    fig.suptitle("Streak Length x Ratio Matrix — Training Data Distribution", fontsize=14)
    fig.tight_layout()
    img_path = os.path.join(SCRIPT_DIR, "experiments", "streak_ratio_matrix.png")
    fig.savefig(img_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {img_path}")


if __name__ == "__main__":
    main()
