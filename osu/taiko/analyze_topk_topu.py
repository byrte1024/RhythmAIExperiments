"""Experiment 44-B: Top-K vs Top-U oracle analysis.

Runs a subsample 8 val pass on exp 44, then for each sample:
- Top-K oracle: pick the best of top K predictions (closest to target)
- Top-U oracle: cluster predictions within 5%, merge confidence, pick best of top U unique clusters

Reports HIT/GOOD/MISS rates for each K and U value, with graphs.
"""
import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_train import (
    OnsetDataset, split_by_song, N_CLASSES, C_EVENTS
)
from detection_model import EventEmbeddingDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, "runs", "detect_experiment_44")
DS_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
STOP = N_CLASSES - 1


def compute_top_u(probs, max_u=10, tolerance=0.05):
    """Compute Top-U unique predictions from a probability distribution.

    Args:
        probs: (N_CLASSES,) softmax probabilities
        max_u: max unique clusters to return
        tolerance: merge bins within this relative distance

    Returns:
        list of (bin_index, merged_confidence) sorted by confidence descending,
        length <= max_u
    """
    # sort classes by confidence descending (exclude STOP)
    order = np.argsort(probs[:STOP])[::-1]

    clusters = []  # list of [representative_bin, total_confidence, weighted_bin_sum]

    for cls in order:
        conf = probs[cls]
        if conf < 1e-6:
            break

        # check if this bin is within tolerance of an existing cluster
        matched = False
        for c in clusters:
            rep_bin = c[0]
            if rep_bin > 0 and abs(cls - rep_bin) / rep_bin <= tolerance:
                c[1] += conf  # add confidence
                c[2] += conf * cls  # weighted sum for centroid
                matched = True
                break

        if not matched:
            clusters.append([cls, conf, conf * cls])

        if len(clusters) >= max_u * 3:  # early stop, we have enough candidates
            break

    # sort by merged confidence descending
    clusters.sort(key=lambda c: c[1], reverse=True)

    # return top max_u as (bin, confidence)
    result = []
    for c in clusters[:max_u]:
        centroid = int(round(c[2] / c[1]))  # confidence-weighted centroid
        result.append((centroid, c[1]))

    return result


def oracle_pick(candidates, target):
    """Pick the candidate closest to target. Returns the picked bin."""
    if not candidates:
        return 0
    best = min(candidates, key=lambda c: abs(c - target))
    return best


def hit_good_miss(pred, target):
    """Compute HIT/GOOD/MISS for a single (pred, target) pair."""
    if target >= STOP or target <= 0:
        return None, None, None
    frame_err = abs(pred - target)
    ratio = (pred + 1) / (target + 1)
    pct_err = abs(ratio - 1.0)
    hit = (pct_err <= 0.03) or (frame_err <= 1)
    good = (pct_err <= 0.10) or (frame_err <= 2)
    miss = pct_err > 0.20
    return hit, good, miss


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # load manifest + config
    with open(os.path.join(DS_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(RUN_DIR, "config.json")) as f:
        config = json.load(f)

    # split
    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)

    val_ds = OnsetDataset(manifest, DS_DIR, val_idx, augment=False, subsample=8,
                          multi_target=False)
    print(f"Val samples (subsample=8): {len(val_ds)}")

    nw = config.get("workers", 3)
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=False, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )

    # load model
    device = torch.device(config.get("device", "cuda"))
    model = EventEmbeddingDetector(
        n_mels=80, d_model=config["d_model"],
        n_layers=config["enc_layers"] + config["fusion_layers"],
        n_heads=config["n_heads"],
        n_classes=N_CLASSES, max_events=C_EVENTS, dropout=config["dropout"],
    ).to(device)

    ckpt_dir = os.path.join(RUN_DIR, "checkpoints")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {ckpts[-1]}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state["model"] if "model" in state else state
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(clean_state)
    model.eval()

    # collect predictions + full probs
    print("Running validation...")
    all_targets = []
    all_probs = []

    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            mel, evt_off, evt_mask, cond, target = batch
            mel = mel.to(device, non_blocking=True)
            evt_off = evt_off.to(device, non_blocking=True)
            evt_mask = evt_mask.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)

            logits = model(mel, evt_off, evt_mask, cond)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits.float(), dim=-1)

            all_targets.append(target.numpy())
            all_probs.append(probs.cpu().numpy())

    targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)
    print(f"Collected {len(targets)} samples")

    # filter non-STOP
    ns = targets < STOP
    targets_ns = targets[ns]
    probs_ns = probs[ns]
    print(f"Non-STOP: {ns.sum()}")

    max_k = 10

    # Top-K oracle
    print("\nComputing Top-K oracle...")
    topk_indices = np.argsort(probs_ns[:, :STOP], axis=1)[:, ::-1][:, :max_k]  # (N, max_k)

    k_hits = np.zeros(max_k)
    k_goods = np.zeros(max_k)
    k_misses = np.zeros(max_k)
    n = len(targets_ns)

    for ki in range(max_k):
        candidates = topk_indices[:, :ki+1]  # (N, ki+1)
        # oracle: pick closest to target for each sample
        diffs = np.abs(candidates - targets_ns[:, None])  # (N, ki+1)
        best_idx = diffs.argmin(axis=1)  # (N,)
        oracle_preds = candidates[np.arange(n), best_idx]  # (N,)

        frame_err = np.abs(oracle_preds - targets_ns)
        ratio = (oracle_preds + 1) / (targets_ns + 1)
        pct_err = np.abs(ratio - 1.0)
        hit = (pct_err <= 0.03) | (frame_err <= 1)
        good = (pct_err <= 0.10) | (frame_err <= 2)
        miss = pct_err > 0.20

        k_hits[ki] = hit.mean()
        k_goods[ki] = good.mean()
        k_misses[ki] = miss.mean()
        print(f"  Top-{ki+1}: HIT={hit.mean()*100:.1f}% GOOD={good.mean()*100:.1f}% MISS={miss.mean()*100:.1f}%")

    # Top-U oracle + threshold analysis
    print("\nComputing Top-U oracle + threshold stats...")
    u_hits = np.zeros(max_k)
    u_goods = np.zeros(max_k)
    u_misses = np.zeros(max_k)

    # thresholds: how many unique clusters pass each confidence threshold?
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    # for each threshold, track count of U clusters above it per sample
    thresh_counts = {t: [] for t in thresholds}

    for i in tqdm(range(n), desc="Top-U", leave=False):
        clusters = compute_top_u(probs_ns[i], max_u=max_k, tolerance=0.05)
        t = targets_ns[i]

        # threshold analysis: how many clusters have normalized confidence >= T?
        total_conf = sum(c[1] for c in clusters)
        if total_conf > 0:
            normed = [(c[0], c[1] / total_conf) for c in clusters]
        else:
            normed = clusters
        for thresh in thresholds:
            count = sum(1 for _, conf in normed if conf >= thresh)
            thresh_counts[thresh].append(count)

        for ui in range(max_k):
            if ui < len(clusters):
                candidates = [c[0] for c in clusters[:ui+1]]
            else:
                candidates = [c[0] for c in clusters]
            pred = oracle_pick(candidates, t)

            frame_err = abs(pred - t)
            ratio = (pred + 1) / (t + 1)
            pct_err = abs(ratio - 1.0)
            hit = (pct_err <= 0.03) or (frame_err <= 1)
            good = (pct_err <= 0.10) or (frame_err <= 2)
            miss = pct_err > 0.20

            u_hits[ui] += hit
            u_goods[ui] += good
            u_misses[ui] += miss

    u_hits /= n
    u_goods /= n
    u_misses /= n

    for ui in range(max_k):
        print(f"  Top-U{ui+1}: HIT={u_hits[ui]*100:.1f}% GOOD={u_goods[ui]*100:.1f}% MISS={u_misses[ui]*100:.1f}%")

    # threshold stats
    print("\n--- Unique clusters above threshold (normalized confidence) ---")
    thresh_stats = {}
    for thresh in thresholds:
        counts = np.array(thresh_counts[thresh])
        mean_c = counts.mean()
        median_c = np.median(counts)
        p10 = np.percentile(counts, 10)
        p90 = np.percentile(counts, 90)
        print(f"  T={thresh:.0%}: mean={mean_c:.2f} median={median_c:.0f} p10={p10:.0f} p90={p90:.0f}")
        thresh_stats[f"t{int(thresh*100):02d}"] = {
            "threshold": thresh,
            "mean": float(mean_c),
            "median": float(median_c),
            "p10": float(p10),
            "p90": float(p90),
        }

    # save JSON
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_44c")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "n_samples": int(n),
        "top_k": {f"k{ki+1}": {"hit": float(k_hits[ki]), "good": float(k_goods[ki]), "miss": float(k_misses[ki])} for ki in range(max_k)},
        "top_u": {f"u{ui+1}": {"hit": float(u_hits[ui]), "good": float(u_goods[ui]), "miss": float(u_misses[ui])} for ui in range(max_k)},
        "threshold_stats": thresh_stats,
    }
    json_path = os.path.join(out_dir, "topk_topu_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # graph: HIT/MISS curves for Top-K vs Top-U
    ks = np.arange(1, max_k + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # HIT
    axes[0].plot(ks, k_hits * 100, "o-", color="#ff4444", label="Top-K", linewidth=2, markersize=6)
    axes[0].plot(ks, u_hits * 100, "s-", color="#4488ff", label="Top-U", linewidth=2, markersize=6)
    axes[0].set_xlabel("K / U")
    axes[0].set_ylabel("HIT %")
    axes[0].set_title("Oracle HIT Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(ks)

    # GOOD
    axes[1].plot(ks, k_goods * 100, "o-", color="#ff4444", label="Top-K", linewidth=2, markersize=6)
    axes[1].plot(ks, u_goods * 100, "s-", color="#4488ff", label="Top-U", linewidth=2, markersize=6)
    axes[1].set_xlabel("K / U")
    axes[1].set_ylabel("GOOD %")
    axes[1].set_title("Oracle GOOD Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(ks)

    # MISS
    axes[2].plot(ks, k_misses * 100, "o-", color="#ff4444", label="Top-K", linewidth=2, markersize=6)
    axes[2].plot(ks, u_misses * 100, "s-", color="#4488ff", label="Top-U", linewidth=2, markersize=6)
    axes[2].set_xlabel("K / U")
    axes[2].set_ylabel("MISS %")
    axes[2].set_title("Oracle MISS Rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(ks)

    fig.suptitle("Top-K vs Top-U Oracle (5% merge tolerance)", fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "topk_topu_graph.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # delta graph: how much does Top-U improve over Top-K?
    fig, ax = plt.subplots(figsize=(10, 6))
    delta_hit = (u_hits - k_hits) * 100
    delta_miss = (k_misses - u_misses) * 100  # positive = U is better (less miss)
    ax.bar(ks - 0.15, delta_hit, width=0.3, color="#4488ff", label="HIT gain (U vs K)")
    ax.bar(ks + 0.15, delta_miss, width=0.3, color="#44cc44", label="MISS reduction (U vs K)")
    ax.set_xlabel("K / U")
    ax.set_ylabel("Percentage points")
    ax.set_title("Top-U Improvement Over Top-K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="white", alpha=0.3)
    ax.set_xticks(ks)
    fig.tight_layout()
    fig_path2 = os.path.join(out_dir, "topu_delta_graph.png")
    fig.savefig(fig_path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path2}")

    # threshold graph: how many unique clusters pass each threshold?
    fig, ax = plt.subplots(figsize=(10, 6))
    t_vals = [t * 100 for t in thresholds]
    means = [thresh_stats[f"t{int(t*100):02d}"]["mean"] for t in thresholds]
    medians = [thresh_stats[f"t{int(t*100):02d}"]["median"] for t in thresholds]
    p10s = [thresh_stats[f"t{int(t*100):02d}"]["p10"] for t in thresholds]
    p90s = [thresh_stats[f"t{int(t*100):02d}"]["p90"] for t in thresholds]
    ax.fill_between(t_vals, p10s, p90s, alpha=0.2, color="#4488ff", label="p10-p90")
    ax.plot(t_vals, means, "o-", color="#ff4444", linewidth=2, markersize=6, label="Mean")
    ax.plot(t_vals, medians, "s--", color="#44cc44", linewidth=2, markersize=6, label="Median")
    ax.set_xlabel("Confidence threshold (%)")
    ax.set_ylabel("# unique clusters above threshold")
    ax.set_title("Unique Prediction Clusters Above Confidence Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path3 = os.path.join(out_dir, "threshold_unique_graph.png")
    fig.savefig(fig_path3, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path3}")


if __name__ == "__main__":
    main()
