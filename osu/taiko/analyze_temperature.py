"""Experiment 44-D: Temperature sampling on Top-K and Top-U.

Runs a subsample 8 val pass on exp 44, then for each sample sweeps 50
temperature values on Top-K and Top-U candidate sets (3, 5, 10, 20).

At each temperature, applies temperature scaling to candidate confidences
and picks argmax of the tempered distribution. Measures HIT/GOOD/MISS.

This is NOT oracle — it measures what the model would actually pick.
"""
import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detection_train import OnsetDataset, split_by_song, N_CLASSES, C_EVENTS
from detection_model import EventEmbeddingDetector

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(SCRIPT_DIR, "runs", "detect_experiment_44")
DS_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
STOP = N_CLASSES - 1


def compute_top_u(probs, max_u=20, tolerance=0.05):
    """Compute Top-U unique predictions. Returns list of (bin, confidence)."""
    order = np.argsort(probs[:STOP])[::-1]
    clusters = []  # [representative_bin, total_confidence, weighted_bin_sum]

    for cls in order:
        conf = probs[cls]
        if conf < 1e-6:
            break
        matched = False
        for c in clusters:
            if c[0] > 0 and abs(cls - c[0]) / c[0] <= tolerance:
                c[1] += conf
                c[2] += conf * cls
                matched = True
                break
        if not matched:
            clusters.append([cls, conf, conf * cls])
        if len(clusters) >= max_u * 3:
            break

    clusters.sort(key=lambda c: c[1], reverse=True)
    result = []
    for c in clusters[:max_u]:
        centroid = int(round(c[2] / c[1]))
        result.append((centroid, c[1]))
    return result


def tempered_sample(candidates, confs, temperature, rng):
    """Apply temperature to confidences and sample from the distribution.

    candidates: list of bin indices
    confs: list of confidences (same length)
    temperature: float > 0. T→0 = argmax, T→∞ = uniform.
    rng: numpy random generator
    """
    if len(candidates) == 0:
        return 0
    if len(candidates) == 1:
        return candidates[0]

    confs = np.array(confs, dtype=np.float64)
    confs = np.maximum(confs, 1e-30)

    # temperature scaling in log space for stability
    log_confs = np.log(confs) / temperature
    log_confs -= log_confs.max()
    tempered = np.exp(log_confs)
    tempered /= tempered.sum()  # normalize to probabilities

    return candidates[rng.choice(len(candidates), p=tempered)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm

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

    # collect predictions
    print("Running validation...")
    all_targets = []
    all_probs = []

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

    ns = targets < STOP
    targets_ns = targets[ns]
    probs_ns = probs[ns]
    n = len(targets_ns)
    print(f"Non-STOP: {n}")

    # precompute Top-K and Top-U candidates for all samples
    ks = [3, 5, 10, 20]
    max_ku = max(ks)

    print("Precomputing Top-K indices...")
    topk_all = np.argsort(probs_ns[:, :STOP], axis=1)[:, ::-1][:, :max_ku]
    topk_confs = np.take_along_axis(probs_ns[:, :STOP], topk_all, axis=1)

    print("Precomputing Top-U clusters...")
    topu_bins = np.zeros((n, max_ku), dtype=np.int64)
    topu_confs = np.zeros((n, max_ku), dtype=np.float64)
    for i in tqdm(range(n), desc="Top-U", leave=False):
        clusters = compute_top_u(probs_ns[i], max_u=max_ku, tolerance=0.05)
        for j, (b, c) in enumerate(clusters[:max_ku]):
            topu_bins[i, j] = b
            topu_confs[i, j] = c

    # temperature sweep with sampling (averaged over N_TRIALS for stability)
    temperatures = np.logspace(np.log10(0.01), np.log10(100), 50)
    N_TRIALS = 5  # average over multiple random seeds
    print(f"\nSweeping {len(temperatures)} temperatures from {temperatures[0]:.3f} to {temperatures[-1]:.1f} ({N_TRIALS} trials each)...")

    results = {"temperatures": temperatures.tolist(), "n_trials": N_TRIALS, "top_k": {}, "top_u": {}}

    def sweep_temperature(cand_bins, cand_confs, k, label):
        """Sweep temperatures for a candidate set. Returns dict with hit/good/miss lists."""
        res = {"hit": [], "good": [], "miss": []}

        for ti, temp in enumerate(temperatures):
            trial_hits, trial_goods, trial_misses = [], [], []

            for trial in range(N_TRIALS):
                rng = np.random.default_rng(seed=42 + trial)

                # vectorized temperature sampling
                confs = cand_confs[:, :k].copy()  # (n, k)
                log_c = np.log(np.maximum(confs, 1e-30)) / temp
                log_c -= log_c.max(axis=1, keepdims=True)
                tempered = np.exp(log_c)
                tempered /= tempered.sum(axis=1, keepdims=True)  # normalize to probs

                # sample from each row
                cumsum = tempered.cumsum(axis=1)
                rand = rng.random(n)[:, None]
                chosen = (cumsum < rand).sum(axis=1)  # index of chosen candidate
                chosen = np.clip(chosen, 0, k - 1)
                preds = cand_bins[np.arange(n), chosen]

                frame_err = np.abs(preds - targets_ns)
                ratio = (preds + 1) / (targets_ns + 1)
                pct_err = np.abs(ratio - 1.0)
                trial_hits.append(((pct_err <= 0.03) | (frame_err <= 1)).mean())
                trial_goods.append(((pct_err <= 0.10) | (frame_err <= 2)).mean())
                trial_misses.append((pct_err > 0.20).mean())

            res["hit"].append(float(np.mean(trial_hits)))
            res["good"].append(float(np.mean(trial_goods)))
            res["miss"].append(float(np.mean(trial_misses)))

        best_hit_idx = np.argmax(res["hit"])
        t1_idx = np.searchsorted(temperatures, 1.0)
        print(f"  {label}: Best HIT={res['hit'][best_hit_idx]*100:.1f}% at T={temperatures[best_hit_idx]:.3f} | "
              f"T=0.01: {res['hit'][0]*100:.1f}% | T=1.0: {res['hit'][t1_idx]*100:.1f}%")
        return res

    for k in ks:
        results["top_k"][f"k{k}"] = sweep_temperature(topk_all, topk_confs, k, f"Top-K {k}")

    for k in ks:
        results["top_u"][f"u{k}"] = sweep_temperature(topu_bins, topu_confs, k, f"Top-U {k}")

    # save JSON
    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_44d")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "temperature_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # graphs
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for row, (mode, label) in enumerate([("top_k", "Top-K"), ("top_u", "Top-U")]):
        colors = ["#ff4444", "#44cc44", "#4488ff", "#ffaa44"]
        for ci, k in enumerate(ks):
            key = f"{'k' if mode == 'top_k' else 'u'}{k}"
            data = results[mode][key]

            # HIT
            axes[row, 0].semilogx(temperatures, np.array(data["hit"]) * 100,
                                   color=colors[ci], linewidth=2, label=f"{label}-{k}")
            # MISS
            axes[row, 1].semilogx(temperatures, np.array(data["miss"]) * 100,
                                   color=colors[ci], linewidth=2, label=f"{label}-{k}")

        axes[row, 0].set_xlabel("Temperature")
        axes[row, 0].set_ylabel("HIT %")
        axes[row, 0].set_title(f"{label}: HIT Rate vs Temperature")
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].axvline(1.0, color="white", alpha=0.3, linestyle="--")

        axes[row, 1].set_xlabel("Temperature")
        axes[row, 1].set_ylabel("MISS %")
        axes[row, 1].set_title(f"{label}: MISS Rate vs Temperature")
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].axvline(1.0, color="white", alpha=0.3, linestyle="--")

    fig.suptitle("Temperature Sampling: Top-K vs Top-U", fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "temperature_graph.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # overlay: Top-K vs Top-U at same K for direct comparison
    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 5))
    for ci, k in enumerate(ks):
        kdata = results["top_k"][f"k{k}"]
        udata = results["top_u"][f"u{k}"]
        axes[ci].semilogx(temperatures, np.array(kdata["hit"]) * 100,
                          color="#ff4444", linewidth=2, label=f"Top-K {k}")
        axes[ci].semilogx(temperatures, np.array(udata["hit"]) * 100,
                          color="#4488ff", linewidth=2, label=f"Top-U {k}")
        axes[ci].set_xlabel("Temperature")
        axes[ci].set_ylabel("HIT %")
        axes[ci].set_title(f"K/U = {k}")
        axes[ci].legend()
        axes[ci].grid(True, alpha=0.3)
        axes[ci].axvline(1.0, color="white", alpha=0.3, linestyle="--")

    fig.suptitle("Top-K vs Top-U HIT at Same K/U", fontsize=14)
    fig.tight_layout()
    fig_path2 = os.path.join(out_dir, "temperature_comparison.png")
    fig.savefig(fig_path2, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path2}")


if __name__ == "__main__":
    main()
