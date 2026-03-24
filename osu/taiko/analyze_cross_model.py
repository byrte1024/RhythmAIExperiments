"""Cross-model failure analysis for experiment 48.

Loads val_data_{label}.npz files from multiple models, compares where they
succeed/fail, and whether they fail the same way.

Usage:
    python analyze_cross_model.py [--labels exp14,exp35c,exp44,exp45]
"""
import os
import sys
import json
import argparse
import numpy as np
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "experiments")
N_CLASSES = 501


def classify(pred, target):
    """Return 'hit', 'good', 'miss', or 'stop'."""
    if target >= N_CLASSES - 1:
        return "stop"
    frame_err = abs(int(pred) - int(target))
    ratio = (pred + 1) / (target + 1)
    pct_err = abs(ratio - 1.0)
    if (pct_err <= 0.03) or (frame_err <= 1):
        return "hit"
    if (pct_err <= 0.10) or (frame_err <= 2):
        return "good"
    return "miss"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="exp14,exp35c,exp44,exp45",
                        help="Comma-separated model labels")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "experiments", "experiment_48"))
    args = parser.parse_args()

    labels = args.labels.split(",")
    os.makedirs(args.output_dir, exist_ok=True)

    # load all model data
    models = {}
    for label in labels:
        path = os.path.join(DATA_DIR, f"val_data_{label}.npz")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        d = np.load(path)
        models[label] = {
            "scores": d["scores"],
            "preds": d["preds"],
            "targets": d["targets"],
        }
        print(f"Loaded {label}: {len(d['scores'])} samples")

    if len(models) < 2:
        print("Need at least 2 models to compare.")
        return

    # verify all have same targets (same val set)
    labels_loaded = list(models.keys())
    ref_targets = models[labels_loaded[0]]["targets"]
    for label in labels_loaded[1:]:
        if not np.array_equal(models[label]["targets"], ref_targets):
            print(f"WARNING: {label} has different targets than {labels_loaded[0]}!")
            print(f"  lengths: {len(ref_targets)} vs {len(models[label]['targets'])}")

    n = len(ref_targets)
    non_stop = ref_targets < (N_CLASSES - 1)
    n_ns = non_stop.sum()
    print(f"\nTotal samples: {n}, non-STOP: {n_ns}")

    # classify each sample per model
    classifications = {}
    for label in labels_loaded:
        preds = models[label]["preds"]
        targets = models[label]["targets"]
        cls = np.array([classify(preds[i], targets[i]) for i in range(n)])
        classifications[label] = cls

    # === 1. Per-model HIT/MISS rates ===
    print(f"\n{'='*60}")
    print("PER-MODEL RATES (non-STOP only)")
    print(f"{'='*60}")
    for label in labels_loaded:
        cls = classifications[label][non_stop]
        hit = (cls == "hit").mean()
        miss = (cls == "miss").mean()
        good = (cls == "good").mean()
        print(f"  {label:8s}: HIT={hit*100:.1f}% GOOD={good*100:.1f}% MISS={miss*100:.1f}%")

    # === 2. Pairwise agreement ===
    print(f"\n{'='*60}")
    print("PAIRWISE AGREEMENT (non-STOP, both HIT or both MISS)")
    print(f"{'='*60}")
    is_hit = {}
    is_miss = {}
    for label in labels_loaded:
        cls = classifications[label]
        is_hit[label] = (cls == "hit") & non_stop
        is_miss[label] = (cls == "miss") & non_stop

    print(f"{'':8s}", end="")
    for l2 in labels_loaded:
        print(f"  {l2:8s}", end="")
    print()
    for l1 in labels_loaded:
        print(f"{l1:8s}", end="")
        for l2 in labels_loaded:
            if l1 == l2:
                print(f"  {'---':>8s}", end="")
            else:
                both_hit = (is_hit[l1] & is_hit[l2]).sum()
                both_miss = (is_miss[l1] & is_miss[l2]).sum()
                agree = (both_hit + both_miss) / n_ns * 100
                print(f"  {agree:7.1f}%", end="")
        print()

    # === 3. Overlap analysis ===
    print(f"\n{'='*60}")
    print("FAILURE OVERLAP (non-STOP)")
    print(f"{'='*60}")

    # samples where ALL models miss
    all_miss = np.ones(n, dtype=bool)
    any_miss = np.zeros(n, dtype=bool)
    for label in labels_loaded:
        all_miss &= (classifications[label] == "miss")
        any_miss |= (classifications[label] == "miss")
    all_miss &= non_stop
    any_miss &= non_stop

    # samples where ALL models hit
    all_hit = np.ones(n, dtype=bool)
    for label in labels_loaded:
        all_hit &= (classifications[label] == "hit")
    all_hit &= non_stop

    print(f"  All models HIT:  {all_hit.sum():7d} ({all_hit.sum()/n_ns*100:.1f}%)")
    print(f"  All models MISS: {all_miss.sum():7d} ({all_miss.sum()/n_ns*100:.1f}%)")
    print(f"  Any model MISS:  {any_miss.sum():7d} ({any_miss.sum()/n_ns*100:.1f}%)")
    print(f"  Shared failure rate: {all_miss.sum()/any_miss.sum()*100:.1f}% of failures are universal")

    # per-model unique failures (this model misses, all others hit)
    print(f"\n  Model-specific failures (only this model misses):")
    for label in labels_loaded:
        others_hit = np.ones(n, dtype=bool)
        for other in labels_loaded:
            if other != label:
                others_hit &= (classifications[other] == "hit")
        unique_fail = is_miss[label] & others_hit
        print(f"    {label:8s}: {unique_fail.sum():5d} ({unique_fail.sum()/n_ns*100:.2f}%)")

    # === 4. When models fail on same sample, do they predict same bin? ===
    print(f"\n{'='*60}")
    print("SHARED FAILURE ANALYSIS")
    print(f"{'='*60}")

    shared_fail_idx = np.where(all_miss)[0]
    if len(shared_fail_idx) > 0:
        print(f"\n  {len(shared_fail_idx)} samples where ALL models miss:")

        # collect predictions for shared failures
        shared_preds = {}
        for label in labels_loaded:
            shared_preds[label] = models[label]["preds"][shared_fail_idx]
        shared_targets = ref_targets[shared_fail_idx]

        # pairwise prediction similarity
        print(f"\n  Prediction agreement (same bin ±5%) on shared failures:")
        for i, l1 in enumerate(labels_loaded):
            for l2 in labels_loaded[i+1:]:
                p1 = shared_preds[l1].astype(float)
                p2 = shared_preds[l2].astype(float)
                # within 5% of each other
                close = np.abs(p1 - p2) / np.maximum(np.maximum(p1, p2), 1) <= 0.05
                print(f"    {l1} vs {l2}: {close.mean()*100:.1f}% predict same bin")

        # direction analysis: do they overshoot or undershoot the same way?
        print(f"\n  Error direction on shared failures:")
        for label in labels_loaded:
            p = shared_preds[label].astype(float)
            t = shared_targets.astype(float)
            ns_mask = shared_targets < N_CLASSES - 1
            if ns_mask.sum() > 0:
                over = (p[ns_mask] > t[ns_mask]).mean()
                under = (p[ns_mask] < t[ns_mask]).mean()
                mean_err = (p[ns_mask] - t[ns_mask]).mean()
                print(f"    {label:8s}: over={over*100:.1f}% under={under*100:.1f}% mean_err={mean_err:+.1f} bins")

        # ratio analysis: what musical ratio are the errors at?
        print(f"\n  Error ratios on shared failures (pred/target):")
        for label in labels_loaded:
            p = shared_preds[label].astype(float)
            t = shared_targets.astype(float)
            ns_mask = shared_targets < N_CLASSES - 1
            if ns_mask.sum() > 0:
                ratios = (p[ns_mask] + 1) / (t[ns_mask] + 1)
                log_ratios = np.log2(np.clip(ratios, 0.125, 8.0))
                # count near musical ratios
                near_half = (np.abs(log_ratios - (-1)) < 0.1).mean()
                near_1 = (np.abs(log_ratios) < 0.1).mean()
                near_2 = (np.abs(log_ratios - 1) < 0.1).mean()
                print(f"    {label:8s}: ~0.5x={near_half*100:.1f}% ~1x={near_1*100:.1f}% ~2x={near_2*100:.1f}% median_ratio={np.median(ratios):.2f}")

    # === 5. Score correlation ===
    print(f"\n{'='*60}")
    print("SCORE CORRELATION (Pearson r)")
    print(f"{'='*60}")
    print(f"{'':8s}", end="")
    for l2 in labels_loaded:
        print(f"  {l2:8s}", end="")
    print()
    for l1 in labels_loaded:
        print(f"{l1:8s}", end="")
        s1 = models[l1]["scores"][non_stop]
        for l2 in labels_loaded:
            if l1 == l2:
                print(f"  {'1.000':>8s}", end="")
            else:
                s2 = models[l2]["scores"][non_stop]
                r = np.corrcoef(s1, s2)[0, 1]
                print(f"  {r:8.3f}", end="")
        print()

    # === 6. Save summary JSON ===
    summary = {
        "models": labels_loaded,
        "n_samples": int(n),
        "n_non_stop": int(n_ns),
        "all_hit": int(all_hit.sum()),
        "all_miss": int(all_miss.sum()),
        "any_miss": int(any_miss.sum()),
        "shared_failure_pct": float(all_miss.sum() / max(any_miss.sum(), 1) * 100),
    }
    for label in labels_loaded:
        cls = classifications[label][non_stop]
        summary[f"{label}_hit"] = float((cls == "hit").mean())
        summary[f"{label}_miss"] = float((cls == "miss").mean())

    json_path = os.path.join(args.output_dir, "cross_model_analysis.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")

    # === 7. Side-by-side heatmap ===
    try:
        from PIL import Image
        heatmaps = []
        for label in labels_loaded:
            path = os.path.join(DATA_DIR, f"val_heatmap_{label}.png")
            if os.path.exists(path):
                heatmaps.append((label, Image.open(path)))

        if len(heatmaps) >= 2:
            from PIL import ImageDraw, ImageFont
            w = heatmaps[0][1].width
            h = heatmaps[0][1].height
            label_h = 25
            combined = Image.new("RGB", (w * len(heatmaps), h + label_h), (22, 22, 30))
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("consola.ttf", 16)
            except:
                font = ImageFont.load_default()
            for i, (label, img) in enumerate(heatmaps):
                combined.paste(img, (i * w, label_h))
                draw.text((i * w + w // 2, 4), label, fill=(200, 200, 210), font=font, anchor="mt")
            out = os.path.join(args.output_dir, "compare_heatmaps.png")
            combined.save(out)
            print(f"Saved: {out}")
    except ImportError:
        print("PIL not available, skipping heatmap comparison")


if __name__ == "__main__":
    main()
