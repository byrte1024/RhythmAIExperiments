"""Experiment 43-B: Compare AR resilience across notable models.

Runs the autoregressive benchmark on exp 14, 35-C, and 42, saving
per-model graphs and comparison data.

Usage:
    python analyze_ar_comparison.py taiko_v2 --subsample 10
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from detection_train import (
    OnsetDataset, N_CLASSES, C_EVENTS, B_BINS, A_BINS, split_by_song,
    run_benchmarks,
)
from detection_model import (
    OnsetDetector, EventEmbeddingDetector, LegacyOnsetDetector,
    AdditiveOnsetDetector, RerankerOnsetDetector,
)


MODELS = {
    "exp14": "runs/detect_experiment_14/checkpoints/best.pt",
    "exp35c": "runs/detect_experiment_35c/checkpoints/best.pt",
    "exp42": "runs/detect_experiment_42/checkpoints/best.pt",
}

N_MELS = 80

def _load_model(ckpt_path, device):
    """Auto-detect model type from checkpoint and load."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]
    state_keys = set(ckpt["model"].keys())

    if "event_presence_emb" in state_keys:
        model = EventEmbeddingDetector(
            d_model=ckpt_args["d_model"],
            n_layers=ckpt_args.get("enc_layers", 4) + ckpt_args.get("fusion_layers", 4),
            n_heads=ckpt_args["n_heads"],
            dropout=0.0,
        )
    elif any("fusion_layers." in k for k in state_keys) or any("gap_encoder." in k for k in state_keys):
        model = OnsetDetector(
            d_model=ckpt_args["d_model"], n_heads=ckpt_args["n_heads"],
            enc_layers=ckpt_args["enc_layers"],
            gap_enc_layers=ckpt_args.get("gap_enc_layers", 2),
            fusion_layers=ckpt_args.get("fusion_layers", 4),
            snippet_frames=ckpt_args.get("snippet_frames", 10),
            dropout=0.0,
        )
    else:
        # legacy model (exp 11-16)
        model = LegacyOnsetDetector(
            n_mels=N_MELS,
            d_model=ckpt_args.get("d_model", 384),
            d_event=ckpt_args.get("d_event", 128),
            enc_layers=ckpt_args.get("enc_layers", 4),
            enc_event_layers=ckpt_args.get("enc_event_layers", 2),
            audio_path_layers=ckpt_args.get("audio_path_layers", 2),
            n_heads=ckpt_args.get("n_heads", 8),
            n_classes=N_CLASSES,
            max_events=C_EVENTS,
            context_path_layers=ckpt_args.get("context_path_layers", 3),
        )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--subsample", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    ds_dir = os.path.join(SCRIPT_DIR, "datasets", args.dataset)
    with open(os.path.join(ds_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    random.seed(42)
    _, val_idx = split_by_song(manifest, val_ratio=0.1)
    val_ds = OnsetDataset(manifest, ds_dir, val_idx, augment=False,
                          subsample=args.subsample)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=args.workers)
    print(f"Val samples: {len(val_ds)}")

    out_dir = os.path.join(SCRIPT_DIR, "experiments", "experiment_43b")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for model_name, ckpt_rel_path in MODELS.items():
        ckpt_path = os.path.join(SCRIPT_DIR, ckpt_rel_path)
        if not os.path.exists(ckpt_path):
            print(f"\nWARNING: {ckpt_path} not found, skipping {model_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_name}: {ckpt_path}")
        print(f"{'='*60}")

        model, ckpt = _load_model(ckpt_path, args.device)
        print(f"  Model: {model.__class__.__name__}")
        print(f"  HIT: {ckpt['val_metrics'].get('hit_rate', 0):.1%}")

        results = run_benchmarks(model, val_loader, args.device,
                                 amp_enabled=False, multi_target=False)

        # extract AR results
        ar = results.get("autoregress", {})
        la = results.get("lightautoregress", {})

        all_results[model_name] = {
            "val_hit": ckpt["val_metrics"].get("hit_rate", 0),
            "autoregress": ar,
            "lightautoregress": la,
            "metronome_acc": results.get("metronome", {}).get("accuracy", 0),
            "no_events_acc": results.get("no_events", {}).get("accuracy", 0),
        }

        # print summary
        if ar:
            print(f"  AR: eHIT={ar.get('event_hit_rate',0):.1%} "
                  f"eMISS={ar.get('event_miss_rate',0):.1%} "
                  f"hall={ar.get('hallucination_rate',0):.1%} "
                  f"density_ratio={ar.get('density_ratio',0):.2f}")
        if la:
            curve = la.get("hit_curve", [])
            print(f"  LightAR: step0={la.get('step0_hit',0):.1%} "
                  f"curve: {' '.join(f'{h:.0%}' for h in curve[:12])} ...")

        del model
        torch.cuda.empty_cache()

    # comparison table
    print(f"\n{'='*80}")
    print(f"  AR RESILIENCE COMPARISON")
    print(f"{'='*80}")

    print(f"\n  {'Metric':30s}", end="")
    for m in all_results:
        print(f"  {m:>12s}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in all_results:
        print(f"  {'-'*12}", end="")
    print()

    metrics = [
        ("Val HIT (single)", lambda r: f"{r['val_hit']:.1%}"),
        ("Metronome benchmark", lambda r: f"{r['metronome_acc']:.1%}"),
        ("no_events benchmark", lambda r: f"{r['no_events_acc']:.1%}"),
        ("AR event HIT", lambda r: f"{r['autoregress'].get('event_hit_rate',0):.1%}"),
        ("AR event MISS", lambda r: f"{r['autoregress'].get('event_miss_rate',0):.1%}"),
        ("AR hallucination", lambda r: f"{r['autoregress'].get('hallucination_rate',0):.1%}"),
        ("AR density ratio", lambda r: f"{r['autoregress'].get('density_ratio',0):.2f}x"),
        ("AR survival@30", lambda r: f"{r['autoregress'].get('survival_30',0):.1%}"),
        ("LightAR step 0", lambda r: f"{r['lightautoregress'].get('step0_hit',0):.1%}"),
    ]

    # add per-step HIT for key steps
    for step in [1, 3, 5, 8, 15]:
        def make_fn(s):
            return lambda r: f"{r['lightautoregress'].get('hit_curve',[])[s]:.1%}" if s < len(r['lightautoregress'].get('hit_curve',[])) else "N/A"
        metrics.append((f"LightAR step {step}", make_fn(step)))

    for label, fn in metrics:
        row = f"  {label:30s}"
        for m, r in all_results.items():
            try:
                row += f"  {fn(r):>12s}"
            except (KeyError, IndexError):
                row += f"  {'N/A':>12s}"
        print(row)

    # light AR curves for plotting
    print(f"\n  Light AR HIT curves:")
    max_steps = 32
    for step in range(max_steps):
        row = f"  Step {step:2d}:"
        for m, r in all_results.items():
            curve = r["lightautoregress"].get("hit_curve", [])
            if step < len(curve):
                row += f"  {curve[step]:>6.1%}"
            else:
                row += f"  {'':>6s}"
        print(row)

    # save
    # strip numpy arrays for JSON
    def clean(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    with open(os.path.join(out_dir, "ar_comparison.json"), "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\n  Saved to {out_dir}/ar_comparison.json")

    # plot comparison curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"exp14": "#6bc46d", "exp35c": "#e8834a", "exp42": "#4a90d9"}

    # light AR HIT curves
    fig, ax = plt.subplots(figsize=(12, 5))
    for m, r in all_results.items():
        curve = r["lightautoregress"].get("hit_curve", [])
        if curve:
            ax.plot(range(len(curve)), curve, "o-", color=colors.get(m, "gray"),
                    linewidth=2, markersize=3, label=f"{m} (val HIT={r['val_hit']:.1%})")
    ax.set_xlabel("AR Step")
    ax.set_ylabel("HIT Rate")
    ax.set_title("Light AR: HIT Rate Degradation Over 32 Steps")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ar_hit_curves.png"), dpi=150)
    plt.close(fig)

    # unique preds over steps
    fig, ax = plt.subplots(figsize=(12, 5))
    for m, r in all_results.items():
        steps = r["lightautoregress"].get("steps", [])
        uniques = [s.get("unique_preds", 0) for s in steps if s.get("n_total", 0) > 0]
        if uniques:
            ax.plot(range(len(uniques)), uniques, "o-", color=colors.get(m, "gray"),
                    linewidth=2, markersize=3, label=m)
    ax.set_xlabel("AR Step")
    ax.set_ylabel("Unique Predictions")
    ax.set_title("Light AR: Prediction Diversity (Metronome Detection)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ar_unique_preds.png"), dpi=150)
    plt.close(fig)

    # pred mean vs target mean
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 5), sharey=True)
    if len(all_results) == 1:
        axes = [axes]
    for idx, (m, r) in enumerate(all_results.items()):
        ax = axes[idx]
        steps = r["lightautoregress"].get("steps", [])
        active = [s for s in steps if s.get("n_total", 0) > 0]
        if active:
            x = [s["step"] for s in active]
            ax.plot(x, [s.get("pred_mean", 0) for s in active], "o-",
                    color=colors.get(m, "gray"), linewidth=2, markersize=3, label="Pred mean")
            ax.plot(x, [s.get("target_mean", 0) for s in active], "o-",
                    color="#888", linewidth=2, markersize=3, label="Target mean")
        ax.set_title(f"{m}")
        ax.set_xlabel("AR Step")
        if idx == 0:
            ax.set_ylabel("Bin Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Light AR: Prediction Drift vs Ground Truth", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ar_pred_drift.png"), dpi=150)
    plt.close(fig)

    print(f"  Graphs saved to {out_dir}/")


if __name__ == "__main__":
    main()
