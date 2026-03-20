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

        # only run the AR benchmarks, not all ablation benchmarks
        # we need to call the AR portion directly
        from detection_train import _serializable
        import torch.nn.functional as F
        from tqdm import tqdm as _tqdm

        # collect batches (10% of val set)
        all_batches = []
        total_samples = 0
        target_samples = len(val_loader.dataset) // 10
        for batch in val_loader:
            all_batches.append(batch)
            total_samples += batch[0].size(0)
            if total_samples >= target_samples:
                break

        # run just the AR inference benchmark
        # (copy the AR logic from run_benchmarks)
        AR_STEPS = 32
        AR_MAX_SAMPLES = 1000

        def _is_hit_val(pred, target):
            if target >= N_CLASSES - 1:
                return pred == target
            fe = abs(pred - target)
            pe = abs((pred + 1) / (target + 1) - 1.0)
            return pe <= 0.03 or fe <= 1

        dataset = val_loader.dataset
        samples = []
        sample_idx = 0
        for batch in all_batches:
            mel, evt_off, evt_mask, cond, target = batch
            B = mel.size(0)
            for b in range(B):
                if sample_idx >= len(dataset.samples):
                    break
                valid = ~evt_mask[b]
                if valid.sum() < 4:
                    sample_idx += 1
                    continue
                ci, ei = dataset.samples[sample_idx]
                evt = dataset.events[ci]
                if ei == 0:
                    cursor_bin = max(0, int(evt[0]) - B_BINS) if len(evt) > 0 else 0
                else:
                    cursor_bin = int(evt[ei - 1])
                future_bins = evt[evt > cursor_bin] - cursor_bin
                future_bins = future_bins[future_bins < B_BINS].astype(np.int64)
                if len(future_bins) < 2:
                    sample_idx += 1
                    continue
                samples.append({
                    "mel": mel[b], "evt_off": evt_off[b],
                    "evt_mask": evt_mask[b], "cond": cond[b],
                    "gt_abs": future_bins, "density_cond": cond[b, 0].item(),
                })
                sample_idx += 1
                if len(samples) >= AR_MAX_SAMPLES:
                    break
            if len(samples) >= AR_MAX_SAMPLES:
                break

        print(f"  AR samples: {len(samples)}")

        # run AR
        per_step_preds = [[] for _ in range(AR_STEPS)]
        per_step_entropy = [[] for _ in range(AR_STEPS)]
        per_step_survived = np.zeros(AR_STEPS)
        light_hit = np.zeros(AR_STEPS)
        light_total = np.zeros(AR_STEPS)
        light_preds = [[] for _ in range(AR_STEPS)]
        light_targets = [[] for _ in range(AR_STEPS)]
        all_predicted_sets = []
        all_gt_sets = []
        density_conds = []
        density_actuals = []

        for sample in _tqdm(samples, desc="  AR bench", leave=False):
            mel_s = sample["mel"].unsqueeze(0).to(args.device)
            evt_off_s = sample["evt_off"].unsqueeze(0).clone().to(args.device)
            evt_mask_s = sample["evt_mask"].unsqueeze(0).clone().to(args.device)
            cond_s = sample["cond"].unsqueeze(0).to(args.device)
            gt_abs = sample["gt_abs"]
            density_conds.append(sample["density_cond"])
            cursor = 0
            predicted_positions = []

            for step in range(AR_STEPS):
                with torch.no_grad():
                    logits = model(mel_s, evt_off_s, evt_mask_s, cond_s)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    probs = torch.softmax(logits.float(), dim=1)
                    pred = logits.argmax(dim=1).item()
                    ent = -(probs * (probs + 1e-10).log()).sum(dim=1).item()

                if pred >= N_CLASSES - 1:
                    break

                per_step_survived[step] += 1
                per_step_preds[step].append(pred)
                per_step_entropy[step].append(ent)
                abs_pos = cursor + pred
                predicted_positions.append(abs_pos)

                if step < len(gt_abs):
                    gt_target = int(gt_abs[step]) - cursor
                    if gt_target > 0:
                        light_total[step] += 1
                        light_preds[step].append(pred)
                        light_targets[step].append(gt_target)
                        if _is_hit_val(pred, gt_target):
                            light_hit[step] += 1

                cursor = abs_pos
                evt_off_s = evt_off_s - pred
                evt_off_np = evt_off_s[0].cpu().numpy()
                evt_mask_np = evt_mask_s[0].cpu().numpy()
                evt_off_np = np.roll(evt_off_np, -1)
                evt_mask_np = np.roll(evt_mask_np, -1)
                evt_off_np[-1] = 0
                evt_mask_np[-1] = False
                evt_off_s = torch.from_numpy(evt_off_np).unsqueeze(0).to(args.device)
                evt_mask_s = torch.from_numpy(evt_mask_np).unsqueeze(0).to(args.device)

            all_predicted_sets.append(np.array(predicted_positions))
            all_gt_sets.append(gt_abs)
            if len(predicted_positions) >= 2:
                total_bins = predicted_positions[-1]
                total_s = total_bins * 4.989 / 1000
                if total_s > 0:
                    density_actuals.append(len(predicted_positions) / total_s)

        # set matching
        total_gt = sum(len(g) for g in all_gt_sets)
        total_pred = sum(len(p) for p in all_predicted_sets)
        event_hit = 0; event_matched = 0; hallucinations = 0
        for pred_set, gt_set in zip(all_predicted_sets, all_gt_sets):
            if len(pred_set) == 0 or len(gt_set) == 0:
                hallucinations += len(pred_set)
                continue
            used_gt = set()
            for p in pred_set:
                best_dist = float("inf"); best_gi = -1
                for gi, g in enumerate(gt_set):
                    if gi in used_gt: continue
                    d = abs(int(p) - int(g))
                    if d < best_dist: best_dist = d; best_gi = gi
                if best_gi >= 0 and best_dist <= 500:
                    used_gt.add(best_gi)
                    if _is_hit_val(int(p), int(gt_set[best_gi])):
                        event_hit += 1
                    event_matched += 1
                else:
                    hallucinations += 1

        ar = {
            "n_samples": len(samples), "total_gt_onsets": total_gt,
            "total_predicted": total_pred,
            "event_hit_rate": event_hit / max(total_gt, 1),
            "event_miss_rate": (total_gt - event_matched) / max(total_gt, 1),
            "hallucination_rate": hallucinations / max(total_pred, 1),
            "pred_per_sample": total_pred / max(len(samples), 1),
            "gt_per_sample": total_gt / max(len(samples), 1),
        }
        if per_step_survived[0] > 0:
            ar["survival_10"] = per_step_survived[min(9, AR_STEPS-1)] / per_step_survived[0]
            ar["survival_30"] = per_step_survived[min(29, AR_STEPS-1)] / per_step_survived[0]
        if density_conds and density_actuals:
            ar["density_conditioned_mean"] = np.mean(density_conds)
            ar["density_actual_mean"] = np.mean(density_actuals)
            ar["density_ratio"] = np.mean(density_actuals) / max(np.mean(density_conds), 0.01)

        # light AR result
        la_steps = []
        hit_rates = []
        for s in range(AR_STEPS):
            si = {"step": s, "n_total": int(light_total[s]),
                  "hit_rate": float(light_hit[s] / max(light_total[s], 1))}
            if light_preds[s]:
                p_arr = np.array(light_preds[s])
                t_arr = np.array(light_targets[s])
                si["pred_mean"] = float(p_arr.mean())
                si["pred_std"] = float(p_arr.std())
                si["target_mean"] = float(t_arr.mean())
                si["frame_err_mean"] = float(np.abs(p_arr - t_arr).mean())
                si["unique_preds"] = int(len(np.unique(p_arr)))
                si["pred_min"] = int(p_arr.min())
                si["pred_max"] = int(p_arr.max())
            la_steps.append(si)
            hit_rates.append(si["hit_rate"])
        la = {"n_samples": len(samples), "steps": la_steps,
              "hit_curve": hit_rates, "step0_hit": hit_rates[0] if hit_rates else 0}

        all_results[model_name] = {
            "val_hit": ckpt["val_metrics"].get("hit_rate", 0),
            "autoregress": ar, "lightautoregress": la,
        }

        print(f"  AR: eHIT={ar['event_hit_rate']:.1%} eMISS={ar['event_miss_rate']:.1%} "
              f"hall={ar['hallucination_rate']:.1%}")
        print(f"  LightAR: step0={la['step0_hit']:.1%} "
              f"curve: {' '.join(f'{h:.0%}' for h in hit_rates[:12])} ...")

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
