"""Compare quality scores of generated charts vs ground truth using the evaluator.

Loads AR eval outputs (CSVs) and ground truth events, scores both with the evaluator,
and reports whether the model ranks real charts higher than generated ones.

Usage:
    # Score using phase 1 (corruption-only) model
    python classifier_eval_ar.py \
        --checkpoint runs/eval_experiment_66_1/checkpoints/best.pt \
        --ar-dir experiments/experiment_62/ar_eval/detect_experiment_62_best \
        --regime song_density

    # Compare p1 vs p2
    python classifier_eval_ar.py \
        --checkpoint runs/eval_experiment_66_1/checkpoints/best.pt \
        --checkpoint2 runs/eval_experiment_66_1_p2/checkpoints/best.pt \
        --ar-dir experiments/experiment_62/ar_eval/detect_experiment_62_best \
        --regime song_density
"""
import os
import json
import csv
import argparse
import numpy as np
import torch
from tqdm import tqdm

from scipy.stats import spearmanr

from classifier_model import ChartQualityEvaluator
from analyze_ar import compute_gt_metrics, compute_tn_metrics, compute_pattern_metrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "datasets", "taiko_v2")
WINDOW_FRAMES = 2000
MAX_EVENTS = 256
BIN_MS = 5.0


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = ChartQualityEvaluator(
        d_model=ckpt_args.get("d_model", 256),
        n_layers=ckpt_args.get("n_layers", 6),
        n_heads=ckpt_args.get("n_heads", 8),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    eval_step = ckpt.get("eval_step", "?")
    return model, eval_step


def load_csv_events(csv_path):
    """Load predicted events from AR CSV → bin indices."""
    events = []
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("time_ms"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 1:
                try:
                    time_ms = float(parts[0])
                    events.append(int(round(time_ms / BIN_MS)))
                except ValueError:
                    continue
    return np.array(events, dtype=np.int64)


def score_chart(model, mel, events, star_rating, device, n_windows=8):
    """Score a chart over multiple windows. Returns mean score."""
    total_frames = mel.shape[1]
    if total_frames <= WINDOW_FRAMES:
        starts = [0]
    else:
        starts = np.linspace(0, total_frames - WINDOW_FRAMES, n_windows, dtype=int)
        starts = sorted(set(starts))

    scores = []
    with torch.no_grad():
        for start in starts:
            end = start + WINDOW_FRAMES
            mel_w = mel[:, start:min(total_frames, end)].astype(np.float32)
            if mel_w.shape[1] < WINDOW_FRAMES:
                mel_w = np.pad(mel_w, ((0, 0), (0, WINDOW_FRAMES - mel_w.shape[1])))

            mask = (events >= start) & (events < end)
            evt_w = events[mask].astype(np.int64) - start
            n_evt = min(len(evt_w), MAX_EVENTS)
            evt_arr = np.zeros(MAX_EVENTS, dtype=np.int64)
            evt_mask = np.ones(MAX_EVENTS, dtype=bool)
            if n_evt > 0:
                evt_arr[:n_evt] = evt_w[:n_evt]
                evt_mask[:n_evt] = False

            mel_t = torch.from_numpy(mel_w).unsqueeze(0).to(device)
            evt_t = torch.from_numpy(evt_arr).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(evt_mask).unsqueeze(0).to(device)
            star_t = torch.tensor([star_rating], dtype=torch.float32, device=device)

            s = model(mel_t, evt_t, mask_t, star_t).item()
            scores.append(s)

    return float(np.mean(scores))


def run_eval(model, model_name, songs, ar_csv_dir, device, n_windows=8):
    """Score GT and generated charts for all songs. Returns list of result dicts."""
    mel_dir = os.path.join(DATASET_DIR, "mels")
    evt_dir = os.path.join(DATASET_DIR, "events")

    # need to find mel file for each song
    with open(os.path.join(DATASET_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    bset_to_mel = {}
    bset_to_star = {}
    for c in manifest["charts"]:
        bset = c["beatmapset_id"]
        if bset not in bset_to_mel:
            bset_to_mel[bset] = c["mel_file"]
            bset_to_star[bset] = c.get("star_rating", 4.0)

    results = []
    gt_wins = 0
    gen_wins = 0
    ties = 0

    for song in tqdm(songs, desc=f"Scoring ({model_name})"):
        bset = song["beatmapset_id"]
        sname = song["safe_name"]

        # find mel
        mel_file = bset_to_mel.get(bset)
        if mel_file is None:
            continue
        mel = np.load(os.path.join(mel_dir, mel_file), mmap_mode="r")

        # ground truth events
        gt_events = np.load(os.path.join(evt_dir, song["event_file"]))

        # generated events
        csv_path = os.path.join(ar_csv_dir, f"{sname}_predicted.csv")
        if not os.path.exists(csv_path):
            continue
        gen_events = load_csv_events(csv_path)

        if len(gen_events) == 0:
            continue

        star = bset_to_star.get(bset, 4.0)

        gt_score = score_chart(model, mel, gt_events, star, device, n_windows)
        gen_score = score_chart(model, mel, gen_events, star, device, n_windows)

        diff = gt_score - gen_score
        if gt_score > gen_score:
            gt_wins += 1
        elif gen_score > gt_score:
            gen_wins += 1
        else:
            ties += 1

        # compute analyze_ar metrics on generated chart
        BIN_MS_AR = 4.9887
        gt_ms = gt_events.astype(np.float64) * BIN_MS_AR
        gen_ms = gen_events.astype(np.float64) * BIN_MS

        gt_metrics = compute_gt_metrics(gen_ms, gt_ms)
        tn_metrics = compute_tn_metrics(gen_ms, gt_ms, np.random.default_rng(42))
        pattern_metrics = compute_pattern_metrics(gen_ms)

        entry = {
            "song": f"{song['artist']} - {song['title']}",
            "bset": bset,
            "gt_score": gt_score,
            "gen_score": gen_score,
            "diff": diff,
            "gt_events": len(gt_events),
            "gen_events": len(gen_events),
            "gt_wins": gt_score > gen_score,
        }
        if gt_metrics:
            entry.update({f"gt_{k}": v for k, v in gt_metrics.items()})
        if tn_metrics:
            entry.update({f"tn_{k}": v for k, v in tn_metrics.items()})
        if pattern_metrics:
            entry.update({f"pat_{k}": v for k, v in pattern_metrics.items()})

        results.append(entry)

    return results, gt_wins, gen_wins, ties


def print_results(results, gt_wins, gen_wins, ties, model_name, regime):
    total = gt_wins + gen_wins + ties
    print(f"\n{'='*70}")
    print(f"Model: {model_name} | Regime: {regime} | Songs: {total}")
    print(f"{'='*70}")
    print(f"  GT wins:  {gt_wins}/{total} ({gt_wins/max(total,1):.1%})")
    print(f"  Gen wins: {gen_wins}/{total} ({gen_wins/max(total,1):.1%})")
    if ties:
        print(f"  Ties:     {ties}/{total}")

    gt_scores = [r["gt_score"] for r in results]
    gen_scores = [r["gen_score"] for r in results]
    diffs = [r["diff"] for r in results]

    print(f"\n  GT  mean={np.mean(gt_scores):+.2f}, std={np.std(gt_scores):.2f}")
    print(f"  Gen mean={np.mean(gen_scores):+.2f}, std={np.std(gen_scores):.2f}")
    print(f"  Diff mean={np.mean(diffs):+.2f} (positive = GT better)")

    # ── correlation of gen_score with all metrics ──
    metric_keys = [
        # GT matching
        ("gt_matched_rate", "matched (<=25ms)"),
        ("gt_close_rate", "close (<=50ms)"),
        ("gt_far_rate", "far (>100ms)"),
        ("gt_hallucination_rate", "hallucination"),
        ("gt_gt_error_mean", "GT error mean"),
        ("gt_density_ratio", "density ratio"),
        # TaikoNation
        ("tn_over_pspace", "Over. P-Space"),
        ("tn_hi_pspace", "HI P-Space"),
        ("tn_dc_human", "DCHuman"),
        ("tn_oc_human", "OCHuman"),
        ("tn_dc_rand", "DCRand"),
        # Pattern variety
        ("pat_gap_std", "gap_std"),
        ("pat_gap_cv", "gap_cv"),
        ("pat_gap_entropy", "gap_entropy"),
        ("pat_dominant_gap_pct", "dominant_gap%"),
        ("pat_max_metro_streak", "metro_streak"),
        ("pat_max_metro_streak_pct", "metro_streak%"),
        ("pat_density", "density"),
    ]

    gen_scores = np.array([r["gen_score"] for r in results])
    print(f"\nCorrelation of gen_score with AR metrics:")
    print(f"  {'Metric':<25s} {'Spearman':>10s} {'p-value':>12s} {'Meaning if positive':>30s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*30}")
    for key, label in metric_keys:
        vals = [r.get(key) for r in results]
        if all(v is not None for v in vals) and len(vals) >= 5:
            arr = np.array(vals, dtype=np.float64)
            if np.std(arr) > 1e-10:
                rho, p = spearmanr(gen_scores, arr)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {label:<25s} {rho:+10.3f} {p:12.2e} {sig:>3s}")

    # same for gt_score - gen_score diff
    diffs_arr = np.array(diffs)
    print(f"\nCorrelation of (GT - Gen) diff with AR metrics:")
    print(f"  {'Metric':<25s} {'Spearman':>10s} {'p-value':>12s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12}")
    for key, label in metric_keys:
        vals = [r.get(key) for r in results]
        if all(v is not None for v in vals) and len(vals) >= 5:
            arr = np.array(vals, dtype=np.float64)
            if np.std(arr) > 1e-10:
                rho, p = spearmanr(diffs_arr, arr)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {label:<25s} {rho:+10.3f} {p:12.2e} {sig:>3s}")

    # per-song breakdown
    results_sorted = sorted(results, key=lambda r: r["diff"], reverse=True)
    print(f"\nPer-song breakdown (sorted by GT advantage):")
    print(f"  {'Song':<50s} {'GT':>7s} {'Gen':>7s} {'Diff':>7s} {'#GT':>5s} {'#Gen':>5s} {'close':>6s} {'hall':>6s}")
    print(f"  {'-'*50} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*6} {'-'*6}")
    for r in results_sorted:
        marker = "+" if r["gt_wins"] else "-"
        close = r.get("gt_close_rate", 0)
        hall = r.get("gt_hallucination_rate", 0)
        print(f"  {r['song'][:50]:<50s} {r['gt_score']:+7.2f} {r['gen_score']:+7.2f} "
              f"{r['diff']:+7.2f}{marker} {r['gt_events']:5d} {r['gen_events']:5d} "
              f"{close:5.1%} {hall:5.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate quality of AR-generated charts")
    parser.add_argument("--checkpoint", required=True, help="Primary model checkpoint")
    parser.add_argument("--checkpoint2", default=None, help="Second model for comparison")
    parser.add_argument("--ar-dir", required=True, help="AR eval directory (contains songs.json + csvs/)")
    parser.add_argument("--regime", default="song_density", help="Density regime subdirectory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ar_dir = os.path.join(SCRIPT_DIR, args.ar_dir) if not os.path.isabs(args.ar_dir) else args.ar_dir
    songs_path = os.path.join(ar_dir, "songs.json")
    csv_dir = os.path.join(ar_dir, "csvs", args.regime)

    with open(songs_path) as f:
        songs = json.load(f)
    print(f"Songs: {len(songs)}, Regime: {args.regime}")
    print(f"CSVs: {csv_dir}")

    # model 1
    model1, step1 = load_model(args.checkpoint, args.device)
    name1 = f"model_1 (eval {step1})"
    print(f"\n{name1}: {args.checkpoint}")
    results1, gw1, gew1, t1 = run_eval(model1, name1, songs, csv_dir, args.device, args.n_windows)
    print_results(results1, gw1, gew1, t1, name1, args.regime)

    # model 2 (optional)
    if args.checkpoint2:
        model2, step2 = load_model(args.checkpoint2, args.device)
        name2 = f"model_2 (eval {step2})"
        print(f"\n{name2}: {args.checkpoint2}")
        results2, gw2, gew2, t2 = run_eval(model2, name2, songs, csv_dir, args.device, args.n_windows)
        print_results(results2, gw2, gew2, t2, name2, args.regime)

        # comparison
        print(f"\n{'='*70}")
        print(f"COMPARISON: {name1} vs {name2}")
        print(f"{'='*70}")
        print(f"  GT win rate:  {gw1}/{gw1+gew1+t1} ({gw1/max(gw1+gew1+t1,1):.1%}) vs "
              f"{gw2}/{gw2+gew2+t2} ({gw2/max(gw2+gew2+t2,1):.1%})")
        d1 = np.mean([r["diff"] for r in results1])
        d2 = np.mean([r["diff"] for r in results2])
        print(f"  Mean diff:    {d1:+.2f} vs {d2:+.2f}")

    # save
    output = args.output or "ar_quality_eval.json"
    out_data = {"regime": args.regime, "model_1": args.checkpoint, "results_1": results1}
    if args.checkpoint2:
        out_data["model_2"] = args.checkpoint2
        out_data["results_2"] = results2
    with open(output, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
